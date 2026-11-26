"""Loop principal de treinamento (AdamW AN-25-103 Seção 5.3.4/5.3.6) e persistência de checkpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tensorflow import keras

from src.data.preprocessing import FeatureScaler, PreparedData
from src.data.sampler import QuotaBatchGenerator
from src.models.affine_dnn import build_model_from_config, load_saved_model
from src.training.callbacks import LogLRCallback
from src.utils.logging_utils import resolve_keras_verbose
from src.utils.paths import OutputPaths
from src.utils.seed import set_all_seeds


def create_generator_for_model(
    cfg: dict[str, Any],
    model_key: str,
    data: PreparedData,
    seed: int | None = None,
    steps_per_epoch: int | None = None,
) -> QuotaBatchGenerator:
    """Cria o `QuotaBatchGenerator` respeitando `train_kappas` (null = todos os κ_λ; lista = subconjunto)."""
    mcfg = cfg["models"][model_key]
    scfg = cfg["sampling"]
    tcfg = cfg["training"]
    train_df = data.splits["train"]

    train_kappas = mcfg.get("train_kappas")
    if train_kappas is None:
        rows = None
        kgrid = cfg["physics"]["kappa_grid"]
    else:
        kl_tr = train_df["kappa_lambda"].to_numpy(np.float64)
        rows = np.where(
            (train_df["label"].to_numpy() != 2)
            | np.isin(np.round(kl_tr, 3), np.round(train_kappas, 3))
        )[0]
        kgrid = [float(k) for k in train_kappas]

    return QuotaBatchGenerator(
        df=train_df,
        x_cont=data.x_cont["train"],
        x_cat=data.x_cat["train"],
        rows=rows,
        batch_size=tcfg["batch_size"],
        steps_per_epoch=int(steps_per_epoch or tcfg["steps_per_epoch"]),
        class_fraction=scfg["class_batch_fraction"],
        quota_mode=scfg["quota_weight_mode"],
        min_one=scfg["quota_min_one"],
        kappa_grid=kgrid,
        kappa_norm=cfg["physics"]["kappa_norm"],
        bkg_kappa_distribution=scfg.get("bkg_kappa_distribution", "identical_sampled"),
        seed=int(cfg["seed"] if seed is None else seed),
    )


def build_optimizer_and_callbacks(
    model: keras.Model,
    backbone: str,
    steps_per_epoch: int,
    epochs: int,
    learning_rate: float = 0.1,
    lr_halve_every_epochs: int = 2,
    an_decay_factor: float = 500.0,
    use_cosine: bool = False,
    es_patience: int = 10,
    lr_patience: int = 2,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    extra_callbacks: list[keras.callbacks.Callback] | None = None,
    verbose: bool = True,
) -> tuple[keras.optimizers.Optimizer, list[keras.callbacks.Callback], dict[str, Any]]:
    """Configura otimizador e callbacks conforme AN-25-103 Seção 5.3.4 / 5.3.6."""
    opt_info: dict[str, Any] = {"backbone": backbone, "use_cosine": use_cosine}

    if backbone == "an_densenet" and not use_cosine:
        dense_layers = [
            l for l in model.layers if isinstance(l, keras.layers.Dense) and l.name.startswith("an_")
        ]
        n_lin = int(sum(int(np.prod(l.kernel.shape)) for l in dense_layers))
        wd = float(an_decay_factor) / max(n_lin, 1)
        if verbose:
            print(
                f"AdamW (AN 5.3.4): weight_decay = {an_decay_factor:g}/{n_lin:,d} = {wd:.3e} "
                "(aplicado apenas nos kernels Dense `an_*` do backbone)"
            )
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=float(learning_rate),
            decay_steps=int(lr_halve_every_epochs) * int(steps_per_epoch),
            decay_rate=0.5,
            staircase=True,
        )
        optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=wd)
        decayed = {id(l.kernel) for l in dense_layers}
        optimizer.exclude_from_weight_decay(
            var_list=[v for v in model.trainable_variables if id(v) not in decayed]
        )
        callbacks: list[keras.callbacks.Callback] = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(es_patience),
                restore_best_weights=True,
            ),
            LogLRCallback(),
        ]
        opt_info.update({"optimizer": "AdamW", "n_linear_weights": n_lin, "weight_decay": wd})
    elif use_cosine:
        total_steps = int(epochs) * int(steps_per_epoch)
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=float(learning_rate),
            decay_steps=total_steps,
            alpha=float(min_lr) / float(learning_rate),
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(es_patience),
                restore_best_weights=True,
            ),
            LogLRCallback(),
        ]
        opt_info.update({"optimizer": "Adam+CosineDecay"})
    else:
        optimizer = keras.optimizers.Adam(learning_rate=float(learning_rate))
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(es_patience),
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=float(lr_factor),
                patience=int(lr_patience),
                min_lr=float(min_lr),
            ),
            LogLRCallback(),
        ]
        opt_info.update({"optimizer": "Adam+ReduceLROnPlateau"})

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    return optimizer, callbacks, opt_info


def train_model(
    cfg: dict[str, Any],
    model_key: str,
    generator: QuotaBatchGenerator,
    scaler: FeatureScaler,
    cat_vocab_sizes: dict[str, int],
    val_data: tuple,
    epochs: int | None = None,
    seed: int | None = None,
    verbose: int | str | None = None,
    name: str | None = None,
    extra_callbacks: list[keras.callbacks.Callback] | None = None,
) -> tuple[keras.Model, dict[str, list[float]], dict[str, Any]]:
    """Constrói, compila e treina um modelo devolvendo `(model, history_dict, metadata)`."""
    tcfg = cfg["training"]
    mcfg = cfg["models"][model_key]
    used_seed = int(cfg["seed"] if seed is None else seed)
    used_epochs = int(tcfg["epochs"] if epochs is None else epochs)
    keras_verbose = resolve_keras_verbose(tcfg.get("verbose", "auto") if verbose is None else verbose)
    display_name = name or mcfg.get("name", model_key)

    set_all_seeds(used_seed)
    model = build_model_from_config(cfg, model_key, scaler, cat_vocab_sizes)

    optimizer, callbacks, opt_info = build_optimizer_and_callbacks(
        model=model,
        backbone=mcfg.get("backbone", "an_densenet"),
        steps_per_epoch=len(generator),
        epochs=used_epochs,
        learning_rate=tcfg["learning_rate"],
        lr_halve_every_epochs=tcfg["lr_halve_every_epochs"],
        an_decay_factor=tcfg["an_decay_factor"],
        use_cosine=tcfg.get("use_cosine_decay", False),
        es_patience=tcfg["es_patience"],
        lr_patience=tcfg["lr_patience"],
        lr_factor=tcfg["lr_factor"],
        min_lr=tcfg["min_lr"],
        extra_callbacks=extra_callbacks,
        verbose=(keras_verbose > 0),
    )

    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")
    n_params = int(model.count_params())

    if keras_verbose > 0:
        print(
            f"\n=== Treinando {display_name} (kind={mcfg['kind']}, "
            f"backbone={mcfg.get('backbone', 'an_densenet')}, params={n_params:,d}, seed={used_seed}) ==="
        )

    val_clean = val_data if (len(val_data) == 2 or val_data[2] is not None) else val_data[:2]
    eval_bs = int(cfg["evaluation"]["eval_batch_size"])

    t0 = time.time()
    history_obj = model.fit(
        generator,
        validation_data=val_clean,
        validation_batch_size=eval_bs,
        epochs=used_epochs,
        callbacks=callbacks,
        verbose=keras_verbose,
    )
    elapsed_s = float(time.time() - t0)

    hist_dict = {k: [float(x) for x in vals] for k, vals in history_obj.history.items()}
    val_losses = np.asarray(hist_dict.get("val_loss", [np.nan]), dtype=np.float64)
    best_idx = int(np.nanargmin(val_losses))

    meta = {
        "name": display_name,
        "model_key": model_key,
        "kind": mcfg["kind"],
        "backbone": mcfg.get("backbone", "an_densenet"),
        "seed": used_seed,
        "n_params": n_params,
        "epochs_requested": used_epochs,
        "epochs_ran": len(val_losses),
        "steps_per_epoch": len(generator),
        "best_epoch": best_idx + 1,
        "best_val_loss": float(val_losses[best_idx]),
        "elapsed_seconds": elapsed_s,
        "optimizer_info": opt_info,
    }
    return model, hist_dict, meta


def save_model_bundle(
    output_paths: OutputPaths,
    model_name: str,
    model: keras.Model,
    history_dict: dict[str, list[float]],
    meta_dict: dict[str, Any],
) -> dict[str, Path]:
    """Salva `.keras`, histórico CSV/JSON e metadados em `outputs/checkpoints/`."""
    ckpt = output_paths.checkpoints
    ckpt.mkdir(parents=True, exist_ok=True)

    model_path = ckpt / f"model_{model_name}.keras"
    hist_csv_path = ckpt / f"history_{model_name}.csv"
    hist_json_path = ckpt / f"history_{model_name}.json"
    meta_path = ckpt / f"meta_{model_name}.json"

    model.save(model_path)
    df_hist = pd.DataFrame(history_dict)
    df_hist.index.name = "epoch_0based"
    df_hist.to_csv(hist_csv_path)

    with open(hist_json_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    return {
        "model": model_path,
        "history_csv": hist_csv_path,
        "history_json": hist_json_path,
        "meta": meta_path,
    }


def load_model_bundle(
    checkpoints_dir: str | Path,
    model_name: str,
) -> tuple[keras.Model, dict[str, list[float]] | None, dict[str, Any] | None]:
    """Carrega modelo `.keras` salvo (Condor ou Notebook) junto com seu histórico e metadados."""
    ckpt = Path(checkpoints_dir)
    candidates = [
        ckpt / f"model_{model_name}.keras",
        ckpt / f"{model_name}.keras",
    ]
    # Fallback de compatibilidade para nomes legados do notebook v3_6
    if model_name == "AffinePNN":
        candidates.append(ckpt / "model_A_quotas.keras")

    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            f"Checkpoint do modelo '{model_name}' não encontrado em {ckpt}. "
            f"Tentados: {[p.name for p in candidates]}"
        )

    model = load_saved_model(model_path)

    hist_json = ckpt / f"history_{model_name}.json"
    hist_csv = ckpt / f"history_{model_name}.csv"
    history = None
    if hist_json.exists():
        with open(hist_json, "r", encoding="utf-8") as f:
            history = json.load(f)
    elif hist_csv.exists():
        df_h = pd.read_csv(hist_csv, index_col=0)
        history = {col: df_h[col].tolist() for col in df_h.columns}

    meta_path = ckpt / f"meta_{model_name}.json"
    meta = None
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return model, history, meta
