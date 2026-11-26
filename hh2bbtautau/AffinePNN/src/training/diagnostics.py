"""Diagnósticos profundos pós-treino (Célula 54 e Seção 13.4):
pesos de validação, caudas de CE, BatchNorm móvel vs mediana, outliers do LBN, memorização e random-label test."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from tensorflow import keras

from src.data.preprocessing import PreparedData, make_inputs
from src.data.sampler import make_random_label_generator
from src.data.weights import ess
from src.training.losses import ce_from_logits, loss_tail_table, predict_logits


def bn_layers(model: keras.Model) -> list[keras.layers.BatchNormalization]:
    return [l for l in model.layers if isinstance(l, keras.layers.BatchNormalization)]


def snapshot_bn(model: keras.Model) -> list[list[np.ndarray]]:
    return [[keras.ops.convert_to_numpy(v).copy() for v in l.weights] for l in bn_layers(model)]


def restore_bn(model: keras.Model, snap: list[list[np.ndarray]]) -> None:
    for l, s in zip(bn_layers(model), snap):
        for v, a in zip(l.weights, s):
            v.assign(a)


def median_bn(model: keras.Model, next_batch: Callable[[], dict[str, np.ndarray]], n_batches: int = 200) -> None:
    """Recalibra temporariamente moving_mean e moving_variance das BNs usando a mediana de `n_batches`."""
    bns = bn_layers(model)
    if not bns:
        return
    probe = keras.Model(model.inputs, [l.input for l in bns])
    mus = [[] for _ in bns]
    vs = [[] for _ in bns]
    for _ in range(int(n_batches)):
        outs = probe(next_batch(), training=True)
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        for j, o in enumerate(outs):
            arr = keras.ops.convert_to_numpy(o)
            mus[j].append(arr.mean(axis=0))
            vs[j].append(arr.var(axis=0))
    for l, mu, v in zip(bns, mus, vs):
        l.moving_mean.assign(np.median(np.stack(mu), axis=0).astype(np.float32))
        l.moving_variance.assign(np.median(np.stack(v), axis=0).astype(np.float32))


def run_validation_diagnostics(
    models: dict[str, keras.Model],
    histories: dict[str, dict[str, list[float]]] | None,
    data: PreparedData,
    val_inputs: dict[str, np.ndarray],
    val_freq_w: np.ndarray,
    gen_ref: object,
    continuous_features: list[str],
    class_labels: dict[int, str],
    kappa_norm: float = 5.0,
    eval_batch_size: int = 8192,
    n_train_per_pool: int = 3000,
    n_bn_batches: int = 200,
    verbose: bool = True,
) -> dict[str, Any]:
    """Executa todas as 6 baterias de diagnóstico da validação pós-treino e devolve tabelas estruturadas."""
    val_df = data.splits["val"]
    train_df = data.splits["train"]
    x_cont_val = data.x_cont["val"]
    x_cont_train = data.x_cont["train"]
    x_cat_train = data.x_cat["train"]

    val_sample = val_df["sample"].astype(str).to_numpy()
    val_era = val_df["era"].astype(str).to_numpy()
    val_pool = np.char.add(np.char.add(val_sample, "|"), val_era)
    yv = val_df["label"].to_numpy().astype(int)
    wv = np.asarray(val_freq_w, dtype=np.float64)
    kv = val_df["kappa_eval"].to_numpy(np.float32)
    n_val = len(yv)

    def _hdr(title: str) -> None:
        if verbose:
            print("\n" + "=" * 96 + f"\n{title}\n" + "=" * 96)

    # 1) Pesos de validação
    _hdr("1) PESOS DE VALIDAÇÃO (val_freq_w)")
    srt = np.sort(wv)[::-1]
    weight_concentration = {
        f"top_{k}": float(100.0 * srt[:k].sum() / wv.sum()) for k in (1, 10, 100, 1000) if k <= n_val
    }
    cls_rows = []
    for lab in np.unique(yv):
        m = yv == lab
        cls_rows.append(
            {
                "classe": class_labels.get(int(lab), str(lab)),
                "n": int(m.sum()),
                "soma_w": float(wv[m].sum()),
                "w_max": float(wv[m].max()),
                "ESS_over_n": float(ess(wv[m]) / m.sum()),
            }
        )
    val_weight_cls_df = pd.DataFrame(cls_rows)

    pool_tab = (
        pd.DataFrame({"pool": val_pool, "label": yv, "w": wv})
        .groupby("pool")
        .agg(label=("label", "first"), n=("w", "size"), soma_w=("w", "sum"), w_max=("w", "max"), w_medio=("w", "mean"))
        .sort_values("soma_w", ascending=False)
    )
    pool_tab["frac_peso_%"] = 100.0 * pool_tab["soma_w"] / wv.sum()

    if verbose:
        print(
            f"N={n_val:,} | média={wv.mean():.4f} | max={wv.max():.1f} | "
            f"p99={np.percentile(wv, 99):.2f} | p99.9={np.percentile(wv, 99.9):.2f} | ESS/N={ess(wv) / n_val:.3f}"
        )
        print(val_weight_cls_df.to_string(index=False))
        print("\nTop 10 pools por peso total na validação:")
        print(pool_tab.head(10).to_string())

    # 2) CE por evento e caudas
    ce_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    loss_tables: dict[str, pd.Series] = {}
    top_events_by_model: dict[str, pd.DataFrame] = {}
    top_pools_by_model: dict[str, pd.DataFrame] = {}

    for tag, m in models.items():
        _hdr(f"2) CE POR EVENTO — {tag}")
        lg = predict_logits(m, val_inputs, batch_size=eval_batch_size)
        ce = ce_from_logits(lg, yv)
        ce_cache[tag] = (lg, ce)

        lt = loss_tail_table(ce, wv, tag)
        loss_tables[tag] = lt

        contrib = wv * ce / n_val
        o = np.argsort(-contrib)[:15]
        top_ev = pd.DataFrame(
            {
                "sample": val_sample[o],
                "era": val_era[o],
                "label": yv[o],
                "kappa": kv[o],
                "w": wv[o],
                "CE": ce[o],
                "contrib": contrib[o],
                "max_abs_logit": np.abs(lg[o]).max(axis=1),
            }
        )
        top_events_by_model[tag] = top_ev

        tp = (
            pd.DataFrame({"pool": val_pool, "c": contrib, "ce": ce, "w": wv})
            .groupby("pool")
            .agg(contrib=("c", "sum"), ce_medio=("ce", "mean"), ce_max=("ce", "max"), w_medio=("w", "mean"), n=("c", "size"))
            .sort_values("contrib", ascending=False)
        )
        tp["frac_loss_%"] = 100.0 * tp["contrib"] / max(float(contrib.sum()), 1e-30)
        top_pools_by_model[tag] = tp

        if verbose:
            p = np.exp(lg - logsumexp(lg, axis=1, keepdims=True))
            sat_pct = 100.0 * float((p.max(axis=1) > 0.9999).mean())
            wrong_conf = int((p[np.arange(n_val), yv] < 1e-4).sum())
            print(lt.to_string())
            print(f"\nSaturação: max-prob>0.9999 em {sat_pct:.2f}% | confiantes e erradas (p_true<1e-4): {wrong_conf:,}")
            print("\nTop 10 pools por contribuição à loss:")
            print(tp.head(10).to_string())

    # 3) BatchNorm móvel vs mediana
    _hdr(f"3) BATCHNORM: estatísticas móveis vs MEDIANA de {n_bn_batches} batches")
    bn_rows = []
    for tag, m in models.items():
        ce0 = ce_cache[tag][1]
        snap = snapshot_bn(m)
        try:
            median_bn(m, lambda: gen_ref[0][0], n_batches=n_bn_batches)
            ce1 = ce_from_logits(predict_logits(m, val_inputs, batch_size=eval_batch_size), yv)
        finally:
            restore_bn(m, snap)
        row = {
            "modelo": tag,
            "val_bn_movel": float(np.sum(wv * ce0) / n_val),
            "val_bn_mediana": float(np.sum(wv * ce1) / n_val),
            "ce_max_movel": float(ce0.max()),
            "ce_max_mediana": float(ce1.max()),
        }
        bn_rows.append(row)
        if verbose:
            print(
                f"{tag}: val (BN móvel)={row['val_bn_movel']:.4f} | "
                f"val (BN mediana)={row['val_bn_mediana']:.4f} | "
                f"CE max: {row['ce_max_movel']:.1f} -> {row['ce_max_mediana']:.1f}"
            )
    bn_comparison_df = pd.DataFrame(bn_rows)

    # 4) Outliers na saída do LBN e nas features de entrada
    _hdr("4) OUTLIERS: saída do LBN (z robusto) e features de entrada")
    lbn_rows = []
    for tag, m in models.items():
        try:
            lbn_layer = m.get_layer("lbn_flat")
            z = keras.Model(m.inputs, lbn_layer.output).predict(
                val_inputs, batch_size=eval_batch_size, verbose=0
            ).astype(np.float64)
            med = np.median(z, axis=0)
            mad = np.median(np.abs(z - med), axis=0) * 1.4826 + 1e-9
            r = np.abs(z - med) / mad
            bad_ev = np.where(r.max(axis=1) > 50)[0]
            lbn_rows.append(
                {
                    "modelo": tag,
                    "abs_z_max_lbn": float(np.abs(z).max()),
                    "canais_robust_gt50": int((r.max(axis=0) > 50).sum()),
                    "total_canais": int(z.shape[1]),
                    "eventos_robust_gt50": int(len(bad_ev)),
                }
            )
            if verbose:
                print(
                    f"{tag}: |z|max LBN = {np.abs(z).max():.3g} | "
                    f"canais |z_rob|>50: {(r.max(axis=0) > 50).sum()}/{z.shape[1]} | "
                    f"eventos >50 MAD: {len(bad_ev):,}"
                )
        except ValueError:
            if verbose:
                print(f"{tag}: modelo sem camada 'lbn_flat'.")
    lbn_outliers_df = pd.DataFrame(lbn_rows)

    zin = np.abs(x_cont_val)
    input_outliers_df = pd.DataFrame(
        {
            "feature": continuous_features,
            "max_abs_z": zin.max(axis=0),
            "p99_9": np.percentile(zin, 99.9, axis=0),
            "frac_zero_pct": 100.0 * (x_cont_val == 0).mean(axis=0),
        }
    ).sort_values("max_abs_z", ascending=False)
    if verbose:
        print("\nTop 10 features de entrada por |z|max na validação:")
        print(input_outliers_df.head(10).to_string(index=False))

    # 5) Memorização nos pools mais pesados (CE treino vs CE val)
    _hdr("5) MEMORIZAÇÃO NOS POOLS MAIS PESADOS (CE treino vs val)")
    tr_pool = np.char.add(
        np.char.add(train_df["sample"].astype(str).to_numpy(), "|"),
        train_df["era"].astype(str).to_numpy(),
    )
    tr_k = train_df["kappa_eval"].to_numpy(np.float32)
    tr_y = train_df["label"].to_numpy().astype(int)
    rng = np.random.default_rng(0)
    mem_rows = []
    for pool in pool_tab.head(10).index:
        it = np.where(tr_pool == pool)[0]
        iv = np.where(val_pool == pool)[0]
        if len(it) == 0 or len(iv) == 0:
            continue
        it_sub = rng.choice(it, size=min(len(it), int(n_train_per_pool)), replace=False)
        xt = make_inputs(x_cont_train[it_sub], {k: v[it_sub] for k, v in x_cat_train.items()}, tr_k[it_sub], kappa_norm=kappa_norm)
        for tag, m in models.items():
            cet = float(ce_from_logits(predict_logits(m, xt, batch_size=eval_batch_size), tr_y[it_sub]).mean())
            cev = float(ce_cache[tag][1][iv].mean())
            mem_rows.append(
                {
                    "pool": pool,
                    "modelo": tag,
                    "n_treino": int(len(it)),
                    "n_val": int(len(iv)),
                    "CE_treino": cet,
                    "CE_val": cev,
                    "razao_val_treino": cev / max(cet, 1e-9),
                }
            )
    memorization_df = pd.DataFrame(mem_rows)
    if verbose and len(memorization_df) > 0:
        print(memorization_df.to_string(index=False))

    # 6) Histórico: épocas suspeitas
    suspicious_epochs: dict[str, list[int]] = {}
    if histories:
        _hdr("6) HISTÓRICO: épocas suspeitas (val_loss > 3x mediana ou > 5x loss)")
        for tag, h in histories.items():
            if not h or "val_loss" not in h:
                continue
            d = pd.DataFrame(h)
            d.index = d.index + 1
            med = float(d["val_loss"].median())
            flag = d[(d["val_loss"] > 3.0 * med) | (d["val_loss"] > 5.0 * d["loss"])]
            suspicious_epochs[tag] = [int(i) for i in flag.index.tolist()]
            if verbose:
                best_ep = int(d["val_loss"].idxmin())
                best_val = float(d["val_loss"].min())
                print(f"{tag}: melhor época={best_ep} (val_loss={best_val:.4f}) | épocas suspeitas={suspicious_epochs[tag]}")

    return {
        "weight_concentration": weight_concentration,
        "val_weight_by_class": val_weight_cls_df,
        "val_pools": pool_tab,
        "loss_tables": pd.DataFrame(loss_tables),
        "top_events_by_model": top_events_by_model,
        "top_pools_by_model": top_pools_by_model,
        "bn_comparison": bn_comparison_df,
        "lbn_outliers": lbn_outliers_df,
        "input_outliers": input_outliers_df,
        "memorization": memorization_df,
        "suspicious_epochs": suspicious_epochs,
    }


def random_label_test(
    cfg: dict[str, Any],
    data: PreparedData,
    val_inputs: dict[str, np.ndarray],
    seeds: Sequence[int] = (1, 2, 3),
    epochs: int = 20,
    steps_per_epoch: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Teste 13.4: treina AffinePNN com labels embaralhados e verifica se loss de treino permanece em ln(3)."""
    from src.training.train import train_model

    n_classes = cfg["physics"]["n_classes"]
    chance = float(np.log(n_classes))
    val_y = data.splits["val"]["label"].to_numpy()
    eval_bs = int(cfg["evaluation"]["eval_batch_size"])
    steps = int(steps_per_epoch or cfg["training"]["steps_per_epoch"])

    rows = []
    for s in seeds:
        gen = make_random_label_generator(
            train_df=data.splits["train"],
            x_cont_train=data.x_cont["train"],
            x_cat_train=data.x_cat["train"],
            kappa_grid=cfg["physics"]["kappa_grid"],
            kappa_norm=cfg["physics"]["kappa_norm"],
            batch_size=cfg["training"]["batch_size"],
            steps_per_epoch=steps,
            seed=int(s),
        )
        y_perm = np.random.default_rng(int(s)).permutation(val_y)
        model_rl, hist, _ = train_model(
            cfg=cfg,
            model_key="affine",
            generator=gen,
            scaler=data.scaler,
            cat_vocab_sizes=data.cat_preprocessor.vocab_sizes,
            val_data=(val_inputs, y_perm, None),
            epochs=int(epochs),
            seed=int(s),
            verbose=0,
            name=f"RandomLabels_seed{s}",
        )
        p = model_rl.predict(val_inputs, batch_size=eval_bs, verbose=0)
        tail_losses = hist["loss"][-min(5, len(hist["loss"])) :]
        rows.append(
            {
                "seed": int(s),
                "loss_treino_final": float(np.mean(tail_losses)),
                "ln_n_classes": chance,
                "AUC_vs_label_real": float(roc_auc_score((val_y == 2).astype(int), p[:, 2])),
            }
        )

    t = pd.DataFrame(rows)
    if verbose:
        print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        ok = bool((t["loss_treino_final"] > chance - 0.02).all())
        print(
            f"\nln({n_classes}) = {chance:.4f} -> "
            + (
                "13.4 OK: loss de treino fica em ln(3), sem vazamento de label no pipeline."
                if ok
                else "13.4 ATENÇÃO: loss abaixo de ln(3) com labels aleatórios!"
            )
        )
    return t
