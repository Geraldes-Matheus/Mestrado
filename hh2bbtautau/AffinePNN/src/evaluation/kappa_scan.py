"""Sumários por kappa_lambda (com bootstrap), matriz de resposta, varredura contínua, interpolação e seeds."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from tensorflow import keras

from src.data.preprocessing import PreparedData, make_inputs
from src.data.sampler import QuotaBatchGenerator
from src.data.weights import oversampling_frequency_weights
from src.evaluation.roc import (
    auc_from_roc,
    auc_w,
    binary_metrics_at_threshold,
    compute_curves_for_kappa,
    weighted_roc,
)
from src.evaluation.score_distributions import score_per_kappa
from src.evaluation.significance import ams_significance_scan


def summarize_scores(
    sc: dict[float, dict[str, np.ndarray]],
    thr: float = 0.5,
    sig_eff_points: Sequence[float] = (0.5, 0.8),
    n_boot: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Calcula tabela completa de métricas binárias por κ_λ (incluindo PR-AUC, F1, σ_ratio e erro de bootstrap)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, d in sc.items():
        y, s, w = d["y"], d["s"], d["w"]
        fpr, tpr = weighted_roc(y, s, w)
        tpr_m = np.maximum.accumulate(tpr)

        bm = binary_metrics_at_threshold(y, s, w, thr=thr)
        curves = compute_curves_for_kappa(y, s, w, n_thr=200)
        sig_res = ams_significance_scan(y, s, w, balanced=True, n_thr=200)

        row: dict[str, float] = {
            "kappa": float(k),
            "auc": auc_w(y, s, np.ones_like(s)),
            "auc_w": auc_from_roc(fpr, tpr),
            "pr_auc": float(curves["pr_auc"]),
            "sigma_ratio": float(sig_res["best_sigma_ratio"]),
            "best_cut": float(sig_res["best_cut"]),
            "acc": bm["acc"],
            "acc_w_bal": bm["acc_w_bal"],
            "eff_s": bm["eff_s"],
            "eff_s_w": bm["eff_s_w"],
            "pur": bm["pur"],
            "pur_w_bal": bm["pur_w_bal"],
            "f1": bm["f1"],
            "f1_w_bal": bm["f1_w_bal"],
            "best_f1_w_bal": float(curves["best_f1"]),
            "best_f1_thr": float(curves["best_thr"]),
        }
        for e in sig_eff_points:
            eps_b = float(np.interp(float(e), tpr_m, fpr))
            row[f"eps_b@{e}"] = eps_b
            row[f"pur_w_bal@{e}"] = float(e / max(e + eps_b, 1e-12))

        if n_boot and int(n_boot) > 0:
            boots = []
            for _ in range(int(n_boot)):
                i = rng.integers(0, len(y), len(y))
                boots.append(auc_w(y[i], s[i], w[i]))
            row["auc_w_err"] = float(np.std(boots))
        else:
            row["auc_w_err"] = 0.0

        rows.append(row)
    return pd.DataFrame(rows)


def compare_models_table(
    table_a: pd.DataFrame,
    table_ref: pd.DataFrame,
    suffix_a: str = "_A",
    suffix_ref: str = "_AN",
) -> pd.DataFrame:
    """Compara diretamente duas tabelas de `summarize_scores` (ex: AffinePNN vs DNN_AN)."""
    cols = [
        "kappa",
        "auc_w",
        "auc_w_err",
        "pr_auc",
        "sigma_ratio",
        "acc_w_bal",
        "eff_s_w",
        "pur_w_bal",
        "f1_w_bal",
        "best_f1_w_bal",
    ]
    cols = [c for c in cols if c in table_a.columns and c in table_ref.columns]
    for c in table_a.columns:
        if c.startswith("eps_b@") and c in table_ref.columns:
            cols.append(c)
    cmp = table_a[cols].merge(table_ref[cols], on="kappa", suffixes=(suffix_a, suffix_ref))
    cmp[f"ganho_auc_w({suffix_a[1:]}-{suffix_ref[1:]})"] = cmp[f"auc_w{suffix_a}"] - cmp[f"auc_w{suffix_ref}"]
    if f"f1_w_bal{suffix_a}" in cmp.columns:
        cmp[f"ganho_f1({suffix_a[1:]}-{suffix_ref[1:]})"] = cmp[f"f1_w_bal{suffix_a}"] - cmp[f"f1_w_bal{suffix_ref}"]
    cmp["erro_comb"] = np.sqrt(cmp[f"auc_w_err{suffix_a}"] ** 2 + cmp[f"auc_w_err{suffix_ref}"] ** 2)
    if f"eps_b@0.5{suffix_a}" in cmp.columns:
        cmp[f"razao_eps_b@0.5({suffix_a[1:]}/{suffix_ref[1:]})"] = (
            cmp[f"eps_b@0.5{suffix_a}"] / cmp[f"eps_b@0.5{suffix_ref}"].replace(0.0, np.nan)
        )
    return cmp


def kappa_dependence(
    model: keras.Model,
    ev_df: pd.DataFrame,
    ev_xc: np.ndarray,
    ev_xk: dict[str, np.ndarray],
    class_labels: dict[int, str],
    kappas: Sequence[float] = (0.0, 1.0, 2.45, 5.0),
    kappa_norm: float = 5.0,
    n_samples: int = 20_000,
    batch_size: int = 8192,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Mede a variação de `<P(HH)>` por classe verdadeira e o desvio-padrão médio por evento ao variar κ_λ."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ev_df), size=min(int(n_samples), len(ev_df)), replace=False)
    y = ev_df["label"].to_numpy()[idx]

    preds = {
        float(k): model.predict(
            make_inputs(ev_xc, ev_xk, float(k), kappa_norm=kappa_norm, idx=idx),
            batch_size=batch_size,
            verbose=0,
        )
        for k in kappas
    }

    rows = []
    for k in kappas:
        fk = float(k)
        row: dict[str, float] = {"kappa_in": fk}
        for c, name in class_labels.items():
            m = y == c
            row[f"mean_P_HH_{name}"] = float(preds[fk][m, 2].mean()) if m.any() else np.nan
        rows.append(row)

    stack = np.stack([preds[float(k)] for k in kappas], axis=0)
    std_by_cls = {
        class_labels[c]: round(float(stack.std(axis=0).mean(axis=0)[c]), 5)
        for c in class_labels
    }
    return pd.DataFrame(rows), std_by_cls


def kappa_response_matrix(
    model: keras.Model,
    ev_df: pd.DataFrame,
    ev_xc: np.ndarray,
    ev_xk: dict[str, np.ndarray],
    kappas: Sequence[float] = (0.0, 1.0, 2.45, 5.0),
    kappa_norm: float = 5.0,
    batch_size: int = 8192,
) -> pd.DataFrame:
    """Matriz de `auc_w` com sinal gerado em `κ_verdadeiro` (linhas) vs `κ_entrada` fornecido à rede (colunas)."""
    cols_dict: dict[float, dict[float, float]] = {}
    for k_in in kappas:
        sc = score_per_kappa(
            model,
            ev_df,
            ev_xc,
            ev_xk,
            kappas=kappas,
            kappa_input=float(k_in),
            kappa_norm=kappa_norm,
            batch_size=batch_size,
        )
        cols_dict[float(k_in)] = {k_true: auc_w(d["y"], d["s"], d["w"]) for k_true, d in sc.items()}
    mat = pd.DataFrame(cols_dict)
    mat.index.name = "κ verdadeiro (sinal)"
    mat.columns.name = "κ de entrada"
    return mat


def continuous_kappa_scan(
    model: keras.Model,
    ev_df: pd.DataFrame,
    ev_xc: np.ndarray,
    ev_xk: dict[str, np.ndarray],
    kappa_scan_values: Sequence[float],
    ref_signal_kappas: Sequence[float] = (0.0, 1.0, 2.45, 5.0),
    kappa_norm: float = 5.0,
    n_bkg_max: int = 80_000,
    batch_size: int = 8192,
    seed: int = 42,
) -> pd.DataFrame:
    """Varredura contínua fina em `kappa_lambda` (ex: [-2, 8]) avaliando resposta média no fundo e em cada amostra HH."""
    rng = np.random.default_rng(seed)
    labels = ev_df["label"].to_numpy()
    kl = ev_df["kappa_lambda"].to_numpy(np.float64)

    bkg_idx = np.where(labels != 2)[0]
    if len(bkg_idx) > n_bkg_max:
        bkg_idx = np.sort(rng.choice(bkg_idx, size=int(n_bkg_max), replace=False))
    bkg_y = labels[bkg_idx]

    sig_indices = {
        float(ks): np.where((labels == 2) & np.isclose(kl, float(ks)))[0]
        for ks in ref_signal_kappas
    }

    rows = []
    for k_in in kappa_scan_values:
        fk = float(k_in)
        p_bkg = model.predict(
            make_inputs(ev_xc, ev_xk, fk, kappa_norm=kappa_norm, idx=bkg_idx),
            batch_size=batch_size,
            verbose=0,
        )
        row: dict[str, float] = {
            "kappa_in": fk,
            "mean_P_HH_DY": float(p_bkg[bkg_y == 0, 2].mean()),
            "mean_P_HH_ttbar": float(p_bkg[bkg_y == 1, 2].mean()),
        }
        for ks, s_idx in sig_indices.items():
            if len(s_idx) == 0:
                continue
            p_sig = model.predict(
                make_inputs(ev_xc, ev_xk, fk, kappa_norm=kappa_norm, idx=s_idx),
                batch_size=batch_size,
                verbose=0,
            )
            row[f"mean_P_HH_sig_kl{ks:g}"] = float(p_sig[:, 2].mean())
            y_comb = np.r_[np.ones(len(s_idx), dtype=np.int8), np.zeros(len(bkg_idx), dtype=np.int8)]
            s_comb = np.r_[p_sig[:, 2], p_bkg[:, 2]]
            w_comb = np.r_[
                ev_df["w_phys"].to_numpy(np.float64)[s_idx],
                ev_df["w_phys"].to_numpy(np.float64)[bkg_idx],
            ]
            row[f"auc_w_sig_kl{ks:g}"] = auc_w(y_comb, s_comb, w_comb)
        rows.append(row)
    return pd.DataFrame(rows)


def interpolation_study(
    cfg: dict[str, Any],
    data: PreparedData,
    val_freq_w: np.ndarray,
    model_full_ref: keras.Model | None = None,
    held_out: float = 2.45,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
    eval_split: str = "val",
    seed: int | None = None,
    verbose: bool = True,
) -> pd.Series:
    """Estudo 13.5: treina AffinePNN e PNN_Concat SEM `held_out` e mede interpolação no ponto omitido."""
    from src.training.train import train_model

    used_seed = int(cfg["seed"] if seed is None else seed)
    used_epochs = int(cfg["training"]["epochs"] if epochs is None else epochs)
    steps = int(steps_per_epoch or cfg["training"]["steps_per_epoch"])
    kgrid = np.asarray(cfg["physics"]["kappa_grid"], dtype=np.float32)
    knorm = float(cfg["physics"]["kappa_norm"])
    eval_bs = int(cfg["evaluation"]["eval_batch_size"])

    train_df = data.splits["train"]
    val_df = data.splits["val"]
    val_y = val_df["label"].to_numpy()
    kl_tr = train_df["kappa_lambda"].to_numpy(np.float64)

    is_ho_tr = (train_df["label"].to_numpy() == 2) & np.isclose(kl_tr, float(held_out))
    rows = np.where(~is_ho_tr)[0]
    reduced = kgrid[~np.isclose(kgrid, float(held_out))]

    vw = val_freq_w.copy()
    vk = val_df["kappa_eval"].to_numpy().copy()
    is_ho_va = (val_y == 2) & np.isclose(val_df["kappa_lambda"].to_numpy(np.float64), float(held_out))
    vw[is_ho_va] = 0.0
    bad = (val_y != 2) & np.isclose(vk, float(held_out))
    vk[bad] = np.random.default_rng(used_seed).choice(reduced, size=int(bad.sum()))
    val_ho = (make_inputs(data.x_cont["val"], data.x_cat["val"], vk, kappa_norm=knorm), val_y, vw)

    trained_models: dict[str, keras.Model] = {}
    for model_key in ("affine", "pnn_concat"):
        gen = QuotaBatchGenerator(
            df=train_df,
            x_cont=data.x_cont["train"],
            x_cat=data.x_cat["train"],
            rows=rows,
            batch_size=cfg["training"]["batch_size"],
            steps_per_epoch=steps,
            class_fraction=cfg["sampling"]["class_batch_fraction"],
            quota_mode=cfg["sampling"]["quota_weight_mode"],
            min_one=cfg["sampling"]["quota_min_one"],
            kappa_grid=reduced,
            kappa_norm=knorm,
            seed=used_seed,
        )
        m, _, _ = train_model(
            cfg=cfg,
            model_key=model_key,
            generator=gen,
            scaler=data.scaler,
            cat_vocab_sizes=data.cat_preprocessor.vocab_sizes,
            val_data=val_ho,
            epochs=used_epochs,
            seed=used_seed,
            verbose=0,
            name=f"{model_key}_no_kl{held_out}",
        )
        trained_models[model_key] = m

    ev_df = data.splits[eval_split]
    ev_xc = data.x_cont[eval_split]
    ev_xk = data.x_cat[eval_split]

    def _auc_at(m: keras.Model, k_in: float | None = None) -> float:
        sc = score_per_kappa(m, ev_df, ev_xc, ev_xk, kappas=[float(held_out)], kappa_input=k_in, kappa_norm=knorm, batch_size=eval_bs)
        d = sc[float(held_out)]
        return auc_w(d["y"], d["s"], d["w"])

    results: dict[str, float] = {}
    if model_full_ref is not None:
        results[f"referencia: AffinePNN completo (viu κ={held_out:g})"] = _auc_at(model_full_ref)
    results["affine sem κ held-out, κ de entrada CORRETO"] = _auc_at(trained_models["affine"])
    results["concat sem κ held-out, κ de entrada CORRETO"] = _auc_at(trained_models["pnn_concat"])
    for k_wrong in reduced:
        results[f"affine sem κ held-out, κ de entrada ERRADO = {float(k_wrong):g}"] = _auc_at(
            trained_models["affine"], float(k_wrong)
        )

    ser = pd.Series(results, name=f"AUC_w em κ={held_out:g}")
    if verbose:
        print(ser.to_string(float_format=lambda x: f"{x:.4f}"))
    return ser


def seed_study(
    cfg: dict[str, Any],
    data: PreparedData,
    seeds: Sequence[int] = (1, 2, 3),
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
    eval_split: str = "val",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estudo 13.6: treina AffinePNN e DNN_AN em múltiplas seeds para medir dispersão do ganho de AUC."""
    from src.training.train import create_generator_for_model, train_model

    knorm = float(cfg["physics"]["kappa_norm"])
    kgrid = cfg["physics"]["kappa_grid"]
    eval_bs = int(cfg["evaluation"]["eval_batch_size"])
    val_df = data.splits["val"]
    val_y = val_df["label"].to_numpy()
    val_inputs = make_inputs(data.x_cont["val"], data.x_cat["val"], val_df["kappa_eval"].to_numpy(), kappa_norm=knorm)

    ev_df = data.splits[eval_split]
    ev_xc = data.x_cont[eval_split]
    ev_xk = data.x_cat[eval_split]

    rows = []
    for s in seeds:
        s_int = int(s)
        for model_key, label_name in (("affine", "AffinePNN"), ("dnn_an", "DNN_AN")):
            gen = create_generator_for_model(cfg, model_key, data, seed=s_int, steps_per_epoch=steps_per_epoch)
            vw = oversampling_frequency_weights(
                val_df,
                gen,
                class_fraction=cfg["sampling"]["class_batch_fraction"],
                clip=cfg["sampling"]["val_freq_clip"],
            )
            m, _, _ = train_model(
                cfg=cfg,
                model_key=model_key,
                generator=gen,
                scaler=data.scaler,
                cat_vocab_sizes=data.cat_preprocessor.vocab_sizes,
                val_data=(val_inputs, val_y, vw),
                epochs=epochs,
                seed=s_int,
                verbose=0,
                name=f"{label_name}_seed{s_int}",
            )
            sc = score_per_kappa(m, ev_df, ev_xc, ev_xk, kappas=kgrid, kappa_norm=knorm, batch_size=eval_bs)
            t = summarize_scores(sc, thr=cfg["evaluation"]["threshold"], n_boot=0)
            t["modelo"] = label_name
            t["seed"] = s_int
            rows.append(t)

    seeds_df = pd.concat(rows, ignore_index=True)
    piv = seeds_df.pivot_table(index=["kappa", "seed"], columns="modelo", values="auc_w")
    gain_df = (piv["AffinePNN"] - piv["DNN_AN"]).groupby("kappa").agg(["mean", "std"]).reset_index()

    if verbose:
        agg = seeds_df.groupby(["modelo", "kappa"])["auc_w"].agg(["mean", "std"])
        print("AUC ponderada: média e desvio entre seeds:")
        print(agg.to_string(float_format=lambda x: f"{x:.4f}"))
        print("\nGanho de AUC (AffinePNN - DNN_AN) por κ:")
        print(gain_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return seeds_df, gain_df
