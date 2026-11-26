"""Curva ROC exata com suporte a pesos negativos (NLO), métricas binárias, PR-AUC e AUC univariada."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def weighted_roc(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Curva ROC ponderada admitindo pesos negativos de MC NLO. `y=1` indica sinal. Retorna `(fpr, tpr)`."""
    y_bool = np.asarray(y).astype(bool)
    s_arr = np.asarray(s, dtype=np.float64)
    w_arr = np.asarray(w, dtype=np.float64)

    o = np.argsort(-s_arr, kind="mergesort")
    s_arr, y_bool, w_arr = s_arr[o], y_bool[o], w_arr[o]
    last = np.r_[np.flatnonzero(np.diff(s_arr)), len(s_arr) - 1]
    tp = np.cumsum(np.where(y_bool, w_arr, 0.0))[last]
    fp = np.cumsum(np.where(~y_bool, w_arr, 0.0))[last]
    return np.r_[0.0, fp / fp[-1]], np.r_[0.0, tp / tp[-1]]


def auc_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Integra a área sob a curva ROC via regra do trapézio."""
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def auc_w(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> float:
    """ROC AUC ponderada suportando pesos com sinal."""
    return auc_from_roc(*weighted_roc(y, s, w))


def binary_metrics_at_threshold(
    y: np.ndarray,
    s: np.ndarray,
    w: np.ndarray,
    thr: float = 0.5,
) -> dict[str, float]:
    """Acurácia, Eficiência (Recall) e Pureza (Precision) em `s >= thr` (sem peso e ponderadas balanceadas)."""
    y_true = np.asarray(y).astype(bool)
    y_pred = np.asarray(s, dtype=np.float64) >= float(thr)
    w_arr = np.asarray(w, dtype=np.float64)

    tp = float(np.sum(y_true & y_pred))
    fp = float(np.sum((~y_true) & y_pred))
    tn = float(np.sum((~y_true) & (~y_pred)))
    fn = float(np.sum(y_true & (~y_pred)))

    acc = (tp + tn) / max(len(y_true), 1)
    eff_s = tp / max(tp + fn, 1.0)
    pur = tp / max(tp + fp, 1.0)

    w_sig_tot = float(w_arr[y_true].sum())
    w_bkg_tot = float(w_arr[~y_true].sum())

    eff_s_w = float(w_arr[y_true & y_pred].sum()) / w_sig_tot if w_sig_tot != 0 else 0.0
    fpr_b_w = float(w_arr[(~y_true) & y_pred].sum()) / w_bkg_tot if w_bkg_tot != 0 else 0.0

    pur_w_bal = eff_s_w / max(eff_s_w + fpr_b_w, 1e-12)
    acc_w_bal = 0.5 * (eff_s_w + (1.0 - fpr_b_w))
    f1_unw = 2.0 * pur * eff_s / max(pur + eff_s, 1e-12)
    f1_w_bal = 2.0 * pur_w_bal * eff_s_w / max(pur_w_bal + eff_s_w, 1e-12)

    return {
        "acc": acc,
        "acc_w_bal": acc_w_bal,
        "eff_s": eff_s,
        "eff_s_w": eff_s_w,
        "pur": pur,
        "pur_w_bal": pur_w_bal,
        "f1": f1_unw,
        "f1_w_bal": f1_w_bal,
    }


def compute_curves_for_kappa(
    y: np.ndarray,
    s: np.ndarray,
    w: np.ndarray,
    n_thr: int = 200,
) -> dict[str, np.ndarray | float]:
    """Calcula Precision, Recall, F1-score vs Threshold, curva PR contínua e rejeição `1/eps_B`."""
    y_bool = np.asarray(y).astype(bool)
    s_arr = np.asarray(s, dtype=np.float64)
    w_arr = np.asarray(w, dtype=np.float64)

    o = np.argsort(-s_arr, kind="mergesort")
    s_sorted, y_sorted, w_sorted = s_arr[o], y_bool[o], w_arr[o]

    w_sig_tot = float(w_sorted[y_sorted].sum())
    w_bkg_tot = float(w_sorted[~y_sorted].sum())

    tp_cum = np.cumsum(np.where(y_sorted, w_sorted, 0.0)) / w_sig_tot
    fp_cum = np.cumsum(np.where(~y_sorted, w_sorted, 0.0)) / w_bkg_tot

    thr_grid = np.linspace(0.01, 0.99, int(n_thr))
    idx = np.searchsorted(-s_sorted, -thr_grid, side="right") - 1
    valid = idx >= 0

    recall_thr = np.zeros_like(thr_grid)
    eps_b_thr = np.zeros_like(thr_grid)
    recall_thr[valid] = np.clip(tp_cum[idx[valid]], 0.0, 1.0)
    eps_b_thr[valid] = np.clip(fp_cum[idx[valid]], 0.0, 1.0)

    precision_thr = np.where(
        recall_thr + eps_b_thr > 1e-12,
        recall_thr / (recall_thr + eps_b_thr),
        1.0,
    )
    f1_thr = np.where(
        precision_thr + recall_thr > 1e-12,
        2.0 * precision_thr * recall_thr / (precision_thr + recall_thr),
        0.0,
    )

    last = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s_sorted) - 1]
    tpr_full = np.r_[0.0, np.clip(tp_cum[last], 0.0, 1.0)]
    fpr_full = np.r_[0.0, np.clip(fp_cum[last], 0.0, 1.0)]
    tpr_mono = np.maximum.accumulate(tpr_full)
    fpr_mono = np.maximum.accumulate(fpr_full)

    eff_grid = np.linspace(0.05, 0.995, 300)
    fpr_interp = np.interp(eff_grid, tpr_mono, fpr_mono)
    prec_curve = eff_grid / np.maximum(eff_grid + fpr_interp, 1e-12)
    inv_eps_b = 1.0 / np.maximum(fpr_interp, 1e-5)

    pr_auc = float(
        np.sum(np.diff(eff_grid) * (prec_curve[1:] + prec_curve[:-1]) / 2.0)
        / (eff_grid[-1] - eff_grid[0])
    )

    best_idx = int(np.argmax(f1_thr))
    return {
        "thr": thr_grid,
        "recall": recall_thr,
        "precision": precision_thr,
        "f1": f1_thr,
        "eps_b": eps_b_thr,
        "best_thr": float(thr_grid[best_idx]),
        "best_f1": float(f1_thr[best_idx]),
        "eff_grid": eff_grid,
        "prec_curve": prec_curve,
        "inv_eps_b": inv_eps_b,
        "pr_auc": pr_auc,
    }


def univariate_feature_aucs(
    train_df: pd.DataFrame,
    x_cont_train: np.ndarray,
    x_cat_train: dict[str, np.ndarray],
    continuous_features: list[str],
    cat_vocab_sizes: dict[str, int],
    max_samples: int = 500_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Teste 13.3: calcula o poder discriminante univariado (ROC AUC HH vs Fundo) de cada feature isolada."""
    rng = np.random.default_rng(seed)
    n_sub = min(int(max_samples), len(train_df))
    sub = rng.choice(len(train_df), size=n_sub, replace=False)
    y_sub = (train_df["label"].to_numpy()[sub] == 2).astype(int)
    X_sub = np.concatenate(
        [
            x_cont_train[sub],
            np.hstack([x_cat_train[n][sub] for n in cat_vocab_sizes]).astype(np.float32),
        ],
        axis=1,
    )
    names = list(continuous_features) + [f"cat:{n}" for n in cat_vocab_sizes]
    rows = []
    for j, name in enumerate(names):
        auc_val = float(roc_auc_score(y_sub, X_sub[:, j]))
        rows.append({"feature": name, "auc": auc_val, "abs_delta_0p5": abs(auc_val - 0.5)})
    return pd.DataFrame(rows).sort_values("abs_delta_0p5", ascending=False).reset_index(drop=True)
