"""Curva ROC e métricas binárias com pesos não-negativos validados, AUC sem peso, PR-AUC e AUC univariada.

Nota metodológica sobre pesos negativos de NLO:
  Uma curva ROC ou Precision-Recall convencional exige contribuições não-negativas por evento para que
  `TPR(t)` e `FPR(t)` sejam funções monotônicas não-decrescentes do corte `t`. Com pesos negativos de MC NLO
  (como em Drell-Yan `amcatnlo`), somas cumulativas assinadas podem decrescer e violar a interpretação
  probabilística da AUC. Por isso, este módulo adota explicitamente pesos truncados não-negativos
  `w_clf = max(w_phys, 0)` (verificando soma estritamente positiva em sinal e fundo) para TODAS as métricas
  de classificação (ROC, PR, F1, thresholds, bootstrap), reportando sempre a AUC sem pesos (`auc`) ao lado.
  Os pesos NLO originais com sinal são preservados exclusivamente para estimativas de rendimento físico
  esperado (`S`, `B`, `S/B`, `S/sqrt(B)` e significância de Asimov) em `src/evaluation/significance.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def prepare_non_negative_weights(
    y: np.ndarray,
    w: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Converte `y` para booleano, trunca pesos negativos em zero (`max(w, 0)`) e valida soma positiva por classe."""
    y_bool = np.asarray(y).astype(bool)
    if not y_bool.any() or y_bool.all():
        raise ValueError("Ambas as classes (sinal y=1 e fundo y=0) devem conter pelo menos um evento.")

    if w is None:
        w_pos = np.ones(len(y_bool), dtype=np.float64)
    else:
        w_raw = np.asarray(w, dtype=np.float64)
        w_pos = np.clip(np.where(np.isfinite(w_raw), w_raw, 0.0), 0.0, None)

    w_sig_tot = float(w_pos[y_bool].sum())
    w_bkg_tot = float(w_pos[~y_bool].sum())
    if w_sig_tot <= 0.0 or w_bkg_tot <= 0.0:
        raise ValueError(
            f"Soma de pesos não-negativos inválida para cálculo de ROC/PR: "
            f"sum_w_pos(sig)={w_sig_tot:.6g}, sum_w_pos(bkg)={w_bkg_tot:.6g}. "
            "Cada classe deve manter soma estritamente positiva."
        )
    return y_bool, w_pos, w_sig_tot, w_bkg_tot


def weighted_roc(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Curva ROC monotônica ponderada por pesos não-negativos `max(w, 0)`. `y=1` indica sinal. Retorna `(fpr, tpr)`."""
    y_bool, w_pos, w_sig_tot, w_bkg_tot = prepare_non_negative_weights(y, w)
    s_arr = np.asarray(s, dtype=np.float64)

    o = np.argsort(-s_arr, kind="mergesort")
    s_arr, y_bool, w_pos = s_arr[o], y_bool[o], w_pos[o]
    last = np.r_[np.flatnonzero(np.diff(s_arr)), len(s_arr) - 1]
    tp = np.cumsum(np.where(y_bool, w_pos, 0.0))[last]
    fp = np.cumsum(np.where(~y_bool, w_pos, 0.0))[last]
    fpr = np.clip(np.r_[0.0, fp / w_bkg_tot], 0.0, 1.0)
    tpr = np.clip(np.r_[0.0, tp / w_sig_tot], 0.0, 1.0)
    return fpr, tpr


def auc_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Integra a área sob uma curva ROC monotônica via regra do trapézio."""
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def auc_unweighted(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC sem pesos (w = 1 para todos os eventos)."""
    return auc_from_roc(*weighted_roc(y, s, None))


def auc_w(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    """ROC AUC ponderada com pesos não-negativos `max(w, 0)` validados por classe."""
    return auc_from_roc(*weighted_roc(y, s, w))


def binary_metrics_at_threshold(
    y: np.ndarray,
    s: np.ndarray,
    w: np.ndarray | None = None,
    thr: float = 0.5,
) -> dict[str, float]:
    """Acurácia, Eficiência (Recall), Pureza (Precision) e F1 em `s >= thr` (sem peso e ponderadas com `w >= 0`)."""
    y_true, w_pos, w_sig_tot, w_bkg_tot = prepare_non_negative_weights(y, w)
    y_pred = np.asarray(s, dtype=np.float64) >= float(thr)

    tp = float(np.sum(y_true & y_pred))
    fp = float(np.sum((~y_true) & y_pred))
    tn = float(np.sum((~y_true) & (~y_pred)))
    fn = float(np.sum(y_true & (~y_pred)))

    acc = (tp + tn) / max(len(y_true), 1)
    eff_s = tp / max(tp + fn, 1.0)
    pur = tp / max(tp + fp, 1.0)

    eff_s_w = float(w_pos[y_true & y_pred].sum()) / w_sig_tot
    fpr_b_w = float(w_pos[(~y_true) & y_pred].sum()) / w_bkg_tot

    pur_w_bal = eff_s_w / max(eff_s_w + fpr_b_w, 1e-12)
    acc_w_bal = 0.5 * (eff_s_w + (1.0 - fpr_b_w))
    f1_unw = 2.0 * pur * eff_s / max(pur + eff_s, 1e-12)
    f1_w_bal = 2.0 * pur_w_bal * eff_s_w / max(pur_w_bal + eff_s_w, 1e-12)

    return {
        "acc": acc,
        "acc_w_bal": acc_w_bal,
        "eff_s": eff_s,
        "eff_s_w": eff_s_w,
        "eps_b_w": fpr_b_w,
        "pur": pur,
        "pur_w_bal": pur_w_bal,
        "f1": f1_unw,
        "f1_w_bal": f1_w_bal,
    }


def compute_curves_for_kappa(
    y: np.ndarray,
    s: np.ndarray,
    w: np.ndarray | None = None,
    n_thr: int = 200,
) -> dict[str, np.ndarray | float]:
    """Calcula Precision, Recall, F1-score vs Threshold, curva PR monotônica e rejeição `1/eps_B` (com `w >= 0`)."""
    y_bool, w_pos, w_sig_tot, w_bkg_tot = prepare_non_negative_weights(y, w)
    s_arr = np.asarray(s, dtype=np.float64)

    o = np.argsort(-s_arr, kind="mergesort")
    s_sorted, y_sorted, w_sorted = s_arr[o], y_bool[o], w_pos[o]

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

    eff_grid = np.linspace(0.05, 0.995, 300)
    fpr_interp = np.interp(eff_grid, tpr_full, fpr_full)
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
