"""Rendimentos físicos esperados (S, B, S/B, S/sqrt(B)) e Significance Ratio σ_ratio (Anzalone et al., Eq. 3-4)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def physical_yields(
    sc: dict[float, dict[str, np.ndarray]],
    eff_points: Sequence[float] = (0.5, 0.8),
) -> pd.DataFrame:
    """Para cada κ_λ, determina o corte de score que retém `eff_sinal` e calcula S, B, S/B e S/sqrt(B)."""
    rows = []
    for k, d in sc.items():
        y = d["y"].astype(bool)
        s = d["s"]
        w = d["w"]
        S_tot = float(w[y].sum())
        B_tot = float(max(w[~y].sum(), 0.0))
        o = np.argsort(-s[y])
        cum = np.cumsum(w[y][o]) / S_tot
        for e in eff_points:
            thr_idx = min(int(np.searchsorted(cum, float(e))), len(cum) - 1)
            thr = float(s[y][o][thr_idx])
            sel = s >= thr
            S = float(w[y & sel].sum())
            B = float(max(w[~y & sel].sum(), 0.0))
            rows.append(
                {
                    "kappa": float(k),
                    "eff_sinal": float(e),
                    "corte": thr,
                    "S_total": S_tot,
                    "B_total": B_tot,
                    "S": S,
                    "B": B,
                    "S/B": S / B if B > 0 else np.nan,
                    "S/sqrt(B)": S / np.sqrt(B) if B > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def ams_significance_scan(
    y: np.ndarray,
    s: np.ndarray,
    w: np.ndarray,
    balanced: bool = True,
    n_thr: int = 250,
) -> dict[str, np.ndarray | float]:
    """Implementa AMS(t) e a Significance Ratio σ_ratio normalizada em [0, 1] (Anzalone et al., Eq. 3 e Eq. 4).

    Quando `balanced=True` (convenção da Seção 6 do artigo), normaliza Σw_s = Σw_b = 1 para que
    `σ_ratio` não dependa da escala absoluta de luminosidade nem do desbalanceamento extremo entre classes:
        σ_ratio(t) = (s_t * sqrt(s_max)) / (s_max * sqrt(s_t + b_t))
    Para um classificador aleatório (AUC = 0.5), σ_ratio -> 1/sqrt(2) ≈ 0.7071; ideal -> 1.0.
    """
    y_bool = np.asarray(y).astype(bool)
    s_arr = np.asarray(s, dtype=np.float64)
    w_arr = np.asarray(w, dtype=np.float64).copy()

    s_tot_raw = float(w_arr[y_bool].sum())
    b_tot_raw = float(w_arr[~y_bool].sum())

    if balanced:
        w_eff = np.zeros_like(w_arr)
        w_eff[y_bool] = w_arr[y_bool] / max(s_tot_raw, 1e-30)
        w_eff[~y_bool] = w_arr[~y_bool] / max(b_tot_raw, 1e-30)
        s_max = 1.0
    else:
        w_eff = w_arr
        s_max = max(s_tot_raw, 1e-30)

    o = np.argsort(-s_arr, kind="mergesort")
    s_sorted, y_sorted, w_sorted = s_arr[o], y_bool[o], w_eff[o]

    s_cum = np.cumsum(np.where(y_sorted, w_sorted, 0.0))
    b_cum = np.cumsum(np.where(~y_sorted, w_sorted, 0.0))

    thr_grid = np.linspace(0.005, 0.995, int(n_thr))
    idx = np.searchsorted(-s_sorted, -thr_grid, side="right") - 1
    valid = idx >= 0

    st = np.zeros_like(thr_grid)
    bt = np.zeros_like(thr_grid)
    st[valid] = np.maximum(s_cum[idx[valid]], 0.0)
    bt[valid] = np.maximum(b_cum[idx[valid]], 0.0)

    denom = np.sqrt(np.maximum(st + bt, 1e-30))
    ams = np.where(st + bt > 1e-15, st / denom, 0.0)
    sigma_ratio = ams / np.sqrt(s_max)

    best_idx = int(np.argmax(sigma_ratio))
    return {
        "thr": thr_grid,
        "st": st,
        "bt": bt,
        "ams": ams,
        "sigma_ratio": sigma_ratio,
        "best_cut": float(thr_grid[best_idx]),
        "best_sigma_ratio": float(sigma_ratio[best_idx]),
        "eff_s_at_best": float(st[best_idx] / s_max),
        "eff_b_at_best": float(bt[best_idx]),
    }


def significance_ratio_table(
    sc: dict[float, dict[str, np.ndarray]],
    n_thr: int = 250,
) -> pd.DataFrame:
    """Tabela de `σ_ratio` e corte ótimo `t*` (Anzalone et al., Eq. 4) para cada hipótese de `kappa_lambda`."""
    rows = []
    for k, d in sc.items():
        res = ams_significance_scan(d["y"], d["s"], d["w"], balanced=True, n_thr=n_thr)
        rows.append(
            {
                "kappa": float(k),
                "sigma_ratio": float(res["best_sigma_ratio"]),
                "best_cut_t*": float(res["best_cut"]),
                "eff_s@t*": float(res["eff_s_at_best"]),
                "eps_b@t*": float(res["eff_b_at_best"]),
            }
        )
    return pd.DataFrame(rows)
