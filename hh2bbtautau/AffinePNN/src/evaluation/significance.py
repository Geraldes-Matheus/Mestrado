"""Rendimentos físicos esperados com pesos NLO assinados (`S`, `B`, `S/B`, `S/sqrt(B)`, Asimov `Z_A`)
e Significance Ratio `σ_ratio` balanceada (Anzalone et al., Eq. 3–4).

Separação metodológica:
- `physical_yields` e `physical_significance_scan` preservam os pesos NLO originais com sinal
  (`N_esperado = sum_i w_phys_i`) para estimar rendimentos físicos reais na luminosidade integrada
  e calcular `S/B`, `S/sqrt(B)` e a significância de Asimov `Z_A`.
  Apenas a localização do ponto de trabalho `thr(eff_sinal)` usa a distribuição não-negativa do sinal
  para garantir monotonicidade estrita da eficiência.
- `ams_significance_scan(..., balanced=True)` (métrica adimensional `σ_ratio` da Eq. 4 de Anzalone et al.)
  opera como métrica pura de forma/classificação sobre pesos não-negativos normalizados (`Σw_s = Σw_b = 1`).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.evaluation.roc import prepare_non_negative_weights


def asimov_significance(s: float | np.ndarray, b: float | np.ndarray) -> float | np.ndarray:
    """Significância de descoberta de Asimov (Cowan et al., EPJC 71, 1554):
    `Z_A = sqrt(2 * ((s + b) * ln(1 + s / b) - s))` definida para `s > 0` e `b > 0`.
    """
    s_arr = np.asarray(s, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    valid = (s_arr > 0.0) & (b_arr > 0.0)
    out = np.full(np.broadcast(s_arr, b_arr).shape, np.nan, dtype=np.float64)
    if np.any(valid):
        sv = np.broadcast_to(s_arr, out.shape)[valid]
        bv = np.broadcast_to(b_arr, out.shape)[valid]
        term = 2.0 * ((sv + bv) * np.log1p(sv / bv) - sv)
        out[valid] = np.sqrt(np.maximum(term, 0.0))
    return float(out) if out.ndim == 0 else out


def physical_yields(
    sc: dict[float, dict[str, np.ndarray]],
    eff_points: Sequence[float] = (0.5, 0.8, 0.9),
) -> pd.DataFrame:
    """Para cada κ_λ, determina o corte de score que retém `eff_sinal` (com pesos positivos monotônicos)
    e calcula os rendimentos físicos esperados `S`, `B`, `S/B`, `S/sqrt(B)` e Asimov `Z_A`
    preservando os pesos NLO originais com sinal (`w_phys`).
    """
    rows = []
    for k, d in sc.items():
        y, w_pos, s_pos_tot, _ = prepare_non_negative_weights(d["y"], d["w"])
        s = np.asarray(d["s"], dtype=np.float64)
        w_signed = np.asarray(d["w"], dtype=np.float64)

        S_tot_signed = float(w_signed[y].sum())
        B_tot_signed = float(w_signed[~y].sum())

        o = np.argsort(-s[y], kind="mergesort")
        cum_pos = np.cumsum(w_pos[y][o]) / s_pos_tot

        for e in eff_points:
            thr_idx = min(int(np.searchsorted(cum_pos, float(e))), len(cum_pos) - 1)
            thr = float(s[y][o][thr_idx])
            sel = s >= thr
            S = float(w_signed[y & sel].sum())
            B_raw = float(w_signed[(~y) & sel].sum())
            B_pos_valid = B_raw > 0.0
            rows.append(
                {
                    "kappa": float(k),
                    "eff_sinal": float(e),
                    "corte": thr,
                    "S_total": S_tot_signed,
                    "B_total": B_tot_signed,
                    "S": S,
                    "B": B_raw,
                    "S/B": (S / B_raw) if B_pos_valid else np.nan,
                    "S/sqrt(B)": (S / np.sqrt(B_raw)) if B_pos_valid and S > 0 else np.nan,
                    "Z_asimov": float(asimov_significance(S, B_raw)) if B_pos_valid and S > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def compare_physical_yields(
    yields_a: pd.DataFrame,
    yields_ref: pd.DataFrame,
    suffix_a: str = "_A",
    suffix_ref: str = "_AN",
) -> pd.DataFrame:
    """Compara diretamente rendimentos físicos e significância de Asimov entre dois modelos (ex: AffinePNN vs DNN_AN)."""
    cols = ["kappa", "eff_sinal", "corte", "S", "B", "S/B", "S/sqrt(B)", "Z_asimov"]
    cols = [c for c in cols if c in yields_a.columns and c in yields_ref.columns]
    cmp = yields_a[cols].merge(yields_ref[cols], on=["kappa", "eff_sinal"], suffixes=(suffix_a, suffix_ref))
    if f"Z_asimov{suffix_a}" in cmp.columns:
        cmp["razao_Z_asimov"] = cmp[f"Z_asimov{suffix_a}"] / cmp[f"Z_asimov{suffix_ref}"].replace(0.0, np.nan)
    if f"B{suffix_a}" in cmp.columns:
        cmp["reducao_B_fator"] = cmp[f"B{suffix_ref}"] / cmp[f"B{suffix_a}"].replace(0.0, np.nan)
    return cmp


def ams_significance_scan(
    y: np.ndarray,
    s: np.ndarray,
    w: np.ndarray,
    balanced: bool = True,
    n_thr: int = 250,
) -> dict[str, np.ndarray | float]:
    """Implementa AMS(t) e a Significance Ratio σ_ratio normalizada em [0, 1] (Anzalone et al., Eq. 3 e Eq. 4).

    Quando `balanced=True` (convenção da Seção 6 do artigo), usa pesos não-negativos normalizados
    `Σw_s = Σw_b = 1` para garantir monotonicidade e independência de escala absoluta de luminosidade:
        σ_ratio(t) = (s_t * sqrt(s_max)) / (s_max * sqrt(s_t + b_t))
    Para um classificador aleatório (AUC = 0.5), σ_ratio -> 1/sqrt(2) ≈ 0.7071; ideal -> 1.0.
    Quando `balanced=False`, varre a significância física usando os pesos NLO assinados originais.
    """
    s_arr = np.asarray(s, dtype=np.float64)

    if balanced:
        y_bool, w_pos, s_tot_pos, b_tot_pos = prepare_non_negative_weights(y, w)
        w_eff = np.zeros_like(w_pos)
        w_eff[y_bool] = w_pos[y_bool] / s_tot_pos
        w_eff[~y_bool] = w_pos[~y_bool] / b_tot_pos
        s_max = 1.0
    else:
        y_bool = np.asarray(y).astype(bool)
        w_eff = np.asarray(w, dtype=np.float64).copy()
        s_max = max(float(w_eff[y_bool].sum()), 1e-30)

    o = np.argsort(-s_arr, kind="mergesort")
    s_sorted, y_sorted, w_sorted = s_arr[o], y_bool[o], w_eff[o]

    s_cum = np.cumsum(np.where(y_sorted, w_sorted, 0.0))
    b_cum = np.cumsum(np.where(~y_sorted, w_sorted, 0.0))

    thr_grid = np.linspace(0.005, 0.995, int(n_thr))
    idx = np.searchsorted(-s_sorted, -thr_grid, side="right") - 1
    valid = idx >= 0

    st = np.zeros_like(thr_grid)
    bt = np.zeros_like(thr_grid)
    st[valid] = s_cum[idx[valid]]
    bt[valid] = b_cum[idx[valid]]

    st_pos = np.maximum(st, 0.0)
    bt_pos = np.maximum(bt, 0.0)
    denom = np.sqrt(np.maximum(st_pos + bt_pos, 1e-30))
    ams = np.where((st > 0.0) & (bt >= 0.0) & (st_pos + bt_pos > 1e-15), st_pos / denom, 0.0)
    sigma_ratio = ams / np.sqrt(s_max)
    z_asimov = asimov_significance(st, bt)

    best_idx = int(np.argmax(sigma_ratio))
    return {
        "thr": thr_grid,
        "st": st,
        "bt": bt,
        "ams": ams,
        "z_asimov": z_asimov,
        "sigma_ratio": sigma_ratio,
        "best_cut": float(thr_grid[best_idx]),
        "best_sigma_ratio": float(sigma_ratio[best_idx]),
        "eff_s_at_best": float(st_pos[best_idx] / s_max),
        "eff_b_at_best": float(bt_pos[best_idx]),
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
