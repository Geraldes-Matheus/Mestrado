"""Rendimentos físicos esperados com pesos NLO assinados (`S`, `B`, `S/B`, `S/sqrt(B)`, Asimov nominal `Z_A`),
incertezas estatísticas de Monte Carlo (`σ_MC = sqrt(Σ w_i^2)`) e Significance Ratio `σ_ratio` (Anzalone et al., Eq. 3–4).

Separação metodológica:
- `physical_yields` preserva os pesos NLO originais com sinal (`N_esperado = sum_i w_phys_i`) para estimar
  rendimentos físicos reais na luminosidade integrada, reportando simultaneamente:
    * `eff_s_pos`    : eficiência alcançada na convenção de pesos não-negativos (`Σ_sel w^+ / Σ_tot w^+`),
    * `eff_s_signed` : eficiência física líquida com pesos NLO assinados (`Σ_sel w / Σ_tot w`),
    * `S_err_mc`, `B_err_mc`, `sum_w2_S`, `sum_w2_B` : incerteza estatística MC NLO (`σ_MC^2 = Σ w_i^2`),
    * `Z_asimov`     : significância de descoberta de Asimov NOMINAL calculada com `S` e `B` assinados
      (sem parâmetros de nuisance sistemáticos ou estatísticos de MC; retorna `NaN` se `S <= 0` ou `B <= 0`).
- `ams_significance_scan(..., balanced=True)` (métrica adimensional `σ_ratio` da Eq. 4 de Anzalone et al.)
  opera como métrica pura de forma/classificação sobre pesos não-negativos normalizados (`Σw_s = Σw_b = 1`),
  com curvas acumuladas monotonicamente não-decrescentes.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.evaluation.roc import prepare_non_negative_weights


def asimov_significance(s: float | np.ndarray, b: float | np.ndarray) -> float | np.ndarray:
    """Significância de descoberta de Asimov nominal (Cowan et al., EPJC 71, 1554):
    `Z_A = sqrt(2 * ((s + b) * ln(1 + s / b) - s))` definida apenas para `s > 0` e `b > 0`
    (retorna `NaN` caso `s <= 0` ou `b <= 0`; não incorpora incertezas sistemáticas/MC como nuisances).
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
    """Para cada κ_λ, determina o limiar de score correspondente ao working point `eff_sinal` na
    convenção não-negativa (`w^+ = max(w_phys, 0)`) e avalia nos eventos selecionados:
      - eficiências efetivas `eff_s_pos` (`w^+`) e `eff_s_signed` (`w_phys` assinado),
      - rendimentos NLO assinados `S` e `B` com incertezas estatísticas de MC `S_err_mc` e `B_err_mc` (`sqrt(Σ w_i^2)`),
      - `S/B` (com incerteza propagada `S/B_err_mc`), `S/sqrt(B)` e significância nominal de Asimov `Z_asimov`.
    """
    rows = []
    for k, d in sc.items():
        y, w_pos, s_pos_tot, _ = prepare_non_negative_weights(d["y"], d["w"])
        s = np.asarray(d["s"], dtype=np.float64)
        w_signed = np.asarray(d["w"], dtype=np.float64)
        w2 = w_signed ** 2

        S_tot_signed = float(w_signed[y].sum())
        B_tot_signed = float(w_signed[~y].sum())

        s_sig = s[y]
        w_pos_sig = w_pos[y]
        o = np.argsort(-s_sig, kind="mergesort")
        s_sig_sorted = s_sig[o]
        # Agrupa empates de score para refletir exatamente `s >= thr`
        last = np.r_[np.flatnonzero(np.diff(s_sig_sorted)), len(s_sig_sorted) - 1]
        cum_pos_unique = np.cumsum(w_pos_sig[o])[last] / s_pos_tot
        s_thr_candidates = s_sig_sorted[last]

        for e in eff_points:
            thr_idx = min(int(np.searchsorted(cum_pos_unique, float(e), side="left")), len(s_thr_candidates) - 1)
            thr = float(s_thr_candidates[thr_idx])
            sel = s >= thr
            sel_sig = y & sel
            sel_bkg = (~y) & sel

            eff_s_pos = float(w_pos[sel_sig].sum()) / s_pos_tot
            S = float(w_signed[sel_sig].sum())
            B_raw = float(w_signed[sel_bkg].sum())

            sum_w2_S = float(w2[sel_sig].sum())
            sum_w2_B = float(w2[sel_bkg].sum())
            S_err_mc = float(np.sqrt(sum_w2_S))
            B_err_mc = float(np.sqrt(sum_w2_B))

            eff_s_signed = (S / S_tot_signed) if S_tot_signed > 0.0 else np.nan
            eff_b_signed = (B_raw / B_tot_signed) if B_tot_signed > 0.0 else np.nan

            B_pos_valid = B_raw > 0.0
            sb_val = (S / B_raw) if B_pos_valid else np.nan
            if B_pos_valid and S > 0.0:
                sb_err = abs(sb_val) * float(np.sqrt((S_err_mc / S) ** 2 + (B_err_mc / B_raw) ** 2))
            else:
                sb_err = np.nan

            rows.append(
                {
                    "kappa": float(k),
                    "eff_sinal": float(e),
                    "corte": thr,
                    "eff_s_pos": eff_s_pos,
                    "eff_s_signed": eff_s_signed,
                    "eff_b_signed": eff_b_signed,
                    "S_total": S_tot_signed,
                    "B_total": B_tot_signed,
                    "S": S,
                    "S_err_mc": S_err_mc,
                    "sum_w2_S": sum_w2_S,
                    "B": B_raw,
                    "B_err_mc": B_err_mc,
                    "sum_w2_B": sum_w2_B,
                    "S/B": sb_val,
                    "S/B_err_mc": sb_err,
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
    """Compara diretamente rendimentos físicos, incertezas MC e significância nominal de Asimov entre dois modelos."""
    cols = [
        "kappa",
        "eff_sinal",
        "corte",
        "eff_s_pos",
        "eff_s_signed",
        "S",
        "S_err_mc",
        "B",
        "B_err_mc",
        "S/B",
        "S/sqrt(B)",
        "Z_asimov",
    ]
    cols = [c for c in cols if c in yields_a.columns and c in yields_ref.columns]
    cmp = yields_a[cols].merge(yields_ref[cols], on=["kappa", "eff_sinal"], suffixes=(suffix_a, suffix_ref))
    if f"Z_asimov{suffix_a}" in cmp.columns:
        cmp["razao_Z_asimov"] = cmp[f"Z_asimov{suffix_a}"] / cmp[f"Z_asimov{suffix_ref}"].replace(0.0, np.nan)
    if f"B{suffix_a}" in cmp.columns:
        b_a = cmp[f"B{suffix_a}"].where(cmp[f"B{suffix_a}"] > 0.0, np.nan)
        b_ref = cmp[f"B{suffix_ref}"].where(cmp[f"B{suffix_ref}"] > 0.0, np.nan)
        cmp["reducao_B_fator"] = b_ref / b_a
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
