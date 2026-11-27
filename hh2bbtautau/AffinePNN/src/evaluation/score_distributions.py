"""Predição por hipótese de kappa_lambda, teste KS de overtraining e estudo de mass sculpting (AN Seção 5.5)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from tensorflow import keras

from src.data.preprocessing import make_inputs


def score_per_kappa(
    model: keras.Model,
    df: pd.DataFrame,
    x_cont: np.ndarray,
    x_cat: dict[str, np.ndarray],
    kappas: Sequence[float] = (0.0, 1.0, 2.45, 5.0),
    kappa_input: float | None = None,
    kappa_norm: float = 5.0,
    batch_size: int = 8192,
) -> dict[float, dict[str, np.ndarray]]:
    """Avalia `P(HH)` para cada hipótese `k`: sinal gerado em `k` contra TODO o fundo (DY + ttbar) recebendo `k`."""
    labels = df["label"].to_numpy()
    kl = df["kappa_lambda"].to_numpy(np.float64)
    w = df["w_phys"].to_numpy(np.float64)
    is_hh = labels == 2

    out: dict[float, dict[str, np.ndarray]] = {}
    for k in kappas:
        sig = is_hh & np.isclose(kl, float(k))
        idx = np.where(sig | (~is_hh))[0]
        k_in = float(k) if kappa_input is None else float(kappa_input)
        p = model.predict(
            make_inputs(x_cont, x_cat, k_in, kappa_norm=kappa_norm, idx=idx),
            batch_size=int(batch_size),
            verbose=0,
        )
        out[float(k)] = {
            "y": sig[idx].astype(np.int8),
            "s": p[:, 2].astype(np.float64),
            "probs": p.astype(np.float32),
            "pred_cls": np.argmax(p, axis=1).astype(np.int8),
            "w": w[idx],
            "cls": labels[idx],
            "idx": idx,
        }
    return out


def overtraining_ks(
    model: keras.Model,
    train_df: pd.DataFrame,
    x_cont_train: np.ndarray,
    x_cat_train: dict[str, np.ndarray],
    ev_inputs: dict[str, np.ndarray],
    ev_y: np.ndarray,
    class_labels: dict[int, str],
    kappa_norm: float = 5.0,
    n_samples: int = 100_000,
    batch_size: int = 8192,
    seed: int = 42,
    verbose: bool = True,
    return_scores: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Teste de Kolmogorov-Smirnov (KS) comparando scores de treino vs validação/teste em modo eval."""
    rng = np.random.default_rng(seed)
    n_sub = min(int(n_samples), len(train_df))
    idx = rng.choice(len(train_df), size=n_sub, replace=False)

    p_tr = model.predict(
        make_inputs(
            x_cont_train,
            x_cat_train,
            train_df["kappa_eval"].to_numpy()[idx],
            kappa_norm=kappa_norm,
            idx=idx,
        ),
        batch_size=batch_size,
        verbose=0,
    )
    y_tr = train_df["label"].to_numpy()[idx]
    p_ev = model.predict(ev_inputs, batch_size=batch_size, verbose=0)

    rows = []
    for c, name in class_labels.items():
        stat, pval = ks_2samp(p_tr[y_tr == c, c], p_ev[ev_y == c, c])
        status = "OK" if stat < 0.03 else "ATENCAO"
        rows.append({"classe": name, "ks_stat": float(stat), "p_value": float(pval), "status": status})
        if verbose:
            print(f"  {name:6s}: KS = {stat:.4f}  (p = {pval:.2e})  {'OK' if stat < 0.03 else 'ATENÇÃO: olhar'}")
    ks_df = pd.DataFrame(rows)
    if return_scores:
        return ks_df, {"p_train": p_tr, "y_train": y_tr, "p_eval": p_ev, "y_eval": np.asarray(ev_y)}
    return ks_df


def reconstruct_invariant_masses(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstrói m_tautau, m_bb e m_HH a partir dos 4-vetores regressados (`reg_dnn_moe_*`) nos parquets."""
    res: dict[str, np.ndarray] = {}
    mapping = {
        "m_tautau": "reg_dnn_moe_htt",
        "m_bb": "reg_dnn_moe_hbb",
        "m_hh": "reg_dnn_moe_htthbb",
    }
    for out_col, prefix in mapping.items():
        cols = [f"{prefix}_{c}" for c in ("e", "px", "py", "pz")]
        if all(c in df.columns for c in cols):
            e = df[cols[0]].to_numpy(np.float64)
            px = df[cols[1]].to_numpy(np.float64)
            py = df[cols[2]].to_numpy(np.float64)
            pz = df[cols[3]].to_numpy(np.float64)
            m2 = e ** 2 - (px ** 2 + py ** 2 + pz ** 2)
            valid = np.isfinite(m2) & (e > 0)
            m = np.where(valid, np.sqrt(np.maximum(m2, 0.0)), np.nan)
            res[out_col] = m.astype(np.float32)
    return pd.DataFrame(res, index=df.index)


def mass_sculpting_slices(
    df: pd.DataFrame,
    scores_hh: np.ndarray,
    score_edges: Sequence[float] = (0.0, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0001),
) -> pd.DataFrame:
    """Agrupa estatísticas de m_tautau, m_bb e m_hh nas fatias de score da AN-25-103 (Seção 5.5)."""
    masses = reconstruct_invariant_masses(df)
    y = df["label"].to_numpy()
    w = np.clip(df["w_phys"].to_numpy(np.float64), 0.0, None)
    s = np.asarray(scores_hh, dtype=np.float64)

    rows = []
    for lo, hi in zip(score_edges[:-1], score_edges[1:]):
        hi_label = min(hi, 1.0)
        slice_name = f"[{lo:.2f}, {hi_label:.2f}]"
        for bkg_cls, cls_name in ((0, "DY"), (1, "ttbar"), (2, "HH")):
            m = (y == bkg_cls) & (s >= lo) & (s < hi)
            if int(m.sum()) < 5:
                continue
            wm = w[m]
            w_sum = wm.sum()
            row: dict[str, object] = {"slice": slice_name, "classe": cls_name, "N": int(m.sum())}
            for mcol in masses.columns:
                vals = masses[mcol].to_numpy(np.float64)[m]
                ok = np.isfinite(vals) & (vals > 0)
                if ok.any() and wm[ok].sum() > 0:
                    mu = float(np.average(vals[ok], weights=wm[ok]))
                    var = float(np.average((vals[ok] - mu) ** 2, weights=wm[ok]))
                    row[f"{mcol}_mean"] = mu
                    row[f"{mcol}_std"] = float(np.sqrt(max(var, 0.0)))
                else:
                    row[f"{mcol}_mean"] = np.nan
                    row[f"{mcol}_std"] = np.nan
            row["w_sum"] = float(w_sum)
            rows.append(row)
    return pd.DataFrame(rows)
