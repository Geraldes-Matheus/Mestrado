"""Cálculo e diagnóstico de pesos físicos (NLO com sinal), pesos de avaliação e frequência de oversampling."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def ess(w: np.ndarray) -> float:
    """Effective Sample Size: (sum w)^2 / sum(w^2)."""
    w_arr = np.asarray(w, dtype=np.float64)
    denom = float(np.sum(w_arr ** 2))
    if denom <= 0.0:
        return 0.0
    return float((w_arr.sum() ** 2) / denom)


def aggregate_weight(w: np.ndarray, mode: str = "signed") -> float:
    """Agrega os pesos físicos de um pool (subprocesso x era)."""
    w_arr = np.asarray(w, dtype=np.float64)
    if mode == "signed":
        return float(max(w_arr.sum(), 0.0))  # seção de choque líquida
    if mode == "pos":
        return float(np.clip(w_arr, 0.0, None).sum())
    if mode == "abs":
        return float(np.abs(w_arr).sum())
    raise ValueError(f"Modo de peso inválido: {mode!r}. Use 'signed', 'pos' ou 'abs'.")


def class_weight_summary(
    df: pd.DataFrame,
    class_labels: dict[int, str],
    weight_col: str = "w_phys",
) -> pd.DataFrame:
    """Resumo de pesos por classe (soma com sinal, soma positiva, fração negativa e ESS)."""
    rows = []
    for c, name in class_labels.items():
        w = df.loc[df["label"] == c, weight_col].to_numpy(np.float64)
        n = len(w)
        if n == 0:
            continue
        e = ess(w)
        sum_signed = float(w.sum())
        rows.append(
            {
                "classe": name,
                "N": n,
                "soma_signed": sum_signed,
                "soma_pos": float(np.clip(w, 0.0, None).sum()),
                "frac_neg": float((w < 0).mean()),
                "w_max/soma": float(np.abs(w).max() / max(abs(sum_signed), 1e-30)),
                "ESS": e,
                "ESS/N": e / n,
            }
        )
    return pd.DataFrame(rows)


def subprocess_weight_summary(
    df: pd.DataFrame,
    label: int,
    weight_col: str = "w_phys",
) -> pd.DataFrame:
    """Compara fração física com sinal (`share_signed`) vs apenas positivos (`share_pos`) por subprocesso."""
    sub = df[df["label"] == label]
    g = sub.groupby("sample", observed=True)[weight_col]
    t = pd.DataFrame(
        {
            "N": g.size(),
            "soma_signed": g.sum(),
            "soma_pos": g.apply(lambda s: s.clip(lower=0.0).sum()),
            "frac_neg": g.apply(lambda s: (s < 0).mean()),
        }
    )
    t["pos/signed"] = t["soma_pos"] / t["soma_signed"].replace(0.0, np.nan)
    pos_signed = t["soma_signed"].clip(lower=0.0)
    t["share_signed"] = pos_signed / max(float(pos_signed.sum()), 1e-30)
    t["share_pos"] = t["soma_pos"] / max(float(t["soma_pos"].sum()), 1e-30)
    return t.sort_values("share_signed", ascending=False)


def make_eval_weights(
    df: pd.DataFrame,
    class_labels: dict[int, str],
    neg_mode: str = "clip",
    signal_per_kappa: bool = True,
) -> np.ndarray:
    """Constrói peso de AVALIAÇÃO (nunca de treino) com média 1 e soma igual nas 3 classes.

    No sinal (classe 2), divide igualmente entre cada hipótese de `kappa_lambda`.
    """
    w = df["w_phys"].to_numpy(np.float64).copy()
    y = df["label"].to_numpy()
    kl = df["kappa_lambda"].to_numpy(np.float64)

    if neg_mode == "clip":
        w = np.clip(w, 0.0, None)
    elif neg_mode == "abs":
        w = np.abs(w)
    elif neg_mode != "signed":
        raise ValueError(f"neg_mode inválido: {neg_mode!r}")

    out = np.zeros_like(w)
    for c in class_labels:
        m = y == c
        if not m.any():
            continue
        if c == 2 and signal_per_kappa:
            ks = np.unique(np.round(kl[m], 3))
            ks = ks[np.isfinite(ks)]
            for k in ks:
                mk = m & np.isclose(kl, k)
                s_mk = w[mk].sum()
                if s_mk > 0:
                    out[mk] = w[mk] / s_mk / len(ks)
        else:
            s_m = w[m].sum()
            if s_m > 0:
                out[m] = w[m] / s_m

    total = out.sum()
    if total <= 0:
        raise ValueError("Soma dos pesos de avaliação <= 0.")
    return (out * (len(out) / total)).astype(np.float32)


def oversampling_frequency_weights(
    df: pd.DataFrame,
    gen: object,
    class_fraction: dict[int, float] | None = None,
    clip: float | None = None,
) -> np.ndarray:
    """Pesos = frequência relativa de oversampling do gerador (AN-25-103 Seção 5.3.6, Fig. 50).

    Para cada evento: P(classe) * P(pool | classe) / N_pool, normalizado para média 1.
    """
    if class_fraction is None:
        class_fraction = {0: 1.0 / 3.0, 1: 1.0 / 3.0, 2: 1.0 / 3.0}

    y = df["label"].to_numpy()
    sample = df["sample"].astype(str).to_numpy()
    era = df["era"].astype(str).to_numpy()
    kl = df["kappa_lambda"].to_numpy(np.float64)
    w = np.zeros(len(df), dtype=np.float64)

    for label in (0, 1):
        m = y == label
        if not m.any():
            continue
        key = pd.Series(np.char.add(np.char.add(sample[m], "|"), era[m]))
        prob = dict(zip(gen.bkg_keys[label], gen.bkg_probs[label]))
        counts = key.map(key.value_counts()).to_numpy(np.float64)
        w[m] = class_fraction[label] * key.map(prob).fillna(0.0).to_numpy(np.float64) / counts

    kappas = list(gen.hh_pools)
    for k in kappas:
        mk = (y == 2) & np.isclose(kl, float(k))
        if not mk.any():
            continue
        key = pd.Series(era[mk])
        prob = dict(zip(gen.hh_keys[k], gen.hh_probs[k]))
        counts = key.map(key.value_counts()).to_numpy(np.float64)
        w[mk] = (class_fraction[2] / len(kappas)) * key.map(prob).fillna(0.0).to_numpy(np.float64) / counts

    if not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("Pesos de frequência de oversampling inválidos (NaN/Inf ou soma <= 0).")

    w *= len(w) / w.sum()
    if clip is not None:
        w = np.clip(w, 0.0, float(clip))
        w *= len(w) / w.sum()
    return w.astype(np.float32)


def weight_report(
    df: pd.DataFrame,
    w: np.ndarray,
    class_labels: dict[int, str],
    title: str = "",
    print_fn: Callable[[str], None] | None = print,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gera tabelas de diagnóstico de um vetor de pesos por classe e por kappa_lambda no sinal."""
    y = df["label"].to_numpy()
    rows = []
    for c, name in class_labels.items():
        wc = w[y == c].astype(np.float64)
        if len(wc) == 0:
            continue
        e = ess(wc)
        rows.append(
            {
                "classe": name,
                "N": len(wc),
                "soma_w": float(wc.sum()),
                "w_med": float(np.median(wc)),
                "w_max": float(wc.max()),
                "ESS": e,
                "ESS/N": e / len(wc),
            }
        )
    cls_df = pd.DataFrame(rows)

    hh = y == 2
    kl = np.round(df["kappa_lambda"].to_numpy(np.float64)[hh], 2)
    kappa_df = pd.DataFrame({"kappa": kl, "w": w[hh].astype(np.float64)}).groupby("kappa")["w"].agg(N="count", soma="sum")
    if len(kappa_df) > 0 and kappa_df["soma"].sum() != 0:
        kappa_df["frac_peso_HH"] = kappa_df["soma"] / kappa_df["soma"].sum()

    if print_fn is not None:
        if title:
            print_fn(f"\n{title}")
        print_fn(cls_df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        print_fn("Peso do sinal por κ:")
        print_fn(kappa_df.to_string(float_format=lambda x: f"{x:.4g}"))

    return cls_df, kappa_df
