"""Plots de desempenho por hipótese de κ_λ: Precision / Recall / F1 vs score, ROC por κ_λ,
distribuições do score em número de eventos (escala log) e matrizes de confusão.

Todas as funções aceitam o dicionário `scores` produzido por
`src.evaluation.score_distributions.score_per_kappa` / `run_full_evaluation(...)["scores"]`:

    scores[nome_modelo][kappa] = {"y", "s", "w", "cls", "probs", ...}

- Passando um único modelo em `models=[...]` → plots da rede pura (ex.: só AffinePNN).
- Passando dois ou mais modelos             → plots comparativos (ex.: AffinePNN vs DNN_AN).

Definições (sinal = HH gerado em κ_λ; fundo = DY + ttbar), com corte `score >= t`:
    recall    = ε_S(t) = S(t) / S_tot
    ε_B(t)    = B(t) / B_tot
    precision = ε_S / (ε_S + ε_B)      (modo "balanced": sinal e fundo com mesmo peso total)
              = S / (S + B)            (modo "physical": rendimentos físicos com w_phys)
    F1        = 2 P R / (P + R)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.plotting import MODEL_COLORS, MODEL_STYLES, _finalize_fig
from src.evaluation.roc import auc_from_roc, weighted_roc

ScoresDict = dict[str, dict[float, dict[str, np.ndarray]]]

KAPPA_COLORS = {0.0: "#9467bd", 1.0: "#2ca02c", 2.45: "#d62728", 5.0: "#17becf"}
METRIC_LABELS = {
    "precision": "Precision",
    "recall": r"Recall ($\epsilon_S$)",
    "f1": "F1-score",
    "eps_b": r"Eficiência de fundo $\epsilon_B$",
    "accuracy": "Acurácia balanceada",
}
METRIC_COLORS = {"precision": "#1f77b4", "recall": "#2ca02c", "f1": "#d62728"}


# ------------------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------------------
def _kappa_color(k: float, i: int = 0) -> str:
    for kk, c in KAPPA_COLORS.items():
        if np.isclose(kk, k):
            return c
    return f"C{i % 10}"


def _model_color(name: str, i: int = 0) -> str:
    return MODEL_COLORS.get(name, f"C{i % 10}")


def _model_style(name: str, i: int = 0) -> str:
    return MODEL_STYLES.get(name, ["-", "--", "-.", ":"][i % 4])


def _ktag(k: float) -> str:
    return f"{float(k):g}".replace(".", "p").replace("-", "m")


def _resolve(scores: ScoresDict, models: Sequence[str] | None, kappas: Sequence[float] | None) -> tuple[list[str], list[float]]:
    names = list(models) if models else list(scores.keys())
    missing = [m for m in names if m not in scores]
    if missing:
        raise KeyError(f"Modelos ausentes em `scores`: {missing}. Disponíveis: {list(scores)}")
    ks = [float(k) for k in kappas] if kappas is not None else [float(k) for k in scores[names[0]].keys()]
    return names, ks


def _get(scores: ScoresDict, name: str, k: float) -> dict[str, np.ndarray]:
    for kk, d in scores[name].items():
        if np.isclose(float(kk), float(k)):
            return d
    raise KeyError(f"κ_λ = {k} não encontrado para o modelo {name}")


def _title_models(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else " vs ".join(names)


# ------------------------------------------------------------------------------
# Cálculo das métricas em função do corte no score
# ------------------------------------------------------------------------------
def metrics_vs_score(
    y: np.ndarray,
    s: np.ndarray,
    w: np.ndarray | None = None,
    n_thr: int = 201,
    mode: str = "balanced",
) -> dict[str, Any]:
    """Precision, Recall, F1, ε_B, rejeição e acurácia balanceada para cortes `s >= t`, t ∈ [0, 1].

    Usa somas cumulativas (O(N log N)) e preserva pesos negativos de NLO nas somas.
    """
    if mode not in ("balanced", "physical"):
        raise ValueError("mode deve ser 'balanced' ou 'physical'")
    y_b = np.asarray(y).astype(bool)
    s_a = np.asarray(s, dtype=np.float64)
    w_a = np.ones_like(s_a) if w is None else np.asarray(w, dtype=np.float64)

    order = np.argsort(s_a, kind="mergesort")
    s_sorted = s_a[order]
    sig_w = np.where(y_b, w_a, 0.0)[order]
    bkg_w = np.where(~y_b, w_a, 0.0)[order]
    cs_sig = np.r_[0.0, np.cumsum(sig_w)]
    cs_bkg = np.r_[0.0, np.cumsum(bkg_w)]
    s_tot, b_tot = cs_sig[-1], cs_bkg[-1]

    thr = np.linspace(0.0, 1.0, int(n_thr))
    n_below = np.searchsorted(s_sorted, thr, side="left")
    S = s_tot - cs_sig[n_below]
    B = b_tot - cs_bkg[n_below]

    recall = np.clip(S / s_tot, 0.0, 1.0) if s_tot != 0 else np.zeros_like(thr)
    eps_b = np.clip(B / b_tot, 0.0, 1.0) if b_tot != 0 else np.zeros_like(thr)

    if mode == "balanced":
        den = recall + eps_b
        precision = np.divide(recall, den, out=np.ones_like(den), where=den > 1e-12)
    else:
        den = S + B
        precision = np.clip(np.divide(S, den, out=np.ones_like(den), where=np.abs(den) > 1e-30), 0.0, 1.0)

    den_f1 = precision + recall
    f1 = np.divide(2.0 * precision * recall, den_f1, out=np.zeros_like(den_f1), where=den_f1 > 1e-12)
    accuracy = 0.5 * (recall + (1.0 - eps_b))
    rejection = 1.0 / np.maximum(eps_b, 1e-6)

    best = int(np.argmax(f1))
    return {
        "thr": thr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "eps_b": eps_b,
        "rejection": rejection,
        "accuracy": accuracy,
        "S": S,
        "B": B,
        "best_idx": best,
        "best_thr": float(thr[best]),
        "best_f1": float(f1[best]),
        "mode": mode,
    }


def _curves(scores: ScoresDict, names: Sequence[str], ks: Sequence[float], mode: str, n_thr: int) -> dict[str, dict[float, dict]]:
    out: dict[str, dict[float, dict]] = {}
    for n in names:
        out[n] = {}
        for k in ks:
            d = _get(scores, n, k)
            out[n][k] = metrics_vs_score(d["y"], d["s"], d["w"], n_thr=n_thr, mode=mode)
    return out


def working_point_table(
    scores: ScoresDict,
    models: Sequence[str] | None = None,
    kappas: Sequence[float] | None = None,
    mode: str = "balanced",
    sig_eff_points: Sequence[float] = (0.5, 0.8, 0.9),
    n_thr: int = 1001,
) -> pd.DataFrame:
    """Tabela com AUC_w, melhor F1 (e seu corte), precision/recall no ótimo e rejeição 1/ε_B em ε_S fixos."""
    names, ks = _resolve(scores, models, kappas)
    rows = []
    for n in names:
        for k in ks:
            d = _get(scores, n, k)
            c = metrics_vs_score(d["y"], d["s"], d["w"], n_thr=n_thr, mode=mode)
            fpr, tpr = weighted_roc(d["y"], d["s"], d["w"])
            b = c["best_idx"]
            row = {
                "modelo": n,
                "kappa": k,
                "auc_w": auc_from_roc(fpr, tpr),
                "best_thr_f1": c["best_thr"],
                "best_f1": c["best_f1"],
                "precision@best": float(c["precision"][b]),
                "recall@best": float(c["recall"][b]),
                "eps_b@best": float(c["eps_b"][b]),
            }
            tpr_m = np.maximum.accumulate(np.clip(tpr, 0.0, 1.0))
            fpr_m = np.maximum.accumulate(np.clip(fpr, 0.0, 1.0))
            for e in sig_eff_points:
                eb = float(np.interp(e, tpr_m, fpr_m))
                row[f"rej@effS={e:g}"] = 1.0 / max(eb, 1e-6)
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# 1) Precision / Recall / F1 vs score
# ------------------------------------------------------------------------------
def plot_metric_vs_score_cmp(
    scores: ScoresDict,
    metric: str = "f1",
    models: Sequence[str] | None = None,
    kappas: Sequence[float] | None = None,
    mode: str = "balanced",
    n_thr: int = 201,
    split_label: str = "",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Uma métrica (precision | recall | f1 | eps_b | accuracy) vs score, um painel por κ_λ, uma curva por modelo."""
    names, ks = _resolve(scores, models, kappas)
    curves = _curves(scores, names, ks, mode, n_thr)
    fig, axes = plt.subplots(1, len(ks), figsize=(4.4 * len(ks), 4.2), sharey=True, squeeze=False)
    for j, k in enumerate(ks):
        ax = axes[0, j]
        for i, n in enumerate(names):
            c = curves[n][k]
            lab = n
            if metric == "f1":
                lab = f"{n} (máx {c['best_f1']:.3f} @ {c['best_thr']:.2f})"
            ax.plot(c["thr"], c[metric], color=_model_color(n, i), ls=_model_style(n, i), lw=2.0, label=lab)
            if metric == "f1":
                ax.plot(c["best_thr"], c["best_f1"], "o", color=_model_color(n, i), ms=5)
        ax.set_title(rf"$\kappa_\lambda = {k:g}$", fontsize=11, fontweight="bold")
        ax.set_xlabel(r"Corte no score da rede $P(\mathrm{HH})$")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.03)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.8, loc="best")
    axes[0, 0].set_ylabel(METRIC_LABELS.get(metric, metric))
    sfx = f" — {split_label}" if split_label else ""
    fig.suptitle(f"{METRIC_LABELS.get(metric, metric)} vs score — {_title_models(names)}{sfx}", fontsize=12.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


def plot_metric_vs_score_kappas(
    scores: ScoresDict,
    model_name: str,
    metric: str = "f1",
    kappas: Sequence[float] | None = None,
    mode: str = "balanced",
    n_thr: int = 201,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Rede pura: uma métrica vs score com todas as hipóteses de κ_λ sobrepostas no mesmo eixo."""
    names, ks = _resolve(scores, [model_name], kappas)
    curves = _curves(scores, names, ks, mode, n_thr)[model_name]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for i, k in enumerate(ks):
        c = curves[k]
        lab = rf"$\kappa_\lambda={k:g}$"
        if metric == "f1":
            lab += f" (máx {c['best_f1']:.3f} @ {c['best_thr']:.2f})"
        ax.plot(c["thr"], c[metric], color=_kappa_color(k, i), lw=2.0, label=lab)
        if metric == "f1":
            ax.plot(c["best_thr"], c["best_f1"], "o", color=_kappa_color(k, i), ms=5)
    ax.set_xlabel(r"Corte no score da rede $P(\mathrm{HH})$")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)
    ax.set_title(f"{model_name}: {METRIC_LABELS.get(metric, metric)} vs score (todas as hipóteses)", fontweight="bold")
    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


def plot_prf1_vs_score(
    scores: ScoresDict,
    models: Sequence[str] | None = None,
    kappas: Sequence[float] | None = None,
    mode: str = "balanced",
    n_thr: int = 201,
    split_label: str = "",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Precision + Recall + F1 no mesmo painel (um painel por κ_λ). Modelos diferenciados pelo estilo de linha."""
    names, ks = _resolve(scores, models, kappas)
    curves = _curves(scores, names, ks, mode, n_thr)
    fig, axes = plt.subplots(1, len(ks), figsize=(4.6 * len(ks), 4.4), sharey=True, squeeze=False)
    for j, k in enumerate(ks):
        ax = axes[0, j]
        for i, n in enumerate(names):
            c = curves[n][k]
            ls = _model_style(n, i)
            short = n if len(names) == 1 else n.replace("AffinePNN", "Affine").replace("DNN_AN", "AN")
            for met in ("precision", "recall", "f1"):
                ax.plot(c["thr"], c[met], color=METRIC_COLORS[met], ls=ls, lw=1.9,
                        label=f"{METRIC_LABELS[met].split(' (')[0]} ({short})" if len(names) > 1 else METRIC_LABELS[met].split(" (")[0])
            ax.axvline(c["best_thr"], color="gray", ls=ls, lw=0.9, alpha=0.7)
            ax.plot(c["best_thr"], c["best_f1"], "o", color=METRIC_COLORS["f1"], ms=5)
        ax.set_title(rf"$\kappa_\lambda = {k:g}$", fontsize=11, fontweight="bold")
        ax.set_xlabel(r"Corte no score da rede $P(\mathrm{HH})$")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.03)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Métrica")
    axes[0, 0].legend(fontsize=7.6, ncol=1 if len(names) == 1 else 2, loc="lower left")
    sfx = f" — {split_label}" if split_label else ""
    fig.suptitle(f"Precision, Recall e F1 vs score — {_title_models(names)}{sfx}", fontsize=12.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


# ------------------------------------------------------------------------------
# 2) Curvas ROC por κ_λ
# ------------------------------------------------------------------------------
def plot_roc_kappa(
    scores: ScoresDict,
    kappa: float,
    models: Sequence[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """ROC de um único κ_λ: (esq.) ε_S vs ε_B linear com AUC_w; (dir.) rejeição 1/ε_B (log) vs ε_S."""
    names, _ = _resolve(scores, models, [kappa])
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for i, n in enumerate(names):
        d = _get(scores, n, kappa)
        fpr, tpr = weighted_roc(d["y"], d["s"], d["w"])
        auc_v = auc_from_roc(fpr, tpr)
        col, ls = _model_color(n, i), _model_style(n, i)
        ax[0].plot(fpr, tpr, color=col, ls=ls, lw=2.0, label=f"{n}: AUC$_w$ = {auc_v:.4f}")
        m = (tpr > 0.02) & (fpr > 0)
        ax[1].plot(tpr[m], 1.0 / fpr[m], color=col, ls=ls, lw=2.0, label=n)
    ax[0].plot([0, 1], [0, 1], ":", color="gray", lw=1.0, label="Aleatório")
    ax[0].set_xlabel(r"Eficiência do fundo $\epsilon_B$")
    ax[0].set_ylabel(r"Eficiência do sinal $\epsilon_S$")
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1.01)
    ax[0].grid(alpha=0.3)
    ax[0].legend(loc="lower right", fontsize=9)
    ax[0].set_title(rf"Curva ROC — $\kappa_\lambda = {kappa:g}$", fontweight="bold")

    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"Eficiência do sinal $\epsilon_S$")
    ax[1].set_ylabel(r"Rejeição do fundo $1/\epsilon_B$")
    ax[1].set_xlim(0.0, 1.0)
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=9)
    ax[1].set_title(rf"Rejeição vs eficiência — $\kappa_\lambda = {kappa:g}$", fontweight="bold")
    fig.suptitle(_title_models(names), fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


def plot_roc_all_kappas(
    scores: ScoresDict,
    model_name: str,
    kappas: Sequence[float] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Rede pura: ROC de todas as hipóteses de κ_λ sobrepostas (linear + rejeição log)."""
    _, ks = _resolve(scores, [model_name], kappas)
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for i, k in enumerate(ks):
        d = _get(scores, model_name, k)
        fpr, tpr = weighted_roc(d["y"], d["s"], d["w"])
        col = _kappa_color(k, i)
        ax[0].plot(fpr, tpr, color=col, lw=2.0, label=rf"$\kappa_\lambda={k:g}$: AUC$_w$={auc_from_roc(fpr, tpr):.4f}")
        m = (tpr > 0.02) & (fpr > 0)
        ax[1].plot(tpr[m], 1.0 / fpr[m], color=col, lw=2.0, label=rf"$\kappa_\lambda={k:g}$")
    ax[0].plot([0, 1], [0, 1], ":", color="gray", lw=1.0)
    ax[0].set_xlabel(r"Eficiência do fundo $\epsilon_B$")
    ax[0].set_ylabel(r"Eficiência do sinal $\epsilon_S$")
    ax[0].grid(alpha=0.3)
    ax[0].legend(loc="lower right", fontsize=8.5)
    ax[0].set_title(f"{model_name}: ROC por hipótese de κ_λ", fontweight="bold")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"Eficiência do sinal $\epsilon_S$")
    ax[1].set_ylabel(r"Rejeição do fundo $1/\epsilon_B$")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=8.5)
    ax[1].set_title(f"{model_name}: rejeição vs eficiência", fontweight="bold")
    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


# ------------------------------------------------------------------------------
# 3) Score da rede vs número de eventos (escala log)
# ------------------------------------------------------------------------------
def _hist(values: np.ndarray, weights: np.ndarray | None, bins: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(values, bins=bins, weights=weights)
    return h.astype(np.float64)


def _draw_score_events(
    ax: plt.Axes,
    d: dict[str, np.ndarray],
    class_labels: dict[int, str],
    bins: np.ndarray,
    weighted: bool,
    signal_scale: float | str | None,
    kappa: float,
) -> None:
    s, w, cls, y = d["s"], d["w"], d["cls"], d["y"].astype(bool)
    wts = w if weighted else None
    bkg_classes = [c for c in class_labels if c != 2]
    bkg_colors = {0: "#7fb3d5", 1: "#f5b041"}
    bottom = np.zeros(len(bins) - 1)
    b_tot = 0.0
    for c in bkg_classes:
        m = (cls == c) & (~y)
        h = np.clip(_hist(s[m], None if wts is None else wts[m], bins), 0.0, None)
        b_tot += h.sum()
        ax.stairs(bottom + h, bins, baseline=bottom, fill=True, color=bkg_colors.get(c, f"C{c}"), alpha=0.85,
                  label=f"{class_labels[c]} (Σ={h.sum():.3g})")
        bottom = bottom + h
    hs = np.clip(_hist(s[y], None if wts is None else wts[y], bins), 0.0, None)
    scale = 1.0
    if signal_scale == "auto":
        scale = b_tot / hs.sum() if hs.sum() > 0 else 1.0
    elif signal_scale is not None:
        scale = float(signal_scale)
    lab = rf"HH $\kappa_\lambda={kappa:g}$ (Σ={hs.sum():.3g})"
    if not np.isclose(scale, 1.0):
        lab += f" ×{scale:.3g}"
    ax.stairs(hs * scale, bins, color="#c0392b", lw=2.2, label=lab)
    ax.set_yscale("log")
    pos = np.r_[bottom[bottom > 0], (hs * scale)[hs > 0]]
    if pos.size:
        ax.set_ylim(max(pos.min() * 0.3, 1e-6), pos.max() * 30)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"Score da rede $P(\mathrm{HH})$")
    ax.set_ylabel("Eventos esperados (Σ w$_{phys}$)" if weighted else "Eventos MC (não ponderados)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.8, loc="upper center")


def plot_score_events_log(
    scores: ScoresDict,
    kappa: float,
    class_labels: dict[int, str],
    models: Sequence[str] | None = None,
    n_bins: int = 40,
    weighted: bool = True,
    signal_scale: float | str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Histograma do score (eventos em escala log): fundos empilhados + sinal HH, um painel por modelo."""
    names, _ = _resolve(scores, models, [kappa])
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    fig, axes = plt.subplots(1, len(names), figsize=(6.4 * len(names), 4.8), sharey=True, squeeze=False)
    for i, n in enumerate(names):
        _draw_score_events(axes[0, i], _get(scores, n, kappa), class_labels, bins, weighted, signal_scale, kappa)
        axes[0, i].set_title(rf"{n} — $\kappa_\lambda = {kappa:g}$", fontweight="bold")
    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


def plot_score_events_log_kappas(
    scores: ScoresDict,
    model_name: str,
    class_labels: dict[int, str],
    kappas: Sequence[float] | None = None,
    n_bins: int = 40,
    weighted: bool = True,
    signal_scale: float | str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Rede pura: grade de histogramas do score em log, um painel por κ_λ."""
    _, ks = _resolve(scores, [model_name], kappas)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ncols = min(len(ks), 2)
    nrows = int(np.ceil(len(ks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.6 * nrows), squeeze=False)
    for j, k in enumerate(ks):
        ax = axes.ravel()[j]
        _draw_score_events(ax, _get(scores, model_name, k), class_labels, bins, weighted, signal_scale, k)
        ax.set_title(rf"$\kappa_\lambda = {k:g}$", fontweight="bold")
    for ax in axes.ravel()[len(ks):]:
        ax.axis("off")
    fig.suptitle(f"{model_name}: score da rede vs eventos (log)", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


def plot_score_events_overlay(
    scores: ScoresDict,
    kappa: float,
    models: Sequence[str] | None = None,
    n_bins: int = 40,
    weighted: bool = True,
    density: bool = False,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Comparação no mesmo eixo: sinal (linha cheia) e fundo total (preenchido/tracejado) de cada modelo."""
    names, _ = _resolve(scores, models, [kappa])
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for i, n in enumerate(names):
        d = _get(scores, n, kappa)
        y = d["y"].astype(bool)
        w = d["w"] if weighted else None
        hs = np.clip(_hist(d["s"][y], None if w is None else w[y], bins), 0, None)
        hb = np.clip(_hist(d["s"][~y], None if w is None else w[~y], bins), 0, None)
        if density:
            hs = hs / max(hs.sum(), 1e-30)
            hb = hb / max(hb.sum(), 1e-30)
        col = _model_color(n, i)
        ax.stairs(hs, bins, color=col, lw=2.2, label=f"Sinal HH — {n}")
        ax.stairs(hb, bins, color=col, lw=1.8, ls="--", label=f"Fundo (DY+tt) — {n}")
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"Score da rede $P(\mathrm{HH})$")
    ax.set_ylabel("Fração de eventos" if density else ("Eventos esperados (Σ w$_{phys}$)" if weighted else "Eventos MC"))
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5)
    ax.set_title(rf"{_title_models(names)} — $\kappa_\lambda = {kappa:g}$", fontweight="bold")
    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


# ------------------------------------------------------------------------------
# 4) Matriz de confusão (DY / ttbar / HH) e resumo de pontos de trabalho
# ------------------------------------------------------------------------------
def plot_confusion_matrices(
    scores: ScoresDict,
    kappa: float,
    class_labels: dict[int, str],
    models: Sequence[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Matriz de confusão multiclasse (argmax), normalizada por classe verdadeira e ponderada por w_phys ≥ 0."""
    names, _ = _resolve(scores, models, [kappa])
    cl = sorted(class_labels)
    fig, axes = plt.subplots(1, len(names), figsize=(4.9 * len(names), 4.3), squeeze=False)
    for i, n in enumerate(names):
        d = _get(scores, n, kappa)
        pred = d.get("pred_cls", np.argmax(d["probs"], axis=1))
        w = np.clip(d["w"], 0.0, None)
        cm = np.zeros((len(cl), len(cl)))
        for a, ct in enumerate(cl):
            mt = d["cls"] == ct
            tot = w[mt].sum()
            for b, cp in enumerate(cl):
                cm[a, b] = w[mt & (pred == cp)].sum() / tot if tot > 0 else 0.0
        ax = axes[0, i]
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for a in range(len(cl)):
            for b in range(len(cl)):
                ax.text(b, a, f"{cm[a, b]:.2f}", ha="center", va="center", color="white" if cm[a, b] > 0.55 else "black", fontsize=10)
        names_cl = [class_labels[c] for c in cl]
        ax.set_xticks(range(len(cl)), names_cl)
        ax.set_yticks(range(len(cl)), names_cl)
        ax.set_xlabel("Classe prevista")
        ax.set_ylabel("Classe verdadeira")
        ax.set_title(rf"{n} — $\kappa_\lambda={kappa:g}$", fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


def plot_working_point_summary(
    table: pd.DataFrame,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Barras por κ_λ: melhor F1, corte ótimo e rejeição 1/ε_B em ε_S = 0.8 para cada modelo."""
    names = list(dict.fromkeys(table["modelo"]))
    ks = sorted(table["kappa"].unique())
    rej_col = next((c for c in table.columns if c.startswith("rej@effS=0.8")), None) or [c for c in table.columns if c.startswith("rej@")][0]
    panels = [("best_f1", "Melhor F1-score"), ("best_thr_f1", "Corte ótimo (máx. F1)"), ("auc_w", "AUC$_w$"), (rej_col, f"Rejeição 1/ε_B ({rej_col.split('@')[1]})")]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    x = np.arange(len(ks))
    width = 0.75 / max(len(names), 1)
    for ax, (col, title) in zip(axes, panels):
        for i, n in enumerate(names):
            sub = table[table["modelo"] == n].set_index("kappa").reindex(ks)
            vals = sub[col].to_numpy(np.float64)
            bars = ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width, color=_model_color(n, i), label=n, alpha=0.9)
            for bb in bars:
                ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height(), f"{bb.get_height():.3g}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x, [rf"$\kappa_\lambda={k:g}$" for k in ks])
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
        if col == rej_col:
            ax.set_yscale("log")
    axes[0].legend(fontsize=8.5)
    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


# ------------------------------------------------------------------------------
# 5) Relatório completo (rede pura + comparação)
# ------------------------------------------------------------------------------
def generate_performance_report(
    scores: ScoresDict,
    out_dir: str | Path,
    class_labels: dict[int, str],
    kappas: Sequence[float] | None = None,
    reference: str = "AffinePNN",
    baseline: str | None = "DNN_AN",
    mode: str = "balanced",
    split_label: str = "",
    fmt: str = "png",
    show: bool = False,
) -> dict[str, Any]:
    """Gera e salva todos os plots de desempenho em duas pastas:

    - `<out_dir>/<reference>_pure/`            → rede de referência sozinha (ex.: AffinePNN pura)
    - `<out_dir>/<reference>_vs_<baseline>/`   → comparação direta (ex.: AffinePNN vs DNN_AN)

    Também salva `working_points.csv` em cada pasta. Retorna caminhos e tabelas.
    """
    out = Path(out_dir)
    _, ks = _resolve(scores, [reference], kappas)
    groups: list[tuple[str, list[str]]] = [(f"{reference}_pure", [reference])]
    if baseline and baseline in scores:
        groups.append((f"{reference}_vs_{baseline}", [reference, baseline]))

    saved: dict[str, list[Path]] = {}
    tables: dict[str, pd.DataFrame] = {}
    for folder, names in groups:
        d = out / folder
        d.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        def _p(name: str) -> Path:
            p = d / f"{name}.{fmt}"
            paths.append(p)
            return p

        for met in ("precision", "recall", "f1"):
            plot_metric_vs_score_cmp(scores, met, names, ks, mode=mode, split_label=split_label, save_path=_p(f"{met}_vs_score"), show=show)
        plot_prf1_vs_score(scores, names, ks, mode=mode, split_label=split_label, save_path=_p("precision_recall_f1_vs_score"), show=show)

        for k in ks:
            tag = _ktag(k)
            plot_roc_kappa(scores, k, names, save_path=_p(f"roc_kl{tag}"), show=show)
            plot_score_events_log(scores, k, class_labels, names, weighted=True, save_path=_p(f"score_events_log_kl{tag}"), show=show)
            plot_score_events_log(scores, k, class_labels, names, weighted=False, save_path=_p(f"score_mc_counts_log_kl{tag}"), show=show)
            plot_confusion_matrices(scores, k, class_labels, names, save_path=_p(f"confusion_matrix_kl{tag}"), show=show)
            if len(names) > 1:
                plot_score_events_overlay(scores, k, names, density=True, save_path=_p(f"score_shape_overlay_kl{tag}"), show=show)

        if len(names) == 1:
            n = names[0]
            for met in ("precision", "recall", "f1"):
                plot_metric_vs_score_kappas(scores, n, met, ks, mode=mode, save_path=_p(f"{met}_vs_score_all_kappas"), show=show)
            plot_roc_all_kappas(scores, n, ks, save_path=_p("roc_all_kappas"), show=show)
            plot_score_events_log_kappas(scores, n, class_labels, ks, save_path=_p("score_events_log_all_kappas"), show=show)

        tab = working_point_table(scores, names, ks, mode=mode)
        tab.to_csv(d / "working_points.csv", index=False)
        plot_working_point_summary(tab, save_path=_p("working_points_summary"), show=show)
        saved[folder] = paths
        tables[folder] = tab

    return {"paths": saved, "tables": tables}
