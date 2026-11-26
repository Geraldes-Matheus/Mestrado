"""Funções modulares de plotagem para notebooks e scripts (com salvamento opcional em PNG/PDF)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.roc import auc_from_roc, compute_curves_for_kappa, weighted_roc
from src.evaluation.significance import ams_significance_scan

MODEL_COLORS = {
    "AffinePNN": "#1f77b4",
    "DNN_AN": "#ff7f0e",
    "PNN_Concat": "#2ca02c",
}
MODEL_STYLES = {
    "AffinePNN": "-",
    "DNN_AN": "--",
    "PNN_Concat": "-.",
}


def _finalize_fig(fig: plt.Figure, save_path: str | Path | None = None, show: bool = True, dpi: int = 160) -> plt.Figure:
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_training_history(
    histories: dict[str, dict[str, list[float]]],
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plota curvas de loss de treino / val_loss ponderada e schedule logarítmico de learning rate."""
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for name, h in histories.items():
        col = MODEL_COLORS.get(name)
        epochs = np.arange(1, len(h["loss"]) + 1)
        ax[0].plot(epochs, h["loss"], lw=1.8, color=col, label=f"{name} treino")
        if "val_loss" in h:
            ax[0].plot(epochs, h["val_loss"], "--", lw=1.8, color=col, label=f"{name} val (ponderada)")
        if "learning_rate" in h:
            ax[1].plot(epochs, h["learning_rate"], lw=1.8, color=col, label=name)

    ax[0].set_xlabel("Época")
    ax[0].set_ylabel("Cross-Entropy Loss")
    ax[0].set_title("Histórico de Loss (Treino vs Validação)")
    ax[0].legend(fontsize=9)
    ax[0].grid(alpha=0.3)

    ax[1].set_yscale("log")
    ax[1].set_xlabel("Época")
    ax[1].set_ylabel("Learning Rate (log)")
    ax[1].set_title("Decaimento do Learning Rate")
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


def plot_roc_and_scores_cmp(
    scores: dict[str, dict[float, dict[str, np.ndarray]]],
    class_labels: dict[int, str],
    n_bins: int = 40,
    save_dir: str | Path | None = None,
    show: bool = True,
) -> list[plt.Figure]:
    """Plota para cada κ_λ: ROC ponderada (com zoom em alta eficiência) + distribuição log de P(HH) por classe."""
    model_names = list(scores.keys())
    kappas = list(scores[model_names[0]].keys())
    class_colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bin_w = bins[1] - bins[0]
    figs: list[plt.Figure] = []

    for k in kappas:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
        ax_ins = ax[0].inset_axes([0.42, 0.16, 0.53, 0.48])

        for i, m_name in enumerate(model_names):
            d = scores[m_name][k]
            fpr, tpr = weighted_roc(d["y"], d["s"], d["w"])
            auc_val = auc_from_roc(fpr, tpr)
            ls = MODEL_STYLES.get(m_name, ["-", "--", "-."][i % 3])
            col = MODEL_COLORS.get(m_name, f"C{i}")
            ax[0].plot(fpr, tpr, linestyle=ls, color=col, lw=1.8, label=f"{m_name}: AUC = {auc_val:.4f}")
            ax_ins.plot(fpr, tpr, linestyle=ls, color=col, lw=1.8)

        ax[0].plot([0, 1], [0, 1], "--", color="gray", lw=1.0)
        ax[0].set_xlabel("Eficiência do fundo (ponderada)")
        ax[0].set_ylabel("Eficiência do sinal")
        ax[0].set_title(f"{' vs '.join(model_names)} — ROC, κ_λ = {k:g}")
        ax[0].legend(loc="lower right", fontsize=9)
        ax[0].grid(alpha=0.3)

        ax_ins.set_xlim(0.0, 0.12)
        ax_ins.set_ylim(0.75, 0.98)
        ax_ins.set_title("Zoom (alta efic.)", fontsize=8)
        ax_ins.tick_params(labelsize=7)
        ax_ins.grid(alpha=0.3)

        for c, cls_name in class_labels.items():
            for i, m_name in enumerate(model_names):
                d = scores[m_name][k]
                m = d["cls"] == c
                h, _ = np.histogram(d["s"][m], bins=bins, weights=d["w"][m])
                dens = np.maximum(h / max(abs(h.sum()), 1e-30) / bin_w, 1e-7)
                short_m = "Affine" if m_name == "AffinePNN" else ("AN" if m_name == "DNN_AN" else m_name)
                ls = MODEL_STYLES.get(m_name, ["-", "--", "-."][i % 3])
                ax[1].step(
                    bins[:-1],
                    dens,
                    where="post",
                    color=class_colors.get(c, f"C{c}"),
                    linestyle=ls,
                    lw=1.6,
                    label=f"{cls_name} ({short_m})",
                )

        ax[1].set_yscale("log")
        ax[1].set_xlabel("P(HH)")
        ax[1].set_ylabel("Densidade (pesos com sinal)")
        ax[1].set_title(f"Distribuição de P(HH) por classe, κ_λ = {k:g}")
        ax[1].legend(ncol=max(len(model_names), 2), fontsize=8.2)
        ax[1].grid(alpha=0.3)

        fig.tight_layout()
        spath = Path(save_dir) / f"roc_and_scores_kl_{str(k).replace('.', 'p')}.png" if save_dir else None
        figs.append(_finalize_fig(fig, save_path=spath, show=show))
    return figs


def plot_roc_and_pr_grid(
    scores: dict[str, dict[float, dict[str, np.ndarray]]],
    kappas: Sequence[float] = (0.0, 1.0, 2.45, 5.0),
    split_label: str = "val",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Painel 2 x |K|: Eficiência de Fundo (log) vs Eficiência de Sinal e Pureza Balanceada vs Eficiência."""
    fig, axes = plt.subplots(2, len(kappas), figsize=(4.5 * len(kappas), 8.5), sharex="row", squeeze=False)
    for j, k in enumerate(kappas):
        fk = float(k)
        ax_roc, ax_pr = axes[0, j], axes[1, j]
        for i, (name, sc) in enumerate(scores.items()):
            d = sc[fk]
            fpr, tpr = weighted_roc(d["y"], d["s"], d["w"])
            tpr_m = np.maximum.accumulate(tpr)
            auc_val = auc_from_roc(fpr, tpr)
            col = MODEL_COLORS.get(name, f"C{i}")
            ls = MODEL_STYLES.get(name, "-")

            ax_roc.plot(tpr, np.clip(fpr, 1e-4, 1.0), color=col, ls=ls, lw=2.0, label=f"{name}\nAUC$_w$={auc_val:.4f}")
            mask = tpr_m > 1e-3
            purity_bal = tpr_m[mask] / np.maximum(tpr_m[mask] + fpr[mask], 1e-12)
            ax_pr.plot(tpr_m[mask], purity_bal, color=col, ls=ls, lw=2.0, label=name)

        ax_roc.set_yscale("log")
        ax_roc.set_ylim(1e-4, 1.0)
        ax_roc.set_xlim(0.1, 1.0)
        ax_roc.set_title(rf"$\kappa_\lambda = {k:g}$", fontsize=12, fontweight="bold")
        ax_roc.grid(True, which="both", alpha=0.3)
        ax_roc.legend(loc="lower right", fontsize=8.5)
        if j == 0:
            ax_roc.set_ylabel(r"Eficiência de Fundo $\epsilon_B$ (log)")

        ax_pr.set_ylim(0.50, 1.01)
        ax_pr.set_xlim(0.1, 1.0)
        ax_pr.grid(True, alpha=0.3)
        ax_pr.legend(loc="lower left", fontsize=8.5)
        if j == 0:
            ax_pr.set_ylabel("Pureza Balanceada (Precision)")
        ax_pr.set_xlabel(r"Eficiência de Sinal $\epsilon_S$ (Recall)")

    fig.suptitle(f"Curvas ROC e Pureza vs Eficiência — {split_label}", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


def plot_metrics_bars_cmp(
    tables: dict[str, pd.DataFrame],
    kappas: Sequence[float] = (0.0, 1.0, 2.45, 5.0),
    split_label: str = "val",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Compara lado a lado AUC_w, Significance Ratio σ_ratio, Pureza e Eficiência por κ_λ."""
    metrics_to_plot = [
        ("auc_w", "Área da ROC Ponderada (AUC$_w$)"),
        ("sigma_ratio", r"Significance Ratio ($\sigma_{\mathrm{ratio}}$, Anzalone Eq. 4)"),
        ("pur_w_bal", "Pureza / Precision Balanceada (corte 0.5)"),
        ("eff_s_w", "Eficiência / Recall de Sinal (corte 0.5)"),
    ]
    model_names = list(tables.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6))
    x_pos = np.arange(len(kappas))
    width = 0.72 / max(len(model_names), 1)

    for ax, (col_name, title) in zip(axes.ravel(), metrics_to_plot):
        ymin, ymax = 1.0, 0.0
        for i, m_name in enumerate(model_names):
            vals = tables[m_name][col_name].to_numpy(np.float64)
            ymin = min(ymin, float(vals.min()))
            ymax = max(ymax, float(vals.max()))
            offset = (i - (len(model_names) - 1) / 2.0) * width
            color = MODEL_COLORS.get(m_name, f"C{i}")
            bars = ax.bar(x_pos + offset, vals, width, color=color, alpha=0.9, label=m_name)
            for b in bars:
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    b.get_height() + 0.002,
                    f"{b.get_height():.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.8,
                    color=color,
                    fontweight="bold",
                )
        ax.set_ylim(max(0.0, ymin - 0.03), min(1.03, ymax + 0.028))
        ax.set_xticks(x_pos)
        ax.set_xticklabels([rf"$\kappa_\lambda={k:g}$" for k in kappas])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(f"Resumo Comparativo por Hipótese de $\\kappa_\\lambda$ ({split_label})", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


def plot_threshold_tradeoff_panels(
    scores: dict[str, dict[float, dict[str, np.ndarray]]],
    kappas: Sequence[float] = (0.0, 1.0, 2.45, 5.0),
    split_label: str = "val",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Painéis 3 x |K|: Precision, Recall e F1-score em função do Threshold em P(HH)."""
    curves = {
        m_name: {float(k): compute_curves_for_kappa(d["y"], d["s"], d["w"]) for k, d in sc.items()}
        for m_name, sc in scores.items()
    }
    fig, axes = plt.subplots(3, len(kappas), figsize=(4.5 * len(kappas), 10.8), sharex=True, squeeze=False)
    for j, k in enumerate(kappas):
        fk = float(k)
        ax_p, ax_r, ax_f1 = axes[0, j], axes[1, j], axes[2, j]
        for i, m_name in enumerate(scores):
            c = curves[m_name][fk]
            col = MODEL_COLORS.get(m_name, f"C{i}")
            ls = MODEL_STYLES.get(m_name, "-")
            ax_p.plot(c["thr"], c["precision"], color=col, ls=ls, lw=2.0, label=m_name)
            ax_r.plot(c["thr"], c["recall"], color=col, ls=ls, lw=2.0, label=m_name)
            ax_f1.plot(
                c["thr"],
                c["f1"],
                color=col,
                ls=ls,
                lw=2.0,
                label=f"{m_name} (F1={c['best_f1']:.3f} @ {c['best_thr']:.2f})",
            )
            ax_f1.plot(c["best_thr"], c["best_f1"], marker="o", color=col, markersize=5)

        ax_p.set_title(rf"Precision ($\kappa_\lambda = {k:g}$)", fontsize=11, fontweight="bold")
        ax_p.set_ylim(0.50, 1.01)
        ax_p.grid(True, alpha=0.3)
        ax_p.legend(loc="lower right", fontsize=8)
        if j == 0:
            ax_p.set_ylabel("Precision Balanceada")

        ax_r.set_title(rf"Recall $\epsilon_S$ ($\kappa_\lambda = {k:g}$)", fontsize=11, fontweight="bold")
        ax_r.set_ylim(0.0, 1.02)
        ax_r.grid(True, alpha=0.3)
        ax_r.legend(loc="lower left", fontsize=8)
        if j == 0:
            ax_r.set_ylabel(r"Recall ($\epsilon_S$)")

        ax_f1.set_title(rf"F1-score ($\kappa_\lambda = {k:g}$)", fontsize=11, fontweight="bold")
        ax_f1.set_ylim(0.40, 0.98)
        ax_f1.set_xlim(0.0, 1.0)
        ax_f1.set_xlabel(r"Threshold em $P(\mathrm{HH})$")
        ax_f1.grid(True, alpha=0.3)
        ax_f1.legend(loc="lower center", fontsize=7.8)
        if j == 0:
            ax_f1.set_ylabel("F1-score Balanceado")

    fig.suptitle(f"Precision, Recall e F1-score vs Threshold — {split_label}", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


def plot_threshold_tradeoff_combined(
    scores: dict[str, dict[float, dict[str, np.ndarray]]],
    kappas: Sequence[float] = (0.0, 1.0, 2.45, 5.0),
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Visão combinada em 1 linha: Precision + Recall + F1 vs Threshold por κ_λ."""
    curves = {
        m_name: {float(k): compute_curves_for_kappa(d["y"], d["s"], d["w"]) for k, d in sc.items()}
        for m_name, sc in scores.items()
    }
    fig, axes = plt.subplots(1, len(kappas), figsize=(4.5 * len(kappas), 4.3), sharey=True, squeeze=False)
    for j, k in enumerate(kappas):
        fk = float(k)
        ax = axes[0, j]
        for i, m_name in enumerate(scores):
            c = curves[m_name][fk]
            short = "Affine" if m_name == "AffinePNN" else ("AN" if m_name == "DNN_AN" else m_name)
            ls = MODEL_STYLES.get(m_name, ["-", "--"][i % 2])
            ax.plot(c["thr"], c["precision"], color="#1f77b4", ls=ls, lw=1.7, label=f"Prec ({short})")
            ax.plot(c["thr"], c["recall"], color="#2ca02c", ls=ls, lw=1.7, label=f"Rec ({short})")
            ax.plot(c["thr"], c["f1"], color="#d62728", ls=ls, lw=1.9, label=f"F1 ({short})")
            ax.plot(c["best_thr"], c["best_f1"], marker="o", color="#d62728", markersize=4.5)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.03)
        ax.set_title(rf"Trade-off ($\kappa_\lambda = {k:g}$)", fontsize=11, fontweight="bold")
        ax.set_xlabel(r"Threshold em $P(\mathrm{HH})$")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_ylabel("Métrica Balanceada")
            ax.legend(loc="lower left", fontsize=7.6, ncol=2)

    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


def plot_significance_ratio(
    scores: dict[str, dict[float, dict[str, np.ndarray]]],
    kappas: Sequence[float] = (0.0, 1.0, 2.45, 5.0),
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plota a curva σ_ratio(t) (Anzalone et al., Eq. 4, Fig. 15) e o corte ótimo t* para cada κ_λ."""
    fig, axes = plt.subplots(1, len(kappas), figsize=(4.4 * len(kappas), 4.2), sharey=True, squeeze=False)
    for j, k in enumerate(kappas):
        fk = float(k)
        ax = axes[0, j]
        for i, (m_name, sc) in enumerate(scores.items()):
            d = sc[fk]
            res = ams_significance_scan(d["y"], d["s"], d["w"], balanced=True)
            col = MODEL_COLORS.get(m_name, f"C{i}")
            ls = MODEL_STYLES.get(m_name, "-")
            ax.plot(
                res["thr"],
                res["sigma_ratio"],
                color=col,
                ls=ls,
                lw=2.0,
                label=f"{m_name}\n$\\sigma_{{ratio}}$={res['best_sigma_ratio']:.4f} @ t*={res['best_cut']:.2f}",
            )
            ax.axvline(res["best_cut"], color=col, ls=":", alpha=0.65)
        ax.axhline(1.0 / np.sqrt(2.0), color="gray", ls="--", lw=0.9, alpha=0.6)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.68, 1.0)
        ax.set_title(rf"$\sigma_{{\mathrm{{ratio}}}}(t)$ em $\kappa_\lambda = {k:g}$", fontsize=11, fontweight="bold")
        ax.set_xlabel(r"Threshold $t$ em $P(\mathrm{HH})$")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
        if j == 0:
            ax.set_ylabel(r"Significance Ratio $\sigma_{\mathrm{ratio}}$")

    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


def plot_kappa_dependence(
    dep_df: pd.DataFrame,
    class_labels: dict[int, str],
    model_name: str = "AffinePNN",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plota `<P(HH)>` médio em função do `kappa_lambda` de entrada para cada classe verdadeira."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name in class_labels.values():
        col_name = f"mean_P_HH_{name}"
        if col_name in dep_df.columns:
            ax.plot(dep_df["kappa_in"], dep_df[col_name], "o-", lw=1.8, label=f"{name} verdadeiro")
    ax.set_xlabel(r"$\kappa_\lambda$ de entrada")
    ax.set_ylabel(r"$\langle P(\mathrm{HH}) \rangle$")
    ax.set_title(f"Dependência em $\\kappa_\\lambda$ — {model_name}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


def plot_kappa_response_heatmap(
    matrices: dict[str, pd.DataFrame],
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plota lado a lado a matriz de resposta `κ_verdadeiro x κ_entrada` (AUC_w) para AffinePNN e DNN_AN."""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(5.8 * n, 4.8), squeeze=False)
    for j, (name, mat) in enumerate(matrices.items()):
        ax = axes[0, j]
        vals = mat.to_numpy(np.float64)
        im = ax.imshow(vals, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(np.arange(len(mat.columns)))
        ax.set_xticklabels([f"{c:g}" for c in mat.columns])
        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels([f"{r:g}" for r in mat.index])
        ax.set_xlabel(r"$\kappa_\lambda$ de entrada na rede")
        ax.set_ylabel(r"$\kappa_\lambda$ verdadeiro da amostra HH")
        ax.set_title(f"Matriz de Resposta AUC$_w$ — {name}", fontweight="bold")
        for r_i in range(vals.shape[0]):
            best_c = int(np.argmax(vals[r_i]))
            for c_i in range(vals.shape[1]):
                weight = "bold" if c_i == best_c else "normal"
                ax.text(c_i, r_i, f"{vals[r_i, c_i]:.4f}", ha="center", va="center", fontsize=8.8, fontweight=weight)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


def plot_continuous_kappa_scan(
    scan_df: pd.DataFrame,
    ref_kappas: Sequence[float] = (0.0, 1.0, 2.45, 5.0),
    model_name: str = "AffinePNN",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Visualiza a varredura contínua fina em κ_λ: resposta `<P(HH)>` e `AUC_w(κ_in)` para cada amostra de sinal."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    x = scan_df["kappa_in"].to_numpy()

    ax[0].plot(x, scan_df["mean_P_HH_DY"], "--", lw=1.6, label="Fundo DY")
    ax[0].plot(x, scan_df["mean_P_HH_ttbar"], "--", lw=1.6, label="Fundo ttbar")
    for ks in ref_kappas:
        col_p = f"mean_P_HH_sig_kl{ks:g}"
        if col_p in scan_df.columns:
            ax[0].plot(x, scan_df[col_p], lw=1.9, label=rf"Sinal HH ($\kappa_\lambda^{{\mathrm{{ger}}}}={ks:g}$)")
        col_auc = f"auc_w_sig_kl{ks:g}"
        if col_auc in scan_df.columns:
            line = ax[1].plot(x, scan_df[col_auc], lw=2.0, label=rf"Amostra $\kappa_\lambda^{{\mathrm{{ger}}}}={ks:g}$")[0]
            ax[1].axvline(float(ks), color=line.get_color(), ls=":", alpha=0.55)

    ax[0].set_xlabel(r"$\kappa_\lambda$ fornecido na entrada")
    ax[0].set_ylabel(r"$\langle P(\mathrm{HH}) \rangle$")
    ax[0].set_title(f"{model_name} — Resposta Suave em $\\kappa_\\lambda$ Contínuo")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8.3)

    ax[1].set_xlabel(r"$\kappa_\lambda$ fornecido na entrada")
    ax[1].set_ylabel(r"AUC$_w$ na amostra")
    ax[1].set_title(f"{model_name} — Ótimo de AUC$_w$ vs $\\kappa_\\lambda$ de Entrada")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8.5)

    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


def plot_affine_conditioning_params(
    mod_dict: dict[str, dict[str, np.ndarray]],
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Mostra a intensidade da modulação afim aprendida (`|s(κ) - 1|` e `|b(κ)|`) ao longo dos blocos DenseNet."""
    fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.3))
    for layer_name, d in mod_dict.items():
        k = d["kappa"]
        s_dev = np.mean(np.abs(d["scale"] - 1.0), axis=1)
        b_mag = np.mean(np.abs(d["bias"]), axis=1)
        short = layer_name.replace("affine_conditioning_", "Bloco ")
        ax[0].plot(k, s_dev, "o-", lw=1.6, markersize=3.5, label=short)
        ax[1].plot(k, b_mag, "o-", lw=1.6, markersize=3.5, label=short)

    ax[0].set_xlabel(r"$\kappa_\lambda$")
    ax[0].set_ylabel(r"Média neuronal de $|s(\kappa_\lambda) - 1|$")
    ax[0].set_title("Modulação Multiplicativa (Scale) por Bloco")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8, ncol=2)

    ax[1].set_xlabel(r"$\kappa_\lambda$")
    ax[1].set_ylabel(r"Média neuronal de $|b(\kappa_\lambda)|$")
    ax[1].set_title("Modulação Aditiva (Bias) por Bloco")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8, ncol=2)

    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


def plot_feature_distributions(
    df: pd.DataFrame,
    features: Sequence[str],
    class_labels: dict[int, str],
    n_bins: int = 45,
    ncols: int = 3,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Compara distribuições normalizadas em densidade (ponderadas por `max(w_phys,0)`) entre DY, ttbar e cada κ_λ."""
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows), squeeze=False)
    y = df["label"].to_numpy()
    kl = df["kappa_lambda"].to_numpy(np.float64)
    w = np.clip(df["w_phys"].to_numpy(np.float64), 0.0, None)

    for idx, col in enumerate(features):
        ax = axes[idx // ncols, idx % ncols]
        vals_all = df[col].to_numpy(np.float64)
        valid_all = vals_all[np.isfinite(vals_all)]
        if len(valid_all) == 0:
            continue
        lo, hi = np.percentile(valid_all, [0.5, 99.5])
        if lo == hi:
            lo, hi = float(valid_all.min()), float(valid_all.max()) + 1.0
        bins = np.linspace(lo, hi, int(n_bins) + 1)

        for c in (0, 1):
            m = (y == c) & np.isfinite(vals_all)
            if m.any() and w[m].sum() > 0:
                ax.hist(vals_all[m], bins=bins, weights=w[m], density=True, histtype="stepfilled", alpha=0.25, label=class_labels[c])

        kappas = sorted(np.unique(np.round(kl[(y == 2) & np.isfinite(kl)], 2)))
        for k in kappas:
            mk = (y == 2) & np.isclose(kl, k) & np.isfinite(vals_all)
            if mk.any() and w[mk].sum() > 0:
                ax.hist(vals_all[mk], bins=bins, weights=w[mk], density=True, histtype="step", lw=1.7, label=rf"HH ($\kappa_\lambda={k:g}$)")

        ax.set_title(col.replace("reg_dnn_moe_", ""), fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=7.5)

    for idx in range(len(features), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


def plot_mass_sculpting(
    df: pd.DataFrame,
    masses_df: pd.DataFrame,
    scores_hh: np.ndarray,
    mass_col: str = "m_hh",
    score_edges: Sequence[float] = (0.0, 0.50, 0.75, 0.90, 0.99, 1.0001),
    model_name: str = "AffinePNN",
    max_mass: float = 800.0,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Verifica mass sculpting (AN-25-103 Seção 5.5): forma de m_tau_tau, m_bb ou m_HH do fundo em fatias de P(HH)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    is_bkg = df["label"].to_numpy() != 2
    vals = masses_df[mass_col].to_numpy(np.float64)
    w = np.clip(df["w_phys"].to_numpy(np.float64), 0.0, None)
    s = np.asarray(scores_hh, dtype=np.float64)
    bins = np.linspace(0.0, float(max_mass), 45)

    for lo, hi in zip(score_edges[:-1], score_edges[1:]):
        m = is_bkg & (s >= lo) & (s < hi) & np.isfinite(vals) & (vals > 0) & (vals <= max_mass)
        if int(m.sum()) < 20 or w[m].sum() <= 0:
            continue
        ax.hist(
            vals[m],
            bins=bins,
            weights=w[m],
            density=True,
            histtype="step",
            lw=1.8,
            label=f"P(HH) ∈ [{lo:.2f}, {min(hi, 1.0):.2f}] (N={int(m.sum()):,})",
        )

    ax.set_xlabel(f"{mass_col} [GeV]")
    ax.set_ylabel("Densidade normalizada (fundo DY + ttbar)")
    ax.set_title(f"Checagem de Mass Sculpting ({mass_col}) — {model_name}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.2)
    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


# ==============================================================================
# SUÍTE EXPANDIDA: AFFINEPNN PURA E COMPARAÇÃO AFFINEPNN vs DNN_AN
# - Curvas ROC individuais por κ_λ + sobreposição multi-κ_λ
# - Scores da rede vs Eventos em escala log por κ_λ (MC bruto e peso físico)
# - Curvas Precision-Recall com contornos iso-F1 + barras de Precision/Recall/F1
# - Precision, Recall e F1-score vs Score da rede (plots separados e combinado)
# ==============================================================================

KAPPA_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#8c564b", "#e377c2"]


def _kappa_tag(k: float) -> str:
    return str(k).replace(".", "p").replace("-", "m")


def _draw_iso_f1_contours(ax: plt.Axes, f1_levels: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9)) -> None:
    """Desenha curvas tracejadas de F1 constante no plano Precision vs Recall."""
    r = np.linspace(0.05, 1.0, 300)
    for f1 in f1_levels:
        denom = 2.0 * r - f1
        valid = denom > 1e-4
        r_v = r[valid]
        p_v = (f1 * r_v) / denom[valid]
        keep = (p_v >= 0.35) & (p_v <= 1.01)
        if np.any(keep):
            ax.plot(r_v[keep], p_v[keep], color="gray", ls=":", lw=0.85, alpha=0.55)
            idx_lbl = int(np.argmin(np.abs(r_v[keep] - 0.92)))
            rx, py = r_v[keep][idx_lbl], p_v[keep][idx_lbl]
            if 0.42 <= py <= 0.98:
                ax.annotate(f"F1={f1:.1f}", xy=(rx, py), fontsize=7, color="dimgray", alpha=0.8)


def plot_roc_per_kappa_individual(
    scores: dict[str, dict[float, dict[str, np.ndarray]]],
    kappas: Sequence[float] = (0.0, 1.0, 2.45, 5.0),
    title_prefix: str | None = None,
    save_dir: str | Path | None = None,
    show: bool = True,
) -> list[plt.Figure]:
    """Gera 1 figura ROC dedicada por cada κ_λ (TPR vs FPR linear c/ zoom + ε_B log vs ε_S).

    Funciona tanto para 1 modelo puro (`{"AffinePNN": scores["AffinePNN"]}`) quanto para
    comparação entre modelos (`{"AffinePNN": ..., "DNN_AN": ...}`).
    Quando há apenas 1 modelo, gera também um gráfico extra sobrepondo todas as curvas κ_λ!
    """
    model_names = list(scores.keys())
    prefix = title_prefix or (" vs ".join(model_names))
    figs: list[plt.Figure] = []

    for k in kappas:
        fk = float(k)
        fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.6))
        ax_ins = ax[0].inset_axes([0.43, 0.14, 0.52, 0.48])

        for i, m_name in enumerate(model_names):
            d = scores[m_name][fk]
            fpr, tpr = weighted_roc(d["y"], d["s"], d["w"])
            auc_val = auc_from_roc(fpr, tpr)
            col = MODEL_COLORS.get(m_name, f"C{i}")
            ls = MODEL_STYLES.get(m_name, "-")

            ax[0].plot(fpr, tpr, color=col, ls=ls, lw=2.0, label=f"{m_name} (AUC$_w$ = {auc_val:.4f})")
            ax_ins.plot(fpr, tpr, color=col, ls=ls, lw=1.8)
            ax[1].plot(tpr, np.clip(fpr, 1e-4, 1.0), color=col, ls=ls, lw=2.0, label=f"{m_name} (AUC$_w$ = {auc_val:.4f})")

        ax[0].plot([0, 1], [0, 1], "--", color="gray", lw=1.0, alpha=0.7)
        ax[0].set_xlabel(r"Taxa de Falso Positivo / Eficiência de Fundo $\epsilon_B$")
        ax[0].set_ylabel(r"Eficiência de Sinal $\epsilon_S$ (TPR)")
        ax[0].set_title(rf"{prefix} — Curva ROC ($\kappa_\lambda = {k:g}$)", fontweight="bold")
        ax[0].legend(loc="lower right", fontsize=9)
        ax[0].grid(alpha=0.3)

        ax_ins.set_xlim(0.0, 0.15)
        ax_ins.set_ylim(0.65, 1.0)
        ax_ins.set_title("Zoom (baixa taxa de fundo)", fontsize=8)
        ax_ins.tick_params(labelsize=7.5)
        ax_ins.grid(alpha=0.3)

        ax[1].set_yscale("log")
        ax[1].set_xlim(0.1, 1.0)
        ax[1].set_ylim(1e-4, 1.0)
        ax[1].set_xlabel(r"Eficiência de Sinal $\epsilon_S$")
        ax[1].set_ylabel(r"Eficiência de Fundo $\epsilon_B$ (escala log)")
        ax[1].set_title(rf"{prefix} — Rejeição de Fundo em Log ($\kappa_\lambda = {k:g}$)", fontweight="bold")
        ax[1].legend(loc="lower right", fontsize=9)
        ax[1].grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        spath = Path(save_dir) / f"roc_kl_{_kappa_tag(fk)}.png" if save_dir else None
        figs.append(_finalize_fig(fig, save_path=spath, show=show))

    # Se for avaliação de um único modelo (ex: AffinePNN pura), plota também todas as hipóteses κ_λ sobrepostas
    if len(model_names) == 1:
        m_name = model_names[0]
        fig_all, ax_all = plt.subplots(1, 2, figsize=(12.8, 4.6))
        for j, k in enumerate(kappas):
            fk = float(k)
            d = scores[m_name][fk]
            fpr, tpr = weighted_roc(d["y"], d["s"], d["w"])
            auc_val = auc_from_roc(fpr, tpr)
            col = KAPPA_COLORS[j % len(KAPPA_COLORS)]
            lbl = rf"$\kappa_\lambda = {k:g}$ (AUC$_w$ = {auc_val:.4f})"
            ax_all[0].plot(fpr, tpr, color=col, lw=2.0, label=lbl)
            ax_all[1].plot(tpr, np.clip(fpr, 1e-4, 1.0), color=col, lw=2.0, label=lbl)

        ax_all[0].plot([0, 1], [0, 1], "--", color="gray", lw=1.0)
        ax_all[0].set_xlabel(r"Eficiência de Fundo $\epsilon_B$")
        ax_all[0].set_ylabel(r"Eficiência de Sinal $\epsilon_S$")
        ax_all[0].set_title(f"{m_name} Pura — Curvas ROC sobrepostas por $\\kappa_\\lambda$", fontweight="bold")
        ax_all[0].legend(loc="lower right", fontsize=9)
        ax_all[0].grid(alpha=0.3)

        ax_all[1].set_yscale("log")
        ax_all[1].set_xlim(0.1, 1.0)
        ax_all[1].set_ylim(1e-4, 1.0)
        ax_all[1].set_xlabel(r"Eficiência de Sinal $\epsilon_S$")
        ax_all[1].set_ylabel(r"Eficiência de Fundo $\epsilon_B$ (escala log)")
        ax_all[1].set_title(f"{m_name} Pura — Rejeição de Fundo por $\\kappa_\\lambda$", fontweight="bold")
        ax_all[1].legend(loc="lower right", fontsize=9)
        ax_all[1].grid(True, which="both", alpha=0.3)

        fig_all.tight_layout()
        spath_all = Path(save_dir) / f"roc_{m_name}_all_kappas_overlay.png" if save_dir else None
        figs.append(_finalize_fig(fig_all, save_path=spath_all, show=show))

    return figs


def plot_scores_vs_events_log_per_kappa(
    scores: dict[str, dict[float, dict[str, np.ndarray]]],
    class_labels: dict[int, str],
    kappas: Sequence[float] = (0.0, 1.0, 2.45, 5.0),
    n_bins: int = 40,
    save_dir: str | Path | None = None,
    show: bool = True,
) -> list[plt.Figure]:
    """Plota o Score da rede P(HH) vs Número de Eventos em escala LOG para cada κ_λ:
      - Painel esquerdo: Contagem de eventos MC (sem peso) em escala log por classe {DY, ttbar, HH(κ_λ)}
      - Painel direito : Eventos ponderados físicos (Σ max(w_phys, 0)) em escala log por classe
    Além das figuras individuais por κ_λ, gera um painel resumo 2x2 com todos os κ_λ.
    """
    model_names = list(scores.keys())
    is_single = len(model_names) == 1
    class_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    figs: list[plt.Figure] = []

    for k in kappas:
        fk = float(k)
        fig, ax = plt.subplots(1, 2, figsize=(13.2, 4.6))

        for c, cls_name in class_labels.items():
            for i, m_name in enumerate(model_names):
                d = scores[m_name][fk]
                mask = d["cls"] == c
                s_c = d["s"][mask]
                w_pos = np.clip(d["w"][mask], 0.0, None)

                cnt_raw, _ = np.histogram(s_c, bins=bins)
                cnt_w, _ = np.histogram(s_c, bins=bins, weights=w_pos)

                short_m = "" if is_single else f" ({'Affine' if m_name == 'AffinePNN' else ('AN' if m_name == 'DNN_AN' else m_name)})"
                ls = "-" if is_single else MODEL_STYLES.get(m_name, ["-", "--", "-."][i % 3])
                col = class_colors.get(c, f"C{c}")

                if is_single and c != 2:
                    ax[0].hist(s_c, bins=bins, color=col, alpha=0.18, histtype="stepfilled")
                    ax[1].hist(s_c, bins=bins, weights=w_pos, color=col, alpha=0.18, histtype="stepfilled")

                ax[0].step(
                    bins[:-1],
                    np.maximum(cnt_raw, 0.5),
                    where="post",
                    color=col,
                    ls=ls,
                    lw=1.9,
                    label=f"{cls_name}{short_m} [N={int(mask.sum()):,d}]",
                )
                y_floor = max(float(cnt_w[cnt_w > 0].min()) * 0.3, 1e-5) if np.any(cnt_w > 0) else 1e-5
                ax[1].step(
                    bins[:-1],
                    np.maximum(cnt_w, y_floor),
                    where="post",
                    color=col,
                    ls=ls,
                    lw=1.9,
                    label=f"{cls_name}{short_m} [Y={w_pos.sum():.2f}]",
                )

        title_tag = model_names[0] if is_single else " vs ".join(model_names)
        ax[0].set_yscale("log")
        ax[0].set_xlim(0.0, 1.0)
        ax[0].set_xlabel(r"Score da Rede $P(\mathrm{HH})$")
        ax[0].set_ylabel("Eventos MC por bin (escala log)")
        ax[0].set_title(rf"{title_tag} — Score vs Eventos MC ($\kappa_\lambda = {k:g}$)", fontweight="bold")
        ax[0].legend(fontsize=8.2, ncol=1 if is_single else 2)
        ax[0].grid(True, which="both", alpha=0.3)

        ax[1].set_yscale("log")
        ax[1].set_xlim(0.0, 1.0)
        ax[1].set_xlabel(r"Score da Rede $P(\mathrm{HH})$")
        ax[1].set_ylabel(r"Eventos Ponderados $\sum w_{\mathrm{phys}}^+$ por bin (escala log)")
        ax[1].set_title(rf"{title_tag} — Score vs Rendimento Físico ($\kappa_\lambda = {k:g}$)", fontweight="bold")
        ax[1].legend(fontsize=8.2, ncol=1 if is_single else 2)
        ax[1].grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        spath = Path(save_dir) / f"scores_vs_events_log_kl_{_kappa_tag(fk)}.png" if save_dir else None
        figs.append(_finalize_fig(fig, save_path=spath, show=show))

    # Painel resumo 2x2 (ou 1xN) com todos os κ_λ em eventos MC (escala log)
    ncols = 2 if len(kappas) > 2 else max(len(kappas), 1)
    nrows = int(np.ceil(len(kappas) / ncols))
    fig_grid, axes = plt.subplots(nrows, ncols, figsize=(6.3 * ncols, 4.4 * nrows), squeeze=False)
    for idx, k in enumerate(kappas):
        fk = float(k)
        ax_g = axes[idx // ncols, idx % ncols]
        for c, cls_name in class_labels.items():
            for i, m_name in enumerate(model_names):
                d = scores[m_name][fk]
                mask = d["cls"] == c
                cnt_raw, _ = np.histogram(d["s"][mask], bins=bins)
                short_m = "" if is_single else f" ({'Affine' if m_name == 'AffinePNN' else ('AN' if m_name == 'DNN_AN' else m_name)})"
                ls = "-" if is_single else MODEL_STYLES.get(m_name, ["-", "--", "-."][i % 3])
                ax_g.step(
                    bins[:-1],
                    np.maximum(cnt_raw, 0.5),
                    where="post",
                    color=class_colors.get(c, f"C{c}"),
                    ls=ls,
                    lw=1.8,
                    label=f"{cls_name}{short_m}",
                )
        ax_g.set_yscale("log")
        ax_g.set_xlim(0.0, 1.0)
        ax_g.set_xlabel(r"Score da Rede $P(\mathrm{HH})$")
        ax_g.set_ylabel("Eventos MC (log)")
        ax_g.set_title(rf"$\kappa_\lambda = {k:g}$", fontweight="bold")
        ax_g.grid(True, which="both", alpha=0.3)
        ax_g.legend(fontsize=8, ncol=1 if is_single else 2)

    for idx in range(len(kappas), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    title_tag = f"{model_names[0]} Pura" if is_single else " vs ".join(model_names)
    fig_grid.suptitle(f"{title_tag} — Distribuição de Scores $P(\\mathrm{{HH}})$ vs Eventos (Log)", fontsize=13, fontweight="bold", y=1.01)
    fig_grid.tight_layout()
    spath_grid = Path(save_dir) / "scores_vs_events_log_all_kappas_grid.png" if save_dir else None
    figs.append(_finalize_fig(fig_grid, save_path=spath_grid, show=show))
    return figs


def plot_pr_curves_and_f1_summary(
    scores: dict[str, dict[float, dict[str, np.ndarray]]],
    kappas: Sequence[float] = (0.0, 1.0, 2.45, 5.0),
    thr_eval: float = 0.5,
    save_dir: str | Path | None = None,
    show: bool = True,
) -> list[plt.Figure]:
    """Gera plots dedicados de Precision-Recall (com curvas iso-F1 e marcador de F1 máximo)
    e gráfico de barras de Precision, Recall e F1-score por κ_λ.

    Suporta modo modelo único (`AffinePNN` pura) e modo comparativo (`AffinePNN vs DNN_AN`).
    """
    from src.evaluation.roc import binary_metrics_at_threshold

    model_names = list(scores.keys())
    is_single = len(model_names) == 1
    curves = {
        m_name: {float(k): compute_curves_for_kappa(d["y"], d["s"], d["w"]) for k, d in sc.items()}
        for m_name, sc in scores.items()
    }
    figs: list[plt.Figure] = []

    if is_single:
        m_name = model_names[0]
        fig_pr, ax_pr = plt.subplots(1, 2, figsize=(13.0, 4.8))
        _draw_iso_f1_contours(ax_pr[0])

        prec_vals, rec_vals, f1_half_vals, f1_best_vals = [], [], [], []
        for j, k in enumerate(kappas):
            fk = float(k)
            c = curves[m_name][fk]
            d = scores[m_name][fk]
            bm = binary_metrics_at_threshold(d["y"], d["s"], d["w"], thr=thr_eval)
            col = KAPPA_COLORS[j % len(KAPPA_COLORS)]

            ax_pr[0].plot(
                c["eff_grid"],
                c["prec_curve"],
                color=col,
                lw=2.1,
                label=rf"$\kappa_\lambda={k:g}$ (PR-AUC={c['pr_auc']:.4f} | $F_1^*$={c['best_f1']:.3f})",
            )
            # Marca o ponto de melhor F1
            best_i = int(np.argmax(c["f1"]))
            ax_pr[0].plot(c["recall"][best_i], c["precision"][best_i], marker="*", markersize=10, color=col)

            prec_vals.append(bm["pur_w_bal"])
            rec_vals.append(bm["eff_s_w"])
            f1_half_vals.append(bm["f1_w_bal"])
            f1_best_vals.append(float(c["best_f1"]))

        ax_pr[0].set_xlim(0.05, 1.0)
        ax_pr[0].set_ylim(0.45, 1.02)
        ax_pr[0].set_xlabel(r"Recall / Eficiência de Sinal $\epsilon_S$")
        ax_pr[0].set_ylabel("Precision / Pureza Balanceada")
        ax_pr[0].set_title(f"{m_name} Pura — Curvas Precision-Recall e Iso-F1", fontweight="bold")
        ax_pr[0].legend(loc="lower left", fontsize=8.5)
        ax_pr[0].grid(alpha=0.3)

        x = np.arange(len(kappas))
        w_bar = 0.20
        for idx_m, (vals, lbl, bcol) in enumerate(
            [
                (prec_vals, f"Precision (s≥{thr_eval:g})", "#1f77b4"),
                (rec_vals, f"Recall (s≥{thr_eval:g})", "#2ca02c"),
                (f1_half_vals, f"F1-score (s≥{thr_eval:g})", "#d62728"),
                (f1_best_vals, "Melhor F1-score (s*)", "#9467bd"),
            ]
        ):
            offs = (idx_m - 1.5) * w_bar
            bars = ax_pr[1].bar(x + offs, vals, w_bar, color=bcol, alpha=0.88, label=lbl)
            for b in bars:
                ax_pr[1].text(
                    b.get_x() + b.get_width() / 2.0,
                    b.get_height() + 0.008,
                    f"{b.get_height():.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    rotation=90,
                )

        ax_pr[1].set_xticks(x)
        ax_pr[1].set_xticklabels([rf"$\kappa_\lambda={k:g}$" for k in kappas])
        ax_pr[1].set_ylim(0.0, 1.12)
        ax_pr[1].set_ylabel("Valor da Métrica Balanceada")
        ax_pr[1].set_title(f"{m_name} Pura — Resumo Precision, Recall e F1-score", fontweight="bold")
        ax_pr[1].legend(fontsize=8.2, ncol=2, loc="lower right")
        ax_pr[1].grid(axis="y", alpha=0.3)

        fig_pr.tight_layout()
        spath = Path(save_dir) / f"pr_and_f1_summary_{m_name}.png" if save_dir else None
        figs.append(_finalize_fig(fig_pr, save_path=spath, show=show))
    else:
        # Painel 2x2 de curvas Precision vs Recall comparando os modelos em cada κ_λ
        ncols = 2 if len(kappas) > 2 else max(len(kappas), 1)
        nrows = int(np.ceil(len(kappas) / ncols))
        fig_pr, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.6 * nrows), squeeze=False)
        for idx, k in enumerate(kappas):
            fk = float(k)
            ax = axes[idx // ncols, idx % ncols]
            _draw_iso_f1_contours(ax)
            for i, m_name in enumerate(model_names):
                c = curves[m_name][fk]
                col = MODEL_COLORS.get(m_name, f"C{i}")
                ls = MODEL_STYLES.get(m_name, "-")
                best_i = int(np.argmax(c["f1"]))
                ax.plot(
                    c["eff_grid"],
                    c["prec_curve"],
                    color=col,
                    ls=ls,
                    lw=2.1,
                    label=f"{m_name} (PR-AUC={c['pr_auc']:.4f} | $F_1^*$={c['best_f1']:.3f})",
                )
                ax.plot(c["recall"][best_i], c["precision"][best_i], marker="*", markersize=10, color=col)
            ax.set_xlim(0.05, 1.0)
            ax.set_ylim(0.45, 1.02)
            ax.set_xlabel(r"Recall / Eficiência de Sinal $\epsilon_S$")
            ax.set_ylabel("Precision Balanceada")
            ax.set_title(rf"Precision vs Recall — $\kappa_\lambda = {k:g}$", fontweight="bold")
            ax.legend(loc="lower left", fontsize=8.3)
            ax.grid(alpha=0.3)

        for idx in range(len(kappas), nrows * ncols):
            axes[idx // ncols, idx % ncols].axis("off")

        fig_pr.suptitle(f"{' vs '.join(model_names)} — Curvas Precision-Recall e Iso-F1", fontsize=13, fontweight="bold", y=1.01)
        fig_pr.tight_layout()
        spath_pr = Path(save_dir) / "pr_curves_iso_f1_cmp.png" if save_dir else None
        figs.append(_finalize_fig(fig_pr, save_path=spath_pr, show=show))

        # Barras comparativas dedicadas de Precision, Recall, F1 (s>=0.5) e Melhor F1 (s*)
        fig_bars, axes_b = plt.subplots(2, 2, figsize=(13.2, 8.6))
        x = np.arange(len(kappas))
        w_bar = 0.72 / len(model_names)
        metric_specs = [
            ("pur_w_bal", f"Precision Balanceada (Score ≥ {thr_eval:g})"),
            ("eff_s_w", f"Recall / Eficiência de Sinal (Score ≥ {thr_eval:g})"),
            ("f1_w_bal", f"F1-score Balanceado (Score ≥ {thr_eval:g})"),
            ("best_f1", "F1-score Máximo ($F_1^*$ no corte ótimo)"),
        ]
        for ax_b, (mkey, mtitle) in zip(axes_b.ravel(), metric_specs):
            ymin = 1.0
            for i, m_name in enumerate(model_names):
                vals = []
                for k in kappas:
                    fk = float(k)
                    if mkey == "best_f1":
                        vals.append(float(curves[m_name][fk]["best_f1"]))
                    else:
                        d = scores[m_name][fk]
                        vals.append(binary_metrics_at_threshold(d["y"], d["s"], d["w"], thr=thr_eval)[mkey])
                ymin = min(ymin, min(vals))
                offs = (i - (len(model_names) - 1) / 2.0) * w_bar
                col = MODEL_COLORS.get(m_name, f"C{i}")
                bars = ax_b.bar(x + offs, vals, w_bar, color=col, alpha=0.9, label=m_name)
                for b in bars:
                    ax_b.text(
                        b.get_x() + b.get_width() / 2.0,
                        b.get_height() + 0.004,
                        f"{b.get_height():.4f}",
                        ha="center",
                        va="bottom",
                        fontsize=7.8,
                        color=col,
                        fontweight="bold",
                    )
            ax_b.set_xticks(x)
            ax_b.set_xticklabels([rf"$\kappa_\lambda={k:g}$" for k in kappas])
            ax_b.set_ylim(max(0.0, ymin - 0.06), 1.06)
            ax_b.set_title(mtitle, fontweight="bold")
            ax_b.legend(fontsize=9)
            ax_b.grid(axis="y", alpha=0.3)

        fig_bars.suptitle(f"{' vs '.join(model_names)} — Comparação de Precision, Recall e F1-score", fontsize=13, fontweight="bold", y=1.01)
        fig_bars.tight_layout()
        spath_b = Path(save_dir) / "prf1_bars_cmp.png" if save_dir else None
        figs.append(_finalize_fig(fig_bars, save_path=spath_b, show=show))

    return figs


def plot_prf1_vs_score_separate(
    scores: dict[str, dict[float, dict[str, np.ndarray]]],
    kappas: Sequence[float] = (0.0, 1.0, 2.45, 5.0),
    save_dir: str | Path | None = None,
    show: bool = True,
) -> dict[str, plt.Figure]:
    """Gera 3 figuras SEPARADAS em função do Score da rede P(HH):
      1. `precision` : Precision vs Score da rede P(HH)
      2. `recall`    : Recall vs Score da rede P(HH)
      3. `f1`        : F1-score vs Score da rede P(HH) (com marcador no pico de F1)

    Funciona tanto para a `AffinePNN` pura (1 modelo) quanto para `AffinePNN vs DNN_AN`.
    """
    model_names = list(scores.keys())
    is_single = len(model_names) == 1
    tag = model_names[0] if is_single else "cmp"
    title_prefix = f"{model_names[0]} Pura" if is_single else " vs ".join(model_names)

    curves = {
        m_name: {float(k): compute_curves_for_kappa(d["y"], d["s"], d["w"]) for k, d in sc.items()}
        for m_name, sc in scores.items()
    }

    specs = [
        ("precision", "Precision Balanceada", (0.45, 1.02), "lower right"),
        ("recall", r"Recall / Eficiência de Sinal ($\epsilon_S$)", (0.0, 1.03), "lower left"),
        ("f1", "F1-score Balanceado", (0.20, 1.00), "lower center"),
    ]

    ncols = 2 if len(kappas) > 2 else max(len(kappas), 1)
    nrows = int(np.ceil(len(kappas) / ncols))
    out_figs: dict[str, plt.Figure] = {}

    for metric_key, ylabel, ylims, leg_loc in specs:
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.3 * nrows), sharex=True, squeeze=False)
        for idx, k in enumerate(kappas):
            fk = float(k)
            ax = axes[idx // ncols, idx % ncols]
            for i, m_name in enumerate(model_names):
                c = curves[m_name][fk]
                col = KAPPA_COLORS[idx % len(KAPPA_COLORS)] if is_single else MODEL_COLORS.get(m_name, f"C{i}")
                ls = "-" if is_single else MODEL_STYLES.get(m_name, "-")
                if metric_key == "f1":
                    lbl = f"{m_name} ($F_1^*$={c['best_f1']:.3f} @ s={c['best_thr']:.2f})"
                    ax.plot(c["thr"], c["f1"], color=col, ls=ls, lw=2.1, label=lbl)
                    ax.plot(c["best_thr"], c["best_f1"], marker="o", markersize=6, color=col)
                    ax.axvline(c["best_thr"], color=col, ls=":", lw=1.0, alpha=0.65)
                else:
                    val_05 = float(np.interp(0.5, c["thr"], c[metric_key]))
                    lbl = f"{m_name} (@0.5 = {val_05:.3f})"
                    ax.plot(c["thr"], c[metric_key], color=col, ls=ls, lw=2.1, label=lbl)
                    ax.axvline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)

            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(*ylims)
            ax.set_xlabel(r"Score da Rede $P(\mathrm{HH})$")
            ax.set_ylabel(ylabel)
            ax.set_title(rf"{ylabel.split('(')[0].strip()} vs Score ($\kappa_\lambda = {k:g}$)", fontweight="bold")
            ax.legend(loc=leg_loc, fontsize=8.5)
            ax.grid(True, alpha=0.3)

        for idx in range(len(kappas), nrows * ncols):
            axes[idx // ncols, idx % ncols].axis("off")

        fig.suptitle(f"{title_prefix} — {ylabel} vs Score $P(\\mathrm{{HH}})$ (Plot Separado)", fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        spath = Path(save_dir) / f"{metric_key}_vs_score_{tag}.png" if save_dir else None
        out_figs[metric_key] = _finalize_fig(fig, save_path=spath, show=show)

    return out_figs


def plot_prf1_vs_score_all_in_one(
    scores: dict[str, dict[float, dict[str, np.ndarray]]],
    kappas: Sequence[float] = (0.0, 1.0, 2.45, 5.0),
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot único comparando TODAS as curvas juntas (Precision + Recall + F1-score vs Score P(HH))
    para cada hipótese de κ_λ em layout 2x2.

    Funciona tanto para a `AffinePNN` pura (1 modelo) quanto para `AffinePNN vs DNN_AN` (sólido vs tracejado).
    """
    model_names = list(scores.keys())
    is_single = len(model_names) == 1
    title_prefix = f"{model_names[0]} Pura" if is_single else " vs ".join(model_names)

    curves = {
        m_name: {float(k): compute_curves_for_kappa(d["y"], d["s"], d["w"]) for k, d in sc.items()}
        for m_name, sc in scores.items()
    }

    ncols = 2 if len(kappas) > 2 else max(len(kappas), 1)
    nrows = int(np.ceil(len(kappas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.7 * nrows), squeeze=False)

    for idx, k in enumerate(kappas):
        fk = float(k)
        ax = axes[idx // ncols, idx % ncols]
        for i, m_name in enumerate(model_names):
            c = curves[m_name][fk]
            short = "" if is_single else f" ({'Affine' if m_name == 'AffinePNN' else ('AN' if m_name == 'DNN_AN' else m_name)})"
            ls = "-" if is_single else MODEL_STYLES.get(m_name, ["-", "--"][i % 2])
            lw = 2.1 if i == 0 else 1.8

            ax.plot(c["thr"], c["precision"], color="#1f77b4", ls=ls, lw=lw, label=f"Precision{short}")
            ax.plot(c["thr"], c["recall"], color="#2ca02c", ls=ls, lw=lw, label=f"Recall{short}")
            ax.plot(
                c["thr"],
                c["f1"],
                color="#d62728",
                ls=ls,
                lw=lw + 0.2,
                label=f"F1{short} (máx={c['best_f1']:.3f} @ {c['best_thr']:.2f})",
            )
            ax.plot(c["best_thr"], c["best_f1"], marker="o", color="#d62728", markersize=6)
            ax.axvline(c["best_thr"], color="#d62728", ls=":", lw=1.0, alpha=0.6)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.04)
        ax.set_xlabel(r"Score da Rede $P(\mathrm{HH})$")
        ax.set_ylabel("Precision / Recall / F1-score")
        ax.set_title(rf"Precision, Recall e F1 vs Score — $\kappa_\lambda = {k:g}$", fontweight="bold")
        ax.legend(loc="lower left", fontsize=8.0, ncol=1 if is_single else 2)
        ax.grid(True, alpha=0.3)

    for idx in range(len(kappas), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    fig.suptitle(
        f"{title_prefix} — Comparação Conjunta: Precision, Recall e F1-score vs Score $P(\\mathrm{{HH}})$",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return _finalize_fig(fig, save_path=save_path, show=show)


def generate_complete_plot_suite(
    eval_res: dict[str, Any],
    class_labels: dict[int, str],
    kappas: Sequence[float],
    plots_root: str | Path,
    primary_model: str = "AffinePNN",
    show: bool = False,
) -> dict[str, Path]:
    """Gera e organiza em disco TODAS as figuras da rede `AffinePNN` pura (`affine_pure/`)
    e da comparação `AffinePNN vs DNN_AN` (`comparison/`)."""
    root = Path(plots_root)
    scores = eval_res["scores"]
    tables = eval_res["tables"]
    split_label = eval_res.get("split", "val")
    saved_dirs: dict[str, Path] = {}

    # 1. Suíte dedicada apenas à AffinePNN pura (sem DNN_AN)
    if primary_model in scores:
        pure_dir = root / "affine_pure"
        pure_dir.mkdir(parents=True, exist_ok=True)
        pure_scores = {primary_model: scores[primary_model]}

        plot_roc_per_kappa_individual(
            pure_scores, kappas=kappas, title_prefix=f"{primary_model} Pura", save_dir=pure_dir / "roc_per_kappa", show=show
        )
        plot_scores_vs_events_log_per_kappa(
            pure_scores, class_labels=class_labels, kappas=kappas, save_dir=pure_dir / "scores_vs_events_log", show=show
        )
        plot_pr_curves_and_f1_summary(
            pure_scores, kappas=kappas, save_dir=pure_dir, show=show
        )
        plot_prf1_vs_score_separate(
            pure_scores, kappas=kappas, save_dir=pure_dir / "prf1_vs_score_separate", show=show
        )
        plot_prf1_vs_score_all_in_one(
            pure_scores, kappas=kappas, save_path=pure_dir / f"prf1_vs_score_all_in_one_{primary_model}.png", show=show
        )
        saved_dirs["affine_pure"] = pure_dir

    # 2. Suíte comparativa (AffinePNN vs DNN_AN / demais modelos presentes)
    if len(scores) >= 2:
        cmp_dir = root / "comparison"
        cmp_dir.mkdir(parents=True, exist_ok=True)

        plot_roc_per_kappa_individual(
            scores, kappas=kappas, save_dir=cmp_dir / "roc_per_kappa", show=show
        )
        plot_scores_vs_events_log_per_kappa(
            scores, class_labels=class_labels, kappas=kappas, save_dir=cmp_dir / "scores_vs_events_log", show=show
        )
        plot_pr_curves_and_f1_summary(
            scores, kappas=kappas, save_dir=cmp_dir, show=show
        )
        plot_prf1_vs_score_separate(
            scores, kappas=kappas, save_dir=cmp_dir / "prf1_vs_score_separate", show=show
        )
        plot_prf1_vs_score_all_in_one(
            scores, kappas=kappas, save_path=cmp_dir / "prf1_vs_score_all_in_one_cmp.png", show=show
        )
        plot_roc_and_pr_grid(
            scores, kappas=kappas, split_label=split_label, save_path=cmp_dir / "roc_and_pr_grid.png", show=show
        )
        plot_metrics_bars_cmp(
            tables, kappas=kappas, split_label=split_label, save_path=cmp_dir / "metrics_bars_cmp.png", show=show
        )
        plot_significance_ratio(
            scores, kappas=kappas, save_path=cmp_dir / "significance_ratio_scan.png", show=show
        )
        saved_dirs["comparison"] = cmp_dir

    return saved_dirs
