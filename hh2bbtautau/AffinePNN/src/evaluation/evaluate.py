"""Orquestrador completo de avaliação multiclasse e binária por kappa_lambda (validação ou teste)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tensorflow import keras

from src.data.preprocessing import PreparedData, make_inputs
from src.data.weights import make_eval_weights
from src.evaluation.kappa_scan import compare_models_table, summarize_scores
from src.evaluation.score_distributions import score_per_kappa
from src.evaluation.significance import physical_yields, significance_ratio_table
from src.training.losses import weighted_logloss


def multiclass_summary(
    model: keras.Model,
    inputs: dict[str, np.ndarray],
    y: np.ndarray,
    w_bal: np.ndarray,
    class_labels: dict[int, str],
    batch_size: int = 8192,
) -> tuple[float, float, pd.DataFrame]:
    """Acurácia global e Eficiência (Recall) / Pureza (Precision) por classe {DY, ttbar, HH} via argmax."""
    p = model.predict(inputs, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(p, axis=1)
    y_true = np.asarray(y)
    w_arr = np.asarray(w_bal, dtype=np.float64)

    acc_unw = float((y_pred == y_true).mean())
    acc_w = float(w_arr[y_pred == y_true].sum() / w_arr.sum())

    rows = []
    for c, name in class_labels.items():
        t_c = y_true == c
        p_c = y_pred == c
        tp = float(np.sum(t_c & p_c))
        eff_unw = tp / max(float(np.sum(t_c)), 1.0)
        pur_unw = tp / max(float(np.sum(p_c)), 1.0)

        tp_w = float(w_arr[t_c & p_c].sum())
        eff_w = tp_w / max(float(w_arr[t_c].sum()), 1e-12)
        pur_w = tp_w / max(float(w_arr[p_c].sum()), 1e-12)

        rows.append(
            {
                "classe": name,
                "eficiencia_unw": eff_unw,
                "pureza_unw": pur_unw,
                "eficiencia_w": eff_w,
                "pureza_w": pur_w,
            }
        )
    return acc_unw, acc_w, pd.DataFrame(rows)


def run_full_evaluation(
    models: dict[str, keras.Model],
    data: PreparedData,
    cfg: dict[str, Any],
    split: str | None = None,
    metrics_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Avalia todos os modelos em `split` ('val' ou 'test'), imprime relatório e salva CSVs em `metrics_dir`."""
    ecfg = cfg["evaluation"]
    eval_split = split or ("test" if ecfg.get("final_eval", False) else ecfg.get("eval_split", "val"))
    class_labels = cfg["physics"]["class_labels"]
    kgrid = cfg["physics"]["kappa_grid"]
    knorm = float(cfg["physics"]["kappa_norm"])
    eval_bs = int(ecfg["eval_batch_size"])
    thr = float(ecfg.get("threshold", 0.5))
    n_boot = int(ecfg.get("n_boot", 100))
    eff_pts = ecfg.get("sig_eff_points", [0.5, 0.8])

    ev_df = data.splits[eval_split]
    ev_xc = data.x_cont[eval_split]
    ev_xk = data.x_cat[eval_split]

    ev_inputs = make_inputs(ev_xc, ev_xk, ev_df["kappa_eval"].to_numpy(), kappa_norm=knorm)
    ev_y = ev_df["label"].to_numpy()
    ev_w = make_eval_weights(ev_df, class_labels)

    if verbose:
        print(f"Avaliando conjunto: '{eval_split}' ({len(ev_df):,d} eventos)\n")
        print("1. MÉTRICAS GLOBAIS MULTICLASSE (DY vs ttbar vs HH via argmax):")

    multiclass_results: dict[str, dict[str, Any]] = {}
    scores: dict[str, dict[float, dict[str, np.ndarray]]] = {}
    tables: dict[str, pd.DataFrame] = {}
    yields_tables: dict[str, pd.DataFrame] = {}
    sig_ratio_tables: dict[str, pd.DataFrame] = {}

    for name, m in models.items():
        ll = weighted_logloss(m, ev_inputs, ev_y, ev_w, batch_size=eval_bs)
        acc_unw, acc_w, df_mc = multiclass_summary(m, ev_inputs, ev_y, ev_w, class_labels, batch_size=eval_bs)
        multiclass_results[name] = {
            "logloss_w": ll,
            "acc_unw": acc_unw,
            "acc_w": acc_w,
            "per_class": df_mc,
        }
        if verbose:
            print(f"\n--- {name} ---")
            print(f"  Log-loss ponderada : {ll:.4f}")
            print(f"  Acurácia Global    : sem peso = {acc_unw:.4f} ({acc_unw:.2%}) | ponderada = {acc_w:.4f} ({acc_w:.2%})")
            print(df_mc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        sc = score_per_kappa(m, ev_df, ev_xc, ev_xk, kappas=kgrid, kappa_norm=knorm, batch_size=eval_bs)
        scores[name] = sc
        t = summarize_scores(sc, thr=thr, sig_eff_points=eff_pts, n_boot=n_boot, seed=cfg["seed"])
        t["modelo"] = name
        tables[name] = t
        yields_tables[name] = physical_yields(sc, eff_points=eff_pts)
        sig_ratio_tables[name] = significance_ratio_table(sc)

    if verbose:
        print("\n" + "=" * 88)
        print(f"2. MÉTRICAS BINÁRIAS POR κ_λ (Sinal HH vs Todo o Fundo, corte P(HH) >= {thr:g}):")
        cols_main = [
            "kappa",
            "auc",
            "auc_w",
            "pr_auc",
            "sigma_ratio",
            "acc_w_bal",
            "eff_s_w",
            "pur_w_bal",
            "f1_w_bal",
            "best_f1_w_bal",
        ]
        for name, t in tables.items():
            print(f"\n--- {name} ---")
            print(t[cols_main].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            wp_cols = ["kappa"] + [c for c in t.columns if "@" in c] + ["auc_w_err"]
            print(t[wp_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    comparison_df = None
    if "AffinePNN" in tables and "DNN_AN" in tables:
        comparison_df = compare_models_table(tables["AffinePNN"], tables["DNN_AN"], "_A", "_AN")
        if verbose:
            print("\n3. COMPARAÇÃO DIRETA AffinePNN (A) vs DNN da AN (AN) — ponderada:")
            print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if metrics_dir is not None:
        mdir = Path(metrics_dir)
        mdir.mkdir(parents=True, exist_ok=True)
        pd.concat(tables.values(), ignore_index=True).to_csv(mdir / f"binary_metrics_{eval_split}.csv", index=False)
        mc_rows = []
        for name, mc in multiclass_results.items():
            d = mc["per_class"].copy()
            d.insert(0, "modelo", name)
            d["logloss_w"] = mc["logloss_w"]
            d["acc_global_w"] = mc["acc_w"]
            mc_rows.append(d)
        pd.concat(mc_rows, ignore_index=True).to_csv(mdir / f"multiclass_metrics_{eval_split}.csv", index=False)

        y_rows = []
        for name, yt in yields_tables.items():
            d = yt.copy()
            d.insert(0, "modelo", name)
            y_rows.append(d)
        pd.concat(y_rows, ignore_index=True).to_csv(mdir / f"physical_yields_{eval_split}.csv", index=False)

        if comparison_df is not None:
            comparison_df.to_csv(mdir / f"comparison_AffinePNN_vs_DNN_AN_{eval_split}.csv", index=False)

    return {
        "split": eval_split,
        "multiclass": multiclass_results,
        "scores": scores,
        "tables": tables,
        "yields": yields_tables,
        "significance_ratio": sig_ratio_tables,
        "comparison": comparison_df,
    }
