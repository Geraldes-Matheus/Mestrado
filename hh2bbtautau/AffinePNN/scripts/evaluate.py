#!/usr/bin/env python3
"""Script CLI de avaliação, diagnósticos e geração de figuras a partir de checkpoints salvos.

Exemplos:
  # Avaliar checkpoints na validação e salvar todos os plots e CSVs:
  python scripts/evaluate.py --config configs/default.yaml --diagnostics --kappa-scan

  # Avaliação FINAL congelada no conjunto de teste (exige flag explícita --unlock-test):
  python scripts/evaluate.py --split test --unlock-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.preprocessing import make_inputs, prepare_datasets_from_config
from src.data.weights import oversampling_frequency_weights
from src.evaluation.evaluate import run_full_evaluation
from src.evaluation.kappa_scan import continuous_kappa_scan, kappa_dependence, kappa_response_matrix
from src.evaluation.performance_plots import generate_performance_report
from src.evaluation.plotting import (
    generate_complete_plot_suite,
    plot_affine_conditioning_params,
    plot_continuous_kappa_scan,
    plot_kappa_dependence,
    plot_kappa_response_heatmap,
    plot_metrics_bars_cmp,
    plot_overtraining_distributions,
    plot_physical_yields_and_significance,
    plot_roc_and_pr_grid,
    plot_roc_and_scores_cmp,
    plot_signal_eff_vs_bkg_rejection,
    plot_significance_ratio,
    plot_threshold_tradeoff_combined,
    plot_threshold_tradeoff_panels,
    plot_training_history,
)
from src.evaluation.score_distributions import overtraining_ks
from src.models.affine_dnn import extract_affine_modulation
from src.training.diagnostics import run_validation_diagnostics
from src.training.train import create_generator_for_model, load_model_bundle
from src.utils.config import apply_overrides, load_config
from src.utils.logging_utils import get_logger
from src.utils.paths import OutputPaths
from src.utils.seed import configure_gpu_memory_growth, set_all_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avaliação e Plots — AffinePNN vs DNN_AN")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoints-dir", type=str, default=None, help="Diretório contendo os modelos .keras")
    parser.add_argument("--models", nargs="+", default=["AffinePNN", "DNN_AN"], help="Nomes dos modelos a avaliar")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Conjunto a avaliar")
    parser.add_argument("--unlock-test", action="store_true", help="Confirma uso explícito do conjunto de teste")
    parser.add_argument("--parquet-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true", help="Não gera arquivos de imagem em outputs/plots")
    parser.add_argument("--diagnostics", action="store_true", help="Roda diagnóstico profundo pós-treino (Célula 54)")
    parser.add_argument("--kappa-scan", action="store_true", help="Roda matrizes de resposta e varredura contínua em κ_λ")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.unlock_test:
        raise SystemExit(
            "ERRO: Para avaliar no conjunto 'test', passe explicitamente `--unlock-test` "
            "(só faça isso uma única vez após congelar toda a arquitetura e hiperparâmetros)."
        )

    cfg = load_config(args.config, overrides=args.overrides)
    extra_ov = []
    if args.parquet_dir:
        extra_ov.append(f"paths.parquet_dir={args.parquet_dir}")
    if args.output_dir:
        extra_ov.append(f"paths.output_dir={args.output_dir}")
    if extra_ov:
        cfg = apply_overrides(cfg, extra_ov)

    out_paths = OutputPaths.from_root(cfg["paths"]["output_dir"], create=True)
    ckpt_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else out_paths.checkpoints
    logger = get_logger("affinepnn.evaluate", log_file=out_paths.logs / f"evaluate_{args.split}.log")

    configure_gpu_memory_growth()
    set_all_seeds(cfg["seed"])

    models = {}
    histories = {}
    for m_name in args.models:
        m, h, _ = load_model_bundle(ckpt_dir, m_name)
        models[m_name] = m
        if h is not None:
            histories[m_name] = h
        logger.info("Carregado modelo %s (%d parâmetros)", m_name, m.count_params())

    data = prepare_datasets_from_config(cfg, existing_checkpoints_dir=ckpt_dir, verbose=True)

    eval_res = run_full_evaluation(
        models=models,
        data=data,
        cfg=cfg,
        split=args.split,
        metrics_dir=out_paths.metrics,
        verbose=True,
    )

    split_plots_dir = out_paths.plots / args.split
    split_plots_dir.mkdir(parents=True, exist_ok=True)

    ev_df = data.splits[args.split]
    ev_xc = data.x_cont[args.split]
    ev_xk = data.x_cat[args.split]
    knorm = float(cfg["physics"]["kappa_norm"])
    kgrid = cfg["physics"]["kappa_grid"]
    eval_bs = int(cfg["evaluation"]["eval_batch_size"])
    ev_inputs = make_inputs(ev_xc, ev_xk, ev_df["kappa_eval"].to_numpy(), kappa_norm=knorm)
    ev_y = ev_df["label"].to_numpy()

    # Teste KS e distribuição de Overtraining
    ks_frames = []
    for name, m in models.items():
        print(f"\nTeste KS (treino vs {args.split}) — {name}:")
        ks_df, ks_scores = overtraining_ks(
            m,
            data.splits["train"],
            data.x_cont["train"],
            data.x_cat["train"],
            ev_inputs,
            ev_y,
            cfg["physics"]["class_labels"],
            kappa_norm=knorm,
            batch_size=eval_bs,
            seed=cfg["seed"],
            return_scores=True,
        )
        ks_df.insert(0, "modelo", name)
        ks_frames.append(ks_df)
        if not args.no_plots:
            plot_overtraining_distributions(
                p_train=ks_scores["p_train"],
                y_train=ks_scores["y_train"],
                p_eval=ks_scores["p_eval"],
                y_eval=ks_scores["y_eval"],
                class_labels=cfg["physics"]["class_labels"],
                model_name=name,
                split_label=args.split,
                save_path=split_plots_dir / f"overtraining_distributions_{name}_{args.split}.png",
                show=False,
            )
    if ks_frames:
        import pandas as pd

        pd.concat(ks_frames, ignore_index=True).to_csv(
            out_paths.metrics / f"overtraining_ks_{args.split}.csv", index=False
        )

    if args.kappa_scan:
        matrices = {}
        for name, m in models.items():
            mat = kappa_response_matrix(m, ev_df, ev_xc, ev_xk, kappas=kgrid, kappa_norm=knorm, batch_size=eval_bs)
            matrices[name] = mat
            mat.to_csv(out_paths.metrics / f"kappa_response_matrix_{name}_{args.split}.csv")
            print(f"\nMatriz de resposta em κ_λ ({name}):\n{mat.to_string(float_format=lambda x: f'{x:.4f}')}")

        if not args.no_plots and matrices:
            plot_kappa_response_heatmap(
                matrices, save_path=split_plots_dir / "kappa_response_heatmap.png", show=False
            )

        if "AffinePNN" in models:
            k_min, k_max, k_n = cfg["evaluation"].get("kappa_scan_range", [-2.0, 8.0, 41])
            scan_vals = np.linspace(float(k_min), float(k_max), int(k_n))
            scan_df = continuous_kappa_scan(
                models["AffinePNN"],
                ev_df,
                ev_xc,
                ev_xk,
                kappa_scan_values=scan_vals,
                ref_signal_kappas=kgrid,
                kappa_norm=knorm,
                batch_size=eval_bs,
                seed=cfg["seed"],
            )
            scan_df.to_csv(out_paths.metrics / f"continuous_kappa_scan_{args.split}.csv", index=False)
            if not args.no_plots:
                plot_continuous_kappa_scan(
                    scan_df,
                    ref_kappas=kgrid,
                    save_path=split_plots_dir / "continuous_kappa_scan.png",
                    show=False,
                )
                mod_dict = extract_affine_modulation(models["AffinePNN"], scan_vals, kappa_norm=knorm)
                if mod_dict:
                    plot_affine_conditioning_params(
                        mod_dict, save_path=split_plots_dir / "affine_modulation_layers.png", show=False
                    )

    if args.diagnostics:
        gen_ref = create_generator_for_model(cfg, "affine", data)
        val_df = data.splits["val"]
        val_inputs = make_inputs(data.x_cont["val"], data.x_cat["val"], val_df["kappa_eval"].to_numpy(), kappa_norm=knorm)
        val_freq_w = oversampling_frequency_weights(
            val_df,
            gen_ref,
            class_fraction=cfg["sampling"]["class_batch_fraction"],
            clip=cfg["sampling"]["val_freq_clip"],
        )
        diag = run_validation_diagnostics(
            models=models,
            histories=histories,
            data=data,
            val_inputs=val_inputs,
            val_freq_w=val_freq_w,
            gen_ref=gen_ref,
            continuous_features=cfg["features"]["continuous_features"],
            class_labels=cfg["physics"]["class_labels"],
            kappa_norm=knorm,
            eval_batch_size=eval_bs,
            verbose=True,
        )
        diag["loss_tables"].to_csv(out_paths.metrics / "diagnostics_loss_tails.csv")
        diag["bn_comparison"].to_csv(out_paths.metrics / "diagnostics_bn_comparison.csv", index=False)
        diag["memorization"].to_csv(out_paths.metrics / "diagnostics_pool_memorization.csv", index=False)

    if not args.no_plots:
        if histories:
            plot_training_history(histories, save_path=split_plots_dir / "training_history.png", show=False)
        plot_roc_and_scores_cmp(
            eval_res["scores"], cfg["physics"]["class_labels"], save_dir=split_plots_dir, show=False
        )
        plot_roc_and_pr_grid(
            eval_res["scores"], kappas=kgrid, split_label=args.split, save_path=split_plots_dir / "roc_pr_grid.png", show=False
        )
        plot_metrics_bars_cmp(
            eval_res["tables"], kappas=kgrid, split_label=args.split, save_path=split_plots_dir / "metrics_bars_cmp.png", show=False
        )
        plot_threshold_tradeoff_panels(
            eval_res["scores"], kappas=kgrid, split_label=args.split, save_path=split_plots_dir / "threshold_tradeoff_panels.png", show=False
        )
        plot_threshold_tradeoff_combined(
            eval_res["scores"], kappas=kgrid, save_path=split_plots_dir / "threshold_tradeoff_combined.png", show=False
        )
        plot_significance_ratio(
            eval_res["scores"], kappas=kgrid, save_path=split_plots_dir / "significance_ratio_scan.png", show=False
        )
        plot_signal_eff_vs_bkg_rejection(
            eval_res["scores"], kappas=kgrid, save_path=split_plots_dir / "signal_eff_vs_bkg_rejection.png", show=False
        )
        plot_physical_yields_and_significance(
            eval_res["yields"],
            eval_res["significance_ratio"],
            save_path=split_plots_dir / "physical_yields_and_significance_cmp.png",
            show=False,
        )
        for m_name, m in models.items():
            dep_df, _ = kappa_dependence(
                m, ev_df, ev_xc, ev_xk, cfg["physics"]["class_labels"], kappas=kgrid, kappa_norm=knorm, batch_size=eval_bs
            )
            plot_kappa_dependence(
                dep_df,
                cfg["physics"]["class_labels"],
                model_name=m_name,
                save_path=split_plots_dir / f"kappa_dependence_{m_name}.png",
                show=False,
            )
        logger.info("Plots salvos em %s", split_plots_dir)

        # Suíte expandida: AffinePNN pura + Comparação AffinePNN vs DNN_AN
        suite_dirs = generate_complete_plot_suite(
            eval_res=eval_res,
            class_labels=cfg["physics"]["class_labels"],
            kappas=kgrid,
            plots_root=split_plots_dir,
            primary_model=cfg["evaluation"].get("reference_model", "AffinePNN"),
            show=False,
        )
        logger.info("Suíte completa salva em: %s", {k: str(v) for k, v in suite_dirs.items()})

        # Precision / Recall / F1 vs score, ROC por κ_λ, score vs eventos (log), matrizes de confusão
        ecfg = cfg["evaluation"]
        ref = ecfg.get("reference_model", "AffinePNN")
        if ref in eval_res["scores"]:
            rep = generate_performance_report(
                eval_res["scores"],
                split_plots_dir / "performance",
                class_labels=cfg["physics"]["class_labels"],
                kappas=kgrid,
                reference=ref,
                baseline=ecfg.get("baseline_model", "DNN_AN"),
                mode=ecfg.get("precision_mode", "balanced"),
                split_label=args.split,
                show=False,
            )
            n_figs = sum(len(v) for v in rep["paths"].values())
            logger.info("Relatório de desempenho: %d figuras em %s", n_figs, split_plots_dir / "performance")


if __name__ == "__main__":
    main()
