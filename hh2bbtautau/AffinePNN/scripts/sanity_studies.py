#!/usr/bin/env python3
"""Script CLI para executar estudos pesados de sanidade no CERN HTCondor ou terminal:
  - random_label  (Seção 13.4: labels embaralhados -> loss em ln(3))
  - interpolation (Seção 13.5: treino omitindo held_out κ_λ = 2.45)
  - seeds         (Seção 13.6: estabilidade entre múltiplas seeds; suporta job array por seed!)

Exemplos:
  python scripts/sanity_studies.py --study random_label --seeds 1 2 3
  python scripts/sanity_studies.py --study interpolation --held-out 2.45
  python scripts/sanity_studies.py --study seeds --seeds 101 102 103 104 105
  # Em job array do HTCondor (1 seed por GPU worker):
  python scripts/sanity_studies.py --study seeds --seeds 101
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import make_inputs, prepare_datasets_from_config
from src.data.weights import oversampling_frequency_weights
from src.evaluation.kappa_scan import interpolation_study, seed_study
from src.evaluation.plotting import (
    plot_interpolation_study,
    plot_random_label_test,
    plot_seed_stability,
)
from src.training.diagnostics import random_label_test
from src.training.train import create_generator_for_model, load_model_bundle
from src.utils.config import apply_overrides, load_config
from src.utils.paths import OutputPaths
from src.utils.seed import configure_gpu_memory_growth, set_all_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estudos de Sanidade, Interpolação e Seeds — AffinePNN")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--study", choices=["random_label", "interpolation", "seeds"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Lista de seeds (ex: 1 2 3 ou $(ProcId))")
    parser.add_argument("--held-out", type=float, default=2.45, help="Valor de κ_λ omitido no estudo de interpolação")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--parquet-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    extra_ov = []
    if args.parquet_dir:
        extra_ov.append(f"paths.parquet_dir={args.parquet_dir}")
    if args.output_dir:
        extra_ov.append(f"paths.output_dir={args.output_dir}")
    if extra_ov:
        cfg = apply_overrides(cfg, extra_ov)

    out_paths = OutputPaths.from_root(cfg["paths"]["output_dir"], create=True)
    configure_gpu_memory_growth()
    set_all_seeds(cfg["seed"])

    if args.synthetic:
        from src.data.synthetic import generate_synthetic_parquets

        syn_dir = out_paths.root / "synthetic_parquets"
        generate_synthetic_parquets(syn_dir, cfg, seed=cfg["seed"])
        cfg["paths"]["parquet_dir"] = str(syn_dir)

    ckpt_exists = (out_paths.checkpoints / "normalization_train_only.csv").exists()
    data = prepare_datasets_from_config(
        cfg,
        existing_checkpoints_dir=out_paths.checkpoints if ckpt_exists else None,
        save_checkpoints_dir=None if ckpt_exists else out_paths.checkpoints,
        verbose=True,
    )

    knorm = float(cfg["physics"]["kappa_norm"])
    val_df = data.splits["val"]
    val_inputs = make_inputs(data.x_cont["val"], data.x_cat["val"], val_df["kappa_eval"].to_numpy(), kappa_norm=knorm)

    if args.study == "random_label":
        seeds = args.seeds or cfg["studies"]["random_label_seeds"]
        epochs = args.epochs or cfg["studies"]["random_label_epochs"]
        rl_df = random_label_test(
            cfg=cfg,
            data=data,
            val_inputs=val_inputs,
            seeds=seeds,
            epochs=epochs,
            steps_per_epoch=args.steps_per_epoch,
            verbose=True,
        )
        rl_df.to_csv(out_paths.metrics / "sanity_random_label_test.csv", index=False)
        plot_random_label_test(rl_df, save_path=out_paths.plots / "sanity_random_label_test.png", show=False)

    elif args.study == "interpolation":
        gen_ref = create_generator_for_model(cfg, "affine", data)
        val_freq_w = oversampling_frequency_weights(
            val_df,
            gen_ref,
            class_fraction=cfg["sampling"]["class_batch_fraction"],
            clip=cfg["sampling"]["val_freq_clip"],
        )
        model_full = None
        model_dnn = None
        try:
            model_full, _, _ = load_model_bundle(out_paths.checkpoints, "AffinePNN")
        except FileNotFoundError:
            pass
        try:
            model_dnn, _, _ = load_model_bundle(out_paths.checkpoints, "DNN_AN")
        except FileNotFoundError:
            pass
        ser = interpolation_study(
            cfg=cfg,
            data=data,
            val_freq_w=val_freq_w,
            model_full_ref=model_full,
            model_dnn_ref=model_dnn,
            held_out=args.held_out,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            verbose=True,
        )
        tag = str(args.held_out).replace(".", "p")
        ser.to_frame().to_csv(out_paths.metrics / f"sanity_interpolation_kl{tag}.csv")
        plot_interpolation_study(
            ser, held_out=args.held_out, save_path=out_paths.plots / f"sanity_interpolation_kl{tag}.png", show=False
        )

    elif args.study == "seeds":
        seeds = args.seeds or cfg["studies"]["seeds_for_study"]
        seeds_df, gain_df = seed_study(
            cfg=cfg,
            data=data,
            seeds=seeds,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            verbose=True,
        )
        seed_tag = "_".join(str(s) for s in seeds)
        seeds_df.to_csv(out_paths.metrics / f"sanity_seeds_metrics_{seed_tag}.csv", index=False)
        gain_df.to_csv(out_paths.metrics / f"sanity_seeds_gain_{seed_tag}.csv", index=False)
        plot_seed_stability(
            seeds_df, save_path=out_paths.plots / f"sanity_seeds_stability_{seed_tag}.png", show=False
        )


if __name__ == "__main__":
    main()
