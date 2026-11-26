#!/usr/bin/env python3
"""Script principal de treinamento (compatível com CERN HTCondor e execução local).

Exemplos:
  # Treino padrão completo (AffinePNN + DNN_AN) usando configs/default.yaml:
  python scripts/train.py --config configs/default.yaml

  # Treinar apenas a AffinePNN salvando no EOS:
  python scripts/train.py --models affine --output-dir /eos/user/m/mgeralde/hh_affine_outputs

  # Smoke-test rápido local com parquets sintéticos (2 épocas, 10 steps):
  python scripts/train.py --synthetic --epochs 2 --steps-per-epoch 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.parquet_dataset import check_feature_leakage_names
from src.data.preprocessing import check_four_vectors, make_inputs, prepare_datasets_from_config
from src.data.sampler import pool_report
from src.data.weights import oversampling_frequency_weights, weight_report
from src.evaluation.evaluate import run_full_evaluation
from src.training.losses import an_style_loss
from src.training.train import create_generator_for_model, save_model_bundle, train_model
from src.utils.config import apply_overrides, load_config
from src.utils.logging_utils import get_logger
from src.utils.paths import OutputPaths
from src.utils.seed import configure_gpu_memory_growth, set_all_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinamento AffinePNN vs DNN_AN (CMS HH -> bbtautau)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Caminho do YAML principal")
    parser.add_argument("--models", nargs="+", default=None, help="Chaves de modelos a treinar (ex: affine dnn_an pnn_concat)")
    parser.add_argument("--parquet-dir", type=str, default=None, help="Sobrescreve paths.parquet_dir")
    parser.add_argument("--output-dir", type=str, default=None, help="Sobrescreve paths.output_dir")
    parser.add_argument("--seed", type=int, default=None, help="Sobrescreve seed global")
    parser.add_argument("--epochs", type=int, default=None, help="Sobrescreve training.epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Sobrescreve training.steps_per_epoch")
    parser.add_argument("--synthetic", action="store_true", help="Gera e utiliza parquets sintéticos para teste rápido")
    parser.add_argument("--skip-eval", action="store_true", help="Pula a avaliação automática no conjunto de validação")
    parser.add_argument("--export", action="store_true", help="Força exportação ONNX/SavedModel/Keras ao final")
    parser.add_argument("--skip-export", action="store_true", help="Desativa exportação automática ONNX/SavedModel")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Overrides chave.subchave=valor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)

    extra_overrides: list[str] = []
    if args.parquet_dir:
        extra_overrides.append(f"paths.parquet_dir={args.parquet_dir}")
    if args.output_dir:
        extra_overrides.append(f"paths.output_dir={args.output_dir}")
    if args.seed is not None:
        extra_overrides.append(f"seed={args.seed}")
    if args.epochs is not None:
        extra_overrides.append(f"training.epochs={args.epochs}")
    if args.steps_per_epoch is not None:
        extra_overrides.append(f"training.steps_per_epoch={args.steps_per_epoch}")
    if extra_overrides:
        cfg = apply_overrides(cfg, extra_overrides)

    out_paths = OutputPaths.from_root(cfg["paths"]["output_dir"], create=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = get_logger("affinepnn.train", log_file=out_paths.logs / f"train_{ts}.log")

    gpus = configure_gpu_memory_growth()
    set_all_seeds(cfg["seed"])
    logger.info("Iniciando pipeline de treino | seed=%d | GPUs=%s", cfg["seed"], gpus or "CPU")

    if args.synthetic:
        from src.data.synthetic import generate_synthetic_parquets

        syn_dir = out_paths.root / "synthetic_parquets"
        logger.info("Gerando parquets sintéticos em %s ...", syn_dir)
        generate_synthetic_parquets(syn_dir, cfg, seed=cfg["seed"])
        cfg["paths"]["parquet_dir"] = str(syn_dir)

    # 1. Checagem contra vazamento de colunas de metadados (13.1)
    bad_cols = check_feature_leakage_names(
        cfg["features"]["continuous_features"],
        cfg["features"]["categorical_features"],
        weight_col=cfg["features"]["weight_col"],
        forbidden=cfg["features"].get("forbidden_substrings", ()),
    )
    if bad_cols:
        raise RuntimeError(f"Colunas proibidas detectadas nas features: {bad_cols}")
    logger.info("Checagem 13.1 OK: 49 contínuas + 7 categóricas limpas.")

    # 2. Carregamento, limpeza, split estratificado, normalização treino-only e embeddings
    data = prepare_datasets_from_config(cfg, save_checkpoints_dir=out_paths.checkpoints, verbose=True)

    # 3. Verificação física dos 4-vetores do LBN
    check_four_vectors(
        x_cont=data.x_cont["train"],
        x_cat=data.x_cat["train"],
        scaler=data.scaler,
        continuous_features=cfg["features"]["continuous_features"],
        lbn_particles=cfg["features"]["lbn_particles"],
        strict=True,
        verbose=True,
    )

    # 4. Prepara entradas de validação
    val_df = data.splits["val"]
    val_y = val_df["label"].to_numpy()
    val_inputs = make_inputs(
        data.x_cont["val"],
        data.x_cat["val"],
        val_df["kappa_eval"].to_numpy(),
        kappa_norm=cfg["physics"]["kappa_norm"],
    )

    model_keys = args.models or cfg["training"]["models_to_train"]
    trained_models = {}
    all_meta = {}
    all_hist = {}
    ref_gen = None
    ref_val_freq_w = None

    for m_key in model_keys:
        if m_key not in cfg["models"]:
            raise KeyError(f"Modelo '{m_key}' não definido em configs. Disponíveis: {list(cfg['models'])}")
        m_name = cfg["models"][m_key].get("name", m_key)

        gen = create_generator_for_model(cfg, m_key, data)
        if ref_gen is None:
            ref_gen = gen
            pool_df = pool_report(gen, steps=len(gen), epochs=cfg["training"]["epochs"], verbose=True)
            pool_df.to_csv(out_paths.metrics / "training_pool_report.csv", index=False)

        val_w = oversampling_frequency_weights(
            val_df,
            gen,
            class_fraction=cfg["sampling"]["class_batch_fraction"],
            clip=cfg["sampling"]["val_freq_clip"],
        )
        if ref_val_freq_w is None:
            ref_val_freq_w = val_w
            weight_report(val_df, val_w, cfg["physics"]["class_labels"], title="VALIDAÇÃO: val_freq_w (early stopping)")

        model, hist_dict, meta = train_model(
            cfg=cfg,
            model_key=m_key,
            generator=gen,
            scaler=data.scaler,
            cat_vocab_sizes=data.cat_preprocessor.vocab_sizes,
            val_data=(val_inputs, val_y, val_w),
            name=m_name,
        )

        saved_files = save_model_bundle(out_paths, m_name, model, hist_dict, meta)
        logger.info(
            "Modelo %s salvo em %s (melhor época=%d, val_loss=%.4f, tempo=%.1fs)",
            m_name,
            saved_files["model"],
            meta["best_epoch"],
            meta["best_val_loss"],
            meta["elapsed_seconds"],
        )
        trained_models[m_name] = model
        all_meta[m_name] = meta
        all_hist[m_name] = hist_dict

    # Salva snapshot completo da configuração utilizada
    with open(out_paths.checkpoints / "run_config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "models_trained": all_meta, "timestamp": ts}, f, indent=2)

    # 5. Avaliação pós-treino rápida em validação + loss estilo Fig. 50 da AN
    if not args.skip_eval and trained_models and ref_gen is not None and ref_val_freq_w is not None:
        fig50_df = an_style_loss(
            models=trained_models,
            train_df=data.splits["train"],
            x_cont_train=data.x_cont["train"],
            x_cat_train=data.x_cat["train"],
            val_inputs=val_inputs,
            val_y=val_y,
            val_freq_w=ref_val_freq_w,
            gen_ref=ref_gen,
            kappa_norm=cfg["physics"]["kappa_norm"],
            class_fraction=cfg["sampling"]["class_batch_fraction"],
            val_freq_clip=cfg["sampling"]["val_freq_clip"],
            batch_size=cfg["evaluation"]["eval_batch_size"],
            seed=cfg["seed"],
        )
        print("\nLoss estilo Fig. 50 da AN (sem oversampling, ponderada por frequência):")
        print(fig50_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        fig50_df.to_csv(out_paths.metrics / "an_fig50_loss_comparison.csv", index=False)

        run_full_evaluation(
            models=trained_models,
            data=data,
            cfg=cfg,
            split="val",
            metrics_dir=out_paths.metrics,
            verbose=True,
        )

    # 6. Exportação completa (ONNX + End2End ONNX + SavedModel + Keras + H5 + NPZ + manifesto)
    exp_cfg = cfg.get("export", {})
    should_export = (args.export or exp_cfg.get("enabled_after_training", True)) and not args.skip_export
    if should_export and trained_models:
        from src.models.export import export_complete_model_bundle

        export_dir = out_paths.root / exp_cfg.get("dir", "exported_models")
        for m_name, model in trained_models.items():
            manifest = export_complete_model_bundle(
                model_name=m_name,
                model=model,
                scaler=data.scaler,
                cat_preprocessor=data.cat_preprocessor,
                cfg=cfg,
                export_root_dir=export_dir,
                history_dict=all_hist.get(m_name),
                meta_dict=all_meta.get(m_name),
                opset=int(exp_cfg.get("onnx_opset", 17)),
                verify_parity=True,
            )
            parity = manifest.get("onnx_parity_verification") or {}
            logger.info(
                "Pacote completo '%s' exportado em %s | ONNX max|ΔP| end2end=%.2e [%s]",
                m_name,
                manifest["bundle_dir"],
                parity.get("end2end_raw_onnx_max_abs_diff", -1.0),
                parity.get("status", "OK"),
            )

    logger.info("Execução concluída com sucesso. Artefatos em: %s", out_paths.root)


if __name__ == "__main__":
    main()
