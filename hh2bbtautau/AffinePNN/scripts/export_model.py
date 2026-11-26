#!/usr/bin/env python3
"""Exporta modelos treinados (`outputs/checkpoints/`) para pacotes completos e autossuficientes
em `outputs/exported_models/<ModelName>/` contendo:
  - Keras v3 (`.keras`) + Pesos HDF5 (`.weights.h5`) + TensorFlow `SavedModel`
  - ONNX padrão (`<ModelName>.onnx`) + ONNX End-to-End Físico Bruto (`<ModelName>_end2end_raw.onnx`)
  - Todos os parâmetros aprendidos em NumPy (`learned_parameters_<ModelName>.npz`) + arquitetura JSON
  - Pré-processamento completo (`preprocessing_spec.json`, `normalization_train_only.csv`, `config_encoders.json`)
  - Manifesto auditável (`bundle_manifest.json`) com hashes SHA-256 e verificação numérica no `onnxruntime`.

Exemplo:
  python scripts/export_model.py --models AffinePNN DNN_AN
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import CategoricalPreprocessor, FeatureScaler
from src.models.export import export_complete_model_bundle
from src.training.train import load_model_bundle
from src.utils.config import apply_overrides, load_config
from src.utils.logging_utils import get_logger
from src.utils.paths import OutputPaths
from src.utils.seed import configure_gpu_memory_growth, set_all_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exportação completa de modelos (ONNX + SavedModel + Parâmetros)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoints-dir", type=str, default=None, help="Diretório com os checkpoints .keras")
    parser.add_argument("--export-dir", type=str, default=None, help="Diretório destino (default: outputs/exported_models)")
    parser.add_argument("--models", nargs="+", default=["AffinePNN", "DNN_AN"], help="Nomes dos modelos a exportar")
    parser.add_argument("--opset", type=int, default=17, help="Versão do opset ONNX (default: 17)")
    parser.add_argument("--skip-parity-check", action="store_true", help="Pula verificação no onnxruntime")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    out_paths = OutputPaths.from_root(cfg["paths"]["output_dir"], create=True)

    ckpt_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else out_paths.checkpoints
    export_dir = Path(args.export_dir) if args.export_dir else (out_paths.root / "exported_models")
    logger = get_logger("affinepnn.export", log_file=out_paths.logs / "export_models.log")

    configure_gpu_memory_growth()
    set_all_seeds(cfg["seed"])

    scaler = FeatureScaler.load_csv(ckpt_dir / "normalization_train_only.csv")
    cat_prep = CategoricalPreprocessor.load_json(ckpt_dir / "config_encoders.json")

    for m_name in args.models:
        logger.info("Exportando pacote completo do modelo %s ...", m_name)
        model, history, meta = load_model_bundle(ckpt_dir, m_name)
        manifest = export_complete_model_bundle(
            model_name=m_name,
            model=model,
            scaler=scaler,
            cat_preprocessor=cat_prep,
            cfg=cfg,
            export_root_dir=export_dir,
            history_dict=history,
            meta_dict=meta,
            opset=args.opset,
            verify_parity=not args.skip_parity_check,
        )
        parity = manifest.get("onnx_parity_verification") or {}
        logger.info(
            "Pacote '%s' salvo em %s | params=%d | ONNX max|ΔP| (pre=%.2e, end2end_raw=%.2e) [%s]",
            m_name,
            manifest["bundle_dir"],
            manifest["total_params"],
            parity.get("preprocessed_onnx_max_abs_diff", -1.0),
            parity.get("end2end_raw_onnx_max_abs_diff", -1.0),
            parity.get("status", "SKIPPED"),
        )


if __name__ == "__main__":
    main()
