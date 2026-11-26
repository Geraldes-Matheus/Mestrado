#!/usr/bin/env python3
"""Script CLI de inferência para aplicar um checkpoint salvo sobre arquivos Parquet
e exportar probabilidades P(DY), P(ttbar), P(HH) para hipóteses arbitrárias de kappa_lambda.

Exemplo:
  python scripts/inference.py --model AffinePNN --input-parquet /eos/.../22post_hh_ggf_kl1.parquet --kappas 0 1 2.45 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.parquet_dataset import clean_and_check_data_quality, read_parquet_limited
from src.data.preprocessing import (
    CategoricalPreprocessor,
    FeatureScaler,
    apply_physical_preprocessing,
    make_inputs,
)
from src.training.train import load_model_bundle
from src.utils.config import load_config
from src.utils.paths import OutputPaths
from src.utils.seed import configure_gpu_memory_growth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferência com AffinePNN / DNN_AN em arquivos Parquet")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoints-dir", type=str, default=None)
    parser.add_argument("--model", type=str, default="AffinePNN", help="Nome do checkpoint (ex: AffinePNN ou DNN_AN)")
    parser.add_argument("--input-parquet", type=str, required=True, help="Arquivo Parquet de entrada")
    parser.add_argument("--output-parquet", type=str, default=None, help="Arquivo Parquet de saída em outputs/predictions/")
    parser.add_argument("--kappas", type=float, nargs="+", default=[0.0, 1.0, 2.45, 5.0], help="Hipóteses de κ_λ a inferir")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--use-onnx",
        action="store_true",
        help="Executa inferência usando o pacote ONNX End-to-End em outputs/exported_models/<Model>/ sem carregar Keras",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_paths = OutputPaths.from_root(cfg["paths"]["output_dir"], create=True)
    ckpt_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else out_paths.checkpoints

    fcfg = cfg["features"]
    cont_cols = fcfg["continuous_features"]
    cat_cols = fcfg["categorical_features"]
    in_path = Path(args.input_parquet)

    import pyarrow.parquet as pq

    avail_cols = set(pq.ParquetFile(in_path).schema.names)
    extra_meta = [c for c in fcfg.get("meta_columns", []) if c in avail_cols]
    cols_to_read = cont_cols + cat_cols + extra_meta

    df, n_total = read_parquet_limited(in_path, cols_to_read, max_events=args.max_events, seed=cfg["seed"])

    pred_cols: dict[str, object] = {}
    for c in extra_meta:
        pred_cols[c] = df[c].to_numpy()

    if args.use_onnx:
        from src.models.export import ONNXModelPredictor

        predictor = ONNXModelPredictor(out_paths.exported_models / args.model, use_end2end_raw=True)
        for k in args.kappas:
            tag = str(k).replace(".", "p").replace("-", "m")
            probs = predictor.predict_raw_dataframe(df, float(k))
            pred_cols[f"P_DY_kl{tag}"] = probs[:, 0]
            pred_cols[f"P_ttbar_kl{tag}"] = probs[:, 1]
            pred_cols[f"P_HH_kl{tag}"] = probs[:, 2]
    else:
        configure_gpu_memory_growth()
        model, _, _ = load_model_bundle(ckpt_dir, args.model)
        scaler = FeatureScaler.load_csv(ckpt_dir / "normalization_train_only.csv")
        cat_prep = CategoricalPreprocessor.load_json(ckpt_dir / "config_encoders.json")

        df, _ = clean_and_check_data_quality(
            df,
            cont_cols,
            cat_cols,
            sentinels=fcfg.get("sentinels", [-99999.0, -9999.0, -999.0]),
            verbose=False,
        )
        apply_physical_preprocessing(
            df,
            continuous_features=cont_cols,
            cov_cols=fcfg.get("cov_cols", []),
            missing_masks=fcfg.get("missing_object_masks", {}),
        )

        x_cont = scaler.transform(df)
        x_cat = cat_prep.transform(df)
        knorm = float(cfg["physics"]["kappa_norm"])

        for k in args.kappas:
            tag = str(k).replace(".", "p").replace("-", "m")
            probs = model.predict(
                make_inputs(x_cont, x_cat, float(k), kappa_norm=knorm),
                batch_size=args.batch_size,
                verbose=0,
            )
            pred_cols[f"P_DY_kl{tag}"] = probs[:, 0]
            pred_cols[f"P_ttbar_kl{tag}"] = probs[:, 1]
            pred_cols[f"P_HH_kl{tag}"] = probs[:, 2]

    out_df = pd.DataFrame(pred_cols)
    out_file = (
        Path(args.output_parquet)
        if args.output_parquet
        else (out_paths.predictions / f"preds_{args.model}_{in_path.stem}.parquet")
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_file, index=False)
    print(f"Inferência concluída ({len(out_df):,d}/{n_total:,d} eventos) -> salvo em {out_file}")


if __name__ == "__main__":
    main()
