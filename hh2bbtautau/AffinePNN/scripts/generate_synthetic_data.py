#!/usr/bin/env python3
"""Gera arquivos Parquet sintéticos localmente em `data/synthetic/` para testar os notebooks
e scripts fora do lxplus/EOS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthetic import generate_synthetic_parquets
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Gerador de Parquets sintéticos (HH -> bbtautau)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default="data/synthetic")
    parser.add_argument("--bkg-events", type=int, default=8000, help="Eventos por arquivo de fundo (DY/ttbar)")
    parser.add_argument("--sig-events", type=int, default=2500, help="Eventos por arquivo de sinal HH")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = generate_synthetic_parquets(
        output_dir=PROJECT_ROOT / args.output_dir,
        cfg=cfg,
        events_per_bkg_file=args.bkg_events,
        events_per_sig_file=args.sig_events,
        seed=args.seed,
    )
    print(f"Parquets sintéticos gerados com sucesso em: {out}")


if __name__ == "__main__":
    main()
