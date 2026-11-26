"""Carregamento hierárquico dos YAMLs de configuração e overrides via CLI/ambiente."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from src.utils.paths import PROJECT_ROOT


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Padroniza tipos (chaves inteiras de classes, grid de kappa_lambda e colunas)."""
    cfg = copy.deepcopy(cfg)

    # Normaliza chaves inteiras de classes
    if "physics" in cfg and "class_labels" in cfg["physics"]:
        cfg["physics"]["class_labels"] = {
            int(k): str(v) for k, v in cfg["physics"]["class_labels"].items()
        }
        cfg["physics"]["n_classes"] = len(cfg["physics"]["class_labels"])

    if "physics" in cfg and "kappa_map" in cfg["physics"]:
        kmap = {str(k): float(v) for k, v in cfg["physics"]["kappa_map"].items()}
        kgrid = sorted(set(kmap.values()))
        cfg["physics"]["kappa_map"] = kmap
        cfg["physics"]["kappa_grid"] = kgrid
        cfg["physics"]["kappa_norm"] = float(max(kgrid)) if kgrid else 1.0

    if "sampling" in cfg and "class_batch_fraction" in cfg["sampling"]:
        cfg["sampling"]["class_batch_fraction"] = {
            int(k): float(v) for k, v in cfg["sampling"]["class_batch_fraction"].items()
        }

    # Variáveis de ambiente úteis em jobs do HTCondor / lxplus
    env_parquet = os.environ.get("HH_PARQUET_DIR")
    if env_parquet:
        cfg.setdefault("paths", {})["parquet_dir"] = env_parquet

    env_output = os.environ.get("HH_OUTPUT_DIR")
    if env_output:
        cfg.setdefault("paths", {})["output_dir"] = env_output

    # Lista completa de colunas para leitura dos parquets
    feats = cfg.get("features", {})
    cont = feats.get("continuous_features", [])
    cat = feats.get("categorical_features", [])
    meta = feats.get("meta_columns", [])
    if cont or cat or meta:
        feats["columns_to_load"] = list(cont) + list(cat) + list(meta)

    return cfg


def load_config(
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Carrega `configs/default.yaml` (ou outro YAML) resolvendo `includes` automaticamente."""
    if config_path is None:
        cfg_file = PROJECT_ROOT / "configs" / "default.yaml"
    else:
        cfg_file = Path(config_path)
        if not cfg_file.is_absolute() and not cfg_file.exists():
            cfg_file = PROJECT_ROOT / cfg_file
    cfg_file = cfg_file.resolve()
    cfg_dir = cfg_file.parent

    raw = _read_yaml(cfg_file)

    if "includes" in raw:
        inc = raw["includes"]
        merged: dict[str, Any] = {}
        if "training" in inc:
            merged = _deep_merge(merged, _read_yaml(cfg_dir / inc["training"]))
        if "features" in inc:
            merged["features"] = _read_yaml(cfg_dir / inc["features"])
        if "models" in inc:
            merged["models"] = {
                m_key: _read_yaml(cfg_dir / m_file)
                for m_key, m_file in inc["models"].items()
            }
        # Permite overrides diretos no próprio arquivo manifesto fora de `includes`
        rest = {k: v for k, v in raw.items() if k != "includes"}
        merged = _deep_merge(merged, rest)
    else:
        # Caso o usuário passe `configs/training.yaml` diretamente
        merged = raw
        if "features" not in merged and (cfg_dir / "features.yaml").exists():
            merged["features"] = _read_yaml(cfg_dir / "features.yaml")
        if "models" not in merged:
            models_dict = {}
            for m_key in ("affine", "dnn_an", "pnn_concat"):
                p = cfg_dir / f"{m_key}.yaml"
                if p.exists():
                    models_dict[m_key] = _read_yaml(p)
            if models_dict:
                merged["models"] = models_dict

    if overrides:
        merged = apply_overrides(merged, overrides)

    return _normalize_config(merged)


def _parse_scalar(val_str: str) -> Any:
    low = val_str.strip().lower()
    if low in ("null", "none"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return json.loads(val_str)
    except Exception:
        return val_str


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Aplica overrides no formato `chave.subchave=valor` (ex: `training.epochs=10`)."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override inválido '{item}'. Use o formato chave.subchave=valor.")
        key_path, raw_val = item.split("=", 1)
        keys = [k.strip() for k in key_path.split(".") if k.strip()]
        val = _parse_scalar(raw_val)

        cur = out
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = val

    return _normalize_config(out)
