"""Leitura eficiente em chunks dos arquivos Parquet com reescala física e checagens de sanidade."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def short_cat_name(col: str) -> str:
    """Remove o prefixo `reg_dnn_moe_` das features categóricas."""
    return col.replace("reg_dnn_moe_", "")


def read_parquet_limited(
    path: str | Path,
    columns: list[str],
    max_events: int | None = None,
    chunk_size: int = 200_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, int]:
    """Lê um Parquet em chunks via PyArrow e sorteia sem reposição até `max_events` linhas.

    Retorna `(df, n_total)` onde `n_total` é o total de eventos presentes no arquivo antes do corte.
    """
    pf = pq.ParquetFile(path)
    n_total = int(pf.metadata.num_rows)
    if max_events is None or max_events >= n_total:
        keep = np.arange(n_total, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(n_total, size=int(max_events), replace=False))

    chunks: list[pd.DataFrame] = []
    row_offset = 0
    for batch in pf.iter_batches(batch_size=int(chunk_size), columns=columns):
        end = row_offset + batch.num_rows
        sub = keep[(keep >= row_offset) & (keep < end)] - row_offset
        if len(sub) > 0:
            d = batch.to_pandas().iloc[sub].reset_index(drop=True)
            f64 = d.select_dtypes(include=["float64"]).columns
            if len(f64) > 0:
                d[f64] = d[f64].astype("float32")
            chunks.append(d)
        row_offset = end

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=columns)
    return df, n_total


def load_all_parquets(
    parquet_dir: str | Path,
    eras: Iterable[str],
    groups: Iterable[str],
    columns_to_load: list[str],
    weight_col: str = "normalization_weight",
    kappa_map: dict[str, float] | None = None,
    max_events_per_file: int | None = 1_000_000,
    chunk_size: int = 200_000,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega todos os `{era}_{group}.parquet` e aplica a reescala física `w_phys = w * (n_total / n_lidos)`."""
    pdir = Path(parquet_dir)
    if not pdir.exists():
        raise FileNotFoundError(
            f"Diretório de parquets não encontrado: {pdir}. "
            "Verifique o caminho no EOS/local ou gere dados sintéticos com `scripts/generate_synthetic_data.py`."
        )

    kappa_map = kappa_map or {}
    frames: list[pd.DataFrame] = []
    file_rows: list[dict] = []

    for era in eras:
        for group in groups:
            path = pdir / f"{era}_{group}.parquet"
            if not path.exists():
                if verbose:
                    print(f"[AVISO] não encontrei {path.name} -- pulando")
                continue
            d, n_total = read_parquet_limited(
                path=path,
                columns=columns_to_load,
                max_events=max_events_per_file,
                chunk_size=chunk_size,
                seed=seed,
            )
            if len(d) == 0:
                continue
            factor = float(n_total) / float(len(d))
            d["w_phys"] = d[weight_col].to_numpy(np.float64) * factor
            d["era"] = str(era)
            d["parquet_group"] = str(group)
            if group in kappa_map:
                d["kappa_lambda"] = float(kappa_map[group])
            frames.append(d)
            file_rows.append(
                {
                    "arquivo": path.name,
                    "era": str(era),
                    "grupo": str(group),
                    "n_total": n_total,
                    "n_lidos": len(d),
                    "fator": factor,
                }
            )
            if verbose:
                print(f"  {path.name:32s} lidos {len(d):>9,d} de {n_total:>9,d}  (fator de peso {factor:.2f})")

    if not frames:
        raise RuntimeError(f"Nenhum arquivo parquet válido foi carregado de {pdir}.")

    file_info = pd.DataFrame(file_rows)
    df_all = pd.concat(frames, ignore_index=True)
    del frames

    # κ_λ não é definido fisicamente para eventos de fundo: marca como NaN para evitar vazamento acidental
    df_all.loc[df_all["label"] != 2, "kappa_lambda"] = np.nan
    for c in ("era", "parquet_group", "sample"):
        if c in df_all.columns:
            df_all[c] = df_all[c].astype("category")

    if verbose:
        mem_mb = df_all.memory_usage(deep=True).sum() / 1e6
        print(f"\nTotal carregado: {len(df_all):,d} eventos ({mem_mb:.0f} MB)")

    return df_all, file_info


def clean_and_check_data_quality(
    df: pd.DataFrame,
    continuous_features: list[str],
    categorical_features: list[str],
    sentinels: Iterable[float] = (-99999.0, -9999.0, -999.0),
    missing_masks: dict[str, str] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Executa as checagens e limpezas da Seção 4 (Inf -> NaN, sentinelas -> NaN, NaN em categóricas)."""
    cont = df[continuous_features]
    n_inf = int(np.isinf(cont.to_numpy(np.float32)).sum())
    if n_inf > 0:
        df[continuous_features] = cont.replace([np.inf, -np.inf], np.nan)

    sentinel_list = [float(s) for s in sentinels]
    sentinel_counts: dict[str, int] = {}
    for f in continuous_features:
        x = df[f].to_numpy(np.float32)
        mask_s = np.isin(x, sentinel_list)
        k = int(mask_s.sum())
        if k > 0:
            sentinel_counts[f] = k
            df.loc[mask_s, f] = np.nan

    n_nan_cat = int(df[categorical_features].isna().sum().sum())
    if n_nan_cat > 0:
        df = df.dropna(subset=categorical_features).reset_index(drop=True)

    nan_cont = df[continuous_features].isna().sum()
    nan_cont_nonzero = nan_cont[nan_cont > 0].to_dict()

    missing_obj_info: dict[str, dict] = {}
    if missing_masks:
        for flag, prefix in missing_masks.items():
            if flag in df.columns:
                m = df[flag].to_numpy() == 0
                cols = [c for c in continuous_features if c.startswith(prefix)]
                if m.any() and cols:
                    uniq_sample = np.unique(df.loc[m, cols[0]].dropna().to_numpy())[:5].tolist()
                    missing_obj_info[flag] = {
                        "absent_fraction": float(m.mean()),
                        "sample_col": cols[0],
                        "unique_values_head": uniq_sample,
                    }

    if verbose:
        print(f"Inf nas contínuas: {n_inf}")
        print("Sentinelas convertidas em NaN:", sentinel_counts if sentinel_counts else "nenhuma")
        print(f"NaN nas categóricas: {n_nan_cat}")
        print("NaN nas contínuas (antes de mascarar objetos ausentes):", nan_cont_nonzero if nan_cont_nonzero else "nenhum")
        for flag, info in missing_obj_info.items():
            print(
                f"  {flag}==0 ({info['absent_fraction']:.1%} dos eventos): "
                f"valores únicos em {info['sample_col']} = {info['unique_values_head']}"
            )

    report = {
        "n_inf": n_inf,
        "sentinel_counts": sentinel_counts,
        "n_nan_categorical_dropped": n_nan_cat,
        "nan_continuous_before_mask": nan_cont_nonzero,
        "missing_objects": missing_obj_info,
    }
    return df, report


def check_feature_leakage_names(
    continuous_features: list[str],
    categorical_features: list[str],
    weight_col: str = "normalization_weight",
    forbidden: Iterable[str] = ("weight", "label", "kappa", "sample", "era", "group", "process", "file"),
) -> list[str]:
    """Teste 13.1: garante que nenhuma coluna de metadado ou peso entrou na lista de features."""
    all_features = list(continuous_features) + list(categorical_features)
    forb_low = tuple(s.lower() for s in forbidden)
    bad = [f for f in all_features if any(s in f.lower() for s in forb_low)]
    if weight_col in all_features or "label" in all_features:
        bad.append(weight_col)
    return bad


def row_hashes(df: pd.DataFrame, feature_cols: list[str], chunk: int = 500_000) -> np.ndarray:
    """Calcula hash de 64 bits das features por linha em chunks para checagem de duplicatas (Seção 13.2)."""
    out = []
    for i in range(0, len(df), chunk):
        sub = df.iloc[i : i + chunk][feature_cols].astype(np.float32)
        out.append(pd.util.hash_pandas_object(sub, index=False).to_numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.uint64)


def check_split_duplicates(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """Teste 13.2: verifica duplicatas internas e interseção de eventos idênticos entre train/val/test."""
    hashes = {name: row_hashes(d, feature_cols) for name, d in splits.items()}
    rows = []

    for name, h in hashes.items():
        n_dup = int(len(h) - len(np.unique(h)))
        frac = n_dup / max(len(h), 1)
        rows.append({"comparacao": f"{name} (interno)", "n_eventos": n_dup, "fracao": frac, "status": "INFO"})
        if verbose:
            print(f"13.2 duplicatas dentro de {name:5s}: {n_dup:,d} ({frac:.2e})")

    names = list(hashes.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            n_ov = int(np.intersect1d(hashes[a], hashes[b]).size)
            frac = n_ov / max(len(hashes[b]), 1)
            ok = frac < 1e-4
            rows.append(
                {
                    "comparacao": f"{a} ∩ {b}",
                    "n_eventos": n_ov,
                    "fracao": frac,
                    "status": "OK" if ok else "ATENCAO",
                }
            )
            if verbose:
                print(f"13.2 eventos idênticos em {a} ∩ {b}: {n_ov:,d}  {'OK' if ok else 'ATENÇÃO'}")

    return pd.DataFrame(rows)
