"""Split estratificado, pré-processamento físico, padronização (treino-only), embeddings e checagem de 4-vetores."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.parquet_dataset import (
    clean_and_check_data_quality,
    load_all_parquets,
    short_cat_name,
)


def add_eval_kappa(df: pd.DataFrame, kappa_grid: np.ndarray, seed: int) -> None:
    """Atribui `kappa_eval`: sinal mantém seu κ_λ real; fundo sorteia 1x de `kappa_grid` (fixo para avaliação)."""
    rng = np.random.default_rng(seed)
    k = df["kappa_lambda"].to_numpy(np.float32).copy()
    is_bkg = df["label"].to_numpy() != 2
    k[is_bkg] = rng.choice(np.asarray(kappa_grid, dtype=np.float32), size=int(is_bkg.sum()))
    df["kappa_eval"] = k


def stratified_split(
    df_all: pd.DataFrame,
    split_fractions: dict[str, float],
    kappa_grid: list[float] | np.ndarray,
    seed: int = 42,
    class_labels: dict[int, str] | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Divide `df_all` em `train`, `val` e `test` estratificado por `(parquet_group, era)`.

    Reescala `w_phys` por `1 / frac` em cada split para preservar os rendimentos na luminosidade total.
    """
    f_tr = float(split_fractions["train"])
    f_va = float(split_fractions["val"])
    f_te = float(split_fractions["test"])
    if abs(f_tr + f_va + f_te - 1.0) >= 1e-6:
        raise ValueError(f"As frações de split devem somar 1.0, recebido: {split_fractions}")

    strat = df_all["parquet_group"].astype(str) + "_" + df_all["era"].astype(str)
    idx_all = np.arange(len(df_all))
    idx_tr, idx_tmp = train_test_split(idx_all, test_size=1.0 - f_tr, random_state=seed, stratify=strat)
    idx_va, idx_te = train_test_split(
        idx_tmp,
        test_size=f_te / (f_va + f_te),
        random_state=seed,
        stratify=strat.iloc[idx_tmp],
    )

    def _take(idx: np.ndarray, frac: float) -> pd.DataFrame:
        d = df_all.iloc[np.sort(idx)].reset_index(drop=True)
        d["w_phys"] = d["w_phys"] / frac
        return d

    splits = {
        "train": _take(idx_tr, f_tr),
        "val": _take(idx_va, f_va),
        "test": _take(idx_te, f_te),
    }

    kgrid_arr = np.asarray(kappa_grid, dtype=np.float32)
    for i, (split_name, d) in enumerate(splits.items()):
        add_eval_kappa(d, kgrid_arr, seed + 100 + i)
        if verbose:
            cls_str = ""
            if class_labels:
                cls_str = " | " + " ".join(
                    f"{class_labels[c]}={int((d['label'] == c).sum()):,d}" for c in class_labels
                )
            print(f"{split_name:5s}: {len(d):>9,d} eventos{cls_str}")

    return splits


def apply_physical_preprocessing(
    df: pd.DataFrame,
    continuous_features: list[str],
    cov_cols: list[str],
    missing_masks: dict[str, str],
) -> pd.DataFrame:
    """Pré-processamento físico idempotente (Seção 5):
    1. Objetos ausentes preenchidos com 0 nos parquets (`has_jet_pair==0`, `has_fatjet==0`) viram NaN;
    2. Covariâncias da MET recebem compressão `sign(x) * log1p(|x|)`.
    """
    if df.attrs.get("preprocessed_cont", False):
        return df

    for flag, prefix in missing_masks.items():
        if flag in df.columns:
            cols = [c for c in continuous_features if c.startswith(prefix)]
            if cols:
                df.loc[df[flag] == 0, cols] = np.nan

    for c in cov_cols:
        if c in df.columns:
            x = df[c].to_numpy(np.float32)
            df[c] = np.sign(x) * np.log1p(np.abs(x))

    df.attrs["preprocessed_cont"] = True
    return df


class FeatureScaler:
    """Padronizador z-score ajustado EXCLUSIVAMENTE no conjunto de treino (NaN -> 0 após padronizar)."""

    def __init__(self, features: list[str], mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.features = list(features)
        self.mean = mean
        self.std = std

    def fit(self, train_df: pd.DataFrame) -> "FeatureScaler":
        mean = np.array([np.nanmean(train_df[f].to_numpy(np.float64)) for f in self.features], dtype=np.float32)
        std = np.array([np.nanstd(train_df[f].to_numpy(np.float64)) for f in self.features], dtype=np.float32)
        std[(~np.isfinite(std)) | (std == 0.0)] = 1.0
        mean[~np.isfinite(mean)] = 0.0
        self.mean = mean
        self.std = std
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureScaler precisa ser ajustado (`fit`) ou carregado antes de `transform`.")
        x = df[self.features].to_numpy(np.float32)
        x = (x - self.mean) / self.std
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"mean": self.mean, "std": self.std}, index=self.features)

    def save_csv(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(p)

    @classmethod
    def load_csv(cls, path: str | Path) -> "FeatureScaler":
        df = pd.read_csv(path, index_col=0)
        return cls(
            features=df.index.tolist(),
            mean=df["mean"].to_numpy(np.float32),
            std=df["std"].to_numpy(np.float32),
        )


class CategoricalEncoder:
    """Mapeia valores inteiros discretos para `[0, vocab_size - 1)`, reservando `vocab_size - 1` para desconhecidos."""

    def __init__(self, known_values: list[int] | None = None):
        self.known_values = known_values
        self.value_to_index: dict[int, int] = {}
        self.vocab_size: int = 0

    def fit(self, values: pd.Series | np.ndarray) -> "CategoricalEncoder":
        uniq = set(np.round(np.asarray(values, np.float64)).astype(np.int64).tolist())
        if self.known_values is not None:
            uniq |= {int(v) for v in self.known_values}
        ordered = sorted(uniq)
        self.value_to_index = {int(v): i for i, v in enumerate(ordered)}
        self.vocab_size = len(ordered) + 1
        return self

    def transform(self, values: pd.Series | np.ndarray) -> np.ndarray:
        v = np.round(np.asarray(values, np.float64)).astype(np.int64)
        return (
            pd.Series(v)
            .map(self.value_to_index)
            .fillna(self.vocab_size - 1)
            .astype(np.int32)
            .to_numpy()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "known_values": self.known_values,
            "value_to_index": {str(k): int(v) for k, v in self.value_to_index.items()},
            "vocab_size": int(self.vocab_size),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CategoricalEncoder":
        enc = cls(known_values=data.get("known_values"))
        enc.value_to_index = {int(k): int(v) for k, v in data["value_to_index"].items()}
        enc.vocab_size = int(data["vocab_size"])
        return enc


class CategoricalPreprocessor:
    """Agrupa os `CategoricalEncoder` das 7 variáveis categóricas da Tabela 29."""

    def __init__(self, categorical_features: list[str], domains: dict[str, list[int]] | None = None):
        self.categorical_features = list(categorical_features)
        self.domains = domains or {}
        self.encoders: dict[str, CategoricalEncoder] = {}
        self.vocab_sizes: dict[str, int] = {}

    def fit(self, train_df: pd.DataFrame) -> "CategoricalPreprocessor":
        self.encoders = {
            c: CategoricalEncoder(self.domains.get(c)).fit(train_df[c])
            for c in self.categorical_features
        }
        self.vocab_sizes = {
            short_cat_name(c): self.encoders[c].vocab_size
            for c in self.categorical_features
        }
        return self

    def transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            short_cat_name(c): self.encoders[c].transform(df[c]).reshape(-1, 1)
            for c in self.categorical_features
        }

    def unknown_fraction(self, encoded: dict[str, np.ndarray]) -> dict[str, float]:
        return {
            name: float((arr.ravel() == self.vocab_sizes[name] - 1).mean())
            for name, arr in encoded.items()
        }

    def save_json(
        self,
        path: str | Path,
        continuous_features: list[str] | None = None,
        kappa_norm: float | None = None,
        kappa_grid: list[float] | None = None,
    ) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "continuous_features": continuous_features,
            "categorical_features": self.categorical_features,
            "cat_vocab_sizes": self.vocab_sizes,
            "kappa_norm": kappa_norm,
            "kappa_grid": kappa_grid,
            "encoders": {c: self.encoders[c].to_dict() for c in self.categorical_features},
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "CategoricalPreprocessor":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cat_feats = data["categorical_features"]
        obj = cls(categorical_features=cat_feats)
        for c in cat_feats:
            raw_enc = data["encoders"][c]
            if "value_to_index" in raw_enc:
                obj.encoders[c] = CategoricalEncoder.from_dict(raw_enc)
            else:
                # Compatibilidade retroativa com o formato config_encoders.json do notebook v3_6
                enc = CategoricalEncoder()
                enc.value_to_index = {int(k): int(v) for k, v in raw_enc.items()}
                enc.vocab_size = int(data["cat_vocab_sizes"][short_cat_name(c)])
                obj.encoders[c] = enc
        obj.vocab_sizes = {short_cat_name(c): obj.encoders[c].vocab_size for c in cat_feats}
        return obj


def make_inputs(
    x_cont: np.ndarray,
    x_cat: dict[str, np.ndarray],
    kappa_phys: float | np.ndarray,
    kappa_norm: float = 5.0,
    idx: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Monta o dicionário de tensores de entrada esperado pela rede Keras."""
    if idx is not None:
        x_cont = x_cont[idx]
        x_cat = {name: arr[idx] for name, arr in x_cat.items()}
    n = len(x_cont)
    k = np.broadcast_to(np.asarray(kappa_phys, dtype=np.float32), (n,)) / float(kappa_norm)
    d: dict[str, np.ndarray] = {"x_continuous": x_cont}
    for name, arr in x_cat.items():
        d[f"cat_{name}"] = arr
    d["kappa_lambda"] = np.ascontiguousarray(k.reshape(-1, 1), dtype=np.float32)
    return d


def get_lbn_column_indices(
    continuous_features: list[str],
    lbn_particles: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna `(lbn_idx, other_idx)` separando os 4-vetores do LBN das demais features contínuas."""
    lbn_cols = [f"reg_dnn_moe_{p}_{c}" for p in lbn_particles for c in ("e", "px", "py", "pz")]
    lbn_idx = np.array([continuous_features.index(c) for c in lbn_cols], dtype=np.int32)
    lbn_set = set(lbn_idx.tolist())
    other_idx = np.array([i for i in range(len(continuous_features)) if i not in lbn_set], dtype=np.int32)
    return lbn_idx, other_idx


def check_four_vectors(
    x_cont: np.ndarray,
    x_cat: dict[str, np.ndarray],
    scaler: FeatureScaler,
    continuous_features: list[str],
    lbn_particles: list[str],
    n: int = 200_000,
    strict: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Confere se os 4-vetores recuperados pós-despadronização satisfazem E^2 >= |p|^2 - 1 GeV^2."""
    lbn_idx, _ = get_lbn_column_indices(continuous_features, lbn_particles)
    n_sub = min(int(n), len(x_cont))
    phys = (x_cont[:n_sub][:, lbn_idx] * scaler.std[lbn_idx] + scaler.mean[lbn_idx]).reshape(
        -1, len(lbn_particles), 4
    )
    has_fat = x_cat["has_fatjet"][:n_sub, 0] == 1
    m2 = phys[..., 0] ** 2 - (phys[..., 1:] ** 2).sum(axis=-1)

    rows = []
    for k, name in enumerate(lbn_particles):
        ok = m2[:, k] >= -1.0
        if name == "fatjet":
            ok = ok[has_fat]
        frac = float(ok.mean()) if len(ok) > 0 else 1.0
        rows.append({"particula": name, "frac_fisica": frac, "n_avaliados": int(len(ok))})
        if verbose:
            print(f"  {name:9s}: {frac:.4f} dos eventos com E² >= |p|² - 1 GeV²")
        if strict and frac <= 0.99:
            raise AssertionError(
                f"{name}: apenas {frac:.4f} com E² >= |p|² - 1 GeV² (verifique ordem das colunas e FEATURE_MEAN/STD)."
            )
    return pd.DataFrame(rows)


@dataclass
class PreparedData:
    """Estrutura contendo splits, matrizes normalizadas, encoders e metadados dos arquivos."""

    splits: dict[str, pd.DataFrame]
    x_cont: dict[str, np.ndarray]
    x_cat: dict[str, dict[str, np.ndarray]]
    scaler: FeatureScaler
    cat_preprocessor: CategoricalPreprocessor
    file_info: pd.DataFrame
    quality_report: dict[str, Any]


def prepare_datasets_from_config(
    cfg: dict[str, Any],
    existing_checkpoints_dir: str | Path | None = None,
    save_checkpoints_dir: str | Path | None = None,
    verbose: bool = True,
) -> PreparedData:
    """Executa todo o pipeline de dados (Seções 3–5) a partir do dicionário de configuração."""
    feats_cfg = cfg["features"]
    cont_cols = feats_cfg["continuous_features"]
    cat_cols = feats_cfg["categorical_features"]

    df_all, file_info = load_all_parquets(
        parquet_dir=cfg["paths"]["parquet_dir"],
        eras=cfg["data"]["eras"],
        groups=cfg["data"]["groups"],
        columns_to_load=feats_cfg["columns_to_load"],
        weight_col=feats_cfg["weight_col"],
        kappa_map=cfg["physics"]["kappa_map"],
        max_events_per_file=cfg["data"]["max_events_per_file"],
        chunk_size=cfg["data"]["chunk_size"],
        seed=cfg["seed"],
        verbose=verbose,
    )

    df_all, quality_report = clean_and_check_data_quality(
        df=df_all,
        continuous_features=cont_cols,
        categorical_features=cat_cols,
        sentinels=feats_cfg.get("sentinels", [-99999.0, -9999.0, -999.0]),
        missing_masks=feats_cfg.get("missing_object_masks"),
        verbose=verbose,
    )

    splits = stratified_split(
        df_all=df_all,
        split_fractions=cfg["data"]["split_fractions"],
        kappa_grid=cfg["physics"]["kappa_grid"],
        seed=cfg["seed"],
        class_labels=cfg["physics"]["class_labels"],
        verbose=verbose,
    )
    del df_all

    for d in splits.values():
        apply_physical_preprocessing(
            df=d,
            continuous_features=cont_cols,
            cov_cols=feats_cfg.get("cov_cols", []),
            missing_masks=feats_cfg.get("missing_object_masks", {}),
        )

    if existing_checkpoints_dir is not None:
        ckpt = Path(existing_checkpoints_dir)
        scaler = FeatureScaler.load_csv(ckpt / "normalization_train_only.csv")
        cat_prep = CategoricalPreprocessor.load_json(ckpt / "config_encoders.json")
    else:
        scaler = FeatureScaler(cont_cols).fit(splits["train"])
        cat_prep = CategoricalPreprocessor(
            categorical_features=cat_cols,
            domains=feats_cfg.get("categorical_domains", {}),
        ).fit(splits["train"])

    if save_checkpoints_dir is not None:
        sdir = Path(save_checkpoints_dir)
        scaler.save_csv(sdir / "normalization_train_only.csv")
        cat_prep.save_json(
            sdir / "config_encoders.json",
            continuous_features=cont_cols,
            kappa_norm=cfg["physics"]["kappa_norm"],
            kappa_grid=cfg["physics"]["kappa_grid"],
        )

    x_cont = {k: scaler.transform(d) for k, d in splits.items()}
    x_cat = {k: cat_prep.transform(d) for k, d in splits.items()}

    if verbose:
        zmax = np.abs(x_cont["train"]).max(axis=0)
        bad = pd.Series(zmax, index=cont_cols)
        high_z = bad[bad > 50].sort_values(ascending=False)
        print("Features com max|z| > 50 no treino:", high_z.to_dict() if len(high_z) else "nenhuma")
        unk = cat_prep.unknown_fraction(x_cat["test"])
        print("Vocabulários categóricos:", cat_prep.vocab_sizes)
        print("Fração no bucket 'desconhecido' (teste):", {k: f"{v:.2%}" for k, v in unk.items()})

    return PreparedData(
        splits=splits,
        x_cont=x_cont,
        x_cat=x_cat,
        scaler=scaler,
        cat_preprocessor=cat_prep,
        file_info=file_info,
        quality_report=quality_report,
    )
