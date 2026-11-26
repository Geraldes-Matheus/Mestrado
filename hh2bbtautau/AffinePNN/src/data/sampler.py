"""Gerador de mini-batches com quotas multinomiais (AN-25-103 Seção 5.3.5) e diagnóstico de pools."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tensorflow import keras

from src.data.preprocessing import make_inputs
from src.data.weights import aggregate_weight


def _build_multinomial_probs(weight_by_key: dict[str, float]) -> tuple[list[str], np.ndarray]:
    keys = [k for k, w in weight_by_key.items() if w > 0]
    if keys:
        weights = np.array([weight_by_key[k] for k in keys], dtype=np.float64)
    else:
        keys = list(weight_by_key)
        weights = np.ones(len(keys), dtype=np.float64)
    probs = weights / weights.sum()
    return keys, probs


def _guarantee_min_one(counts: np.ndarray) -> np.ndarray:
    """AN 5.3.5: garante >= 1 evento para subprocessos pequenos retirando do maior subprocesso."""
    counts = np.array(counts, copy=True)
    deficit = int((counts == 0).sum())
    counts[counts == 0] = 1
    for _ in range(deficit):
        counts[int(np.argmax(counts))] -= 1
    return counts


def _group_positions(keys: np.ndarray) -> dict[str, np.ndarray]:
    s = pd.Series(keys)
    return s.groupby(s).indices


class QuotaBatchGenerator(keras.utils.Sequence):
    """Amostragem balanceada por classe e proporcional à seção de choque líquida por (subprocesso | era).

    Não utiliza `sample_weight` na loss (evitando instabilidades por pesos negativos de NLO),
    conforme AN-25-103 Seção 5.3.5, e sorteia `kappa_lambda` do grid para o fundo a cada batch
    ("identical, sampled", Anzalone et al., Seção 4.2).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        x_cont: np.ndarray,
        x_cat: dict[str, np.ndarray],
        rows: np.ndarray | None = None,
        batch_size: int = 2048,
        steps_per_epoch: int = 5000,
        class_fraction: dict[int, float] | None = None,
        quota_mode: str = "signed",
        min_one: bool = False,
        kappa_grid: list[float] | np.ndarray = (0.0, 1.0, 2.45, 5.0),
        kappa_norm: float = 5.0,
        bkg_kappa_distribution: str = "identical_sampled",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        n = len(df)
        if len(x_cont) != n or any(len(a) != n for a in x_cat.values()):
            raise ValueError("`df`, `x_cont` e `x_cat` estão fora de sincronia em número de linhas.")

        if class_fraction is None:
            class_fraction = {0: 1.0 / 3.0, 1: 1.0 / 3.0, 2: 1.0 / 3.0}

        self.x_cont = x_cont
        self.x_cat = x_cat
        self.cat_names = list(x_cat)
        self.y = df["label"].to_numpy().astype(np.int32)
        self.kappa_grid = np.asarray(kappa_grid, dtype=np.float32)
        self.kappa_norm = float(kappa_norm)
        self.bkg_kappa_distribution = bkg_kappa_distribution
        self.batch_size = int(batch_size)
        self.steps = int(steps_per_epoch)
        self.rng = np.random.default_rng(seed)
        self.min_one = bool(min_one)
        self.last_idx: np.ndarray | None = None
        self.fixed_bkg_kappa = (
            df["kappa_eval"].to_numpy(np.float32) if "kappa_eval" in df.columns else None
        )

        keep = np.zeros(n, dtype=bool)
        keep[np.arange(n) if rows is None else rows] = True

        w = df["w_phys"].to_numpy(np.float64)
        sample = df["sample"].astype(str).to_numpy()
        era = df["era"].astype(str).to_numpy()
        kl = df["kappa_lambda"].to_numpy(np.float64)

        self.n_class = {
            0: int(round(self.batch_size * class_fraction[0])),
            1: int(round(self.batch_size * class_fraction[1])),
        }
        self.n_class[2] = self.batch_size - self.n_class[0] - self.n_class[1]

        # Fundo: pools e probabilidades multinomiais por (subprocesso | era)
        self.bkg_pools: dict[int, dict[str, np.ndarray]] = {}
        self.bkg_weights: dict[int, dict[str, float]] = {}
        self.bkg_keys: dict[int, list[str]] = {}
        self.bkg_probs: dict[int, np.ndarray] = {}
        for label in (0, 1):
            idx_m = np.where(keep & (self.y == label))[0]
            pools: dict[str, np.ndarray] = {}
            weights: dict[str, float] = {}
            for key, pos in _group_positions(np.char.add(np.char.add(sample[idx_m], "|"), era[idx_m])).items():
                pools[key] = idx_m[pos]
                weights[key] = aggregate_weight(w[idx_m[pos]], quota_mode)
            self.bkg_pools[label], self.bkg_weights[label] = pools, weights
            self.bkg_keys[label], self.bkg_probs[label] = _build_multinomial_probs(weights)

        # Sinal: pools por hipótese de κ_λ e, dentro de cada κ_λ, multinomial por era
        self.hh_pools: dict[float, dict[str, np.ndarray]] = {}
        self.hh_weights: dict[float, dict[str, float]] = {}
        self.hh_keys: dict[float, list[str]] = {}
        self.hh_probs: dict[float, np.ndarray] = {}
        for kappa in self.kappa_grid:
            idx_m = np.where(keep & (self.y == 2) & np.isclose(kl, float(kappa)))[0]
            if len(idx_m) == 0:
                continue
            pools, weights = {}, {}
            for key, pos in _group_positions(era[idx_m]).items():
                pools[key] = idx_m[pos]
                weights[key] = aggregate_weight(w[idx_m[pos]], quota_mode)
            fk = float(kappa)
            self.hh_pools[fk], self.hh_weights[fk] = pools, weights
            self.hh_keys[fk], self.hh_probs[fk] = _build_multinomial_probs(weights)

        if not self.hh_pools:
            raise RuntimeError("Nenhum evento de sinal HH disponível para popular o QuotaBatchGenerator.")

    def __len__(self) -> int:
        return self.steps

    def _draw_multinomial(
        self,
        n: int,
        pools: dict[str, np.ndarray],
        keys: list[str],
        probs: np.ndarray,
    ) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        counts = self.rng.multinomial(n, probs)
        if self.min_one and n >= len(keys):
            counts = _guarantee_min_one(counts)
        parts = [
            self.rng.choice(pools[k], size=int(q), replace=True)
            for k, q in zip(keys, counts)
            if q > 0
        ]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def _sample_bkg_kappa(self, idx: np.ndarray) -> np.ndarray:
        mode = self.bkg_kappa_distribution
        if mode == "identical_sampled":
            return self.rng.choice(self.kappa_grid, size=len(idx)).astype(np.float32)
        if mode == "identical_fixed" and self.fixed_bkg_kappa is not None:
            return self.fixed_bkg_kappa[idx].astype(np.float32)
        if mode == "uniform_sampled":
            return self.rng.uniform(float(self.kappa_grid.min()), float(self.kappa_grid.max()), size=len(idx)).astype(np.float32)
        raise ValueError(f"bkg_kappa_distribution desconhecida: {mode!r}")

    def _sample_background(self, label: int) -> tuple[np.ndarray, np.ndarray]:
        idx = self._draw_multinomial(
            self.n_class[label],
            self.bkg_pools[label],
            self.bkg_keys[label],
            self.bkg_probs[label],
        )
        return idx, self._sample_bkg_kappa(idx)

    def _sample_signal(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        kappas = list(self.hh_pools)
        base, rema = divmod(n, len(kappas))
        extra = set(self.rng.choice(len(kappas), size=rema, replace=False).tolist()) if rema else set()
        idx_parts, kl_parts = [], []
        for i, k in enumerate(kappas):
            q = base + (1 if i in extra else 0)
            idx = self._draw_multinomial(q, self.hh_pools[k], self.hh_keys[k], self.hh_probs[k])
            idx_parts.append(idx)
            kl_parts.append(np.full(len(idx), k, dtype=np.float32))
        return np.concatenate(idx_parts), np.concatenate(kl_parts)

    def __getitem__(self, index: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
        idx_dy, kl_dy = self._sample_background(0)
        idx_tt, kl_tt = self._sample_background(1)
        idx_hh, kl_hh = self._sample_signal(self.n_class[2])

        idx = np.concatenate([idx_dy, idx_tt, idx_hh])
        kl = np.concatenate([kl_dy, kl_tt, kl_hh])
        perm = self.rng.permutation(len(idx))
        idx, kl = idx[perm], kl[perm]
        self.last_idx = idx
        return make_inputs(self.x_cont, self.x_cat, kl, kappa_norm=self.kappa_norm, idx=idx), self.y[idx]


def verify_generator_quotas(
    gen: QuotaBatchGenerator,
    df: pd.DataFrame,
    n_batches: int = 300,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Compara a fração empírica sorteada nos batches com a fração física (soma líquida de `w_phys`)."""
    drawn = np.concatenate([(gen[i], gen.last_idx)[1] for i in range(int(n_batches))])
    lab = df["label"].to_numpy()
    samp = df["sample"].astype(str).to_numpy()
    w_ph = df["w_phys"].to_numpy(np.float64)

    out: dict[str, pd.DataFrame] = {}
    for label, name in ((0, "DY"), (1, "ttbar")):
        d = drawn[lab[drawn] == label]
        emp = pd.Series(samp[d]).value_counts(normalize=True)
        m = lab == label
        phys = pd.Series(w_ph[m]).groupby(samp[m]).sum()
        phys_pos = phys.clip(lower=0.0)
        phys_norm = phys_pos / max(float(phys_pos.sum()), 1e-30)
        t = pd.DataFrame({"empirica(batches)": emp, "fisica(sum_w)": phys_norm}).fillna(0.0)
        t["diferenca"] = t["empirica(batches)"] - t["fisica(sum_w)"]
        t = t.sort_values("fisica(sum_w)", ascending=False)
        out[name] = t
        if verbose:
            print(f"\n{name}: fração por subprocesso ({n_batches} batches)")
            print(t.to_string(float_format=lambda x: f"{x:.4f}"))
    return out


def pool_report(
    gen: QuotaBatchGenerator,
    steps: int = 5000,
    epochs: int = 40,
    top: int = 12,
    verbose: bool = True,
) -> pd.DataFrame:
    """Estima quantas vezes cada evento de cada pool `(subprocesso | era)` será sorteado no treino inteiro."""
    rows = []
    for label, name in ((0, "DY"), (1, "ttbar")):
        for key, p in zip(gen.bkg_keys[label], gen.bkg_probs[label]):
            n_pool = len(gen.bkg_pools[label][key])
            draws = float(p) * gen.n_class[label] * steps * epochs
            if gen.min_one:
                draws = max(draws, float(steps * epochs))
            rows.append(
                {
                    "classe": name,
                    "pool": key,
                    "N": n_pool,
                    "frac_da_classe": float(p),
                    "sorteios_total": draws,
                    "repeticoes/evento": draws / max(n_pool, 1),
                }
            )

    kappas = list(gen.hh_pools)
    for k in kappas:
        for key, p in zip(gen.hh_keys[k], gen.hh_probs[k]):
            n_pool = len(gen.hh_pools[k][key])
            draws = float(p) * (gen.n_class[2] / len(kappas)) * steps * epochs
            if gen.min_one:
                draws = max(draws, float(steps * epochs))
            rows.append(
                {
                    "classe": "HH",
                    "pool": f"κ={k}|{key}",
                    "N": n_pool,
                    "frac_da_classe": float(p) / len(kappas),
                    "sorteios_total": draws,
                    "repeticoes/evento": draws / max(n_pool, 1),
                }
            )

    t = pd.DataFrame(rows).sort_values("repeticoes/evento", ascending=False).reset_index(drop=True)
    if verbose:
        print(f"Pools com mais repetições por evento (treino de {epochs} épocas x {steps} steps):")
        print(t.head(top).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        n_bad = int((t["repeticoes/evento"] > 50).sum())
        msg = "  ATENÇÃO: risco de memorização." if n_bad else "  OK."
        print(f"\n{n_bad} pool(s) com > 50 repetições/evento.{msg}")
    return t


def make_random_label_generator(
    train_df: pd.DataFrame,
    x_cont_train: np.ndarray,
    x_cat_train: dict[str, np.ndarray],
    kappa_grid: list[float] | np.ndarray,
    kappa_norm: float = 5.0,
    batch_size: int = 2048,
    steps_per_epoch: int = 5000,
    seed: int = 42,
) -> QuotaBatchGenerator:
    """Cria gerador com labels embaralhados e κ_λ independente do label para o teste 13.4."""
    rng = np.random.default_rng(seed)
    d = train_df[["label", "sample", "era", "w_phys"]].copy()
    d["label"] = rng.permutation(d["label"].to_numpy())
    d["sample"] = "all"
    d["kappa_lambda"] = rng.choice(np.asarray(kappa_grid, dtype=np.float64), size=len(d))
    return QuotaBatchGenerator(
        df=d,
        x_cont=x_cont_train,
        x_cat=x_cat_train,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        kappa_grid=kappa_grid,
        kappa_norm=kappa_norm,
        seed=seed,
    )
