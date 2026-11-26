"""Cálculo numericamente estável de Cross-Entropy (CE) a partir dos logits e loss estilo Fig. 50 da AN."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from tensorflow import keras

from src.data.preprocessing import make_inputs
from src.data.weights import oversampling_frequency_weights


def predict_logits(
    model: keras.Model,
    inputs: dict[str, np.ndarray],
    batch_size: int = 8192,
) -> np.ndarray:
    """Extrai os logits pré-softmax do modelo (mesmo para `Activation('softmax')` ou `Softmax`)."""
    out = model.output
    lg = getattr(out, "_keras_logits", None)

    if lg is None and len(model.layers) > 0:
        last = model.layers[-1]
        if isinstance(last, keras.layers.Softmax):
            lg = last.input
        elif isinstance(last, keras.layers.Activation) and getattr(last, "activation", None) == keras.activations.softmax:
            lg = last.input

    if lg is not None:
        sub_model = keras.Model(model.inputs, lg)
        return sub_model.predict(inputs, batch_size=batch_size, verbose=0).astype(np.float64)

    last = model.layers[-1]
    if isinstance(last, keras.layers.Dense):
        feat_model = keras.Model(model.inputs, last.input)
        f = feat_model.predict(inputs, batch_size=batch_size, verbose=0).astype(np.float64)
        W, b = [keras.ops.convert_to_numpy(v) for v in last.weights]
        return f @ W + b

    probs = model.predict(inputs, batch_size=batch_size, verbose=0)
    return np.log(np.clip(probs, 1e-30, 1.0)).astype(np.float64)


def ce_from_logits(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cross-entropy por evento via logsumexp sobre os logits pré-softmax."""
    y_int = np.asarray(y, dtype=np.int64)
    return logsumexp(logits, axis=1) - logits[np.arange(len(y_int)), y_int]


def weighted_logloss(
    model: keras.Model,
    inputs: dict[str, np.ndarray],
    y: np.ndarray,
    w: np.ndarray,
    batch_size: int = 8192,
) -> float:
    """Cross-entropy multiclasse ponderada em modo inferência."""
    p = model.predict(inputs, batch_size=batch_size, verbose=0)
    p_true = np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)
    w_arr = np.asarray(w, dtype=np.float64)
    return float(-(w_arr * np.log(p_true)).sum() / w_arr.sum())


def loss_tail_table(
    ce: np.ndarray,
    wv: np.ndarray,
    tag: str,
    w_clips: Sequence[float | None] = (None, 50, 20, 10),
) -> pd.Series:
    """Compara CE sem peso, com `val_freq_w` completa e com diferentes clips de peso máximo."""
    n = len(ce)
    r: dict[str, float] = {
        "sem_peso": float(ce.mean()),
        "val_freq_w (Keras: sum(w*l)/N)": float(np.sum(wv * ce) / n),
    }
    for c in w_clips:
        if c is None:
            continue
        wc = np.minimum(wv, float(c))
        wc = wc / wc.mean()
        r[f"clip{c:g}"] = float(np.sum(wc * ce) / n)
    r["CE max"] = float(ce.max())
    r["CE mediana"] = float(np.median(ce))
    r["CE p99.9"] = float(np.percentile(ce, 99.9))
    return pd.Series(r, name=tag)


def an_style_loss(
    models: dict[str, keras.Model],
    train_df: pd.DataFrame,
    x_cont_train: np.ndarray,
    x_cat_train: dict[str, np.ndarray],
    val_inputs: dict[str, np.ndarray],
    val_y: np.ndarray,
    val_freq_w: np.ndarray,
    gen_ref: object,
    kappa_norm: float = 5.0,
    class_fraction: dict[int, float] | None = None,
    val_freq_clip: float | None = None,
    n_train: int = 300_000,
    batch_size: int = 8192,
    seed: int = 42,
) -> pd.DataFrame:
    """Loss comparável estilo Fig. 50 da AN-25-103 (Seção 5.3.6):
    Treino e validação medidos SEM oversampling, com pesos = frequência do oversampling e BN em inferência."""
    rng = np.random.default_rng(seed)
    sub_n = min(int(n_train), len(train_df))
    idx = rng.choice(len(train_df), size=sub_n, replace=False)
    tr_sub = train_df.iloc[idx].reset_index(drop=True)
    tr_inputs = make_inputs(
        x_cont_train,
        x_cat_train,
        tr_sub["kappa_eval"].to_numpy(),
        kappa_norm=kappa_norm,
        idx=idx,
    )
    tr_y = tr_sub["label"].to_numpy()
    tr_w = oversampling_frequency_weights(
        tr_sub,
        gen_ref,
        class_fraction=class_fraction,
        clip=val_freq_clip,
    )

    rows = []
    for name, m in models.items():
        lt = weighted_logloss(m, tr_inputs, tr_y, tr_w, batch_size=batch_size)
        lv = weighted_logloss(m, val_inputs, val_y, val_freq_w, batch_size=batch_size)
        rows.append({"modelo": name, "treino": lt, "val": lv, "gap(val-treino)": lv - lt})
    return pd.DataFrame(rows)
