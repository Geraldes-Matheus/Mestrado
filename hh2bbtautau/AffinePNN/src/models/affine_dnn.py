"""Construção unificada da AffinePNN, DNN da AN-25-103 (Single-NN) e PNN clássica (concat)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from tensorflow import keras

from src.data.preprocessing import FeatureScaler, get_lbn_column_indices
from src.models.affine_layers import (
    CUSTOM_OBJECTS,
    AffineConditioning,
    AsinhScale,
    FourVectorsFromFeatures,
    LorentzBoostNetwork,
    TakeColumns,
)
from src.models.embeddings import build_tabular_and_categorical_inputs


def build_model(
    continuous_features: list[str],
    cat_vocab_sizes: dict[str, int],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    kind: str = "affine",
    backbone: str = "an_densenet",
    n_classes: int = 3,
    lbn_particles: list[str] | None = None,
    lbn_n: int = 5,
    lbn_m: int = 10,
    use_asinh_lbn: bool = False,
    asinh_scale: float = 100.0,
    embed_dim: int = 5,
    hidden_units: Sequence[int] = (128,) * 8,
    dropout_rate: float = 0.0,
    use_layernorm: bool = False,
    l2_weight: float = 0.0,
    l2_bias: float = 0.0,
    l2_conditioning: float = 0.0,
) -> keras.Model:
    """Constrói o modelo classificador multiclasse {DY, ttbar, HH}:
    - `kind="affine"`: AffinePNN (Dense -> BN -> ELU -> AffineConditioning(κ_λ))
    - `kind="none"`  : DNN da AN-25-103 (Single-NN sem AffineConditioning e cega a κ_λ)
    - `kind="concat"`: PNN clássica de Baldi et al. (κ_λ concatenado apenas na entrada)
    """
    if kind not in ("affine", "concat", "none"):
        raise ValueError(f"kind inválido: {kind!r}. Esperado 'affine', 'concat' ou 'none'.")
    if backbone not in ("an_densenet", "mlp"):
        raise ValueError(f"backbone inválido: {backbone!r}. Esperado 'an_densenet' ou 'mlp'.")

    if lbn_particles is None:
        lbn_particles = ["vis_tau1", "vis_tau2", "bjet1", "bjet2", "fatjet"]

    x_cont_in, cat_inputs, cat_embeds, cat_by_name, kappa_in = build_tabular_and_categorical_inputs(
        n_continuous=len(continuous_features),
        cat_vocab_sizes=cat_vocab_sizes,
        embed_dim=embed_dim,
    )

    if backbone == "an_densenet":
        lbn_idx, other_idx = get_lbn_column_indices(continuous_features, lbn_particles)
        vecs = FourVectorsFromFeatures(
            lbn_idx,
            feature_mean[lbn_idx],
            feature_std[lbn_idx],
            lbn_n,
            name="four_vectors",
        )([x_cont_in, cat_by_name["has_fatjet"]])
        lbn = keras.layers.Flatten(name="lbn_flat")(
            LorentzBoostNetwork(lbn_n, lbn_m, name="lbn")(vecs)
        )
        if use_asinh_lbn:
            lbn = AsinhScale(asinh_scale, name="lbn_asinh")(lbn)
        lbn = keras.layers.BatchNormalization(name="lbn_bn")(lbn)
        other = TakeColumns(other_idx, name="other_features")(x_cont_in)

        parts = [lbn, other] + cat_embeds + ([kappa_in] if kind == "concat" else [])
        h0 = keras.layers.Concatenate(name="concat_heads")(parts)

        feats = [h0]
        for i, units in enumerate(hidden_units):
            inp = feats[0] if len(feats) == 1 else keras.layers.Concatenate(name=f"dense_in_{i}")(feats)
            h = keras.layers.Dense(
                int(units),
                activation=None,
                kernel_initializer="he_uniform",
                name=f"an_dense_{i}",
            )(inp)
            h = keras.layers.BatchNormalization(name=f"an_bn_{i}")(h)
            h = keras.layers.Activation("elu", name=f"an_elu_{i}")(h)
            if kind == "affine":
                h = AffineConditioning(
                    int(units),
                    l2_weight=l2_conditioning,
                    l2_bias=l2_bias,
                    name=f"affine_conditioning_{i}",
                )([h, kappa_in])
            if dropout_rate > 0:
                h = keras.layers.Dropout(float(dropout_rate), name=f"dropout_{i}")(h)
            feats.append(h)

        logits = keras.layers.Dense(
            int(n_classes),
            activation=None,
            kernel_initializer="he_uniform",
            name="an_logits",
        )(keras.layers.Concatenate(name="dense_in_out")(feats))

        if kind == "none":
            # Mantém assinatura de inputs idêntica à AffinePNN, mas com dL/d(kappa_lambda) == 0 exato
            zero_k = keras.layers.Dense(
                int(n_classes),
                use_bias=False,
                trainable=False,
                kernel_initializer="zeros",
                name="ignore_kappa",
            )(kappa_in)
            logits = logits + zero_k

        out = keras.layers.Activation("softmax", name="class_probs")(logits)
    else:
        parts = [x_cont_in] + cat_embeds + ([kappa_in] if kind == "concat" else [])
        h = keras.layers.Concatenate(name="concat_input")(parts)
        for i, units in enumerate(hidden_units):
            h = keras.layers.Dense(
                int(units),
                activation="relu",
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                kernel_regularizer=keras.regularizers.l2(l2_weight) if l2_weight else None,
                bias_regularizer=keras.regularizers.l2(l2_bias) if l2_bias else None,
                name=f"dense_{i}",
            )(h)
            if use_layernorm:
                h = keras.layers.LayerNormalization(center=False, scale=False, name=f"ln_{i}")(h)
            if kind == "affine":
                h = AffineConditioning(
                    int(units),
                    l2_weight=l2_conditioning,
                    l2_bias=l2_bias,
                    name=f"affine_conditioning_{i}",
                )([h, kappa_in])
            if dropout_rate > 0:
                h = keras.layers.Dropout(float(dropout_rate), name=f"dropout_{i}")(h)
        if kind == "none":
            logits = keras.layers.Dense(int(n_classes), activation=None, name="mlp_logits")(h)
            zero_k = keras.layers.Dense(
                int(n_classes),
                use_bias=False,
                trainable=False,
                kernel_initializer="zeros",
                name="ignore_kappa",
            )(kappa_in)
            out = keras.layers.Activation("softmax", name="class_probs")(logits + zero_k)
        else:
            out = keras.layers.Dense(int(n_classes), activation="softmax", name="class_probs")(h)

    return keras.Model([x_cont_in] + cat_inputs + [kappa_in], out, name=f"{kind}_{backbone}_PNN")


def build_model_from_config(
    cfg: dict[str, Any],
    model_key: str,
    scaler: FeatureScaler,
    cat_vocab_sizes: dict[str, int],
) -> keras.Model:
    """Constrói um modelo usando as chaves de `cfg['models'][model_key]` e `cfg['features']`."""
    mcfg = cfg["models"][model_key]
    fcfg = cfg["features"]
    return build_model(
        continuous_features=fcfg["continuous_features"],
        cat_vocab_sizes=cat_vocab_sizes,
        feature_mean=scaler.mean,
        feature_std=scaler.std,
        kind=mcfg["kind"],
        backbone=mcfg.get("backbone", "an_densenet"),
        n_classes=cfg["physics"]["n_classes"],
        lbn_particles=fcfg.get("lbn_particles", ["vis_tau1", "vis_tau2", "bjet1", "bjet2", "fatjet"]),
        lbn_n=mcfg.get("lbn_n", 5),
        lbn_m=mcfg.get("lbn_m", 10),
        use_asinh_lbn=mcfg.get("use_asinh_lbn", False),
        asinh_scale=mcfg.get("asinh_scale", 100.0),
        embed_dim=mcfg.get("embed_dim", 5),
        hidden_units=mcfg.get("hidden_units", [128] * 8),
        dropout_rate=mcfg.get("dropout_rate", 0.0),
        use_layernorm=mcfg.get("use_layernorm", False),
        l2_weight=mcfg.get("l2_weight", 0.0),
        l2_bias=mcfg.get("l2_bias", 0.0),
        l2_conditioning=mcfg.get("l2_conditioning", 0.0),
    )


def load_saved_model(path: str | Path, compile_model: bool = False) -> keras.Model:
    """Carrega um arquivo `.keras` garantindo registro das camadas customizadas `affinepnn`."""
    return keras.models.load_model(Path(path), custom_objects=CUSTOM_OBJECTS, compile=compile_model)


def extract_affine_modulation(
    model: keras.Model,
    kappa_values: Sequence[float],
    kappa_norm: float = 5.0,
) -> dict[str, dict[str, np.ndarray]]:
    """Extrai os vetores de escala `s(κ)` e deslocamento `b(κ)` aprendidos por cada camada `AffineConditioning`."""
    m_in = (np.asarray(kappa_values, dtype=np.float32) / float(kappa_norm)).reshape(-1, 1)
    out: dict[str, dict[str, np.ndarray]] = {}
    for layer in model.layers:
        if isinstance(layer, AffineConditioning):
            s = keras.ops.convert_to_numpy(layer.scale_layer(m_in)).astype(np.float32)
            b = keras.ops.convert_to_numpy(layer.bias_layer(m_in)).astype(np.float32)
            out[layer.name] = {"kappa": np.asarray(kappa_values, dtype=np.float32), "scale": s, "bias": b}
    return out
