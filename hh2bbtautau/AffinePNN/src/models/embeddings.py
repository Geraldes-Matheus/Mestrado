"""Inputs simbólicos Keras e camadas de Entity Embedding para features categóricas (AN-25-103 Seção 5.3.3)."""

from __future__ import annotations

from tensorflow import keras


def build_tabular_and_categorical_inputs(
    n_continuous: int,
    cat_vocab_sizes: dict[str, int],
    embed_dim: int = 5,
):
    """Cria os inputs de `x_continuous`, das variáveis categóricas com Embedding(dim=5) e de `kappa_lambda`."""
    x_cont_in = keras.Input(shape=(int(n_continuous),), name="x_continuous", dtype="float32")
    cat_inputs = []
    cat_embeds = []
    cat_by_name = {}

    for name, vocab in cat_vocab_sizes.items():
        inp = keras.Input(shape=(1,), name=f"cat_{name}", dtype="int32")
        emb = keras.layers.Embedding(int(vocab), int(embed_dim), name=f"emb_{name}")(inp)
        flat = keras.layers.Flatten(name=f"flat_{name}")(emb)
        cat_inputs.append(inp)
        cat_embeds.append(flat)
        cat_by_name[name] = inp

    kappa_in = keras.Input(shape=(1,), name="kappa_lambda", dtype="float32")
    return x_cont_in, cat_inputs, cat_embeds, cat_by_name, kappa_in
