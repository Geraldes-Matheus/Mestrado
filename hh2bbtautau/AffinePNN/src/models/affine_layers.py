"""Camadas Keras customizadas serializáveis: AffineConditioning (Anzalone et al.) e LBN (AN-25-103)."""

from __future__ import annotations

from tensorflow import keras


@keras.utils.register_keras_serializable(package="affinepnn")
class AffineConditioning(keras.layers.Layer):
    """Condicionamento afim por camada (Eq. 1 de Anzalone et al., arXiv:2202.00424):
    z = h * s(m) + b(m), onde s e b são projeções lineares do parâmetro físico m (κ_λ normalizado).
    """

    def __init__(self, units: int, l2_weight: float = 0.0, l2_bias: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.l2_weight = float(l2_weight)
        self.l2_bias = float(l2_bias)
        reg_w = keras.regularizers.l2(self.l2_weight) if self.l2_weight else None
        reg_b = keras.regularizers.l2(self.l2_bias) if self.l2_bias else None
        self.scale_layer = keras.layers.Dense(
            self.units,
            activation=None,
            kernel_regularizer=reg_w,
            bias_regularizer=reg_b,
            bias_initializer="ones",
            name="scale",
        )
        self.bias_layer = keras.layers.Dense(
            self.units,
            activation=None,
            kernel_regularizer=reg_w,
            bias_regularizer=reg_b,
            bias_initializer="zeros",
            name="bias",
        )

    def build(self, input_shape):
        _, m_shape = input_shape
        self.scale_layer.build(m_shape)
        self.bias_layer.build(m_shape)
        super().build(input_shape)

    def call(self, inputs):
        h, m = inputs
        return h * self.scale_layer(m) + self.bias_layer(m)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "units": self.units,
                "l2_weight": self.l2_weight,
                "l2_bias": self.l2_bias,
            }
        )
        return cfg


@keras.utils.register_keras_serializable(package="affinepnn")
class TakeColumns(keras.layers.Layer):
    """Seleciona um subconjunto fixo de colunas de um tensor 2D."""

    def __init__(self, idx, **kwargs):
        super().__init__(**kwargs)
        self.idx = [int(i) for i in idx]

    def call(self, x):
        return keras.ops.take(x, keras.ops.convert_to_tensor(self.idx, dtype="int32"), axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], len(self.idx))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"idx": self.idx})
        return cfg


@keras.utils.register_keras_serializable(package="affinepnn")
class FourVectorsFromFeatures(keras.layers.Layer):
    """Recupera os 4-vetores físicos (E, px, py, pz) desfazendo a padronização z-score
    e zerando o fatjet quando `has_fatjet != 1`."""

    def __init__(self, idx, mean, std, n_particles: int, fat_last: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.idx = [int(i) for i in idx]
        self.mean = [float(m) for m in mean]
        self.std = [float(s) for s in std]
        self.n_particles = int(n_particles)
        self.fat_last = bool(fat_last)

    def call(self, inputs):
        ops = keras.ops
        x, has_fatjet = inputs
        sel = ops.take(x, ops.convert_to_tensor(self.idx, dtype="int32"), axis=1)
        phys = sel * ops.convert_to_tensor(self.std, dtype="float32") + ops.convert_to_tensor(self.mean, dtype="float32")
        phys = ops.reshape(phys, (-1, self.n_particles, 4))
        present = ops.reshape(ops.cast(ops.equal(has_fatjet, 1), "float32"), (-1, 1, 1))
        is_fat = ops.convert_to_tensor([[[0.0]] * (self.n_particles - 1) + [[1.0]]], dtype="float32")
        return phys * (1.0 - is_fat * (1.0 - present))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.n_particles, 4)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "idx": self.idx,
                "mean": self.mean,
                "std": self.std,
                "n_particles": self.n_particles,
                "fat_last": self.fat_last,
            }
        )
        return cfg


@keras.utils.register_keras_serializable(package="affinepnn")
class LorentzBoostNetwork(keras.layers.Layer):
    """Camada LBN (Erdmann et al. / AN-25-103 Seção 5.3.3):
    Combina linearmente N 4-vetores em M partículas compostas e M referenciais de repouso,
    aplicando boosts de Lorentz explícitos (saída M x 4)."""

    def __init__(self, n_particles: int = 5, n_combined: int = 10, eps: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.n_particles = int(n_particles)
        self.n_combined = int(n_combined)
        self.eps = float(eps)

    def build(self, input_shape):
        init = keras.initializers.RandomUniform(minval=0.0, maxval=1.0)
        self.w_particles = self.add_weight(
            name="particle_weights",
            shape=(self.n_particles, self.n_combined),
            initializer=init,
            trainable=True,
        )
        self.w_restframes = self.add_weight(
            name="restframe_weights",
            shape=(self.n_particles, self.n_combined),
            initializer=init,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        ops = keras.ops
        P = ops.einsum("bnc,nm->bmc", x, ops.abs(self.w_particles))
        R = ops.einsum("bnc,nm->bmc", x, ops.abs(self.w_restframes))
        E_r, p_r = ops.split(R, [1], axis=-1)
        E_p, p_p = ops.split(P, [1], axis=-1)
        m2 = ops.maximum(ops.square(E_r) - ops.sum(ops.square(p_r), axis=-1, keepdims=True), self.eps)
        gamma = E_r / ops.sqrt(m2)
        beta = p_r / ops.maximum(E_r, self.eps)
        bp = ops.sum(beta * p_p, axis=-1, keepdims=True)
        E_new = gamma * (E_p - bp)
        p_new = p_p + (gamma * gamma / (gamma + 1.0) * bp - gamma * E_p) * beta
        return ops.concatenate([E_new, p_new], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_combined, 4)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "n_particles": self.n_particles,
                "n_combined": self.n_combined,
                "eps": self.eps,
            }
        )
        return cfg


@keras.utils.register_keras_serializable(package="affinepnn")
class AsinhScale(keras.layers.Layer):
    """Compressão suave arcsinh(x / scale) opcional antes da BatchNorm do LBN."""

    def __init__(self, scale: float = 100.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = float(scale)

    def call(self, x):
        return keras.ops.arcsinh(x / self.scale)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"scale": self.scale})
        return cfg


CUSTOM_OBJECTS = {
    "AffineConditioning": AffineConditioning,
    "TakeColumns": TakeColumns,
    "FourVectorsFromFeatures": FourVectorsFromFeatures,
    "LorentzBoostNetwork": LorentzBoostNetwork,
    "AsinhScale": AsinhScale,
    "affinepnn>AffineConditioning": AffineConditioning,
    "affinepnn>TakeColumns": TakeColumns,
    "affinepnn>FourVectorsFromFeatures": FourVectorsFromFeatures,
    "affinepnn>LorentzBoostNetwork": LorentzBoostNetwork,
    "affinepnn>AsinhScale": AsinhScale,
}
