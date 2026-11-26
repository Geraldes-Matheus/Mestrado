"""Controle determinístico de seeds (Python, NumPy, TensorFlow/Keras) e GPU."""

from __future__ import annotations

import os
import random


def set_all_seeds(seed: int = 42) -> None:
    """Fixa seeds de Python, NumPy e TensorFlow/Keras de uma só vez."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    try:
        from tensorflow import keras

        keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def configure_gpu_memory_growth() -> list[str]:
    """Ativa memory_growth nas GPUs disponíveis para evitar alocação total imediata da VRAM."""
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # Já inicializado anteriormente na sessão
                pass
        return [gpu.name for gpu in gpus]
    except ImportError:
        return []
