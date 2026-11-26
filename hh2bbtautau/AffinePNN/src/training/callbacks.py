"""Callbacks de treinamento: registro do Learning Rate e sonda diagnóstica EpochProbe."""

from __future__ import annotations

import numpy as np
from tensorflow import keras


class LogLRCallback(keras.callbacks.Callback):
    """Registra o learning_rate efetivo no histórico ao fim de cada época."""

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs is not None and "learning_rate" not in logs:
            lr = self.model.optimizer.learning_rate
            if isinstance(lr, keras.optimizers.schedules.LearningRateSchedule):
                lr = lr(self.model.optimizer.iterations)
            logs["learning_rate"] = float(keras.ops.convert_to_numpy(lr))


class EpochProbe(keras.callbacks.Callback):
    """Sonda opcional época a época para investigar picos de val_loss (BN móvel vs BN mediana e pior pool)."""

    def __init__(
        self,
        inputs: dict[str, np.ndarray],
        y: np.ndarray,
        w: np.ndarray,
        val_pool: np.ndarray,
        gen: object,
        n: int = 200_000,
        batch_size: int = 8192,
        seed: int = 0,
    ):
        super().__init__()
        rng = np.random.default_rng(seed)
        sub_n = min(int(n), len(y))
        self.i = np.sort(rng.choice(len(y), size=sub_n, replace=False))
        self.x = {k: v[self.i] for k, v in inputs.items()}
        self.y = np.asarray(y)[self.i].astype(int)
        self.w = np.asarray(w, dtype=np.float64)[self.i]
        self.p = np.asarray(val_pool)[self.i]
        self.gen = gen
        self.batch_size = int(batch_size)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        from src.training.diagnostics import median_bn, restore_bn, snapshot_bn
        from src.training.losses import ce_from_logits, predict_logits

        m = self.model
        n = len(self.y)
        ce = ce_from_logits(predict_logits(m, self.x, batch_size=self.batch_size), self.y)
        snap = snapshot_bn(m)
        try:
            median_bn(m, lambda: self.gen[0][0], n_batches=100)
            ce2 = ce_from_logits(predict_logits(m, self.x, batch_size=self.batch_size), self.y)
        finally:
            restore_bn(m, snap)
        j = int(np.argmax(self.w * ce))
        print(
            f"\n[EpochProbe ép.{epoch + 1}] pond(BN móvel)={np.sum(self.w * ce) / n:.4f} | "
            f"pond(BN mediana)={np.sum(self.w * ce2) / n:.4f} | sem_peso={ce.mean():.4f} | "
            f"CEmax={ce.max():.1f} | pior evento: {self.p[j]} w={self.w[j]:.1f} CE={ce[j]:.1f}"
        )
