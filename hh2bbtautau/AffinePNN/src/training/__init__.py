"""Módulos de treinamento, callbacks, cálculo de cross-entropy em logits e diagnósticos pós-treino."""

from src.training.callbacks import EpochProbe, LogLRCallback
from src.training.diagnostics import (
    bn_layers,
    median_bn,
    random_label_test,
    restore_bn,
    run_validation_diagnostics,
    snapshot_bn,
)
from src.training.losses import (
    an_style_loss,
    ce_from_logits,
    loss_tail_table,
    predict_logits,
    weighted_logloss,
)
from src.training.train import (
    build_optimizer_and_callbacks,
    create_generator_for_model,
    load_model_bundle,
    save_model_bundle,
    train_model,
)

__all__ = [
    "EpochProbe",
    "LogLRCallback",
    "an_style_loss",
    "bn_layers",
    "build_optimizer_and_callbacks",
    "ce_from_logits",
    "create_generator_for_model",
    "load_model_bundle",
    "loss_tail_table",
    "median_bn",
    "predict_logits",
    "random_label_test",
    "restore_bn",
    "run_validation_diagnostics",
    "save_model_bundle",
    "snapshot_bn",
    "train_model",
    "weighted_logloss",
]
