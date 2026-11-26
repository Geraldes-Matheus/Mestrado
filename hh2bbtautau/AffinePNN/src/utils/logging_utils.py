"""Configuração de logging compatível com terminal interativo, Jupyter e HTCondor."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def is_interactive_session() -> bool:
    """Retorna True em Jupyter/IPython ou terminal TTY; False em batch (HTCondor)."""
    if "ipykernel" in sys.modules or "IPython" in sys.modules:
        try:
            from IPython import get_ipython

            if get_ipython() is not None:
                return True
        except Exception:
            pass
    return bool(hasattr(sys.stdout, "isatty") and sys.stdout.isatty())


def resolve_keras_verbose(verbose_cfg: int | str = "auto") -> int:
    """No HTCondor (sem TTY), retorna verbose=2 (1 linha por época, sem barras ANSI)."""
    if isinstance(verbose_cfg, int):
        return verbose_cfg
    if str(verbose_cfg).lower() == "auto":
        return 1 if is_interactive_session() else 2
    return int(verbose_cfg)


def get_logger(name: str = "affinepnn", log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Cria ou recupera logger formatado sem duplicar handlers em notebooks."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="[%(asctime)s | %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers)
    if not has_stream:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    if log_file is not None:
        log_path = Path(log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        has_file = any(isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == log_path for h in logger.handlers)
        if not has_file:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
