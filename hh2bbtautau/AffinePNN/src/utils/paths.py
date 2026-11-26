"""Gerenciamento centralizado dos diretórios do projeto e de artefatos de saída."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class OutputPaths:
    """Subpastas padronizadas dentro de `output_dir`."""

    root: Path
    checkpoints: Path
    exported_models: Path
    logs: Path
    metrics: Path
    plots: Path
    predictions: Path

    @classmethod
    def from_root(cls, output_dir: str | Path, create: bool = True) -> "OutputPaths":
        root = Path(output_dir)
        if not root.is_absolute():
            root = (PROJECT_ROOT / root).resolve()
        obj = cls(
            root=root,
            checkpoints=root / "checkpoints",
            exported_models=root / "exported_models",
            logs=root / "logs",
            metrics=root / "metrics",
            plots=root / "plots",
            predictions=root / "predictions",
        )
        if create:
            obj.ensure_dirs()
        return obj

    def ensure_dirs(self) -> None:
        for p in (
            self.root,
            self.checkpoints,
            self.exported_models,
            self.logs,
            self.metrics,
            self.plots,
            self.predictions,
        ):
            p.mkdir(parents=True, exist_ok=True)
