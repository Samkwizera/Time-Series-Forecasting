"""Configuration loading shared by all scripts and notebooks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    figures_dir: Path
    tables_dir: Path
    experiments_dir: Path
    models_dir: Path

    def ensure(self) -> "Paths":
        for p in (self.interim_dir, self.processed_dir, self.figures_dir,
                  self.tables_dir, self.experiments_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self


class Config(dict):
    """Dictionary with attribute access and resolved paths.

    Relative paths in the YAML are resolved against the project root so scripts
    behave identically whether launched from the repo root, a notebook, or Colab.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    @property
    def paths(self) -> Paths:  # type: ignore[override]
        root = Path(self.get("_root", PROJECT_ROOT))
        raw = {k: root / v for k, v in self["paths"].items()}
        return Paths(**raw)


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load the YAML config; env var MILAN_CONFIG overrides the default location."""
    cfg_path = Path(path or os.environ.get("MILAN_CONFIG", DEFAULT_CONFIG))
    with open(cfg_path) as fh:
        data = yaml.safe_load(fh)
    data.setdefault("_root", str(PROJECT_ROOT))
    return Config(data)
