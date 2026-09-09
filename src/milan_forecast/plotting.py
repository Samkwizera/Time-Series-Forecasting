"""Shared figure styling and saving."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

PALETTE = {"naive": "#9e9e9e", "seasonal_naive": "#616161", "sarima": "#1f77b4",
           "lightgbm": "#2ca02c", "lstm": "#d62728", "gru": "#e377c2", "weekly_naive": "#bdbdbd"}


def setup_style() -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
                         "axes.titleweight": "bold"})


def save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path
