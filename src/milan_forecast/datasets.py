"""Forecasting data setup shared by all models.

Definitions used throughout the project
---------------------------------------
* A *series* is the hourly target activity of one selected cell (name -> pd.Series).
* An *origin* ``t`` is the last observed hour. A forecast for horizon ``h`` is a
  prediction of ``y[t + h]`` made with data up to and including ``t``.
* Splits are chronological and shared by every model: train < val < test.
* Models work in ``log1p`` space (variance-stabilising, see tsa_stationarity.csv)
  and are evaluated after inverting the transform, in the original units.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calendar import calendar_flags
from .config import Config
from .ingest import load_hourly


@dataclass
class Split:
    train_end: pd.Timestamp
    val_end: pd.Timestamp

    def label(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.select([index <= self.train_end, index <= self.val_end], ["train", "val"], "test")


@dataclass
class ForecastData:
    # name -> hourly series in original units
    series: dict[str, pd.Series]
    split: Split
    horizons: list[int]
    input_window: int
    calendar: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        idx = next(iter(self.series.values())).index
        self.calendar = calendar_flags(idx)

    @property
    def index(self) -> pd.DatetimeIndex:
        return next(iter(self.series.values())).index

    # NB: origin selection lives in models/_common.py (``eval_origins`` / ``fit_origins``) so that
    # the leakage-critical rule exists in exactly one place.

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.series)


def load_forecast_data(cfg: Config, names: list[str] | None = None) -> ForecastData:
    activity = cfg.forecast.target_activity
    with open(cfg.paths.tables_dir / "selected_cells.json") as fh:
        selected = json.load(fh)["cells"]
    if names:
        selected = {k: v for k, v in selected.items() if k in names}
    hourly = load_hourly(cfg, activity)
    full_index = pd.date_range(hourly.index[0], hourly.index[-1], freq="1h")
    hourly = hourly.reindex(full_index).interpolate(limit_direction="both")
    series = {name: hourly[cell].rename(name).astype(np.float32) for name, cell in selected.items()}
    split = Split(pd.Timestamp(cfg.forecast.split.train_end), pd.Timestamp(cfg.forecast.split.val_end))
    return ForecastData(series, split, list(cfg.forecast.horizons), int(cfg.forecast.input_window))


# --------------------------------------------------------------------------- #
# Transform helpers
# --------------------------------------------------------------------------- #
class LogScaler:
    """log1p followed by per-series standardisation with statistics from the training part only."""

    def __init__(self) -> None:
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}

    def fit(self, data: ForecastData) -> "LogScaler":
        labels = data.split.label(data.index)
        for name, s in data.series.items():
            z = np.log1p(s.to_numpy()[labels == "train"])
            self.mean[name] = float(z.mean())
            self.std[name] = float(z.std() + 1e-6)
        return self

    def transform(self, name: str, values: np.ndarray | pd.Series) -> np.ndarray:
        return (np.log1p(np.asarray(values, dtype=np.float64)) - self.mean[name]) / self.std[name]

    def inverse(self, name: str, values: np.ndarray) -> np.ndarray:
        return np.expm1(np.asarray(values, dtype=np.float64) * self.std[name] + self.mean[name]).clip(min=0)


def predictions_frame(model: str, name: str, origins: pd.DatetimeIndex, horizons: list[int],
                      y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Tidy prediction table: one row per (origin, horizon)."""
    rows = []
    for j, h in enumerate(horizons):
        rows.append(pd.DataFrame({"model": model, "series": name, "origin": origins, "horizon": h,
                                  "target_time": origins + pd.Timedelta(hours=h),
                                  "y_true": y_true[:, j], "y_pred": y_pred[:, j]}))
    return pd.concat(rows, ignore_index=True)


def targets_matrix(series: pd.Series, origins: pd.DatetimeIndex, horizons: list[int]) -> np.ndarray:
    pos = series.index.get_indexer(origins)
    return np.column_stack([series.to_numpy()[pos + h] for h in horizons])
