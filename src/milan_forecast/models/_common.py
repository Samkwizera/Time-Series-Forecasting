"""Helpers shared by the model modules: which origins to fit on, which to forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..datasets import ForecastData

HOLDOUT_DAYS = 7


def fit_parts(part: str) -> tuple[str, ...]:
    """Splits whose data may be used for fitting when evaluating on ``part``."""
    if part == "val":
        return ("train",)
    if part == "test":
        return ("train", "val")
    raise ValueError(f"unknown part {part!r}")


def origins_in(data: ForecastData, parts: tuple[str, ...], min_history: int | None = None) -> pd.DatetimeIndex:
    """Origins whose targets (all horizons) fall inside ``parts`` and that have enough history."""
    idx = data.index
    labels = data.split.label(idx)
    max_h = max(data.horizons)
    ok = np.isin(labels, parts)
    mask = np.zeros(len(idx), dtype=bool)
    n = len(idx) - max_h
    # every target t+1 .. t+max_h must be inside the allowed splits
    for i in range(n):
        if ok[i + 1] and ok[i + max_h]:
            mask[i] = True
    mask[: max(int(min_history or data.input_window), 1)] = False
    return idx[mask]


def eval_origins(data: ForecastData, part: str, min_history: int | None = None) -> pd.DatetimeIndex:
    return origins_in(data, (part,), min_history)


def fit_origins(data: ForecastData, part: str, min_history: int | None = None) -> pd.DatetimeIndex:
    return origins_in(data, fit_parts(part), min_history)


def fit_mask(data: ForecastData, part: str) -> np.ndarray:
    """Boolean mask over the full index: observations available for fitting."""
    return np.isin(data.split.label(data.index), fit_parts(part))


def holdout_split(origins: pd.DatetimeIndex, days: int = HOLDOUT_DAYS) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split origins chronologically: everything before the last ``days`` days, and the last ``days``."""
    cut = origins[-1] - pd.Timedelta(days=days)
    return origins[origins <= cut], origins[origins > cut]
