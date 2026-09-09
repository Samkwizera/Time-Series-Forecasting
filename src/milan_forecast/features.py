"""Feature construction for the tabular (LightGBM) model.

Features are built *relative to the forecast origin* so that no information after
the origin leaks into the inputs. The lag set follows the autocorrelation analysis:
the first 24 lags capture short-range dependence and the daily cycle; 48 and 168
capture the two-day and weekly cycles; rolling statistics summarise the level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .datasets import ForecastData, LogScaler

DEFAULT_LAGS = list(range(1, 25)) + [36, 48, 72, 96, 120, 144, 168]
ROLLING_WINDOWS = [6, 24, 168]


def build_origin_features(z: pd.Series, calendar: pd.DataFrame, lags: list[int], rolling: list[int],
                          use_calendar: bool = True, use_rolling: bool = True) -> pd.DataFrame:
    """Feature matrix indexed by origin for one scaled series ``z``."""
    # lag_1 is the origin value itself, so lag_k = k-1 steps back
    cols = {f"lag_{lag}": z.shift(lag - 1) for lag in lags}
    if use_rolling:
        for w in rolling:
            r = z.rolling(w, min_periods=max(2, w // 2))
            cols[f"roll_mean_{w}"] = r.mean()
            cols[f"roll_std_{w}"] = r.std()
        cols["diff_1"] = z.diff()
        cols["diff_24"] = z.diff(24)
    feats = pd.DataFrame(cols, index=z.index)
    if use_calendar:
        feats["hour"] = calendar["hour"].to_numpy()
        feats["dow"] = calendar["dow"].to_numpy()
        feats["is_weekend"] = calendar["is_weekend"].to_numpy().astype(int)
        feats["is_holiday"] = calendar["is_holiday"].to_numpy().astype(int)
    return feats


def target_calendar(calendar: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Calendar attributes of the *target* time, shifted back to align with the origin."""
    shifted = calendar.shift(-horizon)
    out = pd.DataFrame(index=calendar.index)
    out["t_hour"] = shifted["hour"]
    out["t_dow"] = shifted["dow"]
    out["t_is_weekend"] = shifted["is_weekend"].astype(float)
    out["t_is_holiday"] = shifted["is_holiday"].astype(float)
    return out


def make_supervised(data: ForecastData, scaler: LogScaler, horizon: int, origins: dict[str, pd.DatetimeIndex],
                    lags=DEFAULT_LAGS, rolling=ROLLING_WINDOWS, use_calendar=True, use_rolling=True
                    ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Stack all series into one design matrix for a given horizon.

    Returns (X, y_scaled, meta) where meta holds series name and origin per row.
    """
    xs, ys, metas = [], [], []
    for name, s in data.series.items():
        z = pd.Series(scaler.transform(name, s), index=s.index)
        feats = build_origin_features(z, data.calendar, lags, rolling, use_calendar, use_rolling)
        if use_calendar:
            feats = feats.join(target_calendar(data.calendar, horizon))
        feats["series_id"] = list(data.series).index(name)
        idx = origins[name]
        target = z.shift(-horizon).loc[idx]
        xs.append(feats.loc[idx])
        ys.append(target.to_numpy())
        metas.append(pd.DataFrame({"series": name, "origin": idx}))
    X = pd.concat(xs)
    X["series_id"] = X["series_id"].astype("category")
    return X.reset_index(drop=True), np.concatenate(ys), pd.concat(metas, ignore_index=True)
