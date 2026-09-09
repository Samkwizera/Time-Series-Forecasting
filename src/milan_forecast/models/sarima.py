"""Seasonal ARIMA with weekly Fourier regressors (dynamic harmonic regression).

One model per cell on z = log1p(y). The daily cycle is handled by the seasonal part
with period 24; a period of 168 would make the state vector impractically large, so the
weekly cycle enters through K Fourier pairs of period 168 h as exogenous regressors.
An optional holiday dummy is a further regressor.

Parameters are estimated once on the fitting window. The Kalman filter is then run with
those fixed parameters over fit + evaluation data and, from every evaluation origin, the
h-step forecasts are read off (``get_prediction(dynamic=True)``). This mimics an operator
who re-estimates weekly but forecasts hourly.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..datasets import ForecastData, predictions_frame, targets_matrix
from ._common import eval_origins, fit_mask

log = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "order": [1, 0, 1],
    "seasonal_order": [1, 1, 1, 24],
    "fourier_k": 3,
    "holiday_exog": False,
    "maxiter": 200,
}

WEEK_HOURS = 168


def fourier_terms(n: int, k: int, period: int = WEEK_HOURS) -> np.ndarray:
    """K sin/cos pairs of the given period over an integer time index 0..n-1."""
    if k <= 0:
        return np.empty((n, 0))
    t = np.arange(n)
    cols = []
    for j in range(1, k + 1):
        cols.append(np.sin(2 * np.pi * j * t / period))
        cols.append(np.cos(2 * np.pi * j * t / period))
    return np.column_stack(cols)


def build_exog(data: ForecastData, k: int, holiday: bool) -> np.ndarray | None:
    parts = [fourier_terms(len(data.index), k)]
    if holiday:
        parts.append(data.calendar["is_holiday"].to_numpy(dtype=float)[:, None])
    X = np.hstack(parts)
    return X if X.shape[1] else None


def _fit_one(z: np.ndarray, exog: np.ndarray | None, p: dict):
    model = SARIMAX(z, exog=exog, order=tuple(p["order"]), seasonal_order=tuple(p["seasonal_order"]),
                    enforce_stationarity=False, enforce_invertibility=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp=False, maxiter=int(p["maxiter"]))
    return res


def forecast(data: ForecastData, part: str, params: dict) -> tuple[pd.DataFrame, dict]:
    p = {**DEFAULT_PARAMS, **params}
    origins = eval_origins(data, part)
    max_h = max(data.horizons)
    exog_full = build_exog(data, int(p["fourier_k"]), bool(p["holiday_exog"]))
    fit_sel = fit_mask(data, part)
    n_fit = int(fit_sel.sum())
    # the filter must run over fit data plus everything up to the last target
    last_pos = data.index.get_indexer([origins[-1]])[0] + max_h + 1

    frames, info = [], {"aic": {}, "converged": {}, "n_params": None}
    for name, s in data.series.items():
        z = np.log1p(s.to_numpy(dtype=np.float64))
        exog_fit = exog_full[:n_fit] if exog_full is not None else None
        res = _fit_one(z[:n_fit], exog_fit, p)
        info["aic"][name] = float(res.aic)
        info["converged"][name] = bool(res.mle_retvals.get("converged", True))
        info["n_params"] = int(len(res.params))

        exog_ext = exog_full[:last_pos] if exog_full is not None else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ext = res.apply(z[:last_pos], exog=exog_ext, refit=False)
        pos = s.index.get_indexer(origins)
        preds = np.empty((len(origins), len(data.horizons)))
        for i, t in enumerate(pos):
            # dynamic=True: from t+1 onward forecasts feed back on themselves -> genuine h-step forecasts
            fc = ext.get_prediction(start=t + 1, end=t + max_h, dynamic=True).predicted_mean
            preds[i] = [fc[h - 1] for h in data.horizons]
        y_pred = np.expm1(preds).clip(min=0)
        y_true = targets_matrix(s, origins, data.horizons)
        frames.append(predictions_frame("sarima", name, origins, data.horizons, y_true, y_pred))
        log.info("sarima %s: AIC %.1f, %d origins", name, res.aic, len(origins))
    return pd.concat(frames, ignore_index=True), info
