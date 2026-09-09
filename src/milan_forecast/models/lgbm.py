"""Gradient-boosted trees (LightGBM), one regressor per horizon.

All selected cells are pooled into one design matrix with a categorical cell identifier,
so the trees can share structure (hour-of-day effects) while still splitting on the cell
when profiles differ. Features are built relative to the origin (see ``features.py``).
The objective is l1: it targets the conditional median and is robust to the heavy right
tail of peak hours, and it is the loss MAE/MASE are built on.

Number of trees: early stopping on the last week of the fitting window, then a refit on
the whole window with the chosen number of iterations. The evaluation split is never seen.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..datasets import ForecastData, LogScaler, predictions_frame, targets_matrix
from ..features import DEFAULT_LAGS, ROLLING_WINDOWS, make_supervised
from ._common import eval_origins, fit_origins, holdout_split

log = logging.getLogger(__name__)

LAG_SETS = {"short": list(range(1, 25)), "full": DEFAULT_LAGS}

DEFAULT_PARAMS = {
    "lags": "full",
    "use_calendar": True,
    "use_rolling": True,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 0.0,
    "max_rounds": 2000,
    "early_stopping_rounds": 50,
}


def _lgb_params(p: dict, seed: int) -> dict:
    return {"objective": "l1", "metric": "l1", "verbosity": -1, "seed": seed, "num_threads": 0,
            "num_leaves": int(p["num_leaves"]), "learning_rate": float(p["learning_rate"]),
            "min_child_samples": int(p["min_child_samples"]), "feature_fraction": float(p["feature_fraction"]),
            "bagging_fraction": float(p["bagging_fraction"]), "bagging_freq": int(p["bagging_freq"]),
            "lambda_l2": float(p["lambda_l2"])}


def _fix_categories(X: pd.DataFrame, n_series: int) -> pd.DataFrame:
    # category codes must mean the same thing in the training and the prediction frames
    X["series_id"] = pd.Categorical(X["series_id"].astype(int), categories=list(range(n_series)))
    return X


def forecast(data: ForecastData, part: str, params: dict, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    p = {**DEFAULT_PARAMS, **params}
    lags = LAG_SETS[p["lags"]] if isinstance(p["lags"], str) else list(p["lags"])
    min_hist = max(max(lags), max(ROLLING_WINDOWS) if p["use_rolling"] else 0)
    scaler = LogScaler().fit(data)
    names = list(data.series)

    all_fit = fit_origins(data, part, min_history=min_hist)
    inner_fit, holdout = holdout_split(all_fit)
    evaluation = eval_origins(data, part, min_history=min_hist)
    kw = dict(lags=lags, use_calendar=bool(p["use_calendar"]), use_rolling=bool(p["use_rolling"]))

    y_pred = {n: np.empty((len(evaluation), len(data.horizons))) for n in names}
    info = {"best_iteration": {}, "holdout_l1": {}, "n_features": None, "importance": {}}
    for j, h in enumerate(data.horizons):
        X_in, y_in, _ = make_supervised(data, scaler, h, {n: inner_fit for n in names}, **kw)
        X_ho, y_ho, _ = make_supervised(data, scaler, h, {n: holdout for n in names}, **kw)
        X_in, X_ho = _fix_categories(X_in, len(names)), _fix_categories(X_ho, len(names))
        lp = _lgb_params(p, seed)
        booster = lgb.train(lp, lgb.Dataset(X_in, y_in), num_boost_round=int(p["max_rounds"]),
                            valid_sets=[lgb.Dataset(X_ho, y_ho)],
                            callbacks=[lgb.early_stopping(int(p["early_stopping_rounds"]), verbose=False)])
        best = max(booster.best_iteration, 1)
        info["best_iteration"][h] = int(best)
        info["holdout_l1"][h] = float(booster.best_score["valid_0"]["l1"])

        # refit on the entire fitting window with the number of trees chosen above
        X_all, y_all, _ = make_supervised(data, scaler, h, {n: all_fit for n in names}, **kw)
        X_all = _fix_categories(X_all, len(names))
        final = lgb.train(lp, lgb.Dataset(X_all, y_all), num_boost_round=best)
        info["n_features"] = int(X_all.shape[1])
        gain = final.feature_importance("gain")
        top = sorted(zip(X_all.columns, gain), key=lambda kv: -kv[1])[:10]
        info["importance"][h] = {k: round(float(v), 1) for k, v in top}

        X_ev, _, meta = make_supervised(data, scaler, h, {n: evaluation for n in names}, **kw)
        X_ev = _fix_categories(X_ev, len(names))
        z_hat = final.predict(X_ev)
        for n in names:
            sel = (meta["series"] == n).to_numpy()
            y_pred[n][:, j] = scaler.inverse(n, z_hat[sel])
        log.info("lightgbm h=%d: %d trees, holdout l1 %.4f, %d rows", h, best, info["holdout_l1"][h], len(X_all))

    frames = [predictions_frame("lightgbm", n, evaluation, data.horizons,
                                targets_matrix(data.series[n], evaluation, data.horizons), y_pred[n]) for n in names]
    return pd.concat(frames, ignore_index=True), info
