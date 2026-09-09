"""Tests of the forecasting layer on an in-memory synthetic dataset (no ingestion needed)."""

import numpy as np
import pandas as pd
import pytest

from milan_forecast.datasets import ForecastData, LogScaler, Split, predictions_frame, targets_matrix
from milan_forecast.evaluate import diebold_mariano, mase, pairwise_dm, score_predictions, seasonal_naive_scale
from milan_forecast.features import build_origin_features, make_supervised
from milan_forecast.models import baseline, lgbm, rnn, sarima
from milan_forecast.models._common import eval_origins, fit_origins, holdout_split

HORIZONS = [1, 6, 24]


@pytest.fixture(scope="module")
def data() -> ForecastData:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2013-11-01", periods=35 * 24, freq="1h")
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    series = {}
    for k, name in enumerate(["business", "residential"]):
        daily = np.exp(-((hour - (13 if k == 0 else 21)) ** 2) / 30)
        weekly = 1 - 0.4 * (dow >= 5) * (k == 0)
        y = 200 * (0.3 + daily) * weekly * (1 + 0.05 * rng.standard_normal(len(idx)))
        series[name] = pd.Series(y.clip(min=0).astype(np.float32), index=idx, name=name)
    split = Split(pd.Timestamp("2013-11-28 23:00"), pd.Timestamp("2013-12-01 23:00"))
    return ForecastData(series, split, HORIZONS, input_window=72)


def test_origins_do_not_leak_across_splits(data):
    labels = pd.Series(data.split.label(data.index), index=data.index)
    for part in ("val", "test"):
        origins = eval_origins(data, part)
        assert len(origins) > 0
        for h in HORIZONS:
            assert (labels.loc[origins + pd.Timedelta(hours=h)] == part).all()
    fit = fit_origins(data, "val")
    assert (labels.loc[fit + pd.Timedelta(hours=max(HORIZONS))] == "train").all()
    fit_test = fit_origins(data, "test")
    assert set(labels.loc[fit_test + pd.Timedelta(hours=1)]) <= {"train", "val"}


def test_holdout_split_is_chronological(data):
    fit = fit_origins(data, "val")
    a, b = holdout_split(fit, days=7)
    assert len(a) + len(b) == len(fit)
    assert a[-1] < b[0]
    assert (b[-1] - b[0]) < pd.Timedelta(days=7)


def test_scaler_round_trip(data):
    scaler = LogScaler().fit(data)
    s = data.series["business"]
    z = scaler.transform("business", s)
    np.testing.assert_allclose(scaler.inverse("business", z), s.to_numpy(), rtol=1e-5, atol=1e-3)


def test_features_relative_to_origin(data):
    scaler = LogScaler().fit(data)
    s = data.series["business"]
    z = pd.Series(scaler.transform("business", s), index=s.index)
    feats = build_origin_features(z, data.calendar, lags=[1, 2, 24], rolling=[6], use_calendar=True, use_rolling=True)
    t = 100
    assert feats["lag_1"].iloc[t] == z.iloc[t]
    assert feats["lag_2"].iloc[t] == z.iloc[t - 1]
    assert feats["lag_24"].iloc[t] == z.iloc[t - 23]
    assert feats["hour"].iloc[t] == s.index[t].hour


def test_make_supervised_targets_are_future_values(data):
    scaler = LogScaler().fit(data)
    origins = fit_origins(data, "val")
    X, y, meta = make_supervised(data, scaler, 6, {n: origins for n in data.series})
    assert len(X) == len(y) == len(meta) == 2 * len(origins)
    row = meta.iloc[5]
    s = data.series[row["series"]]
    expected = scaler.transform(row["series"], s.loc[[row["origin"] + pd.Timedelta(hours=6)]])[0]
    assert np.isclose(y[5], expected)


def test_metrics_and_mase_scale(data):
    s = data.series["business"]
    scale = seasonal_naive_scale(s)
    assert scale > 0
    y = np.array([1.0, 2.0, 3.0])
    assert mase(y, y, scale) == 0.0
    assert mase(y, y + scale, scale) == pytest.approx(1.0)


def test_baselines_shapes_and_seasonal_naive_beats_naive(data):
    naive = baseline.forecast(data, "val", "naive")
    seasonal = baseline.forecast(data, "val", "seasonal_naive")
    assert set(naive.columns) == {"model", "series", "origin", "horizon", "target_time", "y_true", "y_pred"}
    assert len(naive) == len(seasonal)
    # seasonal naive with h=24 is exactly y[t] because 24 - 24 = 0 steps back from the origin
    s24 = seasonal[seasonal["horizon"] == 24]
    n24 = naive[naive["horizon"] == 24]
    np.testing.assert_allclose(s24["y_pred"].to_numpy(), n24["y_pred"].to_numpy())
    labels = data.split.label(data.index)
    scales = {n: seasonal_naive_scale(v[labels == "train"]) for n, v in data.series.items()}
    m = score_predictions(pd.concat([naive, seasonal]), scales).groupby("model")["mase"].mean()
    assert m["seasonal_naive"] < m["naive"]


def test_diebold_mariano_detects_a_clearly_better_model():
    rng = np.random.default_rng(1)
    e_good = rng.normal(0, 1, 300)
    e_bad = rng.normal(0, 3, 300)
    stat, p = diebold_mariano(e_good, e_bad, h=1)
    assert stat < 0 and p < 0.01
    preds = pd.concat([
        predictions_frame("a", "s", pd.date_range("2013-12-01", periods=300, freq="1h"), [1],
                          np.zeros((300, 1)), e_good[:, None]),
        predictions_frame("b", "s", pd.date_range("2013-12-01", periods=300, freq="1h"), [1],
                          np.zeros((300, 1)), e_bad[:, None])])
    dm = pairwise_dm(preds, reference="b")
    assert dm.iloc[0]["better"] == "model"


def _check_learned(preds, data, part="val"):
    origins = eval_origins(data, part)
    assert len(preds) == len(data.series) * len(HORIZONS) * len(origins)
    assert preds["y_pred"].notna().all() and (preds["y_pred"] >= 0).all()
    for name, s in data.series.items():
        got = preds[(preds["series"] == name) & (preds["horizon"] == 1)].sort_values("origin")["y_true"].to_numpy()
        np.testing.assert_allclose(got, targets_matrix(s, origins, HORIZONS)[:, 0])


def test_sarima_forecast(data):
    preds, info = sarima.forecast(data, "val", {"order": [1, 0, 0], "seasonal_order": [0, 1, 0, 24],
                                                 "fourier_k": 1, "holiday_exog": True, "maxiter": 30})
    _check_learned(preds, data)
    assert set(info["aic"]) == set(data.series)


def test_lightgbm_forecast(data):
    preds, info = lgbm.forecast(data, "val", {"lags": "short", "learning_rate": 0.1, "max_rounds": 300,
                                              "early_stopping_rounds": 20}, seed=0)
    _check_learned(preds, data)
    assert set(info["best_iteration"]) == set(HORIZONS)
    labels = data.split.label(data.index)
    scales = {n: seasonal_naive_scale(v[labels == "train"]) for n, v in data.series.items()}
    assert score_predictions(preds, scales)["mase"].mean() < 1.5


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_rnn_forecast(data, cell):
    preds, info = rnn.forecast(data, "test", {"cell": cell, "hidden_size": 8, "input_window": 48,
                                              "max_epochs": 2, "patience": 1}, seed=0)
    _check_learned(preds, data, part="test")
    assert preds["model"].unique().tolist() == [cell]
    assert info["epochs"] <= 2 and info["n_params"] > 0
