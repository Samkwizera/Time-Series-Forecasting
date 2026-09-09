"""Time-series analysis: stationarity, decomposition, autocorrelation.

The outputs of this module are the evidence behind the modelling choices:

* ADF/KPSS on raw, log and seasonally-differenced series decide the SARIMA
  differencing orders and whether the LSTM/LightGBM targets need a transform.
* STL with a 24 h period (and MSTL with 24 h + 168 h) quantifies how much variance
  is explained by the daily and weekly cycles.
* ACF/PACF locate the lags that carry information (1-3 h, 24 h, 168 h), which
  fixes the lag set used as LightGBM features and the SARIMA seasonal order.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import MSTL, STL
from statsmodels.tsa.stattools import acf, adfuller, kpss

from .config import Config
from .plotting import save, setup_style

log = logging.getLogger(__name__)


def stationarity_tests(series: pd.Series, name: str) -> list[dict]:
    """Run ADF and KPSS on several transforms; returns rows for a table."""
    variants = {
        "raw": series,
        "log1p": np.log1p(series),
        "log1p, diff 1": np.log1p(series).diff().dropna(),
        "log1p, seasonal diff 24": np.log1p(series).diff(24).dropna(),
        "log1p, diff 1 + seasonal diff 24": np.log1p(series).diff(24).diff().dropna(),
    }
    rows = []
    for label, s in variants.items():
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
        try:
            kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
        # KPSS can blow up on near-constant series; report NaN rather than abort the table
        except Exception:  # noqa: BLE001
            kpss_stat, kpss_p = np.nan, np.nan
        verdict = ("stationary" if adf_p < 0.05 and kpss_p > 0.05 else
                   "non-stationary" if adf_p >= 0.05 and kpss_p <= 0.05 else "inconclusive")
        rows.append({"series": name, "transform": label, "adf_stat": adf_stat, "adf_p": adf_p,
                     "kpss_stat": kpss_stat, "kpss_p": kpss_p, "verdict": verdict})
    return rows


def decomposition(series: pd.Series, cfg: Config, name: str) -> dict:
    """STL (24 h) and MSTL (24 h, 168 h); returns variance shares of each component."""
    setup_style()
    y = np.log1p(series.asfreq("1h").interpolate())
    stl = STL(y, period=24, robust=True).fit()
    fig = stl.plot()
    fig.set_size_inches(10, 7)
    fig.suptitle(f"STL decomposition (period = 24 h) of log(1+{name})", y=1.0)
    save(fig, cfg.paths.figures_dir, f"tsa_stl_{name}")

    out = {"series": name, "stl24_trend_var": stl.trend.var(), "stl24_seasonal_var": stl.seasonal.var(),
           "stl24_resid_var": stl.resid.var()}
    # MSTL silently drops the 168h period when there are too few weeks
    if len(y) >= 3 * 168:
        mstl = MSTL(y, periods=(24, 168), stl_kwargs={"robust": True}).fit()
        fig = mstl.plot()
        fig.set_size_inches(10, 8)
        fig.suptitle(f"MSTL decomposition (24 h and 168 h) of log(1+{name})", y=1.0)
        save(fig, cfg.paths.figures_dir, f"tsa_mstl_{name}")
        seasonal = np.asarray(mstl.seasonal)
        if seasonal.ndim == 1:
            seasonal = np.column_stack([seasonal, np.zeros_like(seasonal)])
        out.update({"mstl_daily_var": seasonal[:, 0].var(), "mstl_weekly_var": seasonal[:, 1].var(),
                    "mstl_trend_var": np.asarray(mstl.trend).var(), "mstl_resid_var": np.asarray(mstl.resid).var()})
    total = y.var()
    for k in list(out):
        if k.endswith("_var"):
            out[k.replace("_var", "_share")] = float(out[k] / total)
    return out


def autocorrelation(series: pd.Series, cfg: Config, name: str, nlags: int = 200) -> pd.DataFrame:
    setup_style()
    y = np.log1p(series.asfreq("1h").interpolate())
    d = y.diff(24).dropna()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    plot_acf(y, lags=nlags, ax=axes[0, 0], title=f"ACF log(1+{name})", zero=False)
    plot_pacf(y, lags=min(72, nlags), ax=axes[0, 1], title="PACF (first 72 lags)", zero=False, method="ywm")
    plot_acf(d, lags=nlags, ax=axes[1, 0], title="ACF after seasonal difference (24 h)", zero=False)
    plot_pacf(d, lags=min(72, nlags), ax=axes[1, 1], title="PACF after seasonal difference", zero=False, method="ywm")
    for ax in axes[:, 0]:
        for lag in (24, 48, 72, 168):
            if lag <= nlags:
                ax.axvline(lag, color="grey", ls=":", lw=0.8)
    save(fig, cfg.paths.figures_dir, f"tsa_acf_{name}")
    vals = acf(y, nlags=nlags, fft=True)
    table = pd.DataFrame({"lag": np.arange(nlags + 1), "acf": vals})
    key = table[table["lag"].isin([1, 2, 3, 6, 12, 24, 48, 168])].assign(series=name)
    return key


def run_all(hourly: pd.DataFrame, city: pd.DataFrame, selected: dict[str, int], cfg: Config) -> None:
    """Run every analysis on the citywide total and each selected cell; write tables."""
    activity = cfg.forecast.target_activity
    targets = {"citywide": city[activity].resample("1h").sum()}
    targets.update({name: hourly[cell] for name, cell in selected.items()})
    stat_rows, decomp_rows, acf_rows = [], [], []
    for name, s in targets.items():
        stat_rows += stationarity_tests(s, name)
        decomp_rows.append(decomposition(s, cfg, name))
        acf_rows.append(autocorrelation(s, cfg, name))
    tables = cfg.paths.tables_dir
    pd.DataFrame(stat_rows).to_csv(tables / "tsa_stationarity.csv", index=False)
    pd.DataFrame(decomp_rows).to_csv(tables / "tsa_decomposition.csv", index=False)
    pd.concat(acf_rows).to_csv(tables / "tsa_acf_key_lags.csv", index=False)
    log.info("stationarity table:\n%s", pd.DataFrame(stat_rows).to_string(index=False, float_format="%.3f"))
