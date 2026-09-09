"""Comparative evaluation and failure analysis across the final models."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import Config
from .eda import calendar_flags, italian_holidays
from .evaluate import pairwise_dm, score_predictions, summarise
from .plotting import PALETTE, save, setup_style

log = logging.getLogger(__name__)


def load_final_predictions(cfg: Config, runs: dict[str, str], part: str = "test") -> pd.DataFrame:
    """Read the prediction tables of the chosen final run of each model."""
    pred_dir = cfg.paths.experiments_dir / "predictions"
    frames = []
    for model, run in runs.items():
        path = pred_dir / f"{model}_{run}_{part}.parquet"
        if not path.exists():
            log.warning("missing %s, skipping", path)
            continue
        frames.append(pd.read_parquet(path))
    preds = pd.concat(frames, ignore_index=True)
    preds["target_time"] = pd.to_datetime(preds["target_time"])
    return preds


def metrics_tables(preds: pd.DataFrame, scales: dict[str, float], cfg: Config) -> pd.DataFrame:
    metrics = score_predictions(preds, scales)
    metrics.to_csv(cfg.paths.tables_dir / "results_by_series_horizon.csv", index=False)
    summary = summarise(metrics)
    summary.to_csv(cfg.paths.tables_dir / "results_summary.csv", index=False)
    pivot = metrics.pivot_table(index=["series", "horizon"], columns="model", values="mase")
    pivot.to_csv(cfg.paths.tables_dir / "results_mase_pivot.csv")
    return metrics


def plot_metric_by_horizon(metrics: pd.DataFrame, cfg: Config, metric: str = "mase") -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    order = [m for m in PALETTE if m in metrics["model"].unique()] + \
            [m for m in metrics["model"].unique() if m not in PALETTE]
    agg = metrics.groupby(["model", "horizon"])[metric].mean().reset_index()
    sns.barplot(agg, x="horizon", y=metric, hue="model", hue_order=order, ax=axes[0],
                palette={m: PALETTE.get(m, "#8c564b") for m in order})
    axes[0].set_title(f"Mean {metric.upper()} by horizon (over selected cells)")
    axes[0].axhline(1.0, color="k", ls=":", lw=0.8)
    sns.barplot(metrics, x="series", y=metric, hue="model", hue_order=order, ax=axes[1], errorbar=None,
                palette={m: PALETTE.get(m, "#8c564b") for m in order})
    axes[1].set_title(f"Mean {metric.upper()} by cell type (over horizons)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].axhline(1.0, color="k", ls=":", lw=0.8)
    axes[1].get_legend().remove()
    save(fig, cfg.paths.figures_dir, f"results_{metric}_by_horizon_and_cell")


def plot_forecasts(preds: pd.DataFrame, cfg: Config, horizon: int = 1, days: int = 7) -> None:
    """Actual vs forecast for every series over the last ``days`` of the evaluation period."""
    setup_style()
    sub = preds[preds["horizon"] == horizon]
    series_names = list(sub["series"].unique())
    end = sub["target_time"].max()
    start = end - pd.Timedelta(days=days)
    fig, axes = plt.subplots(len(series_names), 1, figsize=(12, 2.0 * len(series_names)), sharex=True)
    for ax, name in zip(np.atleast_1d(axes), series_names):
        s = sub[(sub["series"] == name) & (sub["target_time"] >= start)]
        truth = s.drop_duplicates("target_time").set_index("target_time")["y_true"].sort_index()
        truth.plot(ax=ax, color="k", lw=1.2, label="actual")
        for model, g in s.groupby("model"):
            g.set_index("target_time")["y_pred"].sort_index().plot(ax=ax, lw=0.9, alpha=0.9,
                                                                   color=PALETTE.get(model), label=model)
        hol = italian_holidays(pd.DatetimeIndex(truth.index))
        for day in truth.index[hol.to_numpy()].normalize().unique():
            ax.axvspan(day, day + pd.Timedelta(days=1), color="orange", alpha=0.15, lw=0)
        ax.set_ylabel(name, fontsize=8)
    np.atleast_1d(axes)[0].legend(ncol=6, fontsize=7, loc="upper left")
    np.atleast_1d(axes)[0].set_title(f"{horizon} h-ahead forecasts, last {days} days of the test period (shaded = holidays)")
    save(fig, cfg.paths.figures_dir, f"results_forecasts_h{horizon}")


def error_over_time(preds: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Daily MAE per model to locate the periods where forecasts break down."""
    setup_style()
    p = preds.copy()
    p["abs_err"] = (p["y_true"] - p["y_pred"]).abs()
    p["day"] = p["target_time"].dt.normalize()
    daily = p.groupby(["model", "day"])["abs_err"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for model in daily.columns:
        daily[model].plot(ax=ax, marker="o", ms=3, lw=1, color=PALETTE.get(model), label=model)
    hol = italian_holidays(pd.DatetimeIndex(daily.index))
    for day in daily.index[hol.to_numpy()]:
        ax.axvspan(day, day + pd.Timedelta(days=1), color="orange", alpha=0.15, lw=0)
    ax.set_ylabel("mean absolute error (all horizons, all cells)")
    ax.set_title("Daily forecast error across the test period (shaded = holidays)")
    ax.legend(ncol=6, fontsize=8)
    save(fig, cfg.paths.figures_dir, "failure_daily_error")
    daily.to_csv(cfg.paths.tables_dir / "failure_daily_error.csv")
    return daily


def error_by_hour_and_daytype(preds: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    setup_style()
    p = preds.copy()
    p["abs_err"] = (p["y_true"] - p["y_pred"]).abs()
    flags = calendar_flags(pd.DatetimeIndex(p["target_time"]))
    p["hour"] = flags["hour"].to_numpy()
    p["day_type"] = flags["day_type"].to_numpy()
    # normalise by the mean level of each series so cells are comparable
    level = p.groupby("series")["y_true"].transform("mean")
    p["rel_err"] = p["abs_err"] / level
    table = p.groupby(["model", "day_type", "hour"])["rel_err"].mean().reset_index()
    models = list(table["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.2), sharey=True)
    for ax, model in zip(np.atleast_1d(axes), models):
        for day_type, ls in [("weekday", "-"), ("weekend", "--"), ("holiday", ":")]:
            t = table[(table["model"] == model) & (table["day_type"] == day_type)]
            if len(t):
                ax.plot(t["hour"], t["rel_err"], ls=ls, label=day_type, color=PALETTE.get(model, "k"))
        ax.set_title(model)
        ax.set_xlabel("hour of target")
    np.atleast_1d(axes)[0].set_ylabel("relative abs. error")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("Where the errors concentrate: by hour of day and day type", y=1.03)
    save(fig, cfg.paths.figures_dir, "failure_hour_daytype")
    table.to_csv(cfg.paths.tables_dir / "failure_hour_daytype.csv", index=False)
    return table


def worst_cases(preds: pd.DataFrame, cfg: Config, n: int = 10) -> pd.DataFrame:
    p = preds.copy()
    p["abs_err"] = (p["y_true"] - p["y_pred"]).abs()
    p["rel_err"] = p["abs_err"] / p.groupby("series")["y_true"].transform("mean")
    worst = (p.sort_values("rel_err", ascending=False)
             .groupby("model").head(n)[["model", "series", "horizon", "target_time", "y_true", "y_pred", "rel_err"]])
    worst.to_csv(cfg.paths.tables_dir / "failure_worst_cases.csv", index=False)
    return worst


def residual_diagnostics(preds: pd.DataFrame, cfg: Config, horizon: int = 1) -> pd.DataFrame:
    """Bias and error-vs-level per model; shows whether models systematically under-forecast peaks."""
    setup_style()
    p = preds[preds["horizon"] == horizon].copy()
    p["err"] = p["y_pred"] - p["y_true"]
    models = list(p["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.2), sharey=True)
    for ax, model in zip(np.atleast_1d(axes), models):
        g = p[p["model"] == model]
        ax.scatter(g["y_true"], g["err"], s=4, alpha=0.3, color=PALETTE.get(model, "k"))
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{model} (bias {g['err'].mean():+.1f})")
        ax.set_xlabel("actual")
    np.atleast_1d(axes)[0].set_ylabel("forecast - actual")
    fig.suptitle(f"Residuals against actual level, {horizon} h ahead", y=1.03)
    save(fig, cfg.paths.figures_dir, f"failure_residuals_h{horizon}")
    bias = p.groupby(["model", "series"])["err"].agg(["mean", "std"]).reset_index()
    bias.to_csv(cfg.paths.tables_dir / "failure_bias.csv", index=False)
    return bias


def dm_tests(preds: pd.DataFrame, cfg: Config, reference: str = "seasonal_naive") -> pd.DataFrame:
    table = pairwise_dm(preds, reference)
    table.to_csv(cfg.paths.tables_dir / f"dm_vs_{reference}.csv", index=False)
    return table


def compute_cost_table(cfg: Config) -> pd.DataFrame:
    """Training time and peak memory per model, read from memory_log.csv."""
    log_path = cfg.paths.tables_dir / "memory_log.csv"
    if not log_path.exists():
        return pd.DataFrame()
    mem = pd.read_csv(log_path)
    train = mem[mem["stage"].str.startswith("train:")].copy()
    train[["_", "model", "run", "part"]] = train["stage"].str.split(":", expand=True)
    cost = (train.groupby(["model", "part"]).agg(seconds=("seconds", "median"), peak_rss_mb=("rss_peak_mb", "max"),
                                                  runs=("seconds", "size")).reset_index())
    cost.to_csv(cfg.paths.tables_dir / "compute_cost.csv", index=False)
    return cost
