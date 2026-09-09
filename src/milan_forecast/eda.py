"""Exploratory analysis: temporal, spatial and statistical characterisation.

Every public function returns the data it plotted so that the same numbers can be
written to ``reports/tables`` and quoted in the report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .config import Config
from .plotting import save, setup_style

log = logging.getLogger(__name__)

CLUSTER_LABEL_ORDER = ["business", "residential", "nightlife", "mixed/suburban"]


def italian_holidays(index: pd.DatetimeIndex) -> pd.Series:
    """Boolean series: True on Italian public holidays incl. Sant'Ambrogio (Milan, 7 Dec)."""
    years = sorted(set(index.year))
    cal = holidays.country_holidays("IT", subdiv="MI", years=years)
    return pd.Series(index.normalize().isin(pd.DatetimeIndex(list(cal.keys()))), index=index)


def calendar_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    flags = pd.DataFrame(index=index)
    flags["hour"] = index.hour
    flags["dow"] = index.dayofweek
    flags["is_weekend"] = flags["dow"] >= 5
    flags["is_holiday"] = italian_holidays(index).to_numpy()
    flags["day_type"] = np.select([flags["is_holiday"], flags["is_weekend"]], ["holiday", "weekend"], "weekday")
    return flags


# --------------------------------------------------------------------------- #
# Temporal analyses (citywide)
# --------------------------------------------------------------------------- #
def plot_citywide_series(city: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    setup_style()
    hourly = city.resample("1h").sum()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    hourly["internet"].plot(ax=axes[0], lw=0.8, color="#d62728")
    axes[0].set_ylabel("Internet activity / h")
    for col, c in zip(["sms_in", "sms_out", "call_in", "call_out"], ["#1f77b4", "#aec7e8", "#2ca02c", "#98df8a"]):
        hourly[col].plot(ax=axes[1], lw=0.8, label=col, color=c)
    axes[1].legend(ncol=4, fontsize=8)
    axes[1].set_ylabel("SMS / call activity / h")
    hol = italian_holidays(hourly.index)
    for day in hourly.index[hol].normalize().unique():
        for ax in axes:
            ax.axvspan(day, day + pd.Timedelta(days=1), color="orange", alpha=0.18, lw=0)
    axes[0].set_title("Citywide activity, hourly totals (shaded = Italian public holidays)")
    save(fig, cfg.paths.figures_dir, "eda_citywide_hourly")
    return hourly


def daily_profiles(city: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Mean activity by hour of day for weekdays, weekends and holidays."""
    setup_style()
    hourly = city.resample("1h").sum()
    flags = calendar_flags(hourly.index)
    acts = list(cfg.data.activities)
    prof = hourly.join(flags[["hour", "day_type"]]).groupby(["day_type", "hour"])[acts].mean()
    fig, axes = plt.subplots(1, len(acts), figsize=(3.2 * len(acts), 3.2), sharex=True)
    for ax, act in zip(np.atleast_1d(axes), acts):
        for day_type, style in [("weekday", "-"), ("weekend", "--"), ("holiday", ":")]:
            if day_type in prof.index.get_level_values(0):
                prof.loc[day_type, act].plot(ax=ax, ls=style, label=day_type)
        ax.set_title(act)
        ax.set_xlabel("hour of day")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Citywide mean hourly profile by day type", y=1.03)
    save(fig, cfg.paths.figures_dir, "eda_daily_profiles")
    prof.to_csv(cfg.paths.tables_dir / "daily_profiles.csv")
    return prof


def hour_dow_heatmap(city: pd.DataFrame, cfg: Config, activity: str | None = None) -> pd.DataFrame:
    setup_style()
    activity = activity or cfg.forecast.target_activity
    hourly = city[activity].resample("1h").sum()
    table = hourly.groupby([hourly.index.dayofweek, hourly.index.hour]).mean().unstack()
    table.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(table)]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    sns.heatmap(table, ax=ax, cmap="viridis", cbar_kws={"label": f"mean {activity} / h"})
    ax.set_xlabel("hour of day")
    ax.set_title(f"Weekly rhythm of citywide {activity} activity")
    save(fig, cfg.paths.figures_dir, f"eda_hour_dow_{activity}")
    return table


def daily_totals_anomalies(city: pd.DataFrame, cfg: Config, activity: str | None = None) -> pd.DataFrame:
    """Daily totals with a robust z-score relative to the same weekday, to flag unusual days."""
    setup_style()
    activity = activity or cfg.forecast.target_activity
    daily = city[activity].resample("1D").sum().to_frame("total")
    daily["dow"] = daily.index.dayofweek
    med = daily.groupby("dow")["total"].transform("median")
    mad = daily.groupby("dow")["total"].transform(lambda s: (s - s.median()).abs().median() + 1e-9)
    daily["robust_z"] = 0.6745 * (daily["total"] - med) / mad
    daily["is_holiday"] = italian_holidays(daily.index).to_numpy()
    fig, ax = plt.subplots(figsize=(11, 3.2))
    colors = np.where(daily["is_holiday"], "orange", np.where(daily["dow"] >= 5, "#9e9e9e", "#1f77b4"))
    ax.bar(daily.index, daily["total"], color=colors, width=0.8)
    flagged = daily[daily["robust_z"].abs() > 3.5]
    for ts, row in flagged.iterrows():
        ax.annotate(ts.strftime("%d %b"), (ts, row["total"]), fontsize=7, ha="center", va="bottom", rotation=90)
    ax.set_ylabel(f"daily {activity}")
    ax.set_title("Daily totals (blue weekday, grey weekend, orange holiday; labels = |robust z| > 3.5)")
    save(fig, cfg.paths.figures_dir, f"eda_daily_totals_{activity}")
    daily.to_csv(cfg.paths.tables_dir / f"daily_totals_{activity}.csv")
    return daily


# --------------------------------------------------------------------------- #
# Spatial analyses
# --------------------------------------------------------------------------- #
def _to_grid(values: pd.Series, side: int) -> np.ndarray:
    """Square ids run from 1 at the bottom-left, row-major; return image with north up."""
    grid = np.zeros(side * side, dtype=float)
    ids = values.index.to_numpy().astype(int) - 1
    grid[ids] = values.to_numpy()
    return grid.reshape(side, side)[::-1]


def spatial_maps(hourly: pd.DataFrame, cfg: Config, activity: str) -> pd.DataFrame:
    setup_style()
    side = cfg.data.grid_side
    flags = calendar_flags(hourly.index)
    weekday = hourly[flags["day_type"].to_numpy() == "weekday"]
    mean_map = weekday.mean()
    snapshots = {h: weekday[weekday.index.hour == h].mean() for h in (4, 11, 15, 21)}
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    im = axes[0].imshow(np.log1p(_to_grid(mean_map, side)), cmap="magma")
    axes[0].set_title(f"log(1+mean {activity})")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    vmax = np.log1p(max(s.max() for s in snapshots.values()))
    for ax, (h, snap) in zip(axes[1:], snapshots.items()):
        ax.imshow(np.log1p(_to_grid(snap, side)), cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"weekday {h:02d}:00")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Spatial distribution of {activity} activity over the {side}x{side} grid (north up)", y=1.02)
    save(fig, cfg.paths.figures_dir, f"eda_spatial_{activity}")

    # Concentration of traffic across cells
    sorted_share = np.sort(mean_map.to_numpy())[::-1].cumsum() / mean_map.sum()
    n = len(sorted_share)
    conc = pd.DataFrame({
        "metric": ["top_1pct_share", "top_10pct_share", "top_50pct_share", "gini"],
        "value": [sorted_share[max(int(0.01 * n) - 1, 0)], sorted_share[int(0.10 * n) - 1],
                  sorted_share[int(0.50 * n) - 1], _gini(mean_map.to_numpy())],
    })
    conc.to_csv(cfg.paths.tables_dir / f"spatial_concentration_{activity}.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    ax.plot(np.linspace(0, 100, n), 100 * sorted_share)
    ax.plot([0, 100], [0, 100], ls=":", color="grey")
    ax.set_xlabel("% of cells (busiest first)")
    ax.set_ylabel(f"% of {activity} activity")
    ax.set_title(f"Concentration of traffic (Gini = {conc.value.iloc[-1]:.2f})")
    save(fig, cfg.paths.figures_dir, f"eda_concentration_{activity}")
    return conc


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


# --------------------------------------------------------------------------- #
# Cell clustering and selection
# --------------------------------------------------------------------------- #
def normalised_profiles(hourly: pd.DataFrame, min_mean: float | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Weekday 24-h profile of each cell, scaled to sum 1 so shape is compared, not volume."""
    flags = calendar_flags(hourly.index)
    weekday = hourly[flags["day_type"].to_numpy() == "weekday"]
    prof = weekday.groupby(weekday.index.hour).mean().T  # cells x 24
    mean_level = hourly.mean()
    if min_mean is None:
        min_mean = mean_level.quantile(0.5)  # ignore the quiet outer half of the grid
    keep = mean_level[mean_level >= min_mean].index
    prof = prof.loc[keep]
    prof = prof.div(prof.sum(axis=1), axis=0).fillna(0)
    return prof, mean_level


def _name_clusters(centroids: pd.DataFrame) -> dict[int, str]:
    """Heuristic names from the centroid shape (peak hour and day/evening ratio)."""
    names = {}
    for k, c in centroids.iterrows():
        peak = int(c.idxmax())
        day = c.loc[9:17].sum()
        evening = c.loc[19:23].sum()
        night = c.loc[0:4].sum()
        if night > 0.16 or peak <= 2:
            names[k] = "nightlife"
        elif c.max() / max(c.min(), 1e-9) < 3.0:
            names[k] = "mixed/suburban"
        elif peak in range(9, 18) and day > 1.35 * evening * (9 / 5):
            names[k] = "business"
        elif peak >= 18 or evening * (9 / 5) > day:
            names[k] = "residential"
        else:
            names[k] = "mixed/suburban"
    # Ensure names are unique by appending an index when the heuristic collides.
    seen: dict[str, int] = {}
    for k in list(names):
        base = names[k]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            names[k] = f"{base}-{seen[base]}"
    return names


def cluster_cells(hourly: pd.DataFrame, cfg: Config, activity: str) -> tuple[pd.Series, pd.DataFrame]:
    """K-means on normalised weekday profiles; writes centroid plot, cluster map and silhouette table."""
    setup_style()
    prof, mean_level = normalised_profiles(hourly)
    k = cfg.eda.n_clusters
    seed = cfg.forecast.seed
    scores = {}
    for kk in range(2, max(k + 3, 5)):
        if kk >= len(prof):
            break
        lab = KMeans(kk, n_init=10, random_state=seed).fit_predict(prof)
        scores[kk] = silhouette_score(prof, lab)
    pd.Series(scores, name="silhouette").rename_axis("k").to_csv(cfg.paths.tables_dir / "cluster_silhouette.csv")
    km = KMeans(k, n_init=10, random_state=seed).fit(prof)
    labels = pd.Series(km.labels_, index=prof.index, name="cluster")
    centroids = pd.DataFrame(km.cluster_centers_, columns=prof.columns)
    names = _name_clusters(centroids)
    labels = labels.map(names)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    for kk, c in centroids.iterrows():
        axes[0].plot(c.index, c.values, label=f"{names[kk]} (n={int((km.labels_ == kk).sum())})")
    axes[0].set_xlabel("hour of day (weekday)")
    axes[0].set_ylabel("share of daily activity")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Cluster centroids of normalised daily profiles")
    side = cfg.data.grid_side
    cmap = plt.get_cmap("tab10")
    img = np.full(side * side, np.nan)
    order = {name: i for i, name in enumerate(sorted(names.values()))}
    img[labels.index.to_numpy().astype(int) - 1] = labels.map(order).to_numpy()
    axes[1].imshow(img.reshape(side, side)[::-1], cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    handles = [plt.Line2D([], [], marker="s", ls="", color=cmap(i), label=n) for n, i in order.items()]
    axes[1].legend(handles=handles, fontsize=7, loc="lower left")
    axes[1].set_title("Spatial layout of clusters (grey = below median activity)")
    save(fig, cfg.paths.figures_dir, f"eda_clusters_{activity}")
    labels.to_frame().assign(mean_level=mean_level.loc[labels.index]).to_csv(cfg.paths.tables_dir / "cell_clusters.csv")
    return labels, centroids


def select_cells(hourly: pd.DataFrame, labels: pd.Series, cfg: Config) -> dict[str, int]:
    """Pick representative cells: landmarks from config plus the busiest cell of each cluster."""
    mean_level = hourly.mean()
    selected: dict[str, int] = {}
    for name, cell in dict(cfg.eda.landmarks).items():
        if int(cell) in hourly.columns:
            selected[name] = int(cell)
    n_per = cfg.eda.n_cells_per_cluster
    for cluster in sorted(labels.unique()):
        members = labels[labels == cluster].index
        ranked = mean_level.loc[members].sort_values(ascending=False)
        picked = 0
        for cell in ranked.index:
            if int(cell) in selected.values():
                continue
            key = cluster.replace("/", "_") + ("" if picked == 0 else f"_{picked + 1}")
            selected[key] = int(cell)
            picked += 1
            if picked >= n_per:
                break
    with open(cfg.paths.tables_dir / "selected_cells.json", "w") as fh:
        json.dump({"cells": selected, "clusters": {str(c): labels.get(c, "landmark") for c in selected.values()}}, fh, indent=1)
    log.info("selected cells: %s", selected)
    return selected


def plot_selected_cells(hourly: pd.DataFrame, selected: dict[str, int], cfg: Config, activity: str) -> None:
    setup_style()
    fig, axes = plt.subplots(len(selected), 1, figsize=(11, 1.6 * len(selected)), sharex=True)
    for ax, (name, cell) in zip(np.atleast_1d(axes), selected.items()):
        hourly[cell].plot(ax=ax, lw=0.7)
        ax.set_ylabel(f"{name}\n#{cell}", fontsize=8)
    np.atleast_1d(axes)[0].set_title(f"Hourly {activity} activity in the selected cells")
    save(fig, cfg.paths.figures_dir, f"eda_selected_cells_{activity}")


def summary_statistics(hourly: pd.DataFrame, city: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per-activity totals and share, plus per-cell dispersion of the target activity."""
    acts = list(cfg.data.activities)
    totals = city[acts].sum()
    table = pd.DataFrame({"total": totals, "share_pct": 100 * totals / totals.sum()})
    table.loc["cells_nonzero_target", "total"] = int((hourly.sum() > 0).sum())
    table.loc["cell_mean_cv", "total"] = float((hourly.std() / hourly.mean().replace(0, np.nan)).median())
    table.loc["n_hours", "total"] = len(hourly)
    table.loc["n_missing_hours", "total"] = int(pd.date_range(hourly.index[0], hourly.index[-1], freq="1h").difference(hourly.index).size)
    table.to_csv(cfg.paths.tables_dir / "summary_statistics.csv")
    return table
