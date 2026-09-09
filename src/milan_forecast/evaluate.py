"""Metrics, statistical comparison and experiment logging."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import Config


# --------------------------------------------------------------------------- #
# Point metrics
# --------------------------------------------------------------------------- #
def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def smape(y, p):
    denom = (np.abs(y) + np.abs(p)) / 2
    mask = denom > 0
    return float(100 * np.mean(np.abs(y[mask] - p[mask]) / denom[mask]))


def mase(y, p, scale):
    """MAE scaled by the in-sample MAE of the 24 h seasonal naive forecast (Hyndman & Koehler)."""
    return float(mae(y, p) / scale) if scale > 0 else float("nan")


def seasonal_naive_scale(train_series: pd.Series, season: int = 24) -> float:
    v = train_series.to_numpy()
    return float(np.mean(np.abs(v[season:] - v[:-season])))


def score_predictions(preds: pd.DataFrame, scales: dict[str, float]) -> pd.DataFrame:
    """Aggregate a tidy prediction table into metrics per (model, series, horizon)."""
    rows = []
    for (model, name, h), g in preds.groupby(["model", "series", "horizon"], sort=True):
        y, p = g["y_true"].to_numpy(dtype=float), g["y_pred"].to_numpy(dtype=float)
        rows.append({"model": model, "series": name, "horizon": h, "n": len(g), "mae": mae(y, p),
                     "rmse": rmse(y, p), "smape": smape(y, p), "mase": mase(y, p, scales.get(name, np.nan))})
    return pd.DataFrame(rows)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean over series per (model, horizon), plus an all-horizon row."""
    per_h = metrics.groupby(["model", "horizon"])[["mae", "rmse", "smape", "mase"]].mean().reset_index()
    overall = metrics.groupby("model")[["mae", "rmse", "smape", "mase"]].mean().reset_index().assign(horizon="all")
    return pd.concat([per_h, overall], ignore_index=True)


# --------------------------------------------------------------------------- #
# Diebold-Mariano test
# --------------------------------------------------------------------------- #
def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 1) -> tuple[float, float]:
    """DM statistic and two-sided p-value for H0: equal expected loss.

    ``e1``/``e2`` are forecast errors aligned on the same targets; loss is |e|^power.
    Uses a Newey-West (Bartlett) long-run variance with h-1 lags, and the small-sample
    correction of Harvey, Leybourne and Newbold (1997).
    """
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    mean = d.mean()
    lags = max(h - 1, 0)
    gamma = [np.sum((d[k:] - mean) * (d[:n - k] - mean)) / n for k in range(lags + 1)]
    var = gamma[0] + 2 * sum((1 - k / (lags + 1)) * gamma[k] for k in range(1, lags + 1))
    if var <= 0:
        return float("nan"), float("nan")
    dm = mean / np.sqrt(var / n)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    return float(hln), float(p)


def pairwise_dm(preds: pd.DataFrame, reference: str) -> pd.DataFrame:
    """DM test of every model against ``reference`` for each series and horizon."""
    rows = []
    ref = preds[preds["model"] == reference].set_index(["series", "horizon", "target_time"])
    for model in preds["model"].unique():
        if model == reference:
            continue
        other = preds[preds["model"] == model].set_index(["series", "horizon", "target_time"])
        joined = ref[["y_true", "y_pred"]].join(other[["y_pred"]], rsuffix="_other", how="inner").reset_index()
        for (name, h), g in joined.groupby(["series", "horizon"]):
            e_ref = g["y_true"].to_numpy() - g["y_pred"].to_numpy()
            e_oth = g["y_true"].to_numpy() - g["y_pred_other"].to_numpy()
            stat, p = diebold_mariano(e_oth, e_ref, h=int(h))
            rows.append({"model": model, "reference": reference, "series": name, "horizon": h,
                         "dm_stat": stat, "p_value": p,
                         "better": "model" if stat < 0 else "reference"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Experiment logging
# --------------------------------------------------------------------------- #
class ExperimentLogger:
    """Writes one JSON per run and appends a row to experiments/experiment_log.md.

    The markdown log is the human-readable trail required for the tuning section:
    every row has the parameters, the validation score and the reasoning ("why next").
    """

    def __init__(self, cfg: Config, model: str):
        self.dir = cfg.paths.experiments_dir / "runs" / model
        self.dir.mkdir(parents=True, exist_ok=True)
        self.md = cfg.paths.experiments_dir / "experiment_log.md"
        self.model = model
        self._t0 = time.perf_counter()

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def log(self, run_id: str, params: dict, metrics: dict, note: str = "", extra: dict | None = None) -> Path:
        elapsed = round(time.perf_counter() - self._t0, 1)
        record = {"model": self.model, "run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "params": params, "metrics": metrics, "train_seconds": elapsed, "note": note, **(extra or {})}
        path = self.dir / f"{run_id}.json"
        path.write_text(json.dumps(record, indent=1, default=str))
        new = not self.md.exists()
        with open(self.md, "a") as fh:
            if new:
                fh.write("# Experiment log\n\nOne row per training run. `val_*` metrics are on the validation split; "
                         "the *Reasoning / next step* column records why the following run was configured as it was.\n\n")
                fh.write("| model | run | params | val MAE | val sMAPE | val MASE | s | reasoning / next step |\n")
                fh.write("|---|---|---|---|---|---|---|---|\n")
            p = ", ".join(f"{k}={v}" for k, v in params.items())
            fh.write(f"| {self.model} | {run_id} | {p} | {metrics.get('mae', float('nan')):.3g} | "
                     f"{metrics.get('smape', float('nan')):.2f} | {metrics.get('mase', float('nan')):.3f} | {elapsed} | {note} |\n")
        return path
