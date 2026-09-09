#!/usr/bin/env python
"""Stage 6: comparative evaluation and failure analysis of the final models.

    python scripts/06_compare.py --runs sarima=s3 lightgbm=g4 lstm=l5

The run ids refer to the test-split runs produced by scripts/04_train.py; the three
baselines are always included.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milan_forecast import analysis  # noqa: E402
from milan_forecast.config import load_config  # noqa: E402
from milan_forecast.datasets import load_forecast_data  # noqa: E402
from milan_forecast.evaluate import seasonal_naive_scale  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--runs", nargs="+", default=[], help="model=run pairs, e.g. sarima=s3 lightgbm=g4 lstm=l5")
    parser.add_argument("--part", default="test")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    cfg.paths.ensure()

    runs = {"baselines": "default"}
    for pair in args.runs:
        model, run = pair.split("=")
        runs[model] = run
    preds = analysis.load_final_predictions(cfg, runs, args.part)

    data = load_forecast_data(cfg)
    labels = data.split.label(data.index)
    scales = {n: seasonal_naive_scale(s[labels == "train"]) for n, s in data.series.items()}

    metrics = analysis.metrics_tables(preds, scales, cfg)
    analysis.plot_metric_by_horizon(metrics, cfg, "mase")
    analysis.plot_metric_by_horizon(metrics, cfg, "smape")
    for h in sorted(preds["horizon"].unique()):
        analysis.plot_forecasts(preds, cfg, horizon=int(h))
    analysis.error_over_time(preds, cfg)
    analysis.error_by_hour_and_daytype(preds, cfg)
    analysis.worst_cases(preds, cfg)
    analysis.residual_diagnostics(preds, cfg, horizon=1)
    dm = analysis.dm_tests(preds, cfg, reference="seasonal_naive")
    learned = [m for m in preds["model"].unique() if m not in ("naive", "seasonal_naive", "weekly_naive")]
    if "lightgbm" in learned:
        analysis.dm_tests(preds, cfg, reference="lightgbm")
    cost = analysis.compute_cost_table(cfg)

    summary = metrics.groupby(["model", "horizon"])[["mae", "smape", "mase"]].mean()
    logging.info("results summary:\n%s", summary.to_string(float_format="%.3f"))
    logging.info("DM vs seasonal naive (p<0.05 count by model):\n%s",
                 dm[dm["p_value"] < 0.05].groupby(["model", "better"]).size().to_string())
    if len(cost):
        logging.info("compute cost:\n%s", cost.to_string(index=False))


if __name__ == "__main__":
    main()
