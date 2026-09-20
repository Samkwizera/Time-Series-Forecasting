#!/usr/bin/env python
"""Stage 4: train one model configuration and store its forecasts.

Every invocation is one experiment: it fits the model, scores it on the chosen
split, writes a tidy prediction table to experiments/predictions/<model>_<run>_<part>.parquet
and appends a row to experiments/experiment_log.md.

Examples
--------
    python scripts/04_train.py --model baselines --part test
    python scripts/04_train.py --model sarima --part val --run s1 --params '{"order": [2,0,1]}' --note "PACF suggests AR(2)"
    python scripts/04_train.py --model lightgbm --part val --run g1 --params '{"lags": "short"}' --train-metrics
    python scripts/04_train.py --model lstm --part val --run l1 --params '{"hidden_size": 128}'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from milan_forecast.config import load_config  # noqa: E402
from milan_forecast.datasets import load_forecast_data  # noqa: E402
from milan_forecast.evaluate import ExperimentLogger, score_predictions, seasonal_naive_scale  # noqa: E402
from milan_forecast.memory import track  # noqa: E402
from milan_forecast.models import baseline, lgbm, rnn, sarima  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", required=True, choices=["baselines", "sarima", "lightgbm", "lstm", "gru"])
    parser.add_argument("--part", default="val", choices=["val", "test"])
    parser.add_argument("--run", default="default", help="short run id used in file names and the log")
    parser.add_argument("--params", default="{}",
                        help="JSON dict of model parameters overriding the defaults, or @path/to/params.json "
                             "(useful on Windows shells that strip quotes)")
    parser.add_argument("--note", default="", help="reasoning recorded in the experiment log")
    parser.add_argument("--observed", default="",
                        help="what this run's numbers showed, recorded in the log next to the reasoning")
    parser.add_argument("--train-metrics", action="store_true",
                        help="also score the model in-sample, so the log carries the train/val gap "
                             "(validation runs only: the in-sample fit is on the train split)")
    parser.add_argument("--series", nargs="*", default=None, help="restrict to these selected-cell names")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    warnings.filterwarnings("ignore", category=UserWarning)
    cfg = load_config(args.config)
    paths = cfg.paths.ensure()
    data = load_forecast_data(cfg, args.series)
    raw = args.params
    if raw.startswith("@"):
        raw = Path(raw[1:]).read_text()
    params = json.loads(raw)
    pred_dir = paths.experiments_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    labels = data.split.label(data.index)
    scales = {n: seasonal_naive_scale(s[labels == "train"]) for n, s in data.series.items()}

    def run_part(part: str) -> tuple[pd.DataFrame, dict]:
        if args.model == "baselines":
            return pd.concat([baseline.forecast(data, part, m) for m in ("naive", "seasonal_naive", "weekly_naive")]), {}
        if args.model == "sarima":
            return sarima.forecast(data, part, params)
        if args.model == "lightgbm":
            return lgbm.forecast(data, part, params, seed=cfg.forecast.seed)
        return rnn.forecast(data, part, params, seed=cfg.forecast.seed)

    if args.model in ("lstm", "gru"):
        params = {"cell": args.model, **params}

    logger = ExperimentLogger(cfg, args.model)
    with track(f"train:{args.model}:{args.run}:{args.part}", paths.tables_dir / "memory_log.csv") as ctx:
        preds, info = run_part(args.part)
        ctx["sample"]()

    metrics = score_predictions(preds, scales)
    overall = metrics.groupby("model")[["mae", "rmse", "smape", "mase"]].mean()
    logging.info("%s metrics on %s:\n%s", args.model, args.part, metrics.to_string(index=False, float_format="%.3f"))
    logging.info("mean over series and horizons:\n%s", overall.to_string(float_format="%.3f"))

    preds.to_parquet(pred_dir / f"{args.model}_{args.run}_{args.part}.parquet", index=False)
    metrics.to_csv(pred_dir / f"{args.model}_{args.run}_{args.part}_metrics.csv", index=False)
    train_mase = None
    if args.train_metrics and args.part != "val":
        # a test run fits on train+val, so scoring it on train alone would compare two
        # different fits and the gap would mean nothing
        logging.warning("--train-metrics only applies to validation runs; skipping the in-sample pass")
    elif args.train_metrics:
        # refit on the same window and score in-sample: the gap against the validation
        # MASE below is what tells overfitting apart from an underpowered feature set
        train_preds, _ = run_part("train")
        train_mase = float(score_predictions(train_preds, scales).groupby("model")["mase"].mean().iloc[0])
        logging.info("in-sample MASE %.3f vs %s MASE %.3f", train_mase, args.part, overall["mase"].iloc[0])

    if args.model != "baselines":
        summary = overall.iloc[0].to_dict()
        per_h = metrics.groupby("horizon")["mase"].mean().round(3).to_dict()
        defaults = {"sarima": sarima, "lightgbm": lgbm}.get(args.model, rnn).DEFAULT_PARAMS
        # log the full effective config, not just the overrides, so runs are comparable later
        effective = {**defaults, **params}
        logger.log(f"{args.run}_{args.part}", effective, summary, note=args.note, observed=args.observed,
                   train_mase=train_mase,
                   extra={"part": args.part, "mase_by_horizon": per_h, "info": info})


if __name__ == "__main__":
    main()
