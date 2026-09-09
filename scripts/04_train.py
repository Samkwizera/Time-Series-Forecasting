#!/usr/bin/env python
"""Stage 4: train one model configuration and store its forecasts.

Every invocation is one experiment: it fits the model, scores it on the chosen
split, writes a tidy prediction table to experiments/predictions/<model>_<run>_<part>.parquet
and appends a row to experiments/experiment_log.md.

Examples
--------
    python scripts/04_train.py --model baselines --part test
    python scripts/04_train.py --model sarima --part val --run s1 --params '{"order": [2,0,1]}' --note "PACF suggests AR(2)"
    python scripts/04_train.py --model lightgbm --part val --run g1 --params '{"lags": "short"}'
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
    parser.add_argument("--params", default="{}", help="JSON dict of model parameters overriding the defaults")
    parser.add_argument("--note", default="", help="reasoning recorded in the experiment log")
    parser.add_argument("--series", nargs="*", default=None, help="restrict to these selected-cell names")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    warnings.filterwarnings("ignore", category=UserWarning)
    cfg = load_config(args.config)
    paths = cfg.paths.ensure()
    data = load_forecast_data(cfg, args.series)
    params = json.loads(args.params)
    pred_dir = paths.experiments_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    labels = data.split.label(data.index)
    scales = {n: seasonal_naive_scale(s[labels == "train"]) for n, s in data.series.items()}

    logger = ExperimentLogger(cfg, args.model)
    with track(f"train:{args.model}:{args.run}:{args.part}", paths.tables_dir / "memory_log.csv") as ctx:
        info: dict = {}
        if args.model == "baselines":
            preds = pd.concat([baseline.forecast(data, args.part, m) for m in ("naive", "seasonal_naive", "weekly_naive")])
        elif args.model == "sarima":
            preds, info = sarima.forecast(data, args.part, params)
        elif args.model == "lightgbm":
            preds, info = lgbm.forecast(data, args.part, params, seed=cfg.forecast.seed)
        else:
            params = {"cell": args.model, **params}
            preds, info = rnn.forecast(data, args.part, params, seed=cfg.forecast.seed)
        ctx["sample"]()

    metrics = score_predictions(preds, scales)
    overall = metrics.groupby("model")[["mae", "rmse", "smape", "mase"]].mean()
    logging.info("%s metrics on %s:\n%s", args.model, args.part, metrics.to_string(index=False, float_format="%.3f"))
    logging.info("mean over series and horizons:\n%s", overall.to_string(float_format="%.3f"))

    preds.to_parquet(pred_dir / f"{args.model}_{args.run}_{args.part}.parquet", index=False)
    metrics.to_csv(pred_dir / f"{args.model}_{args.run}_{args.part}_metrics.csv", index=False)
    if args.model != "baselines":
        summary = overall.iloc[0].to_dict()
        per_h = metrics.groupby("horizon")["mase"].mean().round(3).to_dict()
        defaults = {"sarima": sarima, "lightgbm": lgbm}.get(args.model, rnn).DEFAULT_PARAMS
        # log the full effective config, not just the overrides, so runs are comparable later
        effective = {**defaults, **params}
        logger.log(f"{args.run}_{args.part}", effective, summary, note=args.note,
                   extra={"part": args.part, "mase_by_horizon": per_h, "info": info})


if __name__ == "__main__":
    main()
