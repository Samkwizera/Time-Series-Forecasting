#!/usr/bin/env python
"""Stage 5: run the documented tuning rounds and, optionally, an Optuna search for LightGBM.

    python scripts/05_tune.py --model sarima
    python scripts/05_tune.py --model lightgbm --optuna 30
    python scripts/05_tune.py --model lstm --rounds l2 l3

Each round from experiments/tuning_plan.yaml is executed on the validation split
through scripts/04_train.py, so every run is logged exactly like a manual one.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from milan_forecast.config import load_config  # noqa: E402


def run_round(model: str, run: str, params: dict, why: str, config: str | None, extra: list[str]) -> float:
    cmd = [sys.executable, str(ROOT / "scripts" / "04_train.py"), "--model", model, "--part", "val",
           "--run", run, "--params", json.dumps(params), "--note", why] + extra
    if config:
        cmd += ["--config", config]
    logging.info("round %s: %s", run, params)
    subprocess.run(cmd, check=True)
    cfg = load_config(config)
    record = json.loads((cfg.paths.experiments_dir / "runs" / model / f"{run}_val.json").read_text())
    mase = record["metrics"]["mase"]
    logging.info("round %s -> val MASE %.3f", run, mase)
    return mase


def optuna_search(model: str, n_trials: int, config: str | None, extra: list[str], base: dict) -> None:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {**base,
                  "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
                  "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                  "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
                  "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                  "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True)}
        return run_round(model, f"opt{trial.number:02d}", params,
                         f"Optuna trial {trial.number} (TPE sampler over leaves, lr, min_child, feature_fraction, l2)",
                         config, extra)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    cfg = load_config(config)
    out = cfg.paths.experiments_dir / f"optuna_{model}.json"
    out.write_text(json.dumps({"best_value": study.best_value, "best_params": study.best_params,
                               "trials": [{"number": t.number, "value": t.value, "params": t.params}
                                          for t in study.trials]}, indent=1))
    logging.info("best trial: MASE %.3f with %s", study.best_value, study.best_params)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", required=True, choices=["sarima", "lightgbm", "lstm"])
    parser.add_argument("--rounds", nargs="*", default=None, help="subset of round ids to run")
    parser.add_argument("--optuna", type=int, default=0, help="number of Optuna trials (lightgbm only)")
    parser.add_argument("--plan", default=str(ROOT / "experiments" / "tuning_plan.yaml"))
    parser.add_argument("--series", nargs="*", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    extra = ["--series", *args.series] if args.series else []
    plan = yaml.safe_load(open(args.plan))[args.model]
    results = {}
    for rnd in plan:
        if args.rounds and rnd["run"] not in args.rounds:
            continue
        results[rnd["run"]] = run_round(args.model, rnd["run"], rnd["params"], rnd["why"], args.config, extra)
    if results:
        best = min(results, key=results.get)
        logging.info("manual rounds: %s -> best %s", {k: round(v, 3) for k, v in results.items()}, best)
    if args.optuna and args.model == "lightgbm":
        base = {"lags": "full", "use_calendar": True, "use_rolling": True}
        optuna_search(args.model, args.optuna, args.config, extra, base)


if __name__ == "__main__":
    main()
