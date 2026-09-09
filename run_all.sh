#!/usr/bin/env bash
# Full reproduction: ingest -> EDA -> TSA -> baselines -> tuning rounds -> final test runs -> comparison.
# Usage:
#   ./run_all.sh                                   # real data in data/raw
#   MILAN_CONFIG=config/smoke.yaml ./run_all.sh 28 # synthetic smoke run with 28 days
# Final run ids can be overridden, e.g. SARIMA_RUN=s3 LGBM_RUN=opt12 LSTM_RUN=l4 ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

SYN_DAYS="${1:-0}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-20}"
PY="${PYTHON:-python}"

if [ "$SYN_DAYS" != "0" ]; then
  $PY scripts/01_ingest.py --synthetic-days "$SYN_DAYS"
else
  $PY scripts/00_download.py --verify
  $PY scripts/01_ingest.py
fi
$PY scripts/02_eda.py
$PY scripts/03_tsa.py

$PY scripts/04_train.py --model baselines --part val
$PY scripts/04_train.py --model baselines --part test

$PY scripts/05_tune.py --model sarima
$PY scripts/05_tune.py --model lightgbm --optuna "$OPTUNA_TRIALS"
$PY scripts/05_tune.py --model lstm

# pick the best validation run of each model unless overridden
best_run() {
  $PY - "$1" <<'EOF'
import json, sys, glob, os
model = sys.argv[1]
cfg_path = os.environ.get("MILAN_CONFIG", "config/default.yaml")
import yaml
exp_dir = yaml.safe_load(open(cfg_path))["paths"]["experiments_dir"]
runs = [json.load(open(p)) for p in glob.glob(f"{exp_dir}/runs/{model}/*_val.json")]
best = min(runs, key=lambda r: r["metrics"]["mase"])
print(best["run_id"].removesuffix("_val"))
EOF
}
SARIMA_RUN="${SARIMA_RUN:-$(best_run sarima)}"
LGBM_RUN="${LGBM_RUN:-$(best_run lightgbm)}"
LSTM_RUN="${LSTM_RUN:-$(best_run lstm)}"
echo "final runs: sarima=$SARIMA_RUN lightgbm=$LGBM_RUN lstm=$LSTM_RUN"

params_of() {
  $PY - "$1" "$2" <<'EOF'
import json, sys, os, yaml
model, run = sys.argv[1], sys.argv[2]
cfg_path = os.environ.get("MILAN_CONFIG", "config/default.yaml")
exp_dir = yaml.safe_load(open(cfg_path))["paths"]["experiments_dir"]
print(json.dumps(json.load(open(f"{exp_dir}/runs/{model}/{run}_val.json"))["params"]))
EOF
}
$PY scripts/04_train.py --model sarima   --part test --run "$SARIMA_RUN" --params "$(params_of sarima "$SARIMA_RUN")" --note "final test run of best validation config"
$PY scripts/04_train.py --model lightgbm --part test --run "$LGBM_RUN"   --params "$(params_of lightgbm "$LGBM_RUN")" --note "final test run of best validation config"
LSTM_PARAMS="$(params_of lstm "$LSTM_RUN")"
LSTM_CELL="$($PY -c "import json,sys; print(json.loads(sys.argv[1]).get('cell','lstm'))" "$LSTM_PARAMS")"
$PY scripts/04_train.py --model "$LSTM_CELL" --part test --run "$LSTM_RUN" --params "$LSTM_PARAMS" --note "final test run of best validation config"

$PY scripts/06_compare.py --runs sarima="$SARIMA_RUN" lightgbm="$LGBM_RUN" "$LSTM_CELL"="$LSTM_RUN"
echo "done: figures in reports/figures, tables in reports/tables, log in experiments/experiment_log.md"
