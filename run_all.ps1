# Full reproduction on Windows / PowerShell; mirrors run_all.sh stage for stage.
#   .\run_all.ps1                                          # real data in data/raw
#   $env:MILAN_CONFIG="config/smoke.yaml"; .\run_all.ps1 28  # synthetic smoke run with 28 days
# Final run ids can be overridden with $env:SARIMA_RUN, $env:LGBM_RUN, $env:LSTM_RUN.
param([int]$SynDays = 0)
# Not "Stop": Python logs to stderr, which PowerShell 5.1 would otherwise treat as a terminating error.
# Failures are caught through the exit code in Run instead.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$optunaTrials = if ($env:OPTUNA_TRIALS) { $env:OPTUNA_TRIALS } else { "20" }
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }

function Run { param([string[]]$CmdArgs)
    & $py @CmdArgs
    if ($LASTEXITCODE -ne 0) { throw "failed: $py $($CmdArgs -join ' ')" }
}

if ($SynDays -ne 0) {
    Run @("scripts/01_ingest.py", "--synthetic-days", "$SynDays")
} else {
    Run @("scripts/00_download.py", "--verify")
    Run @("scripts/01_ingest.py")
}
Run @("scripts/02_eda.py")
Run @("scripts/03_tsa.py")

Run @("scripts/04_train.py", "--model", "baselines", "--part", "val")
Run @("scripts/04_train.py", "--model", "baselines", "--part", "test")

Run @("scripts/05_tune.py", "--model", "sarima")
Run @("scripts/05_tune.py", "--model", "lightgbm", "--optuna", $optunaTrials)
Run @("scripts/05_tune.py", "--model", "lstm")

# pick the best validation run of each model unless overridden
$helper = @"
import json, sys, glob, os, yaml
what, model = sys.argv[1], sys.argv[2]
cfg_path = os.environ.get("MILAN_CONFIG", "config/default.yaml")
exp_dir = yaml.safe_load(open(cfg_path))["paths"]["experiments_dir"]
if what == "best":
    runs = [json.load(open(p)) for p in glob.glob(f"{exp_dir}/runs/{model}/*_val.json")]
    print(min(runs, key=lambda r: r["metrics"]["mase"])["run_id"].removesuffix("_val"))
else:
    print(json.dumps(json.load(open(f"{exp_dir}/runs/{model}/{sys.argv[3]}_val.json"))["params"]))
"@
$helperPath = Join-Path $env:TEMP "milan_run_helper.py"
Set-Content -Path $helperPath -Value $helper

function Best($model) { (& $py $helperPath best $model).Trim() }
# PowerShell 5.1 strips the quotes of inline JSON, so parameters travel through a file (@path)
function ParamsOf($model, $run) {
    $path = Join-Path $env:TEMP "milan_params_${model}_${run}.json"
    (& $py $helperPath params $model $run).Trim() | Set-Content -Path $path -NoNewline
    return "@$path"
}

$sarimaRun = if ($env:SARIMA_RUN) { $env:SARIMA_RUN } else { Best "sarima" }
$lgbmRun   = if ($env:LGBM_RUN)   { $env:LGBM_RUN }   else { Best "lightgbm" }
$lstmRun   = if ($env:LSTM_RUN)   { $env:LSTM_RUN }   else { Best "lstm" }
Write-Host "final runs: sarima=$sarimaRun lightgbm=$lgbmRun lstm=$lstmRun"

$note = "final test run of best validation config"
Run @("scripts/04_train.py", "--model", "sarima",   "--part", "test", "--run", $sarimaRun, "--params", (ParamsOf "sarima" $sarimaRun), "--note", $note)
Run @("scripts/04_train.py", "--model", "lightgbm", "--part", "test", "--run", $lgbmRun,   "--params", (ParamsOf "lightgbm" $lgbmRun), "--note", $note)
$lstmParams = ParamsOf "lstm" $lstmRun
$lstmCell = (Get-Content $lstmParams.Substring(1) -Raw | ConvertFrom-Json).cell
if (-not $lstmCell) { $lstmCell = "lstm" }
Run @("scripts/04_train.py", "--model", $lstmCell, "--part", "test", "--run", $lstmRun, "--params", $lstmParams, "--note", $note)

Run @("scripts/06_compare.py", "--runs", "sarima=$sarimaRun", "lightgbm=$lgbmRun", "$lstmCell=$lstmRun")
Run @("scripts/07_report_assets.py")
Write-Host "done: figures in reports/figures, tables in reports/tables, log in experiments/experiment_log.md"
