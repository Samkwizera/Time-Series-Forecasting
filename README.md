# Forecasting Mobile Network Traffic in Milan

Empirical study of short-term mobile traffic forecasting on the Telecom Italia
"Telecommunications - SMS, Call, Internet - MI" dataset (Milan, 100 x 100 grid,
10-minute intervals, 1 Nov 2013 - 1 Jan 2014, 62 daily files, 20.8 GB).

Research question: *how does forecasting accuracy for hourly Internet activity
differ between a statistical model (SARIMAX), a gradient-boosted tree model
(LightGBM) and a recurrent network (LSTM/GRU) across grid cells with different
activity-profile labels and across 1 h, 6 h and 24 h horizons?* The labels are
inferred from traffic profiles; they are not verified land-use categories.

Repository: https://github.com/Samkwizera/Time-Series-Forecasting

## Headline result

On the test window (22 Dec - 1 Jan, deliberately a holiday stress test) the **24-hour
seasonal naive is the best method overall** (mean MASE 0.863). SARIMAX (1.427) and
LightGBM (1.432) are indistinguishable in aggregate; the GRU is clearly worst (2.223).
SARIMAX is the strongest method in the study at the 1-hour horizon (0.399). The reason
is that a holiday fortnight is mainly a level shift, and the seasonal naive re-anchors
on yesterday's observed value while the learned models hold a level estimated from six
ordinary weeks and over-forecast. See the report's *Why the seasonal naive wins* section.

Validation selected SARIMAX `(1,0,1)(1,1,1)_24` **without** Fourier or holiday
regressors, LightGBM run `g3` (full lag set + calendar, no rolling features), and a
one-layer GRU (run `l5`) - the last by a margin too small to distinguish it from the
comparable LSTM on a single seed.

## Repository layout

```
config/            default.yaml (real data), smoke.yaml (synthetic 20x20 grid for tests and CI)
src/milan_forecast/
  config.py        YAML loading and path resolution
  memory.py        peak-RSS / timing instrumentation -> reports/tables/memory_log.csv
  ingest.py        streaming TSV -> per-day Parquet -> hourly (time x cell) matrices
  synthetic.py     schema-identical synthetic fixture (tests only, never for results)
  eda.py           temporal / spatial analysis, k-means cell typing, cell selection
  tsa.py           ADF/KPSS, STL/MSTL, ACF/PACF
  datasets.py      splits, rolling origins, log1p scaler, tidy prediction tables
  features.py      lag / rolling / calendar features relative to the forecast origin
  evaluate.py      MAE, RMSE, sMAPE, MASE, Diebold-Mariano test, experiment logger
  analysis.py      comparative plots, error-over-time, residual and worst-case analysis
  models/          baseline.py, sarima.py, lgbm.py, rnn.py (LSTM/GRU), _common.py (fit/eval origins)
scripts/           00_download 01_ingest 02_eda 03_tsa 04_train 05_tune 06_compare 07_report_assets
run_all.sh         the whole pipeline in one command (bash); run_all.ps1 is the PowerShell equivalent
notebooks/         pipeline.ipynb (clean) and pipeline_executed.ipynb (completed Kaggle run)
experiments/       tuning_plan.yaml, final_params/*.json, experiment_log.md, runs/*.json, predictions/
reports/           figures/ and tables/ written by the scripts (inputs to the report)
report/            report.tex (IEEE), references.bib, build.sh -> report.pdf
video/outline.md   7-10 minute presentation plan tied to the figures
tests/             pytest: ingestion on a synthetic fixture, and the forecasting layer
                   (split leakage, features, metrics, DM test, all four model modules)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: python -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest                                                   # ~1.5 min, no real data needed
```

Tested with Python 3.10-3.14. CPU-only PyTorch is enough for the LSTM experiments; on a
GPU machine install the matching CUDA wheel and the code picks it up automatically.

## Getting the data

Harvard Dataverse requires a guestbook response before serving this dataset, so
the download happens once through a browser (or use a Kaggle mirror, see below):

1. Open https://doi.org/10.7910/DVN/EGZHFV, **Access Dataset > Download ZIP**, complete the guestbook.
2. Unpack the 62 `sms-call-internet-mi-YYYY-MM-DD.txt` files into `data/raw/`.
3. `python scripts/00_download.py --verify` checks presence and sizes against the Dataverse manifest
   (`--md5` also verifies checksums; with a Dataverse API token `--token` downloads missing files).

The exact size is 20,804,803,507 bytes = 20.8 GB = 19.4 GiB (Dataverse displays GiB).

## Running the pipeline

```bash
./run_all.sh                     # everything, real data; ~1 h ingestion + tuning on CPU
.\run_all.ps1                    # same thing from PowerShell
```

How each model is fitted and evaluated (shared protocol, see `src/milan_forecast/models/__init__.py`):
`--part val` fits on the training split and forecasts every origin whose targets fall in
the validation split; `--part test` fits on train+validation and forecasts the test split.
Early stopping (LightGBM trees, LSTM epochs) uses the last week of the fitting window,
never the split being evaluated.

or stage by stage:

```bash
python scripts/01_ingest.py                        # TSV -> Parquet -> hourly matrices (<4 GB RAM)
python scripts/02_eda.py                           # figures + cell clustering + selected_cells.json
python scripts/03_tsa.py                           # stationarity, decomposition, autocorrelation
python scripts/04_train.py --model baselines --part test
python scripts/05_tune.py --model sarima           # rounds from experiments/tuning_plan.yaml
python scripts/05_tune.py --model lightgbm --optuna 20
python scripts/05_tune.py --model lstm
python scripts/04_train.py --model sarima --part test --run s2 --params @experiments/final_params/sarima_s2.json
python scripts/04_train.py --model lightgbm --part test --run g3 --params @experiments/final_params/lightgbm_g3.json
python scripts/04_train.py --model gru --part test --run l5 --params @experiments/final_params/gru_l5.json
python scripts/06_compare.py --runs sarima=s2 lightgbm=g3 gru=l5
python scripts/07_report_assets.py && report/build.sh
```

Every training run appends a row (parameters, validation metrics, reasoning) to
`experiments/experiment_log.md` and writes `experiments/runs/<model>/<run>.json`.
To continue the iterative loop, add a round to `experiments/tuning_plan.yaml`
with its `why` and re-run `05_tune.py --rounds <id>`.

### Kaggle / Colab

Open `notebooks/pipeline.ipynb`. On Kaggle attach a dataset
containing the 62 raw files (there are public mirrors of the Telecom Italia
Milan data; check it holds all 62 days, some only contain the first week) and set
`RAW_DIR`; on Colab mount Drive. Kaggle's ~30 GB RAM and free GPU quota make it
the more comfortable option. The notebook ends by zipping `reports/` and
`experiments/` for download.

### Smoke test without the real data

```bash
MILAN_CONFIG=config/smoke.yaml ./run_all.sh 28                  # synthetic 20x20 grid, 28 days
$env:MILAN_CONFIG="config/smoke.yaml"; .\run_all.ps1 28         # PowerShell
```

Expect roughly 25 min on a laptop CPU (SARIMA rolling forecasts and the LSTM rounds dominate).

Outputs go to `reports/smoke/` and `experiments/smoke/` (git-ignored). The
synthetic data only exercises the code path; no number from it belongs in the report.

## Building the report

```bash
report/build.sh            # needs tectonic or TeX Live; reads reports/tables/* via scripts/07_report_assets.py
```

On Windows without bash: `python scripts/07_report_assets.py` then `tectonic report/report.tex`
(install tectonic with `winget install tectonic` or use Overleaf with the `report/` folder).

The report never hard-codes a result: `scripts/07_report_assets.py` turns the CSVs
into `report/generated/*.tex` (tables and `\newcommand` macros). Passages wrapped
in `\revise{}` can be printed in red during drafting by setting `\revisemode` to 1
in `report.tex`. The committed submission setting is 0.

## Reproducibility notes

- Seeds are fixed (`forecast.seed`), splits are date-based, and every stage skips
  outputs that already exist (`--force` recomputes).
- Peak memory and wall time of every stage are appended to `reports/tables/memory_log.csv`.
- Time zone handling: raw timestamps are UTC milliseconds; they are converted to
  Europe/Rome at ingestion, and the hourly matrix is re-aggregated by timestamp
  because daily files are cut at UTC midnight.

## Pushing to GitHub

```bash
git remote add github https://github.com/Samkwizera/Time-Series-Forecasting.git
git push -u github main
```
