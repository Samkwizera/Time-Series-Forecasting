# Forecasting Mobile Network Traffic in Milan

Empirical study of short-term mobile traffic forecasting on the Telecom Italia
"Telecommunications - SMS, Call, Internet - MI" dataset (Milan, 100 x 100 grid,
10-minute intervals, 1 Nov 2013 - 1 Jan 2014, 62 daily files, 20.8 GB).

Research question: *how does forecasting accuracy for hourly Internet activity
differ between a statistical model (SARIMA), a gradient-boosted tree model
(LightGBM) and a recurrent network (LSTM) across grid cells with different
land-use profiles and across 1 h, 6 h and 24 h horizons?*

Repository: https://github.com/Samkwizera/Time-Series-Forecasting

## Repository layout

```
config/          default.yaml (real data) and smoke.yaml (synthetic 20x20 grid for tests)
src/milan_forecast/
  config.py      YAML loading, path resolution
  memory.py      RSS / timing instrumentation written to reports/tables/memory_log.csv
  ingest.py      streaming TSV -> Parquet aggregation, hourly (time x cell) matrices
  synthetic.py   schema-identical synthetic fixture used by the tests
  eda.py, tsa.py, features.py, datasets.py, evaluate.py, plotting.py, models/   (stages 2-6)
scripts/         numbered entry points, run in order (00 -> 07)
notebooks/       thin Colab notebooks that call the scripts
experiments/     experiment_log.md and one JSON per training run
reports/         figures/ and tables/ produced by the scripts (inputs to the report)
report/          LaTeX source of the research report (IEEE style) and references.bib
video/           outline of the 7-10 minute presentation
tests/           pytest suite (runs on the synthetic fixture in < 10 s)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest                      # sanity check on synthetic data
```

## Getting the data

Harvard Dataverse requires a guestbook response before serving the files, so the
download is done once through the browser:

1. Open https://doi.org/10.7910/DVN/EGZHFV, click **Access Dataset > Download ZIP**
   (or select individual days) and complete the guestbook.
2. Unpack the 62 `sms-call-internet-mi-YYYY-MM-DD.txt` files into `data/raw/`
   (on Colab: into a Google Drive folder and set `paths.raw_dir` in `config/default.yaml`).
3. Verify: `python scripts/00_download.py --verify` (add `--md5` for checksums).

With a personal Dataverse API token the same script downloads missing files directly:
`python scripts/00_download.py --token <TOKEN>`.

## Running the pipeline

```bash
python scripts/01_ingest.py                # 20.8 GB TSV -> ~1 GB Parquet -> hourly matrices (< 4 GB RAM)
python scripts/01_ingest.py --max-days 7   # quick development subset
```

Later stages (EDA, time-series analysis, model training, comparison, report) are
documented below as they are added.

### Smoke test without the real data

```bash
MILAN_CONFIG=config/smoke.yaml python scripts/01_ingest.py --synthetic-days 28
```

## Pushing to GitHub

```bash
git remote add github https://github.com/Samkwizera/Time-Series-Forecasting.git
git push -u github main
```
