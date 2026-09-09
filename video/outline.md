# Video presentation outline (7-10 minutes)

Record with the repository open on one side and the figures from `reports/figures/` on the other.
Every claim below should be shown on screen with the file that supports it. Fill in the numbers
from `reports/tables/results_summary.csv` and `experiments/experiment_log.md` after the real run.

## 0:00 - 0:45  Problem and question
- Operators dimension cells for peak load; forecasting the next hours lets them pre-allocate capacity and sleep idle carriers.
- Cells differ by land use (office, residential, nightlife). Question: does the best forecasting model depend on cell type and horizon?
- Show `README.md` header with the research question.

## 0:45 - 2:15  Data and memory management (show `src/milan_forecast/ingest.py`)
- Telecom Italia Milan: 100x100 grid, 10-min intervals, 62 daily files, 20.8 GB.
- Naive `pandas.read_csv` estimate vs. actual peak RSS: read the two rows of `reports/tables/memory_log.csv`
  and `naive_cost_estimate.csv` on screen.
- Three decisions: stream + sum over country_code early (drops ~90 % of rows), downcast to float32/uint16,
  persist per-day Parquet and a wide hourly matrix. The raw TSVs are read exactly once.
- Mention the Polars streaming engine and the pandas chunked fallback (`aggregate_day_pandas`), and why the
  UTC/local-time boundary needs the second groupby (`build_wide`).

## 2:15 - 3:45  What the data looks like (show EDA figures)
- `eda_citywide_hourly.png`: daily and weekly cycles, holiday shading, December decline.
- `eda_spatial_internet.png` + `eda_concentration_internet.png`: extreme concentration (quote Gini and top-10 % share).
- `eda_clusters_internet.png`: k-means on normalised daily profiles gives the cell types; explain how the
  labels were assigned from centroid shape and that they are descriptive, not ground truth.
- `tsa_acf_citywide.png` + stationarity table: seasonal difference at 24 makes the series stationary;
  ACF peaks at 24k and 168 -> this fixes SARIMA orders, the LightGBM lag set and the LSTM window.

## 3:45 - 5:15  Models and why (show `src/milan_forecast/models/`)
- Common protocol: log1p, chronological split, rolling origins, horizons 1/6/24 h, same calendar info to every model.
- SARIMAX: period-24 seasonal ARIMA + weekly Fourier regressors + holiday dummy. Why not period 168.
  Rolling forecasts with fixed parameters via `results.apply` (an operator refitting weekly).
- LightGBM: one model per horizon, pooled cells with a categorical id, l1 objective, features relative to the origin.
- LSTM: one-week window, cyclical calendar inputs, cell embedding, direct multi-horizon head.
- Baselines: naive, seasonal naive 24 h and 168 h; MASE uses the 24 h seasonal naive as scale.

## 5:15 - 6:30  Iterative tuning (show `experiments/tuning_plan.yaml` and `experiment_log.md`)
- Walk through one chain, e.g. SARIMA s1 -> s2 -> s3: what the ACF suggested, what each change did to validation MASE.
- LightGBM: manual feature-group ablation first (lags -> seasonal lags -> calendar -> rolling), then Optuna for capacity.
- LSTM: window 24 -> 168 was the decisive change; capacity increases did not help with six weeks of data.
- One important technical decision to discuss explicitly: putting the test window on Christmas/New Year, and the
  consequence that validation (an ordinary week) cannot reward holiday handling.

## 6:30 - 8:30  Results and failure case (show `results_*` and `failure_*` figures)
- `results_mase_by_horizon_and_cell.png` and the summary table: ranking by horizon and by cell type;
  DM test counts against seasonal naive.
- `results_forecasts_h1.png`: qualitative fit per cell.
- Failure case: `failure_daily_error.png` - all models break on 24-26 Dec and 31 Dec / 1 Jan; explain why
  (three holidays in training, holiday dummy cannot calibrate the drop). `failure_residuals_h1.png` -
  under-forecast of peaks by SARIMAX/LSTM vs. unbiased but noisier LightGBM.
- Compute cost table: time and memory per model.

## 8:30 - 9:30  Limitations, conclusion, reproducibility
- Limitations: few cells, one activity, two months / three holidays, descriptive cluster labels.
- Conclusion in two sentences answering the research question.
- Show `run_all.sh` and the notebook: one command reproduces everything; tests run on a synthetic fixture.
- AI-use disclosure in one sentence, consistent with the report.
