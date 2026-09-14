# Video presentation outline (7-10 minutes)

Record with the repository open on one side and the figures from `reports/figures/` on the other.
Every claim below should be shown on screen with the file that supports it. The figures and
numbers below come from the completed Kaggle run.

## 0:00 - 0:45  Problem and question
- Operators dimension cells for peak load; forecasting the next hours lets them pre-allocate capacity and sleep idle carriers.
- Cells differ by land use (office, residential, nightlife). Question: does the best forecasting model depend on cell type and horizon?
- Show `README.md` header with the research question.

## 0:45 - 2:15  Data and memory management (show `src/milan_forecast/ingest.py`)
- Telecom Italia Milan: 100x100 grid, 10-min intervals, 62 daily files, 20.8 GB.
- A naive full-period `pandas` load is estimated at 16.7 GB. The daily streaming loop peaked at
  1,551 MB, while the citywide 10-minute aggregation reached the pipeline maximum of 5,244 MB.
  Show `reports/tables/memory_log.csv` and `naive_cost_estimate.csv`.
- Three decisions: stream + sum over country_code early (drops ~90 % of rows), downcast to float32/uint16,
  persist per-day Parquet and a wide hourly matrix. The raw TSVs are read exactly once.
- Mention the Polars streaming engine and the pandas chunked fallback (`aggregate_day_pandas`), and why the
  UTC/local-time boundary needs the second groupby (`build_wide`).

## 2:15 - 3:45  What the data looks like (show EDA figures)
- `eda_citywide_hourly.png`: daily and weekly cycles, holiday shading, December decline.
- `eda_spatial_internet.png` + `eda_concentration_internet.png`: extreme concentration (quote Gini and top-10 % share).
- `eda_clusters_internet.png`: the silhouette score favours two clusters (0.47), but the analysis retains
  four (0.28) to separate two business-shaped and two mixed/suburban-shaped profiles. Explain that this is
  an interpretability choice and that the labels are descriptive, not ground truth.
- `tsa_acf_citywide.png` + stationarity table: seasonal difference at 24 makes the series stationary;
  ACF peaks at 24k and 168 motivate the SARIMA candidates, the LightGBM lag set and a one-week RNN candidate.

## 3:45 - 5:15  Models and why (show `src/milan_forecast/models/`)
- Common protocol: log1p, chronological split, rolling origins and horizons of 1/6/24 h.
- SARIMAX: explain why weekly Fourier and holiday regressors were tested, then show that validation selected
  s2, the period-24 model without either regressor. Mention the fixed-parameter rolling forecasts.
- LightGBM: one model per horizon, pooled cells with a categorical id, l1 objective, features relative to the origin.
- Recurrent family: LSTM and GRU candidates with cyclical calendar inputs, a cell embedding and a direct
  multi-horizon head. Validation selected the one-layer, 64-unit GRU with a one-week window.
- Baselines: naive, seasonal naive 24 h and 168 h; MASE uses the 24 h seasonal naive as scale.

## 5:15 - 6:30  Iterative tuning (show `experiments/tuning_plan.yaml` and `experiment_log.md`)
- Walk through SARIMAX s1 (0.717) -> s2 (0.659) -> s3 (0.666): the seasonal MA term helped, but weekly
  Fourier regressors did not improve validation MASE, so s2 was selected.
- LightGBM: calendar-aware g3 was best at 0.604; rolling features worsened the score to 0.664, and the best
  Optuna trial reached only 0.633.
- Recurrent models: the 168-hour LSTM was worse than the 24-hour model. A larger LSTM recovered, and the GRU
  was narrowly best in the family at 0.711.
- One important technical decision to discuss explicitly: putting the test window on Christmas/New Year, and the
  consequence that validation (an ordinary week) cannot reward holiday handling.

## 6:30 - 8:30  Results and failure case (show `results_*` and `failure_*` figures)
- `results_mase_by_horizon_and_cell.png` and the summary table: the 24-hour seasonal naive is best overall
  (MASE 0.863). SARIMAX is the best learned model (1.427), narrowly ahead of LightGBM (1.432); the GRU is 2.223.
- Explain the horizon effect: SARIMAX is the strongest learned model at 1 and 6 hours, while LightGBM is the
  strongest learned model at 24 hours. Do not claim that the seven-cell sample proves a land-use effect.
- DM tests mostly favour seasonal naive: SARIMAX is significantly better in 7 of 21 comparisons and worse
  in 10; LightGBM is better in 2 and worse in 14; the GRU is better in 2 and worse in 15.
- `results_forecasts_h1.png`: qualitative fit per cell.
- Failure case: errors rise around Christmas and New Year, although each model peaks on a different day.
  `failure_residuals_h1.png` shows positive mean residuals for the learned models, so they over-forecast during
  the holiday test window. The largest relative failures are 24-hour GRU forecasts for Bocconi on 23 and
  30 December, where ordinary daytime levels were predicted during a large activity drop.
- Compute cost table: time and memory per model.

## 8:30 - 9:30  Limitations, conclusion, reproducibility
- Limitations: seven cells, one activity, two months, few holidays, descriptive cluster labels, the choice of
  four clusters despite a two-cluster silhouette optimum, and one non-converged final SARIMAX cell fit.
- Conclusion in two sentences answering the research question.
- Show `run_all.sh` and the notebook: one command reproduces everything; tests run on a synthetic fixture.
- AI-use disclosure in one sentence, consistent with the report.
