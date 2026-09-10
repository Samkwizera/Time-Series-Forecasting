#!/usr/bin/env python
"""Stage 2: exploratory analysis. Produces figures in reports/figures and tables in reports/tables.

    python scripts/02_eda.py
    MILAN_CONFIG=config/smoke.yaml python scripts/02_eda.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from milan_forecast import eda  # noqa: E402
from milan_forecast.config import load_config  # noqa: E402
from milan_forecast.ingest import load_hourly  # noqa: E402
from milan_forecast.memory import track  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    paths = cfg.paths.ensure()
    activity = cfg.forecast.target_activity

    with track("eda", paths.tables_dir / "memory_log.csv"):
        city_path = paths.processed_dir / "citywide_10min.parquet"
        if not city_path.exists():
            raise FileNotFoundError(
                f"{city_path} is missing. Stage 1 (scripts/01_ingest.py) has to finish first; "
                "EDA cannot run on the raw TSV files."
            )
        city = pd.read_parquet(city_path)
        hourly = load_hourly(cfg, activity)

        eda.plot_citywide_series(city, cfg)
        eda.daily_profiles(city, cfg)
        eda.hour_dow_heatmap(city, cfg, activity)
        eda.daily_totals_anomalies(city, cfg, activity)
        eda.spatial_maps(hourly, cfg, activity)
        labels, _ = eda.cluster_cells(hourly, cfg, activity)
        selected = eda.select_cells(hourly, labels, cfg)
        eda.plot_selected_cells(hourly, selected, cfg, activity)
        stats = eda.summary_statistics(hourly, city, cfg)
    logging.info("summary statistics:\n%s", stats.to_string())
    logging.info("figures written to %s", paths.figures_dir)


if __name__ == "__main__":
    main()
