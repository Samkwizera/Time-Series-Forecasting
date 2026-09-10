#!/usr/bin/env python
"""Stage 3: time-series analysis (stationarity, decomposition, autocorrelation).

    python scripts/03_tsa.py
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

from milan_forecast import tsa  # noqa: E402
from milan_forecast.config import load_config  # noqa: E402
from milan_forecast.ingest import load_hourly  # noqa: E402
from milan_forecast.memory import track  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    warnings.filterwarnings("ignore", message=".*InterpolationWarning.*")
    warnings.filterwarnings("ignore", category=UserWarning)
    cfg = load_config(args.config)
    paths = cfg.paths.ensure()
    selected_path = paths.tables_dir / "selected_cells.json"
    if not selected_path.exists():
        raise FileNotFoundError(
            f"{selected_path} is missing. Run scripts/02_eda.py first (it writes this file)."
        )
    with open(selected_path) as fh:
        selected = json.load(fh)["cells"]
    with track("tsa", paths.tables_dir / "memory_log.csv"):
        city = pd.read_parquet(paths.processed_dir / "citywide_10min.parquet")
        hourly = load_hourly(cfg, cfg.forecast.target_activity)
        tsa.run_all(hourly, city, selected, cfg)


if __name__ == "__main__":
    main()
