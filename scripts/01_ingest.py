#!/usr/bin/env python
"""Stage 1: convert raw daily TSV files into aggregated Parquet and hourly matrices.

Examples
--------
Real data (after downloading the 62 files into data/raw)::

    python scripts/01_ingest.py

Development subset of the real data (first 7 days)::

    python scripts/01_ingest.py --max-days 7

Synthetic smoke test::

    MILAN_CONFIG=config/smoke.yaml python scripts/01_ingest.py --synthetic-days 14
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from milan_forecast.config import load_config  # noqa: E402
from milan_forecast.ingest import (build_processed, ingest_raw, list_raw_files,  # noqa: E402
                                   naive_load_cost_mb)
from milan_forecast.synthetic import write_synthetic_raw  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--engine", choices=["polars", "pandas"], default="polars")
    parser.add_argument("--max-days", type=int, default=None, help="only ingest the first N daily files")
    parser.add_argument("--synthetic-days", type=int, default=0, help="generate N synthetic raw days first")
    parser.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.max_days is not None:
        cfg["data"]["max_days"] = args.max_days

    if args.synthetic_days:
        files = write_synthetic_raw(cfg, n_days=args.synthetic_days)
        logging.info("wrote %d synthetic raw files to %s", len(files), cfg.paths.raw_dir)

    files = list_raw_files(cfg.paths.raw_dir, cfg.data.max_days)
    if files:
        est_mb, est_rows = naive_load_cost_mb(files[0], cfg)
        logging.info("naive pandas load of %s would need ~%.0f MB for ~%.1fM rows",
                     files[0].name, est_mb, est_rows / 1e6)
        pd.DataFrame([{"file": files[0].name, "raw_size_mb": files[0].stat().st_size / 2**20,
                       "est_rows": est_rows, "naive_pandas_mb": est_mb,
                       "n_files": len(files), "naive_all_days_gb": est_mb * len(files) / 1024}]
                     ).to_csv(cfg.paths.ensure().tables_dir / "naive_cost_estimate.csv", index=False)

    outputs = ingest_raw(cfg, engine=args.engine, force=args.force)
    logging.info("interim parquet files: %d", len(outputs))
    processed = build_processed(cfg, force=args.force)
    for name, path in processed.items():
        logging.info("%-20s %s (%.1f MB)", name, path, path.stat().st_size / 2**20)


if __name__ == "__main__":
    main()
