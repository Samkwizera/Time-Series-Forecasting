"""End-to-end ingestion test on a tiny synthetic fixture."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from milan_forecast.config import load_config
from milan_forecast.ingest import (aggregate_day_pandas, aggregate_day_polars, build_processed,
                                   ingest_raw, load_hourly)
from milan_forecast.synthetic import write_synthetic_raw

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    cfg = load_config(ROOT / "config" / "smoke.yaml")
    root = tmp_path_factory.mktemp("milan")
    cfg["_root"] = str(root)
    cfg["data"]["grid_side"] = 5
    cfg["data"]["n_cells"] = 25
    write_synthetic_raw(cfg, n_days=3)
    return cfg


def test_engines_agree(cfg):
    path = sorted(cfg.paths.raw_dir.glob("*.txt"))[0]
    a = aggregate_day_polars(path, cfg)
    b = aggregate_day_pandas(path, cfg)
    assert len(a) == len(b) == 25 * 144
    np.testing.assert_allclose(a["internet"].to_numpy(), b["internet"].to_numpy(), rtol=1e-4)
    assert a["square_id"].dtype == np.uint16
    assert a["internet"].dtype == np.float32


def test_country_codes_are_summed(cfg):
    path = sorted(cfg.paths.raw_dir.glob("*.txt"))[0]
    raw = pd.read_csv(path, sep="\t", header=None, names=cfg.data.raw_columns)
    agg = aggregate_day_polars(path, cfg)
    assert np.isclose(raw["internet"].sum(), agg["internet"].sum(), rtol=1e-4)
    assert not agg.duplicated(["square_id", "timestamp"]).any()


def test_processed_outputs(cfg):
    ingest_raw(cfg)
    outputs = build_processed(cfg)
    hourly = load_hourly(cfg, "internet")
    assert hourly.shape == (3 * 24, 25)
    assert hourly.index.is_monotonic_increasing
    assert str(hourly.index[0]) == "2013-11-01 00:00:00"
    city = pd.read_parquet(outputs["citywide_10min"])
    assert len(city) == 3 * 144
    assert np.isclose(city["internet"].sum(), hourly.to_numpy().sum(), rtol=1e-3)
