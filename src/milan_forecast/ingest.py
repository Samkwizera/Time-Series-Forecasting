"""Memory-efficient ingestion of the raw Telecom Italia TSV files.

Raw layout (one file per day, no header, tab separated)::

    square_id  time_interval(ms epoch UTC)  country_code  sms_in sms_out call_in call_out internet

A day holds 6-8 million rows because each (cell, interval) pair is repeated once
per country code. Loading a day naively with pandas costs ~1 GB; loading all 62
days would need >50 GB. The strategy here is:

1. Stream each file (Polars lazy scan, or pandas chunks as a fallback).
2. Sum over ``country_code`` immediately, which removes ~90 % of the rows.
3. Downcast to float32 / uint16 and write one Parquet file per day.
4. Build compact wide matrices (time x cell) at 10-minute and hourly resolution.

Only the aggregated Parquet files are read afterwards, so EDA and modelling never
touch the raw TSVs again.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Config
from .memory import downcast, frame_memory_mb, track

log = logging.getLogger(__name__)

RAW_FILE_RE = re.compile(r"sms-call-internet-mi-(\d{4}-\d{2}-\d{2})\.txt$")
KEY_COLUMNS = ["square_id", "time_interval"]


def list_raw_files(raw_dir: Path, max_days: int | None = None) -> list[Path]:
    files = sorted(p for p in raw_dir.glob("sms-call-internet-mi-*.txt") if RAW_FILE_RE.search(p.name))
    if max_days:
        files = files[:max_days]
    return files


def day_of(path: Path) -> str:
    match = RAW_FILE_RE.search(path.name)
    if not match:
        raise ValueError(f"Unexpected raw file name: {path.name}")
    return match.group(1)


# --------------------------------------------------------------------------- #
# Per-day aggregation
# --------------------------------------------------------------------------- #
def aggregate_day_polars(path: Path, cfg: Config) -> pd.DataFrame:
    """Aggregate one raw day with Polars' streaming engine (preferred)."""
    import polars as pl

    cols = cfg.data.raw_columns
    acts = cfg.data.activities
    schema = {c: (pl.Float32 if c in acts else pl.Int64) for c in cols}
    lazy = (
        pl.scan_csv(path, separator="\t", has_header=False, new_columns=cols,
                    schema_overrides=schema, null_values=[""])
        .group_by(KEY_COLUMNS)
        .agg([pl.col(a).sum().alias(a) for a in acts])
        .sort(KEY_COLUMNS)
    )
    df = lazy.collect(engine="streaming").to_pandas()
    return _finalise_day(df, cfg)


def aggregate_day_pandas(path: Path, cfg: Config, sample=None) -> pd.DataFrame:
    """Fallback aggregation with pandas chunks; bounded memory regardless of file size."""
    cols = cfg.data.raw_columns
    acts = cfg.data.activities
    dtypes = {c: np.float32 for c in acts}
    dtypes.update({"square_id": np.int32, "time_interval": np.int64, "country_code": np.int32})
    partial = []
    reader = pd.read_csv(path, sep="\t", header=None, names=cols, dtype=dtypes,
                         chunksize=cfg.data.chunk_rows, usecols=KEY_COLUMNS + acts)
    for chunk in reader:
        partial.append(chunk.groupby(KEY_COLUMNS, sort=False)[acts].sum())
        if sample:
            sample()
    # a (cell, interval) group can straddle two chunks, hence the second groupby
    df = pd.concat(partial).groupby(level=KEY_COLUMNS).sum().reset_index()
    return _finalise_day(df, cfg)


def _finalise_day(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df["timestamp"] = (pd.to_datetime(df.pop("time_interval"), unit="ms", utc=True)
                       .dt.tz_convert(cfg.data.timezone).dt.tz_localize(None))
    df["square_id"] = df["square_id"].astype(np.uint16)
    for a in cfg.data.activities:
        df[a] = df[a].fillna(0).astype(np.float32)
    df = df.sort_values(["timestamp", "square_id"]).reset_index(drop=True)
    return downcast(df)


def naive_load_cost_mb(path: Path, cfg: Config, n_rows: int = 200_000) -> tuple[float, int]:
    """Estimate the cost of a naive full pandas load from the first ``n_rows`` rows.

    Used only to produce the "before" column of the memory table in the report.
    Returns (estimated MB for the whole file, estimated row count).
    """
    head = pd.read_csv(path, sep="\t", header=None, names=cfg.data.raw_columns, nrows=n_rows)
    per_row_bytes = frame_memory_mb(head) * 2**20 / len(head)
    with open(path, "rb") as fh:
        sample = fh.read(4_000_000)
    bytes_per_line = len(sample) / max(sample.count(b"\n"), 1)
    est_rows = int(path.stat().st_size / bytes_per_line)
    return per_row_bytes * est_rows / 2**20, est_rows


def ingest_raw(cfg: Config, engine: str = "polars", force: bool = False) -> list[Path]:
    """Convert every raw daily TSV into an aggregated Parquet file."""
    paths = cfg.paths.ensure()
    files = list_raw_files(paths.raw_dir, cfg.data.max_days)
    if not files:
        raise FileNotFoundError(f"No raw files in {paths.raw_dir}. See README for download steps.")
    mem_log = paths.tables_dir / "memory_log.csv"
    outputs = []
    for path in files:
        day = day_of(path)
        out = paths.interim_dir / f"day={day}.parquet"
        outputs.append(out)
        if out.exists() and not force:
            continue
        with track(f"ingest:{day}", mem_log, note=f"{path.stat().st_size / 2**20:.0f} MB raw") as ctx:
            if engine == "polars":
                df = aggregate_day_polars(path, cfg)
            else:
                df = aggregate_day_pandas(path, cfg, ctx["sample"])
            df.to_parquet(out, index=False, compression="zstd")
        log.info("%s -> %d rows, %.1f MB in memory, %.1f MB on disk", day, len(df),
                 frame_memory_mb(df), out.stat().st_size / 2**20)
    return outputs


# --------------------------------------------------------------------------- #
# Wide matrices
# --------------------------------------------------------------------------- #
def load_interim(cfg: Config, days: Iterable[str] | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    paths = cfg.paths
    files = sorted(paths.interim_dir.glob("day=*.parquet"))
    if days is not None:
        wanted = set(days)
        files = [f for f in files if f.stem.split("=")[1] in wanted]
    return pd.concat((pd.read_parquet(f, columns=columns) for f in files), ignore_index=True)


def build_wide(cfg: Config, activity: str, freq: str | None = None) -> pd.DataFrame:
    """Pivot the interim data for one activity into a (time x cell) float32 frame.

    Missing (cell, interval) pairs mean zero activity in this dataset, so they are
    filled with 0 rather than NaN. ``freq`` resamples (sum) to a coarser grid.
    """
    n_cells = cfg.data.n_cells
    files = sorted(cfg.paths.interim_dir.glob("day=*.parquet"))
    blocks = []
    for f in files:
        day = pd.read_parquet(f, columns=["timestamp", "square_id", activity])
        wide = (day.pivot_table(index="timestamp", columns="square_id", values=activity,
                                aggfunc="sum", fill_value=0.0)
                .reindex(columns=np.arange(1, n_cells + 1), fill_value=0.0)
                .astype(np.float32))
        if freq:
            wide = wide.resample(freq).sum().astype(np.float32)
        blocks.append(wide)
    out = pd.concat(blocks).sort_index()
    # files are cut at UTC midnight, so the same local hour can show up in two files
    out = out.groupby(level=0).sum().astype(np.float32)
    out.columns.name = "square_id"
    return out


def build_processed(cfg: Config, force: bool = False) -> dict[str, Path]:
    """Create the processed artefacts used by EDA and modelling."""
    paths = cfg.paths.ensure()
    mem_log = paths.tables_dir / "memory_log.csv"
    outputs: dict[str, Path] = {}
    for activity in cfg.data.activities:
        hourly_path = paths.processed_dir / f"hourly_{activity}.parquet"
        outputs[f"hourly_{activity}"] = hourly_path
        if hourly_path.exists() and not force:
            continue
        with track(f"wide_hourly:{activity}", mem_log):
            wide = build_wide(cfg, activity, freq=cfg.eda.hourly_freq)
            wide.columns = wide.columns.astype(str)
            wide.to_parquet(hourly_path, compression="zstd")
    # keep a 10-min citywide series for EDA; the full 10k-cell width at that resolution is too big
    city_path = paths.processed_dir / "citywide_10min.parquet"
    outputs["citywide_10min"] = city_path
    if not city_path.exists() or force:
        with track("citywide_10min", mem_log):
            df = load_interim(cfg, columns=["timestamp"] + list(cfg.data.activities))
            city = df.groupby("timestamp")[list(cfg.data.activities)].sum().astype(np.float32)
            city.to_parquet(city_path, compression="zstd")
    return outputs


def load_hourly(cfg: Config, activity: str | None = None) -> pd.DataFrame:
    activity = activity or cfg.forecast.target_activity
    df = pd.read_parquet(cfg.paths.processed_dir / f"hourly_{activity}.parquet")
    df.columns = df.columns.astype(int)
    df.columns.name = "square_id"
    return df
