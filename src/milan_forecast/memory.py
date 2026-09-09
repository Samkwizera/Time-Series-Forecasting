"""Memory and timing instrumentation used to produce evidence for the report.

Every heavy stage wraps its work in :func:`track` so that peak RSS, elapsed time
and the input/output sizes are appended to ``reports/tables/memory_log.csv``.
"""

from __future__ import annotations

import csv
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import psutil

log = logging.getLogger(__name__)


@dataclass
class StageRecord:
    stage: str
    seconds: float
    rss_start_mb: float
    rss_peak_mb: float
    rss_end_mb: float
    note: str = ""


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 2**20


@contextmanager
def track(stage: str, log_path: Path | None = None, note: str = "") -> Iterator[dict]:
    """Measure wall time and resident memory around a block of code.

    Peak RSS is sampled at the start and end of the block and updated by callers
    that call ``ctx["sample"]()`` inside long loops (for example once per chunk).
    """
    proc = psutil.Process()
    start = time.perf_counter()
    rss0 = rss_mb()
    peak = [rss0]

    def sample() -> None:
        peak[0] = max(peak[0], proc.memory_info().rss / 2**20)

    ctx = {"sample": sample}
    try:
        yield ctx
    finally:
        sample()
        rec = StageRecord(stage, round(time.perf_counter() - start, 2), round(rss0, 1),
                          round(peak[0], 1), round(rss_mb(), 1), note)
        log.info("%s: %.1fs, RSS start %.0f MB, peak %.0f MB", rec.stage, rec.seconds,
                 rec.rss_start_mb, rec.rss_peak_mb)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            new = not log_path.exists()
            with open(log_path, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(asdict(rec)))
                if new:
                    writer.writeheader()
                writer.writerow(asdict(rec))


def frame_memory_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 2**20


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink numeric columns to the smallest safe dtype.

    Activity values are non-negative floats that never exceed float32 range and
    are only meaningful to a few decimals, so float32 halves their footprint.
    Square ids fit in uint16 (1..10000).
    """
    out = df.copy(deep=False)
    for col in out.columns:
        dtype = out[col].dtype
        if np.issubdtype(dtype, np.floating):
            out[col] = out[col].astype(np.float32)
        elif np.issubdtype(dtype, np.integer):
            cmax = out[col].max()
            cmin = out[col].min()
            if cmin >= 0 and cmax < 2**16:
                out[col] = out[col].astype(np.uint16)
            elif cmin >= -2**31 and cmax < 2**31:
                out[col] = out[col].astype(np.int32)
    return out
