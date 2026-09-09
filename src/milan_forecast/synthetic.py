"""Synthetic raw files with the exact Dataverse schema, for tests and smoke runs.

The generator is *not* a substitute for the real data: it exists so the whole
pipeline (ingest -> EDA -> models -> report tables) can be exercised end to end
on a laptop or CI runner in under a minute. Numbers in the report must come from
the real dataset.

The profiles mimic the qualitative structure documented for Milan (Barlacchi et
al. 2015): strong daily cycle, weekday/weekend contrast, a business core with
day-time peaks, residential rings with evening peaks, and a nightlife cluster.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

# Italy dominates, plus a few foreign codes and the "unknown" 0 code seen in the real files
COUNTRY_CODES = [39, 33, 44, 49, 0]


def _cell_profiles(grid_side: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (base_level[n_cells], type[n_cells]) with a spatial structure."""
    ys, xs = np.divmod(np.arange(grid_side * grid_side), grid_side)
    centre = (grid_side - 1) / 2
    dist = np.hypot(xs - centre, ys - centre) / (grid_side / 2)
    base = 400 * np.exp(-3 * dist**2) + 5 + rng.gamma(1.0, 3.0, size=dist.size)
    # 0 = business core, 1 = residential ring, 2 = suburbs, 3 = scattered nightlife spots
    cell_type = np.where(dist < 0.25, 0, np.where(dist < 0.6, 1, 2))
    nightlife = rng.choice(dist.size, size=max(1, dist.size // 50), replace=False)
    cell_type[nightlife] = 3
    return base.astype(np.float32), cell_type


def _daily_shape(hours: np.ndarray, cell_type: int) -> np.ndarray:
    # business: office hours peak
    if cell_type == 0:
        return 0.2 + 0.8 * np.exp(-((hours - 13.5) / 3.2) ** 2)
    # residential: small morning bump, big evening peak
    if cell_type == 1:
        return 0.25 + 0.35 * np.exp(-((hours - 8) / 1.5) ** 2) + 0.75 * np.exp(-((hours - 20.5) / 2.5) ** 2)
    # nightlife: peak wraps around midnight
    if cell_type == 3:
        return 0.15 + 0.9 * np.exp(-((((hours + 2) % 24) - 1.5) / 2.0) ** 2)
    # suburban: flat-ish
    return 0.3 + 0.5 * np.exp(-((hours - 12) / 4.0) ** 2)


def write_synthetic_raw(cfg: Config, n_days: int = 5, seed: int = 0) -> list[Path]:
    """Write ``n_days`` raw TSV files into ``cfg.paths.raw_dir`` and return their paths."""
    rng = np.random.default_rng(seed)
    grid_side = cfg.data.grid_side
    n_cells = grid_side * grid_side
    assert n_cells == cfg.data.n_cells, "grid_side**2 must equal n_cells"
    base, cell_type = _cell_profiles(grid_side, rng)
    step = pd.Timedelta(minutes=cfg.data.interval_minutes)
    intervals_per_day = int(pd.Timedelta("1D") / step)
    start = pd.Timestamp(cfg.data.start_date, tz=cfg.data.timezone)
    raw_dir = cfg.paths.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for d in range(n_days):
        day_start = start + pd.Timedelta(days=d)
        times = pd.DatetimeIndex([day_start + step * i for i in range(intervals_per_day)])
        epoch_ms = (times.tz_convert("UTC").tz_localize(None) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
        hours = (times.hour + times.minute / 60).to_numpy()
        weekend = day_start.dayofweek >= 5
        shapes = np.stack([_daily_shape(hours, t) for t in range(4)])
        level = shapes[cell_type] * base[:, None]
        if weekend:
            level *= np.where(cell_type == 0, 0.45, 0.9)[:, None]
        noise = rng.gamma(shape=20.0, scale=1 / 20.0, size=level.shape)
        activity = level * noise
        frames = []
        for code, share in zip(COUNTRY_CODES, [0.85, 0.05, 0.04, 0.03, 0.03]):
            part = activity * share
            frame = pd.DataFrame({
                "square_id": np.repeat(np.arange(1, n_cells + 1), intervals_per_day),
                "time_interval": np.tile(epoch_ms.to_numpy().astype(np.int64), n_cells),
                "country_code": code,
                "sms_in": (part * 0.12).ravel(),
                "sms_out": (part * 0.10).ravel(),
                "call_in": (part * 0.08).ravel(),
                "call_out": (part * 0.09).ravel(),
                "internet": part.ravel(),
            })
            # real files only have rows for foreign codes where there was actual traffic,
            # so thin them out instead of emitting a full grid per code
            if code != 39:
                frame = frame.sample(frac=0.15, random_state=int(rng.integers(1 << 31)))
            frames.append(frame)
        day_df = pd.concat(frames).sample(frac=1.0, random_state=seed + d)
        path = raw_dir / f"sms-call-internet-mi-{day_start.date()}.txt"
        day_df.to_csv(path, sep="\t", header=False, index=False, float_format="%.4f")
        out.append(path)
    return out
