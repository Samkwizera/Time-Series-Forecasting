"""Naive baselines. They set the bar every learned model has to clear.

* ``naive``: the last observed value, y[t].
* ``seasonal_naive``: the same hour yesterday, y[t + h - 24]. Also the MASE scaling series.
* ``weekly_naive``: the same hour last week, y[t + h - 168].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..datasets import ForecastData, predictions_frame, targets_matrix
from ._common import eval_origins

SEASON = {"naive": 0, "seasonal_naive": 24, "weekly_naive": 168}


def forecast(data: ForecastData, part: str, method: str = "seasonal_naive") -> pd.DataFrame:
    if method not in SEASON:
        raise ValueError(f"unknown baseline {method!r}; choose from {list(SEASON)}")
    season = SEASON[method]
    origins = eval_origins(data, part, min_history=max(season, data.input_window))
    frames = []
    for name, s in data.series.items():
        values = s.to_numpy()
        pos = s.index.get_indexer(origins)
        y_true = targets_matrix(s, origins, data.horizons)
        cols = []
        for h in data.horizons:
            if season == 0:
                cols.append(values[pos])
            else:
                # for h > season the "same hour last period" is itself in the future:
                # step back whole periods until the reference lies at or before the origin
                back = int(np.ceil(h / season)) * season
                cols.append(values[pos + h - back])
        frames.append(predictions_frame(method, name, origins, data.horizons, y_true, np.column_stack(cols)))
    return pd.concat(frames, ignore_index=True)
