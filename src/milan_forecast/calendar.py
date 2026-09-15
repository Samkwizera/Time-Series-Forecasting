"""Calendar attributes shared by the analysis and the modelling layers.

Kept separate from ``eda.py`` so that importing the forecasting stack does not pull in
matplotlib/seaborn: ``datasets.py`` needs these flags, but nothing about them is plotting.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd


def italian_holidays(index: pd.DatetimeIndex) -> pd.Series:
    """Boolean series: True on Italian public holidays incl. Sant'Ambrogio (Milan, 7 Dec)."""
    years = sorted(set(index.year))
    cal = holidays.country_holidays("IT", subdiv="MI", years=years)
    return pd.Series(index.normalize().isin(pd.DatetimeIndex(list(cal.keys()))), index=index)


def calendar_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    flags = pd.DataFrame(index=index)
    flags["hour"] = index.hour
    flags["dow"] = index.dayofweek
    flags["is_weekend"] = flags["dow"] >= 5
    flags["is_holiday"] = italian_holidays(index).to_numpy()
    flags["day_type"] = np.select([flags["is_holiday"], flags["is_weekend"]], ["holiday", "weekend"], "weekday")
    return flags
