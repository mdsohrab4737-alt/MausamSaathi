from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    'rainfall_1d', 'rainfall_3d', 'rainfall_7d', 'rainy_days_7d',
    'temperature_c', 'humidity_pct', 'wind_ms', 'pressure_hpa',
    'enso_index', 'iod_index', 'mjo_index', 'rainfall_anomaly_pct',
    'day_sin', 'day_cos',
]
TARGET_COLUMNS = ['onset_label', 'break_label', 'heavy_rain_label']


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out['date'])
    day = dates.dt.dayofyear.astype(float)
    out['day_sin'] = np.sin(2 * np.pi * day / 365.25)
    out['day_cos'] = np.cos(2 * np.pi * day / 365.25)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_time_features(df)
    missing = [c for c in FEATURE_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f'Missing feature columns: {missing}')
    return out[FEATURE_COLUMNS].astype(float)
