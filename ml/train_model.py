from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    average_precision_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from features import FEATURE_COLUMNS, TARGET_COLUMNS, build_features

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'demo_training.csv'
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def generate_demo_dataset(path: Path) -> pd.DataFrame:
    """Generate a realistic-looking synthetic demo dataset.

    This is explicitly for demonstrating the ML pipeline. Production training
    must replace it with observed historical weather/rainfall data and
    documented meteorological event labels.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    dates = pd.date_range('2017-01-01', '2025-12-31', freq='D')
    n = len(dates)
    month = dates.month.to_numpy()

    # Seasonal monsoon signal for an India-like regional demo.
    monsoon_strength = np.clip(
        np.exp(-((month - 7.5) / 1.9) ** 2), 0, 1
    )
    enso_monthly = rng.normal(0, 0.9, n)
    iod_monthly = rng.normal(0, 0.8, n)
    mjo = np.sin(np.arange(n) / 18.0) + rng.normal(0, 0.25, n)

    base_rain = monsoon_strength * (10 + 18 * (1 + 0.25 * (-enso_monthly) + 0.18 * iod_monthly + 0.12 * mjo))
    rainfall_1d = np.maximum(0, rng.gamma(shape=1.3, scale=np.maximum(base_rain, 0.5)))
    rainfall_3d = pd.Series(rainfall_1d).rolling(3, min_periods=1).sum().to_numpy()
    rainfall_7d = pd.Series(rainfall_1d).rolling(7, min_periods=1).sum().to_numpy()
    rainy_days_7d = pd.Series((rainfall_1d >= 2).astype(int)).rolling(7, min_periods=1).sum().to_numpy()

    temperature = 31.5 - 4.2 * monsoon_strength + rng.normal(0, 1.4, n)
    humidity = 57 + 28 * monsoon_strength + 2.5 * mjo + rng.normal(0, 4, n)
    wind = 2.4 + 2.0 * monsoon_strength + rng.normal(0, 0.6, n)
    pressure = 1009 - 4.5 * monsoon_strength + rng.normal(0, 1.8, n)
    climatology = 7 + 18 * monsoon_strength
    anomaly = (rainfall_7d - pd.Series(climatology).rolling(7, min_periods=1).mean().to_numpy()) / (climatology + 1) * 100

    # Demonstration labels only. These are deliberately simple proxy rules,
    # not official meteorological onset/break/heavy-rain definitions.
    month_day = dates.dayofyear.to_numpy()
    onset_window = ((month_day >= 150) & (month_day <= 235))
    onset_label = (
        onset_window
        & (rainfall_7d >= 40)
        & (rainy_days_7d >= 4)
        & (humidity >= 65)
    ).astype(int)

    # Break proxy: low monsoon activity using recent rainfall, humidity and
    # MJO support. A small amount of label noise mimics observational uncertainty.
    break_window = (month >= 7) & (month <= 9)
    break_label = (
        (break_window & (rainfall_7d < 28) & (humidity < 78))
        | ((rainfall_7d < 22) & (humidity < 82))
        | ((mjo < -0.25) & (rainfall_7d < 35) & break_window)
    ).astype(int)
    noise_mask = rng.random(n) < 0.04
    break_label = np.where(noise_mask, 1 - break_label, break_label)

    heavy_rain_label = (
        (rainfall_1d >= 35) | (rainfall_3d >= 75)
    ).astype(int)

    # Small noise to avoid a perfectly deterministic demo target.
    flip = rng.random((n, 3)) < 0.025
    labels = np.column_stack([onset_label, break_label, heavy_rain_label])
    labels = np.where(flip, 1 - labels, labels)

    df = pd.DataFrame({
        'date': dates,
        'rainfall_1d': rainfall_1d,
        'rainfall_3d': rainfall_3d,
        'rainfall_7d': rainfall_7d,
        'rainy_days_7d': rainy_days_7d,
        'temperature_c': temperature,
        'humidity_pct': humidity,
        'wind_ms': wind,
        'pressure_hpa': pressure,
        'enso_index': enso_monthly,
        'iod_index': iod_monthly,
        'mjo_index': mjo,
        'rainfall_anomaly_pct': anomaly,
        'onset_label': labels[:, 0],
        'break_label': labels[:, 1],
        'heavy_rain_label': labels[:, 2],
    })
    df.to_csv(path, index=False)
    return df


def fit_one(df: pd.DataFrame, target: str) -> dict:
    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:valid_end].copy()
    test = df.iloc[valid_end:].copy()

    X_train = build_features(train)
    y_train = train[f'{target}_label'].astype(int)
    X_valid = build_features(valid)
    y_valid = valid[f'{target}_label'].astype(int)
    X_test = build_features(test)
    y_test = test[f'{target}_label'].astype(int)

    model = XGBClassifier(
        n_estimators=240,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=2,
    )
    model.fit(X_train, y_train)

    valid_raw = model.predict_proba(X_valid)[:, 1]
    test_raw = model.predict_proba(X_test)[:, 1]

    # Calibrate on validation only; final test remains untouched for scoring.
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(valid_raw, y_valid)
    test_cal = calibrator.predict(test_raw)

    test_pred = (test_cal >= 0.5).astype(int)
    metrics = {
        'brier': round(float(brier_score_loss(y_test, test_cal)), 4),
        'log_loss': round(float(log_loss(y_test, np.clip(test_cal, 1e-5, 1 - 1e-5))), 4),
        'roc_auc': round(float(roc_auc_score(y_test, test_cal)), 4),
        'pr_auc': round(float(average_precision_score(y_test, test_cal)), 4),
        'f1': round(float(f1_score(y_test, test_pred)), 4),
        'positive_rate_test': round(float(y_test.mean()), 4),
        'test_samples': int(len(test)),
    }

    joblib.dump(model, MODEL_DIR / f'{target}_model.joblib')
    joblib.dump(calibrator, MODEL_DIR / f'{target}_calibrator.joblib')

    return metrics


def main() -> None:
    df = generate_demo_dataset(DATA_PATH)
    metrics = {}
    for target in ('onset', 'break', 'heavy_rain'):
        metrics[target] = fit_one(df, target)

    summary = {
        'purpose': 'SIH 26086 ML proof-of-concept',
        'data': {
            'source': 'synthetic demonstration data',
            'rows': int(len(df)),
            'date_start': str(df['date'].min().date()),
            'date_end': str(df['date'].max().date()),
            'split': '70% train / 15% validation / 15% test, chronological',
        },
        'features': FEATURE_COLUMNS,
        'targets': TARGET_COLUMNS,
        'models': {
            'classifier': 'XGBClassifier',
            'calibration': 'IsotonicRegression fit on validation predictions',
        },
        'metrics': metrics,
        'warning': 'Replace synthetic data and proxy labels with official observed data and documented meteorological event criteria for production.',
    }
    with (MODEL_DIR / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
