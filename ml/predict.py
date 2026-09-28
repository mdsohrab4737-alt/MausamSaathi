from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from features import build_features

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models'
METRICS_PATH = MODEL_DIR / 'metrics.json'
TARGETS = {
    'onset': 'onset',
    'break': 'break',
    'heavy_rain': 'heavy_rain',
}


def load_bundle(target: str):
    model = joblib.load(MODEL_DIR / f'{target}_model.joblib')
    calibrator = joblib.load(MODEL_DIR / f'{target}_calibrator.joblib')
    return model, calibrator


def predict_probabilities(row: dict) -> dict:
    df = pd.DataFrame([row])
    X = build_features(df)
    result = {}
    for output_name, model_name in TARGETS.items():
        model, calibrator = load_bundle(model_name)
        raw = float(model.predict_proba(X)[:, 1][0])
        calibrated = float(calibrator.predict([raw])[0])
        result[f'{output_name}_raw'] = round(raw * 100, 1)
        result[f'{output_name}_probability'] = round(calibrated * 100, 1)
    return result


def load_metrics() -> dict:
    with METRICS_PATH.open('r', encoding='utf-8') as f:
        return json.load(f)
