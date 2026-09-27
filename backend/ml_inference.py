from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_test" / "processed"

# IMPORTANT:
# Training and inference must use the same engineered feature table.
FEATURES_CSV = PROCESSED / "bilaspur_training_master_2024_2025.csv"

MODEL_DIR = PROCESSED / "models" / "final"

HEAVY_RAIN_THRESHOLD_MM = 64.5

MODEL_PATHS = {
    7: {
        "mean": MODEL_DIR / "bilaspur_7d_rainfall_mean_rf_core.pkl",
        "max": MODEL_DIR / "bilaspur_7d_rainfall_max_rf_core.pkl",
    },
    14: {
        "mean": MODEL_DIR / "bilaspur_14d_rainfall_mean_rf_core.pkl",
        "max": MODEL_DIR / "bilaspur_14d_rainfall_max_rf_core.pkl",
    },
    21: {
        "mean": MODEL_DIR / "bilaspur_21d_rainfall_mean_rf_core.pkl",
        "max": MODEL_DIR / "bilaspur_21d_rainfall_max_rf_core.pkl",
    },
    30: {
        "mean": MODEL_DIR / "bilaspur_30d_rainfall_mean_rf_core.pkl",
        "max": None,
    },
}


def _load_model_bundle(path: Path) -> tuple[Any, list[str]]:
    if not path or not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    obj = joblib.load(path)

    if isinstance(obj, dict):
        model = (
            obj.get("model")
            or obj.get("estimator")
            or obj.get("regressor")
        )

        features = (
            obj.get("features")
            or obj.get("feature_names")
            or obj.get("feature_columns")
        )

        if model is None:
            raise ValueError(f"No model object found inside {path}")

        if features is None:
            features = getattr(model, "feature_names_in_", None)

        if features is None:
            raise ValueError(f"No feature list found for {path}")

        return model, list(features)

    model = obj
    features = getattr(model, "feature_names_in_", None)

    if features is None:
        raise ValueError(f"Could not determine features for {path}")

    return model, list(features)


def _tree_quantiles(model: Any, X: pd.DataFrame) -> dict[str, float]:
    """
    Empirical spread from individual Random Forest trees.

    This is an ensemble-spread proxy, not a calibrated probability interval.
    NumPy input is used for individual DecisionTreeRegressor objects to keep
    feature-name handling consistent with the saved model.
    """
    estimators = getattr(model, "estimators_", None)

    if not estimators:
        pred = float(model.predict(X)[0])
        return {
            "p10": pred,
            "p25": pred,
            "p50": pred,
            "p75": pred,
            "p90": pred,
        }

    X_np = X.to_numpy(dtype=np.float64)

    tree_predictions = np.asarray(
        [tree.predict(X_np)[0] for tree in estimators],
        dtype=np.float64,
    )

    return {
        "p10": float(np.percentile(tree_predictions, 10)),
        "p25": float(np.percentile(tree_predictions, 25)),
        "p50": float(np.percentile(tree_predictions, 50)),
        "p75": float(np.percentile(tree_predictions, 75)),
        "p90": float(np.percentile(tree_predictions, 90)),
    }


def _threshold_share(
    model: Any,
    X: pd.DataFrame,
    threshold: float,
) -> float | None:

    estimators = getattr(model, "estimators_", None)

    if not estimators:
        return None

    X_np = X.to_numpy(dtype=np.float64)

    tree_predictions = np.asarray(
        [tree.predict(X_np)[0] for tree in estimators],
        dtype=np.float64,
    )

    return float(np.mean(tree_predictions >= threshold) * 100.0)


def _load_dataset() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Training feature dataset not found:\n{FEATURES_CSV}"
        )

    df = pd.read_csv(
        FEATURES_CSV,
        parse_dates=["date"],
    )

    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found in inference dataset.")

    return df


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if pd.isna(value):
        return None

    return value


def run_demo_forecast(
    as_of: str,
    horizon: int,
) -> dict[str, Any]:

    if horizon not in MODEL_PATHS:
        raise ValueError(
            "Horizon must be 7, 14, 21 or 30 days."
        )

    df = _load_dataset()

    target_date = pd.Timestamp(as_of)

    matches = df.index[
        df["date"] == target_date
    ].tolist()

    if not matches:
        available_start = df["date"].min().date()
        available_end = df["date"].max().date()

        raise ValueError(
            f"Date {as_of} not found. "
            f"Available range: {available_start} to {available_end}."
        )

    row_index = matches[0]
    row = df.iloc[[row_index]].copy()

    # ---------------------------------------------------------
    # MEAN RAINFALL MODEL
    # ---------------------------------------------------------

    mean_model, mean_features = _load_model_bundle(
        MODEL_PATHS[horizon]["mean"]
    )

    missing_mean_features = [
        feature
        for feature in mean_features
        if feature not in row.columns
    ]

    if missing_mean_features:
        raise ValueError(
            "Missing mean-model features:\n"
            + ", ".join(missing_mean_features)
        )

    X_mean = row[mean_features].astype(float)

    if X_mean.isna().any().any():
        bad_features = X_mean.columns[
            X_mean.isna().any()
        ].tolist()

        raise ValueError(
            "Selected date has missing model inputs:\n"
            + ", ".join(bad_features)
        )

    mean_quantiles = _tree_quantiles(
        mean_model,
        X_mean,
    )

    total_quantiles = {
        key: value * horizon
        for key, value in mean_quantiles.items()
    }

    # ---------------------------------------------------------
    # MAX RAINFALL MODEL
    # ---------------------------------------------------------

    max_result = None

    max_model_path = MODEL_PATHS[horizon]["max"]

    if max_model_path is not None and max_model_path.exists():

        max_model, max_features = _load_model_bundle(
            max_model_path
        )

        missing_max_features = [
            feature
            for feature in max_features
            if feature not in row.columns
        ]

        if missing_max_features:
            raise ValueError(
                "Missing max-rainfall model features:\n"
                + ", ".join(missing_max_features)
            )

        X_max = row[max_features].astype(float)

        if X_max.isna().any().any():
            bad_features = X_max.columns[
                X_max.isna().any()
            ].tolist()

            raise ValueError(
                "Selected date has missing max-model inputs:\n"
                + ", ".join(bad_features)
            )

        max_quantiles = _tree_quantiles(
            max_model,
            X_max,
        )

        threshold_share = _threshold_share(
            max_model,
            X_max,
            HEAVY_RAIN_THRESHOLD_MM,
        )

        max_result = {
            "forecast_mm": max_quantiles,
            "heavy_rain_threshold_mm": HEAVY_RAIN_THRESHOLD_MM,
            "threshold_exceedance_share_pct": threshold_share,
            "interpretation": (
                "Share of Random Forest trees whose predicted future "
                "maximum rainfall reaches the heavy-rain threshold. "
                "This is an ensemble-spread proxy, not a calibrated "
                "probability."
            ),
        }

    # ---------------------------------------------------------
    # INPUT SNAPSHOT
    # ---------------------------------------------------------

    snapshot_columns = [
        "rainfall_mean_mm",
        "rainfall_max_mm",
        "rainfall_3d_mm",
        "rainfall_7d_mm",
        "dmi",
        "roni",
        "pc1",
        "pc2",
        "amplitude",
        "mjo_phase",
        "temperature_c",
        "dewpoint_c",
        "relative_humidity_pct",
        "pressure_hpa",
        "wind_speed_ms",
    ]

    snapshot = {}

    for column in snapshot_columns:
        if column in row.columns:
            snapshot[column] = _json_safe(
                row.iloc[0][column]
            )

    # ---------------------------------------------------------
    # FINAL RESPONSE
    # ---------------------------------------------------------

    return {
        "location": "Bilaspur, Chhattisgarh",
        "as_of_date": target_date.strftime("%Y-%m-%d"),
        "horizon_days": horizon,

        "data_mode": "historical_hindcast_demo",

        "model_family": "RandomForestRegressor",

        "feature_set": (
            "rainfall + atmosphere + calendar"
        ),

        "rainfall_mean_mm": mean_quantiles,

        "rainfall_total_mm": total_quantiles,

        "rainfall_max": max_result,

        "input_snapshot": snapshot,

        "notes": [
            (
                "This is a historical hindcast/demo using "
                "the saved 2024-2025 dataset."
            ),
            (
                "It is not a current operational weather forecast."
            ),
            (
                "The 30-day maximum-rainfall model is intentionally "
                "not deployed."
            ),
            (
                "Prediction intervals are Random Forest ensemble "
                "spread proxies."
            ),
        ],
    }


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "MausamSaathi saved-model inference test"
        )
    )

    parser.add_argument(
        "--date",
        default="2025-09-01",
        help="Historical as-of date, e.g. 2025-09-01",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=14,
        choices=[7, 14, 21, 30],
    )

    args = parser.parse_args()

    result = run_demo_forecast(
        as_of=args.date,
        horizon=args.horizon,
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )
