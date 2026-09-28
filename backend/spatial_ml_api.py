from __future__ import annotations

import json
import io

try:
    from .cloud_published_runtime import cloud_runtime_configured, ensure_current_runtime
except ImportError:
    from cloud_published_runtime import cloud_runtime_configured, ensure_current_runtime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse


ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# LIVE 2026 SPATIAL DATA
# Built by Step 9A from:
#   canonical 486-local-body hierarchy
#   + canonical 486-boundary GeoJSON
#   + 2026-09-19 live ML predictions
# ---------------------------------------------------------------------------

ISSUE_DATE = None
EXPECTED_BODIES = 486
HORIZONS = (7, 14, 21, 30)
INDICATORS = (
    "heavy_rain_panchayat_mean",
    "heavy_rain_grid",
    "onset",
    "break_proxy",
    "active",
)
PROBABILITY_TYPES = ("raw", "calibrated")

PREDICTIONS_PATH = (
    ROOT
    / "data_test"
    / "processed"
    / "current_climate_2026"
    / "mausam_current_ml_predictions.csv"
)

FEATURE_SNAPSHOT_PATH = (
    ROOT
    / "data_test"
    / "processed"
    / "current_climate_2026"
    / "mausam_current_ml_features.csv"
)

RAINFALL_ANOMALY_PATH = (
    ROOT
    / "data_test"
    / "processed"
    / "current_climate_2026"
    / "mausam_current_rainfall_anomaly.csv"
)

SPATIAL_GEOJSON_PATH = (
    ROOT
    / "data_test"
    / "processed"
    / "current_spatial_2026"
    / "mausam_current_spatial_current.geojson"
)

SPATIAL_LAYER_DIR = (
    ROOT
    / "data_test"
    / "processed"
    / "current_spatial_2026"
    / "layers_current"
)

router = APIRouter(tags=["spatial-ml"])


def _load_predictions() -> pd.DataFrame:
    if cloud_runtime_configured(ROOT):
        try:
            ensure_current_runtime(
                ROOT,
                ROOT / "data_test" / "processed" / "current_climate_2026",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Published cloud runtime synchronization failed: {exc}",
            ) from exc

    global ISSUE_DATE
    if not PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Live 2026 spatial ML prediction snapshot is not available: "
                f"{PREDICTIONS_PATH}"
            ),
        )

    try:
        df = pd.read_csv(PREDICTIONS_PATH, dtype={"local_body_code": str})
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read live 2026 spatial ML prediction snapshot: {exc}",
        ) from exc

    required = {"date", "local_body_code", "local_body_name"}
    for horizon in HORIZONS:
        for indicator in INDICATORS:
            required.add(f"{indicator}_{horizon}d_raw")
            required.add(f"{indicator}_{horizon}d_calibrated")

    missing = sorted(required - set(df.columns))
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Live spatial ML snapshot missing required columns: {missing}",
        )

    if len(df) != EXPECTED_BODIES:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Live spatial ML snapshot has {len(df)} rows; "
                f"expected {EXPECTED_BODIES}."
            ),
        )

    df["local_body_code"] = df["local_body_code"].astype(str).str.strip()
    if df["local_body_code"].eq("").any():
        raise HTTPException(
            status_code=500,
            detail="Live spatial ML snapshot contains blank local_body_code values.",
        )

    if df["local_body_code"].nunique() != EXPECTED_BODIES:
        raise HTTPException(
            status_code=500,
            detail="Live spatial ML snapshot does not contain 486 unique local bodies.",
        )

    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        raise HTTPException(
            status_code=500,
            detail="Live spatial ML snapshot contains invalid dates.",
        )

    unique_dates = dates.dt.strftime("%Y-%m-%d").unique().tolist()
    if len(unique_dates) != 1:
        raise HTTPException(
            status_code=500,
            detail=(
                "Live spatial ML snapshot must contain exactly one issue date; "
                f"found {unique_dates}."
            ),
        )
    ISSUE_DATE = unique_dates[0]

    # Validate every probability field is numeric and within [0, 1].
    probability_columns = [
        f"{indicator}_{horizon}d_{ptype}"
        for horizon in HORIZONS
        for indicator in INDICATORS
        for ptype in PROBABILITY_TYPES
    ]

    numeric = df[probability_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = [
            col for col in probability_columns if numeric[col].isna().any()
        ]
        raise HTTPException(
            status_code=500,
            detail=f"Live spatial ML snapshot has invalid probability values in: {bad}",
        )

    outside = (numeric < 0.0) | (numeric > 1.0)
    if outside.any().any():
        bad = [col for col in probability_columns if outside[col].any()]
        raise HTTPException(
            status_code=500,
            detail=f"Live spatial ML snapshot has probability values outside [0,1]: {bad}",
        )

    return df


def _load_features() -> pd.DataFrame:
    if cloud_runtime_configured(ROOT):
        try:
            ensure_current_runtime(
                ROOT,
                ROOT / "data_test" / "processed" / "current_climate_2026",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Published cloud runtime synchronization failed: {exc}",
            ) from exc

    if not FEATURE_SNAPSHOT_PATH.exists():
        return pd.DataFrame(columns=["local_body_code", "rainfall_anomaly_mm"])

    try:
        features = pd.read_csv(
            FEATURE_SNAPSHOT_PATH,
            dtype={"local_body_code": str},
            usecols=["local_body_code", "rainfall_anomaly_mm"],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read live feature snapshot: {exc}",
        ) from exc

    features["local_body_code"] = features["local_body_code"].astype(str).str.strip()
    return features


def _load_spatial_geojson() -> dict[str, Any]:
    if cloud_runtime_configured(ROOT):
        try:
            ensure_current_runtime(
                ROOT,
                ROOT / "data_test" / "processed" / "current_climate_2026",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Published cloud runtime synchronization failed: {exc}",
            ) from exc

    if not SPATIAL_GEOJSON_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Live 2026 spatial GeoJSON is not available: "
                f"{SPATIAL_GEOJSON_PATH}"
            ),
        )

    try:
        with SPATIAL_GEOJSON_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read live spatial GeoJSON: {exc}",
        ) from exc

    if data.get("type") != "FeatureCollection":
        raise HTTPException(
            status_code=500,
            detail="Live spatial GeoJSON is not a FeatureCollection.",
        )

    features = data.get("features")
    if not isinstance(features, list) or len(features) != EXPECTED_BODIES:
        raise HTTPException(
            status_code=500,
            detail=(
                "Live spatial GeoJSON feature count mismatch. "
                f"Expected {EXPECTED_BODIES}."
            ),
        )

    return data


def _probability_column(
    indicator: str,
    horizon: int,
    probability_type: str,
) -> str:
    if horizon not in HORIZONS:
        raise HTTPException(
            status_code=400,
            detail="Horizon must be 7, 14, 21 or 30 days.",
        )

    if indicator not in INDICATORS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid indicator. Choose one of: {', '.join(INDICATORS)}",
        )

    if probability_type not in PROBABILITY_TYPES:
        raise HTTPException(
            status_code=400,
            detail="probability_type must be 'raw' or 'calibrated'.",
        )

    return f"{indicator}_{horizon}d_{probability_type}"


def _load_rainfall_anomaly_predictions() -> pd.DataFrame:
    if cloud_runtime_configured(ROOT):
        try:
            ensure_current_runtime(
                ROOT,
                ROOT / "data_test" / "processed" / "current_climate_2026",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Published cloud runtime synchronization failed: {exc}",
            ) from exc

    if not RAINFALL_ANOMALY_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Live 2026 horizon-specific rainfall anomaly predictions "
                "are not available: "
                f"{RAINFALL_ANOMALY_PATH}"
            ),
        )

    try:
        anomalies = pd.read_csv(
            RAINFALL_ANOMALY_PATH,
            dtype={"local_body_code": str},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read live rainfall anomaly predictions: {exc}",
        ) from exc

    required = {
        "date",
        "local_body_code",
        "local_body_name",
        "rainfall_anomaly_7d_mm",
        "rainfall_anomaly_14d_mm",
        "rainfall_anomaly_21d_mm",
        "rainfall_anomaly_30d_mm",
    }

    missing = sorted(required - set(anomalies.columns))
    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                "Live rainfall anomaly prediction file is missing required "
                f"columns: {missing}"
            ),
        )

    if len(anomalies) != EXPECTED_BODIES:
        raise HTTPException(
            status_code=500,
            detail=(
                "Live rainfall anomaly prediction file has "
                f"{len(anomalies)} rows; expected {EXPECTED_BODIES}."
            ),
        )

    anomalies["local_body_code"] = (
        anomalies["local_body_code"].astype(str).str.strip()
    )

    if anomalies["local_body_code"].eq("").any():
        raise HTTPException(
            status_code=500,
            detail=(
                "Live rainfall anomaly prediction file contains blank "
                "local_body_code values."
            ),
        )

    if anomalies["local_body_code"].nunique() != EXPECTED_BODIES:
        raise HTTPException(
            status_code=500,
            detail=(
                "Live rainfall anomaly prediction file does not contain "
                "486 unique local bodies."
            ),
        )

    dates = pd.to_datetime(anomalies["date"], errors="coerce")
    if dates.isna().any():
        raise HTTPException(
            status_code=500,
            detail="Live rainfall anomaly prediction file contains invalid dates.",
        )

    unique_dates = dates.dt.strftime("%Y-%m-%d").unique().tolist()
    if len(unique_dates) != 1:
        raise HTTPException(
            status_code=500,
            detail=(
                "Live rainfall anomaly file must contain exactly one issue date; "
                f"found {unique_dates}."
            ),
        )
    if unique_dates[0] != ISSUE_DATE:
        raise HTTPException(
            status_code=500,
            detail=(
                "Live rainfall anomaly date does not match current prediction date. "
                f"Prediction={ISSUE_DATE}, anomaly={unique_dates[0]}."
            ),
        )

    anomaly_columns = [
        "rainfall_anomaly_7d_mm",
        "rainfall_anomaly_14d_mm",
        "rainfall_anomaly_21d_mm",
        "rainfall_anomaly_30d_mm",
    ]
    numeric = anomalies[anomaly_columns].apply(pd.to_numeric, errors="coerce")

    if numeric.isna().any().any():
        bad = [col for col in anomaly_columns if numeric[col].isna().any()]
        raise HTTPException(
            status_code=500,
            detail=(
                "Live rainfall anomaly predictions contain invalid numeric "
                f"values in: {bad}"
            ),
        )

    if not numeric.notna().all().all():
        raise HTTPException(
            status_code=500,
            detail="Live rainfall anomaly predictions contain invalid values.",
        )

    if (numeric == float("inf")).any().any() or (numeric == float("-inf")).any().any():
        raise HTTPException(
            status_code=500,
            detail="Live rainfall anomaly predictions contain infinite values.",
        )

    return anomalies


def _rainfall_anomaly_map(horizon: int) -> dict[str, float | None]:
    if horizon not in HORIZONS:
        raise HTTPException(
            status_code=400,
            detail="Horizon must be 7, 14, 21 or 30 days.",
        )

    anomalies = _load_rainfall_anomaly_predictions()
    column = f"rainfall_anomaly_{horizon}d_mm"

    return {
        str(row["local_body_code"]): float(row[column])
        for _, row in anomalies.iterrows()
    }


@router.get("/ml-spatial-forecast")
def ml_spatial_forecast(
    horizon: int = Query(14, description="Forecast horizon in days"),
    probability_type: Literal["raw", "calibrated"] = Query(
        "raw",
        description="Raw model probability or calibrated probability",
    ),
    indicator: str | None = Query(
        None,
        description=(
            "Optional indicator. Omit it to return all five spatial indicators."
        ),
    ),
    local_body_code: str | None = Query(
        None,
        description="Optional local-body code for a single Panchayat result.",
    ),
):
    if horizon not in HORIZONS:
        raise HTTPException(
            status_code=400,
            detail="Horizon must be 7, 14, 21 or 30 days.",
        )

    if indicator is not None and indicator not in INDICATORS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid indicator. Choose one of: {', '.join(INDICATORS)}",
        )

    column = (
        _probability_column(indicator, horizon, probability_type)
        if indicator is not None
        else None
    )

    df = _load_predictions()
    anomaly_map = _rainfall_anomaly_map(horizon)

    if local_body_code is not None:
        code = str(local_body_code).strip()
        matched = df[df["local_body_code"] == code].copy()

        if matched.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Local body not found: {local_body_code}",
            )

        if len(matched) != 1:
            raise HTTPException(
                status_code=500,
                detail=f"Expected exactly one prediction for local body {code}.",
            )

        row = matched.iloc[0]

        if indicator is not None:
            value = float(row[column])
            return {
                "status": "ok",
                "snapshot_date": ISSUE_DATE,
                "historical_hindcast": False,
                "live": True,
                "horizon": horizon,
                "probability_type": probability_type,
                "indicator": indicator,
                "local_body_count": 1,
                "predictions": [
                    {
                        "local_body_code": code,
                        "local_body_name": str(row["local_body_name"]),
                        "probability": value,
                    }
                ],
                "rainfall_anomaly_mm": anomaly_map.get(code),
                "rainfall_anomaly_source": str(RAINFALL_ANOMALY_PATH),
                "source": str(PREDICTIONS_PATH),
            }

        # When a local body is requested without an indicator, return all five
        # live indicators for that single Panchayat. This is what the frontend
        # uses to populate the outlook cards and rainfall anomaly.
        selected_indicators: dict[str, list[dict[str, object]]] = {}
        for name in INDICATORS:
            selected_column = _probability_column(
                name, horizon, probability_type
            )
            selected_indicators[name] = [
                {
                    "local_body_code": code,
                    "local_body_name": str(row["local_body_name"]),
                    "probability": float(row[selected_column]),
                }
            ]

        return {
            "status": "ok",
            "snapshot_date": ISSUE_DATE,
            "historical_hindcast": False,
            "live": True,
            "horizon": horizon,
            "probability_type": probability_type,
            "indicator": None,
            "local_body_count": 1,
            "indicators": selected_indicators,
            "rainfall_anomaly_mm": anomaly_map.get(code),
            "rainfall_anomaly_source": str(RAINFALL_ANOMALY_PATH),
            "source": str(PREDICTIONS_PATH),
        }

    if indicator is not None:
        result_rows = []
        for _, row in df.iterrows():
            value = float(row[column])
            result_rows.append(
                {
                    "local_body_code": str(row["local_body_code"]),
                    "local_body_name": str(row["local_body_name"]),
                    "probability": value,
                }
            )

        return {
            "status": "ok",
            "snapshot_date": ISSUE_DATE,
            "historical_hindcast": False,
            "live": True,
            "horizon": horizon,
            "probability_type": probability_type,
            "indicator": indicator,
            "local_body_count": len(result_rows),
            "predictions": result_rows,
            "source": str(PREDICTIONS_PATH),
        }

    indicators: dict[str, list[dict[str, object]]] = {}

    for name in INDICATORS:
        current_column = _probability_column(name, horizon, probability_type)
        values: list[dict[str, object]] = []

        for _, row in df.iterrows():
            values.append(
                {
                    "local_body_code": str(row["local_body_code"]),
                    "local_body_name": str(row["local_body_name"]),
                    "probability": float(row[current_column]),
                }
            )

        indicators[name] = values

    return {
        "status": "ok",
        "snapshot_date": ISSUE_DATE,
        "historical_hindcast": False,
        "live": True,
        "horizon": horizon,
        "probability_type": probability_type,
        "indicator": None,
        "local_body_count": int(len(df)),
        "indicators": indicators,
        "source": str(PREDICTIONS_PATH),
    }


@router.get("/current-state")
def current_state():
    if cloud_runtime_configured(ROOT):
        try:
            ensure_current_runtime(
                ROOT,
                ROOT / "data_test" / "processed" / "current_climate_2026",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Published cloud runtime synchronization failed: {exc}",
            ) from exc

    state_path = (
        ROOT
        / "data_test"
        / "processed"
        / "current_climate_2026"
        / "current_state.json"
    )

    if not state_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Current state manifest is not available: {state_path}",
        )

    try:
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read current state manifest: {exc}",
        ) from exc

    return state


@router.get("/ml-spatial-forecast/meta")
def ml_spatial_forecast_meta():
    _load_predictions()
    return {
        "status": "ok",
        "snapshot_date": ISSUE_DATE,
        "historical_hindcast": False,
        "live": True,
        "horizons": list(HORIZONS),
        "indicators": list(INDICATORS),
        "probability_types": list(PROBABILITY_TYPES),
        "local_body_count": EXPECTED_BODIES,
        "prediction_source": str(PREDICTIONS_PATH),
        "feature_source": str(FEATURE_SNAPSHOT_PATH),
        "rainfall_anomaly_source": str(RAINFALL_ANOMALY_PATH),
        "rainfall_anomaly_horizons": list(HORIZONS),
        "geojson_source": str(SPATIAL_GEOJSON_PATH),
    }


@router.get("/ml-spatial-geojson")
def ml_spatial_geojson(
    horizon: int | None = Query(
        None,
        description="Optional horizon. When supplied, return a compact layer.",
    ),
    indicator: str | None = Query(
        None,
        description="Optional indicator. Required when horizon is supplied.",
    ),
    probability_type: Literal["raw", "calibrated"] = Query(
        "calibrated",
        description="Probability type for compact layer output.",
    ),
):
    data = _load_spatial_geojson()

    # No filter: return the complete current 2026 spatial FeatureCollection.
    if horizon is None and indicator is None:
        response = JSONResponse(content=data)
        response.headers["X-Mausam-Data-Date"] = ISSUE_DATE
        response.headers["X-Mausam-Live"] = "true"
        response.headers["X-Mausam-Local-Bodies"] = str(EXPECTED_BODIES)
        return response

    if horizon is None or indicator is None:
        raise HTTPException(
            status_code=400,
            detail="horizon and indicator must be supplied together.",
        )

    target_key = _probability_column(indicator, horizon, probability_type)
    properties_key = f"{indicator}_{horizon}d__{probability_type}_probability"

    features = []
    for feature in data["features"]:
        props = feature.get("properties") or {}
        if target_key not in {
            f"{name}_{horizon}d__raw_probability"
            for name in INDICATORS
        } | {
            f"{name}_{horizon}d__calibrated_probability"
            for name in INDICATORS
        }:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Spatial GeoJSON property missing for {indicator} "
                    f"{horizon}d {probability_type}: {properties_key}"
                ),
            )

        compact_props = {
            "local_body_code": str(props.get("local_body_code", "")),
            "local_body_name": str(props.get("local_body_name", "")),
            "mausam_date": str(props.get("mausam_date", ISSUE_DATE)),
            "mausam_block_code": props.get("mausam_block_code"),
            "mausam_block_name": props.get("mausam_block_name"),
            properties_key: props.get(properties_key),
        }

        for key in (
            "village_count",
            "subdistrict_names",
            "boundary_source",
            "join_method",
            "name_join_used",
        ):
            if key in props:
                compact_props[key] = props[key]

        features.append(
            {
                "type": "Feature",
                "properties": compact_props,
                "geometry": feature.get("geometry"),
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "snapshot_date": ISSUE_DATE,
            "live": True,
            "historical_hindcast": False,
            "horizon": horizon,
            "indicator": indicator,
            "probability_type": probability_type,
            "local_body_count": len(features),
            "source": str(SPATIAL_GEOJSON_PATH),
        },
    }


@router.get("/ml-spatial-geojson/meta")
def ml_spatial_geojson_meta():
    _load_predictions()

    layer_files = [
        f"{horizon}d_{indicator}_{horizon}d.geojson"
        for horizon in HORIZONS
        for indicator in INDICATORS
    ]

    return {
        "status": "ok",
        "snapshot_date": ISSUE_DATE,
        "live": True,
        "historical_hindcast": False,
        "local_body_count": EXPECTED_BODIES,
        "horizons": list(HORIZONS),
        "indicators": list(INDICATORS),
        "probability_types": list(PROBABILITY_TYPES),
        "geojson": str(SPATIAL_GEOJSON_PATH),
        "layer_directory": str(SPATIAL_LAYER_DIR),
        "layer_files": layer_files,
        "layer_count": len(layer_files),
    }
