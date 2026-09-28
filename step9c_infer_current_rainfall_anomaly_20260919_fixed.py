import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import Booster, DMatrix


ROOT = Path(__file__).resolve().parent

# Daily refresh uses stable "current" artifacts. The issue date is read from
# the feature snapshot itself, so this module is reusable for future dates.
SNAPSHOT = ROOT / r"data_test/processed/current_climate_2026/mausam_current_ml_features.csv"
OUT_DIR = ROOT / r"data_test/processed/current_climate_2026"
OUTPUT_CSV = OUT_DIR / "mausam_current_rainfall_anomaly.csv"
AUDIT_JSON = OUT_DIR / "mausam_current_rainfall_anomaly_audit.json"


MODELS = {
    7: {
        "meta": ROOT / r"data_test/processed/models/step5d_full_7d/rainfall_7d_anomaly_total_xgboost_time_split_1979_2015.metadata.json",
        "model": ROOT / r"data_test/processed/models/step5d_full_7d/rainfall_7d_anomaly_total_xgboost_time_split_1979_2015.json",
        "target": "future_rainfall_7d_anomaly_mm",
        "output": "rainfall_anomaly_7d_mm",
    },
    14: {
        "meta": ROOT / r"data_test/processed/models/step5e_full_14d/rainfall_14d_anomaly_total_xgboost_time_split_1979_2015.metadata.json",
        "model": ROOT / r"data_test/processed/models/step5e_full_14d/rainfall_14d_anomaly_total_xgboost_time_split_1979_2015.json",
        "target": "future_rainfall_14d_anomaly_mm",
        "output": "rainfall_anomaly_14d_mm",
    },
    21: {
        "meta": ROOT / r"data_test/processed/models/step5f_full_21d/rainfall_21d_anomaly_total_xgboost_time_split_1979_2015.metadata.json",
        "model": ROOT / r"data_test/processed/models/step5f_full_21d/rainfall_21d_anomaly_total_xgboost_time_split_1979_2015.json",
        "target": "future_rainfall_21d_anomaly_mm",
        "output": "rainfall_anomaly_21d_mm",
    },
    30: {
        "meta": ROOT / r"data_test/processed/models/step5g_full_30d/rainfall_30d_anomaly_total_xgboost_time_split_1979_2015.metadata.json",
        "model": ROOT / r"data_test/processed/models/step5g_full_30d/rainfall_30d_anomaly_total_xgboost_time_split_1979_2015.json",
        "target": "future_rainfall_30d_anomaly_mm",
        "output": "rainfall_anomaly_30d_mm",
    },
}


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def main() -> None:
    print("=" * 78)
    print("MAUSAMSAATHI SIH 2026 - CURRENT RAINFALL ANOMALY INFERENCE")
    print("=" * 78)

    if not SNAPSHOT.exists():
        fail(f"Feature snapshot not found: {SNAPSHOT}")

    df = pd.read_csv(SNAPSHOT)

    required_meta = ["date", "local_body_code", "local_body_name"]
    if list(df.columns[:3]) != required_meta:
        fail(
            "Snapshot metadata columns mismatch. "
            f"Expected first 3 columns {required_meta}, got {list(df.columns[:3])}"
        )

    feature_columns = list(df.columns[3:])
    if len(feature_columns) != 102:
        fail(f"Expected 102 model features, found {len(feature_columns)}")

    if df.empty:
        fail("Feature snapshot is empty.")

    if df["local_body_code"].isna().any():
        fail("local_body_code contains missing values.")

    if df["local_body_code"].duplicated().any():
        dupes = df.loc[
            df["local_body_code"].duplicated(),
            "local_body_code",
        ].tolist()[:10]
        fail(f"Duplicate local_body_code values found. Examples: {dupes}")

    if df["local_body_code"].nunique() != 486:
        fail(
            f"Expected 486 local bodies, "
            f"found {df['local_body_code'].nunique()}"
        )

    snapshot_dates = (
        df["date"].astype(str).dropna().unique().tolist()
    )
    if len(snapshot_dates) != 1:
        fail(
            "Expected a single issue date in the current feature snapshot, "
            f"found {snapshot_dates}"
        )

    issue_date = snapshot_dates[0]
    print(f"INPUT SNAPSHOT: {SNAPSHOT}")
    print(f"ROWS: {len(df)}")
    print(f"MODEL FEATURES: {len(feature_columns)}")
    print(f"ISSUE DATE: {issue_date}")
    print(f"LOCAL BODIES: {df['local_body_code'].nunique()}")

    X = df[feature_columns].copy()

    # Convert to numeric explicitly so inference fails loudly instead of
    # silently coercing a bad value later.
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors="raise")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        fail("Feature matrix contains NaN/Inf values.")

    result = df[["date", "local_body_code", "local_body_name"]].copy()
    audit = {
        "status": "PASS",
        "issue_date": issue_date,
        "snapshot": str(SNAPSHOT.relative_to(ROOT)),
        "output": str(OUTPUT_CSV.relative_to(ROOT)),
        "rows": int(len(df)),
        "local_body_count": int(df["local_body_code"].nunique()),
        "feature_count": int(len(feature_columns)),
        "models": {},
    }

    for horizon in (7, 14, 21, 30):
        cfg = MODELS[horizon]

        if not cfg["meta"].exists():
            fail(f"{horizon}D metadata not found: {cfg['meta']}")
        if not cfg["model"].exists():
            fail(f"{horizon}D model not found: {cfg['model']}")

        with cfg["meta"].open("r", encoding="utf-8") as f:
            meta = json.load(f)

        meta_features = meta.get("features")
        if meta.get("feature_count") != 102:
            fail(
                f"{horizon}D metadata feature_count is "
                f"{meta.get('feature_count')}, expected 102"
            )

        if meta_features != feature_columns:
            fail(
                f"{horizon}D model feature schema does not match "
                "current snapshot."
            )

        if meta.get("target") != cfg["target"]:
            fail(
                f"{horizon}D metadata target mismatch: "
                f"{meta.get('target')} != {cfg['target']}"
            )

        booster = Booster()
        booster.load_model(str(cfg["model"]))

        model_features = int(booster.num_features())
        if model_features != 102:
            fail(
                f"{horizon}D model reports {model_features} features, "
                "expected 102"
            )

        matrix = DMatrix(X, feature_names=feature_columns)
        pred = booster.predict(matrix)
        pred = np.asarray(pred, dtype=float).reshape(-1)

        if len(pred) != len(df):
            fail(
                f"{horizon}D prediction row count {len(pred)} "
                f"!= snapshot row count {len(df)}"
            )

        if not np.isfinite(pred).all():
            fail(f"{horizon}D predictions contain NaN/Inf values.")

        result[cfg["output"]] = pred

        stats = {
            "target": cfg["target"],
            "feature_count": 102,
            "model_feature_count": model_features,
            "rows": int(len(pred)),
            "min_mm": float(np.min(pred)),
            "max_mm": float(np.max(pred)),
            "mean_mm": float(np.mean(pred)),
        }
        audit["models"][f"{horizon}d"] = stats

        print(
            f"{horizon:>2}D: PASS | target={cfg['target']} | "
            f"min={stats['min_mm']:.4f} mm | "
            f"max={stats['max_mm']:.4f} mm | "
            f"mean={stats['mean_mm']:.4f} mm"
        )

    # Keep full precision in CSV; this is the model output layer.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False, float_format="%.10f")

    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_JSON.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    # Final shape / uniqueness checks on the generated output.
    written = pd.read_csv(OUTPUT_CSV)

    expected_columns = [
        "date",
        "local_body_code",
        "local_body_name",
        "rainfall_anomaly_7d_mm",
        "rainfall_anomaly_14d_mm",
        "rainfall_anomaly_21d_mm",
        "rainfall_anomaly_30d_mm",
    ]
    if list(written.columns) != expected_columns:
        fail(
            f"Output columns mismatch. Expected {expected_columns}, "
            f"got {list(written.columns)}"
        )

    if len(written) != 486:
        fail(
            f"Output rows mismatch: expected 486, got {len(written)}"
        )

    if written["local_body_code"].duplicated().any():
        fail("Generated output contains duplicate local_body_code values.")

    anomaly_columns = expected_columns[3:]
    if not np.isfinite(
        written[anomaly_columns].to_numpy(dtype=float)
    ).all():
        fail("Generated output contains NaN/Inf anomaly values.")

    print("-" * 78)
    print(f"OUTPUT CSV: {OUTPUT_CSV}")
    print(f"AUDIT JSON: {AUDIT_JSON}")
    print("OUTPUT ROWS:", len(written))
    print("OUTPUT COLUMNS:", len(written.columns))
    print("GENERATED LOCAL BODIES:", written["local_body_code"].nunique())
    print("ISSUE DATE:", issue_date)
    print("RAINFALL ANOMALY INFERENCE: PASS")
    print("=" * 78)


if __name__ == "__main__":
    main()
