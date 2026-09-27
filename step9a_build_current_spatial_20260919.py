from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent

HIERARCHY_PATH = ROOT / "data_test" / "geo" / "lgd" / "bilaspur_lgd_block_panchayat_hierarchy_2026.json"
BOUNDARY_PATH = ROOT / "data_test" / "geo" / "bilaspur" / "bilaspur_gram_panchayat_boundaries.geojson"
PREDICTION_PATH = ROOT / "data_test" / "processed" / "current_climate_2026" / "mausam_current_ml_predictions_20260919.csv"

OUT_DIR = ROOT / "data_test" / "processed" / "current_spatial_2026"
LAYER_DIR = OUT_DIR / "layers_20260919"

ISSUE_DATE = "2026-09-19"
HORIZONS = (7, 14, 21, 30)
TARGETS = (
    "onset",
    "heavy_rain_panchayat_mean",
    "heavy_rain_grid",
    "active",
    "break_proxy",
)


def fail(message: str) -> None:
    raise SystemExit(f"\n[FAIL] {message}")


def load_json(path: Path) -> Any:
    if not path.exists():
        fail(f"Missing file: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        fail(f"Could not read JSON {path}: {exc}")


def load_predictions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        fail(f"Missing prediction CSV: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        fail(f"Could not read prediction CSV {path}: {exc}")


def as_code(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def finite_probability(value: str, field: str, local_body_code: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        fail(f"Non-numeric probability in {field} for local_body_code={local_body_code}: {value!r}")

    if not math.isfinite(x):
        fail(f"Non-finite probability in {field} for local_body_code={local_body_code}: {value!r}")

    if x < 0.0 or x > 1.0:
        fail(f"Probability outside [0,1] in {field} for local_body_code={local_body_code}: {x}")

    return x


def build_hierarchy() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    data = load_json(HIERARCHY_PATH)

    if not isinstance(data, dict):
        fail("Hierarchy root must be a JSON object.")

    blocks = data.get("janpad_panchayats")
    if not isinstance(blocks, list):
        fail("Hierarchy is missing 'janpad_panchayats' list.")

    code_meta: dict[str, dict[str, Any]] = {}
    code_to_block: dict[str, str] = {}

    for block in blocks:
        if not isinstance(block, dict):
            fail("Invalid block entry in hierarchy.")

        block_name = str(block.get("block_name", "")).strip()
        block_code = as_code(block.get("block_code"))
        gram_panchayats = block.get("gram_panchayats")

        if not block_name or not block_code:
            fail(f"Invalid block metadata: {block}")

        if not isinstance(gram_panchayats, list):
            fail(f"Block {block_name} has no valid 'gram_panchayats' list.")

        for gp in gram_panchayats:
            if not isinstance(gp, dict):
                fail(f"Invalid gram panchayat entry under {block_name}.")

            code = as_code(gp.get("local_body_code"))
            name = str(gp.get("name", "")).strip()

            if not code or not name:
                fail(f"Invalid local body under {block_name}: {gp}")

            if code in code_meta:
                fail(f"Duplicate local_body_code in hierarchy: {code}")

            code_meta[code] = {
                "local_body_code": code,
                "local_body_name": name,
                "block_code": block_code,
                "block_name": block_name,
            }
            code_to_block[code] = block_name

    if len(code_meta) != 486:
        fail(f"Hierarchy local body count is {len(code_meta)}, expected 486.")

    validation = data.get("validation", {})
    if isinstance(validation, dict):
        reported = validation.get("total_gram_panchayats")
        unique_reported = validation.get("unique_local_body_codes")
        if reported is not None and int(reported) != 486:
            fail(f"Hierarchy validation.total_gram_panchayats={reported}, expected 486.")
        if unique_reported is not None and int(unique_reported) != 486:
            fail(f"Hierarchy validation.unique_local_body_codes={unique_reported}, expected 486.")

    print(f"HIERARCHY LOCAL BODIES: {len(code_meta)}")
    print(f"HIERARCHY BLOCKS: {sorted(set(code_to_block.values()))}")
    return code_meta, code_to_block


def build_prediction_map(code_meta: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows = load_predictions(PREDICTION_PATH)

    if len(rows) != 486:
        fail(f"Prediction CSV row count is {len(rows)}, expected 486.")

    if not rows:
        fail("Prediction CSV is empty.")

    required = {"date", "local_body_code", "local_body_name"}
    for horizon in HORIZONS:
        for target in TARGETS:
            required.add(f"{target}_{horizon}d_raw")
            required.add(f"{target}_{horizon}d_calibrated")

    actual_columns = set(rows[0].keys())
    missing_columns = sorted(required - actual_columns)
    if missing_columns:
        fail(f"Prediction CSV is missing required columns: {missing_columns}")

    by_code: dict[str, dict[str, Any]] = {}

    for row in rows:
        code = as_code(row.get("local_body_code"))
        if not code:
            fail("Prediction row contains blank local_body_code.")

        if code in by_code:
            fail(f"Duplicate local_body_code in prediction CSV: {code}")

        if code not in code_meta:
            fail(f"Prediction local_body_code not found in canonical hierarchy: {code}")

        if str(row.get("date", "")).strip() != ISSUE_DATE:
            fail(
                f"Prediction date mismatch for {code}: "
                f"{row.get('date')!r}, expected {ISSUE_DATE!r}"
            )

        prediction_name = str(row.get("local_body_name", "")).strip()
        expected_name = code_meta[code]["local_body_name"]
        if prediction_name != expected_name:
            fail(
                f"Prediction name mismatch for {code}: "
                f"CSV={prediction_name!r}, hierarchy={expected_name!r}"
            )

        normalized: dict[str, Any] = {
            "date": ISSUE_DATE,
            "local_body_code": code,
            "local_body_name": prediction_name,
        }

        for horizon in HORIZONS:
            for target in TARGETS:
                raw_src = f"{target}_{horizon}d_raw"
                cal_src = f"{target}_{horizon}d_calibrated"
                normalized[f"{target}_{horizon}d_raw"] = finite_probability(
                    row[raw_src], raw_src, code
                )
                normalized[f"{target}_{horizon}d_calibrated"] = finite_probability(
                    row[cal_src], cal_src, code
                )

        by_code[code] = normalized

    print(f"PREDICTION ROWS: {len(by_code)}")
    print("PREDICTION DUPLICATE KEYS: 0")
    return by_code


def load_boundaries(code_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    data = load_json(BOUNDARY_PATH)

    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        fail("Boundary GeoJSON must be a FeatureCollection.")

    features = data.get("features")
    if not isinstance(features, list):
        fail("Boundary GeoJSON is missing a features list.")

    if len(features) != 486:
        fail(f"Boundary feature count is {len(features)}, expected 486.")

    seen: set[str] = set()
    output: list[dict[str, Any]] = []

    for feature in features:
        if not isinstance(feature, dict):
            fail("Boundary GeoJSON contains a non-object feature.")

        props = feature.get("properties") or {}
        code = as_code(props.get("local_body_code"))

        if not code:
            fail("Boundary feature has blank local_body_code.")

        if code in seen:
            fail(f"Duplicate local_body_code in boundaries: {code}")

        if code not in code_meta:
            fail(f"Boundary local_body_code not found in canonical hierarchy: {code}")

        geometry = feature.get("geometry")
        if not isinstance(geometry, dict) or not geometry.get("type") or "coordinates" not in geometry:
            fail(f"Boundary geometry missing/invalid for local_body_code={code}")

        boundary_name = str(props.get("local_body_name", "")).strip()
        hierarchy_name = code_meta[code]["local_body_name"]
        if boundary_name != hierarchy_name:
            fail(
                f"Boundary name mismatch for {code}: "
                f"boundary={boundary_name!r}, hierarchy={hierarchy_name!r}"
            )

        seen.add(code)
        output.append(feature)

    print(f"BOUNDARY FEATURES: {len(output)}")
    print("BOUNDARY DUPLICATE KEYS: 0")
    return output


def add_prediction_properties(
    base_properties: dict[str, Any],
    meta: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    # Preserve canonical boundary properties exactly, then add current
    # prediction metadata and the same probability naming convention used
    # by the existing 2023 spatial layer.
    props = dict(base_properties)

    props["mausam_date"] = ISSUE_DATE
    props["mausam_local_body_code"] = prediction["local_body_code"]
    props["mausam_local_body_name"] = prediction["local_body_name"]

    # Add block metadata without altering the existing boundary properties.
    props["mausam_block_code"] = meta["block_code"]
    props["mausam_block_name"] = meta["block_name"]

    for horizon in HORIZONS:
        for target in TARGETS:
            src_raw = f"{target}_{horizon}d_raw"
            src_cal = f"{target}_{horizon}d_calibrated"
            props[f"{target}_{horizon}d__raw_probability"] = prediction[src_raw]
            props[f"{target}_{horizon}d__calibrated_probability"] = prediction[src_cal]

    return props


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)


def build_main_geojson(
    boundaries: list[dict[str, Any]],
    code_meta: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> Path:
    features: list[dict[str, Any]] = []

    for boundary in boundaries:
        code = as_code(boundary["properties"]["local_body_code"])
        prediction = predictions.get(code)
        if prediction is None:
            fail(f"No live prediction for boundary local_body_code={code}")

        feature = {
            "type": "Feature",
            "properties": add_prediction_properties(
                boundary.get("properties") or {},
                code_meta[code],
                prediction,
            ),
            "geometry": boundary["geometry"],
        }
        features.append(feature)

    # Preserve input feature order. This also preserves the canonical
    # boundary ordering used by the source file.
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    out = OUT_DIR / f"mausam_current_spatial_{ISSUE_DATE.replace('-', '')}.geojson"
    write_json(out, geojson)
    print(f"MAIN GEOJSON: {out}")
    print(f"MAIN GEOJSON FEATURES: {len(features)}")
    return out


def build_layer_geojsons(
    boundaries: list[dict[str, Any]],
    code_meta: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> list[Path]:
    outputs: list[Path] = []

    for horizon in HORIZONS:
        for target in TARGETS:
            features: list[dict[str, Any]] = []

            raw_key = f"{target}_{horizon}d__raw_probability"
            cal_key = f"{target}_{horizon}d__calibrated_probability"

            for boundary in boundaries:
                code = as_code(boundary["properties"]["local_body_code"])
                prediction = predictions[code]

                base_props = boundary.get("properties") or {}
                props = {
                    "local_body_code": code,
                    "local_body_name": base_props.get(
                        "local_body_name", code_meta[code]["local_body_name"]
                    ),
                    "mausam_date": ISSUE_DATE,
                    "mausam_local_body_code": code,
                    "mausam_local_body_name": prediction["local_body_name"],
                    "mausam_block_code": code_meta[code]["block_code"],
                    "mausam_block_name": code_meta[code]["block_name"],
                    raw_key: prediction[f"{target}_{horizon}d_raw"],
                    cal_key: prediction[f"{target}_{horizon}d_calibrated"],
                }

                # Preserve useful canonical boundary metadata in each layer.
                for k in ("village_count", "subdistrict_names", "boundary_source", "join_method", "name_join_used"):
                    if k in base_props:
                        props[k] = base_props[k]

                features.append(
                    {
                        "type": "Feature",
                        "properties": props,
                        "geometry": boundary["geometry"],
                    }
                )

            layer = {
                "type": "FeatureCollection",
                "features": features,
            }

            out = LAYER_DIR / f"{horizon}d_{target}_{horizon}d.geojson"
            write_json(out, layer)
            outputs.append(out)

    print(f"INDIVIDUAL LAYERS WRITTEN: {len(outputs)}")
    return outputs


def build_audit(
    boundaries: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    main_path: Path,
    layer_paths: list[Path],
    code_meta: dict[str, dict[str, Any]],
) -> Path:
    boundary_codes = {
        as_code(f["properties"]["local_body_code"]) for f in boundaries
    }
    prediction_codes = set(predictions)
    hierarchy_codes = set(code_meta)

    duplicate_boundary_codes = [
        code
        for code, count in Counter(
            as_code(f["properties"]["local_body_code"]) for f in boundaries
        ).items()
        if count > 1
    ]

    calibrated_fields = [
        f"{target}_{horizon}d__calibrated_probability"
        for horizon in HORIZONS
        for target in TARGETS
    ]
    raw_fields = [
        f"{target}_{horizon}d__raw_probability"
        for horizon in HORIZONS
        for target in TARGETS
    ]

    audit = {
        "project": "MAUSAMSAATHI SIH 2026",
        "step": "9A current spatial prediction layer",
        "issue_date": ISSUE_DATE,
        "status": "PASS",
        "join_key": "local_body_code",
        "canonical_local_bodies": len(hierarchy_codes),
        "boundary_features": len(boundary_codes),
        "prediction_rows": len(prediction_codes),
        "hierarchy_boundary_missing": sorted(hierarchy_codes - boundary_codes),
        "boundary_hierarchy_extra": sorted(boundary_codes - hierarchy_codes),
        "hierarchy_prediction_missing": sorted(hierarchy_codes - prediction_codes),
        "prediction_hierarchy_extra": sorted(prediction_codes - hierarchy_codes),
        "duplicate_boundary_keys": duplicate_boundary_codes,
        "duplicate_prediction_keys": [],
        "horizons_days": list(HORIZONS),
        "targets": list(TARGETS),
        "raw_probability_fields_count": len(raw_fields),
        "calibrated_probability_fields_count": len(calibrated_fields),
        "main_geojson": str(main_path.relative_to(ROOT)),
        "layer_count": len(layer_paths),
        "layer_files": [str(p.relative_to(ROOT)) for p in layer_paths],
        "existing_2023_layer_untouched": True,
    }

    out = OUT_DIR / f"spatial_build_{ISSUE_DATE.replace('-', '')}_audit.json"
    write_json(out, audit)
    print(f"AUDIT: {out}")
    return out


def main() -> None:
    print("=" * 78)
    print("MAUSAMSAATHI SIH 2026 - STEP 9A")
    print("BUILD CURRENT SPATIAL PREDICTION LAYER")
    print("=" * 78)
    print(f"Issue date: {ISSUE_DATE}")
    print()

    code_meta, _ = build_hierarchy()
    predictions = build_prediction_map(code_meta)
    boundaries = load_boundaries(code_meta)

    hierarchy_codes = set(code_meta)
    prediction_codes = set(predictions)
    boundary_codes = {
        as_code(f["properties"]["local_body_code"]) for f in boundaries
    }

    if not (hierarchy_codes == prediction_codes == boundary_codes):
        fail("Final 486-way local_body_code join is not exact.")

    if not (
        len(hierarchy_codes) == 486
        and len(prediction_codes) == 486
        and len(boundary_codes) == 486
    ):
        fail("Final geography/prediction counts are not all 486.")

    print()
    print("EXACT JOIN: PASS")
    print("HIERARCHY: 486")
    print("BOUNDARY: 486")
    print("PREDICTIONS: 486")
    print("DUPLICATES: 0")
    print()

    main_path = build_main_geojson(boundaries, code_meta, predictions)
    layer_paths = build_layer_geojsons(boundaries, code_meta, predictions)
    audit_path = build_audit(
        boundaries, predictions, main_path, layer_paths, code_meta
    )

    print()
    print("=" * 78)
    print("STEP 9A CURRENT SPATIAL BUILD COMPLETE")
    print("=" * 78)
    print("STATUS: PASS")
    print(f"Main GeoJSON: {main_path}")
    print(f"Layer directory: {LAYER_DIR}")
    print(f"Audit: {audit_path}")
    print(f"Features: 486")
    print(f"Layers: {len(layer_paths)}")
    print("Existing 2023 spatial files: UNTOUCHED")
    print()


if __name__ == "__main__":
    main()
