from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import csv
import json
import re

try:
    from .cloud_published_runtime import (
        cloud_runtime_configured,
        ensure_current_runtime,
        published_state,
    )
except ImportError:
    from cloud_published_runtime import (
        cloud_runtime_configured,
        ensure_current_runtime,
        published_state,
    )

try:
    from .ml_inference import run_demo_forecast
except ImportError:
    from ml_inference import run_demo_forecast
try:
    from .spatial_ml_api import router as spatial_ml_router
except ImportError:
    from spatial_ml_api import router as spatial_ml_router

try:
    from .persistence_api import router as persistence_router
except ImportError:
    from persistence_api import router as persistence_router
try:
    from .auth_persistence_api import router as authenticated_persistence_router
except ImportError:
    from auth_persistence_api import router as authenticated_persistence_router


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA_PATH = BASE / "data" / "forecast.json"

LOCATION_MAPPING_PATH = (
    ROOT
    / "data_test"
    / "geo"
    / "lgd"
    / "bilaspur_village_gram_panchayat_mapping_2026-09-21.csv"
)

CURRENT_DIR = ROOT / "data_test" / "processed" / "current_climate_2026"
SPATIAL_PREDICTIONS_PATH = CURRENT_DIR / "mausam_current_ml_predictions.csv"
SPATIAL_FEATURES_PATH = CURRENT_DIR / "mausam_current_ml_features.csv"
RAINFALL_ANOMALY_PATH = CURRENT_DIR / "mausam_current_rainfall_anomaly.csv"
CURRENT_STATE_PATH = CURRENT_DIR / "current_state.json"

HORIZONS = (7, 14, 21, 30)

# Verified mapping from Bilaspur revenue sub-districts appearing in the
# LGD village → local-body source to the four current development blocks.
# The source CSV exposes Subdistrict, while the district administration
# separately lists the four development blocks.
SUBDISTRICT_TO_BLOCK = {
    "Belgahna": "Kota",
    "Beltara": "Belha",
    "Bilaspur": "Belha",
    "Bilha": "Belha",
    "Bodri": "Belha",
    "Kota": "Kota",
    "Masturi": "Masturi",
    "Pachpedi": "Masturi",
    "Ratanpur": "Kota",
    "Sankari": "Takhatpur",
    "Seepat": "Masturi",
    "Takhatpur": "Takhatpur",
}

with DATA_PATH.open("r", encoding="utf-8") as f:
    DATA = json.load(f)


def _slug(value: str) -> str:
    value = str(value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def _read_authoritative_locations():
    """Build the Bilaspur Block -> Panchayat hierarchy from the verified LGD mapping."""
    if not LOCATION_MAPPING_PATH.exists():
        raise RuntimeError(
            f"Authoritative LGD mapping not found: {LOCATION_MAPPING_PATH}"
        )

    rows = []
    with LOCATION_MAPPING_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        required = {
            "District Name (In English)",
            "Subdistrict Name (In English)",
            "Local Body Code",
            "Local Body Name (In English)",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise RuntimeError(
                "LGD mapping is missing required columns: " + ", ".join(missing)
            )

        for row in reader:
            code = str(row["Local Body Code"] or "").strip()
            if code in {"", "0", "0.0"}:
                continue

            try:
                code = str(int(float(code)))
            except ValueError:
                continue

            district = str(row["District Name (In English)"] or "").strip()
            subdistrict = str(row["Subdistrict Name (In English)"] or "").strip()
            panchayat = str(row["Local Body Name (In English)"] or "").strip()
            block = SUBDISTRICT_TO_BLOCK.get(subdistrict)

            if not district or not subdistrict or not panchayat or block is None:
                continue

            rows.append(
                {
                    "state": "Chhattisgarh",
                    "district": district,
                    "block": block,
                    "source_subdistrict": subdistrict,
                    "panchayat": panchayat,
                    "local_body_code": code,
                }
            )

    # The same Local Body Code can cover multiple villages and can appear
    # repeatedly in the source. Keep one canonical row per local body.
    unique = {}
    for row in rows:
        key = (row["state"], row["district"], row["block"], row["local_body_code"])
        existing = unique.get(key)
        if existing is not None:
            if existing["panchayat"] != row["panchayat"]:
                raise RuntimeError(
                    "Conflicting Local Body Names for code "
                    f"{row['local_body_code']}: "
                    f"{existing['panchayat']!r} vs {row['panchayat']!r}"
                )
            continue
        unique[key] = row

    return sorted(
        unique.values(),
        key=lambda item: (
            item["state"].lower(),
            item["district"].lower(),
            item["block"].lower(),
            item["panchayat"].lower(),
            item["local_body_code"],
        ),
    )

AUTHORITATIVE_ROWS = _read_authoritative_locations()


def _location_id(row):
    return "-".join(
        [
            _slug(row["state"]),
            _slug(row["district"]),
            _slug(row["block"]),
            row["local_body_code"],
        ]
    )


LOCATION_LOOKUP = {
    _location_id(row): row
    for row in AUTHORITATIVE_ROWS
}


def _build_locations_response():
    state_map = {}

    for row in AUTHORITATIVE_ROWS:
        state = row["state"]
        district = row["district"]
        block = row["block"]

        state_map.setdefault(state, {})
        state_map[state].setdefault(district, {})
        state_map[state][district].setdefault(block, {})
        state_map[state][district][block][row["local_body_code"]] = {
            "name": row["panchayat"],
            "local_body_code": row["local_body_code"],
            "location_id": _location_id(row),
        }

    states = []

    for state_name in sorted(state_map, key=str.lower):
        districts = []
        for district_name in sorted(state_map[state_name], key=str.lower):
            blocks = []
            for block_name in sorted(
                state_map[state_name][district_name],
                key=str.lower,
            ):
                panchayats = sorted(
                    state_map[state_name][district_name][block_name].values(),
                    key=lambda item: item["name"].lower(),
                )
                blocks.append(
                    {
                        "name": block_name,
                        "panchayats": panchayats,
                    }
                )

            districts.append(
                {
                    "name": district_name,
                    "blocks": blocks,
                }
            )

        states.append(
            {
                "name": state_name,
                "districts": districts,
            }
        )

    block_counts = {}
    for row in AUTHORITATIVE_ROWS:
        block_counts[row["block"]] = block_counts.get(row["block"], 0) + 1

    return {
        "states": states,
        "source": "LGD village-to-local-body mapping + verified Bilaspur block mapping",
        "local_body_relationship_count": len(AUTHORITATIVE_ROWS),
        "unique_local_body_code_count": len(
            {row["local_body_code"] for row in AUTHORITATIVE_ROWS}
        ),
        "block_counts_in_model_mapping": dict(sorted(block_counts.items())),
    }


LOCATIONS_RESPONSE = _build_locations_response()

def _load_current_live_tables():
    'Load the currently published prediction/anomaly state on each request.'
    if cloud_runtime_configured(ROOT):
        try:
            ensure_current_runtime(ROOT, CURRENT_DIR)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Published cloud runtime synchronization failed: {exc}",
            ) from exc

    if not SPATIAL_PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Current live ML prediction state is not available: "
                f"{SPATIAL_PREDICTIONS_PATH}"
            ),
        )

    predictions = {}
    with SPATIAL_PREDICTIONS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = str(row.get("local_body_code") or "").strip()
            if code:
                predictions[code] = row

    if len(predictions) != len(AUTHORITATIVE_ROWS):
        raise HTTPException(
            status_code=500,
            detail=(
                "Current live ML prediction state has "
                f"{len(predictions)} local bodies; "
                f"expected {len(AUTHORITATIVE_ROWS)}."
            ),
        )

    anomalies = {}
    if RAINFALL_ANOMALY_PATH.exists():
        with RAINFALL_ANOMALY_PATH.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = str(row.get("local_body_code") or "").strip()
                if code:
                    anomalies[code] = row

    issue_dates = sorted(
        {
            str(row.get("date") or "").strip()
            for row in predictions.values()
            if str(row.get("date") or "").strip()
        }
    )
    if len(issue_dates) != 1:
        raise HTTPException(
            status_code=500,
            detail=(
                "Current live ML prediction state has inconsistent dates: "
                f"{issue_dates}"
            ),
        )

    state = {}
    if CURRENT_STATE_PATH.exists():
        try:
            state = json.loads(
                CURRENT_STATE_PATH.read_text(encoding="utf-8")
            )
        except Exception:
            state = {}

    return predictions, anomalies, issue_dates[0], state



def _find_authoritative_location(location_id: str):
    return LOCATION_LOOKUP.get(str(location_id).strip())


def _ml_forecast_for_location(location, horizon: int):
    code = str(location["local_body_code"])
    predictions, anomalies, issue_date, state = _load_current_live_tables()

    row = predictions.get(code)
    if row is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No current live ML prediction is available for "
                f"Local Body Code {code}."
            ),
        )

    def probability(indicator: str, probability_type: str = "calibrated") -> float:
        field = f"{indicator}_{horizon}d_{probability_type}"
        raw = row.get(field)

        if raw in (None, ""):
            raise HTTPException(
                status_code=503,
                detail=f"Current live ML field missing: {field}",
            )

        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid current live ML value in {field}: {raw}",
            ) from exc

        if not 0.0 <= value <= 1.0:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Current live ML probability outside [0,1] "
                    f"in {field}: {value}"
                ),
            )

        return round(value * 100.0, 1)

    anomaly_mm = None
    anomaly_source = None
    anomaly_row = anomalies.get(code)

    if anomaly_row is not None:
        anomaly_field = f"rainfall_anomaly_{horizon}d_mm"
        raw_anomaly = anomaly_row.get(anomaly_field)

        if raw_anomaly not in (None, ""):
            try:
                anomaly_mm = float(raw_anomaly)
                anomaly_source = str(RAINFALL_ANOMALY_PATH)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Invalid rainfall anomaly in "
                        f"{anomaly_field}: {raw_anomaly}"
                    ),
                ) from exc

    return {
        "location": {
            "state": location["state"],
            "district": location["district"],
            "block": location["block"],
            "panchayat": location["panchayat"],
            "local_body_code": code,
        },
        "horizon": horizon,
        "forecast": {
            "onset_probability": probability("onset"),
            "break_probability": probability("break_proxy"),
            "heavy_rain_probability": probability(
                "heavy_rain_panchayat_mean"
            ),
            "rainfall_anomaly": anomaly_mm,
            "rainfall_anomaly_mm": anomaly_mm,
            "rainfall_anomaly_source": (
            "data_test/processed/current_climate_2026/mausam_current_rainfall_anomaly.csv"
            if anomaly_source
            else None
        ),
            "confidence": "Current observation-based ML inference",
            "issue_date": issue_date,
            "refresh_status": state.get("status", "published"),
        },
        "map_areas": DATA.get("map_areas", {}).get(
            location["state"],
            [],
        ),
        "prototype": False,
        "historical_hindcast": False,
        "live": True,
        "source": str(SPATIAL_PREDICTIONS_PATH),
    }


def _resolve_forecast(location_id: str, horizon: int):
    if horizon not in HORIZONS:
        raise HTTPException(
            status_code=400,
            detail="Horizon must be 7, 14, 21 or 30 days",
        )

    # Authoritative LGD local-body locations always use the current
    # published live ML state. Legacy forecasts are fallback-only.
    location = _find_authoritative_location(location_id)
    if location is not None:
        return _ml_forecast_for_location(location, horizon)

    record = DATA.get("forecasts", {}).get(location_id)
    if record:
        return {
            "location": record["location"],
            "horizon": horizon,
            "forecast": record["horizons"][str(horizon)],
            "map_areas": DATA.get("map_areas", {}).get(
                record.get("state", ""),
                [],
            ),
            "prototype": True,
            "historical_hindcast": False,
            "live": False,
            "source": "legacy_forecast_demo",
        }

    raise HTTPException(
        status_code=404,
        detail="Location not found",
    )




def _load_current_state_manifest():
    if cloud_runtime_configured(ROOT):
        try:
            return published_state(ROOT)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Published cloud current state is unavailable: {exc}",
            ) from exc

    current = _load_current_state_manifest()
    return current


app = FastAPI(
    title="MausamSaathi Prototype API",
    version="0.2.0",
)

# MAUSAMSAATHI CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(spatial_ml_router)
app.include_router(persistence_router)
app.include_router(authenticated_persistence_router)


# ============================================================
# EXISTING ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    current = _load_current_state_manifest()

    spatial_path = (
        ROOT
        / "data_test"
        / "processed"
        / "current_spatial_2026"
        / "mausam_current_spatial_current.geojson"
    )

    return {
        "status": "ok",
        "mode": "bilaspur_pilot",
        "data_mode": "current_live_published",
        "current_refresh_status": current.get(
            "status",
            "not_published",
        ),
        "current_issue_date": current.get(
            "last_successful_issue_date"
        ),
        "current_predictions": SPATIAL_PREDICTIONS_PATH.exists(),
        "current_spatial": spatial_path.exists(),
    }


@app.get("/locations")
def locations():
    return LOCATIONS_RESPONSE


@app.get("/locations/bilaspur/local-body/{local_body_code}")
def bilaspur_local_body(local_body_code: str):
    for row in AUTHORITATIVE_ROWS:
        if row["local_body_code"] == str(local_body_code).strip():
            return {
                **row,
                "location_id": _location_id(row),
            }

    raise HTTPException(
        status_code=404,
        detail=f"Bilaspur local body not found: {local_body_code}",
    )


@app.get("/forecast/{location_id}")
def forecast(
    location_id: str,
    horizon: int = 14,
):
    if horizon not in HORIZONS:
        raise HTTPException(
            status_code=400,
            detail="Horizon must be 7, 14, 21 or 30 days",
        )

    return _resolve_forecast(location_id, horizon)


# Localized equivalents for the deterministic advisory rule categories.
# Step 11: season-aware crop advisory engine.
# The ML/spatial forecast layer is unchanged. This layer only translates the
# selected live forecast into crop and monsoon-stage aware actions.

HI_ADVISORY_RULE_ACTIONS = {
    "high_rain": [
        "खेत में जल निकासी की तैयारी रखें",
        "जलभराव से बचाव के लिए खेत की नालियां साफ रखें",
        "अनावश्यक सिंचाई से बचें",
        "भारी वर्षा अपडेट पर नजर रखें",
    ],
    "low_onset": [
        "मानसून आरंभ की संभावना बढ़ने तक नई बुवाई टालें",
        "बीज और आवश्यक कृषि सामग्री तैयार रखें",
        "वैकल्पिक सिंचाई स्रोत की योजना रखें",
        "अगले पूर्वानुमान अपडेट की निगरानी करें",
    ],
    "mid_dry": [
        "फसल और मिट्टी की नमी के अनुसार जरूरत होने पर सिंचाई करें",
        "अनावश्यक सिंचाई से बचें",
        "खेत की नमी और फसल की स्थिति नियमित देखें",
        "अगले वर्षा अपडेट पर नजर रखें",
    ],
    "mid_normal": [
        "खड़ी फसल की नियमित निगरानी करें",
        "खेत में जल निकासी और मिट्टी की नमी संतुलित रखें",
        "स्थानीय फसल प्रबंधन गतिविधियां समय पर करें",
        "अगले वर्षा अपडेट पर नजर रखें",
    ],
    "late_dry": [
        "खड़ी फसल में मिट्टी की नमी कम होने पर उपलब्ध सिंचाई से पूरक सिंचाई करें",
        "अनावश्यक सिंचाई से बचें",
        "फसल की वर्तमान अवस्था के अनुसार प्रबंधन करें",
        "अगले पूर्वानुमान अपडेट पर नजर रखें",
    ],
    "late_normal": [
        "खड़ी फसल की स्थिति और पकने की अवस्था की निगरानी करें",
        "जलभराव और अनावश्यक सिंचाई दोनों से बचें",
        "यदि बुवाई अभी बाकी है तो स्थानीय कृषि विशेषज्ञ से उपयुक्त अल्प-अवधि विकल्प पर सलाह लें",
        "अगले मौसम अपडेट पर नजर रखें",
    ],
    "late_sowing_caution": [
        "इस समय नई खरीफ बुवाई को सामान्य बुवाई मानकर शुरू न करें",
        "यदि बुवाई बाकी है तो स्थानीय कृषि विशेषज्ञ/KVK से उपयुक्त अल्प-अवधि विकल्प पूछें",
    ],
    "post_monsoon": [
        "खड़ी खरीफ फसल की वर्तमान अवस्था के अनुसार प्रबंधन करें",
        "कटाई से पहले मौसम अपडेट देखें",
        "अनावश्यक सिंचाई से बचें",
        "अगली फसल की योजना स्थानीय कृषि सलाह के अनुसार बनाएं",
    ],
}


def _advisory_reference_date():
    """Resolve the latest current 2026 ML snapshot date used by the live bridge."""
    current_dir = ROOT / "data_test" / "processed" / "current_climate_2026"
    candidates = []

    if current_dir.exists():
        for path in current_dir.glob("mausam_current_ml_predictions_*.csv"):
            stamp = path.stem.rsplit("_", 1)[-1]
            if len(stamp) == 8 and stamp.isdigit():
                try:
                    from datetime import datetime
                    candidates.append(datetime.strptime(stamp, "%Y%m%d").date())
                except ValueError:
                    pass

    if candidates:
        return max(candidates)

    from datetime import datetime
    return datetime.now().date()


def _advisory_stage(reference_date):
    """Return a product-facing seasonal stage, not an official IMD onset/withdrawal diagnosis."""
    m = reference_date.month
    d = reference_date.day

    if (m, d) < (6, 1):
        return "pre_monsoon", "Pre-Monsoon", "मानसून-पूर्व"
    if (m, d) <= (7, 15):
        return "early_monsoon", "Early Monsoon", "प्रारंभिक मानसून"
    if (m, d) <= (8, 31):
        return "mid_monsoon", "Mid Monsoon", "मध्य मानसून"
    if (m, d) <= (10, 15):
        return "late_monsoon", "Late / End Monsoon", "अंतिम मानसून"
    return "post_monsoon", "Post-Monsoon", "मानसून के बाद"


def _dedupe_actions(items, limit=4):
    result = []
    seen = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
        if len(result) >= limit:
            break
    return result


def _advisory_horizon_band(horizon: int) -> str:
    """Translate forecast lead time into an advisory-planning band."""
    if horizon <= 7:
        return "short"
    if horizon <= 14:
        return "medium"
    return "extended"


def _build_season_aware_advisory(
    crop_name: str,
    crop_data,
    p_onset: float,
    p_break: float,
    p_heavy: float,
    anomaly_mm,
    reference_date,
    horizon: int,
):
    """
    Build season-aware, crop-specific advisory actions.

    The forecast probabilities remain the model output. This function is a
    deterministic decision-support layer. It deliberately changes wording and
    action emphasis by both crop and forecast lead time so Rice and Maize do not
    collapse to the same generic advice when they share a seasonal risk class.
    """
    stage_key, stage_label, stage_label_hi = _advisory_stage(reference_date)
    horizon_band = _advisory_horizon_band(horizon)
    rule_keys = []
    en_actions = []
    hi_actions = []

    is_heavy = p_heavy >= 60
    is_break = p_break >= 60
    anomaly_negative = anomaly_mm is not None and anomaly_mm < 0

    if stage_key == "pre_monsoon":
        if crop_name == "rice":
            if p_onset < 55:
                rule_keys.append("pre_rice_low_onset")
                en_actions += [
                    "Delay new rice sowing until rainfall/onset confidence improves",
                    "Keep seed and nursery inputs ready",
                    "Plan a backup irrigation source if available",
                    "Monitor the next rainfall update",
                ]
                hi_actions += HI_ADVISORY_RULE_ACTIONS["low_onset"]
            else:
                rule_keys.append("pre_rice_ready")
                en_actions += [
                    "Prepare the rice seedbed and field before sowing",
                    "Keep seed and inputs ready for the forecast window",
                    "Check field drainage and bund condition",
                    "Monitor the next rainfall update",
                ]
                hi_actions += [
                    "धान की नर्सरी/खेत की तैयारी पूरी रखें",
                    "पूर्वानुमान के अनुकूल बुवाई के लिए बीज व सामग्री तैयार रखें",
                    "खेत की मेड़ और जल निकासी की स्थिति जांचें",
                    "अगले वर्षा अपडेट पर नजर रखें",
                ]
        else:
            if is_heavy:
                rule_keys.append("pre_maize_heavy")
                en_actions += crop_data["high_rain_actions"]
                hi_actions += HI_ADVISORY_RULE_ACTIONS["high_rain"]
            else:
                rule_keys.append("pre_maize_ready")
                en_actions += [
                    "Prepare the maize field and seed for the expected sowing window",
                    "Avoid starting sowing before field moisture is suitable",
                    "Check drainage before heavy-rain events",
                    "Monitor the next rainfall update",
                ]
                hi_actions += [
                    "मक्का खेत और बीज की निर्धारित बुवाई अवधि के लिए तैयारी रखें",
                    "उचित मिट्टी की नमी से पहले बुवाई शुरू न करें",
                    "भारी वर्षा से पहले जल निकासी की जांच करें",
                    "अगले वर्षा अपडेट पर नजर रखें",
                ]

    elif stage_key == "early_monsoon":
        if is_heavy:
            rule_keys.append("early_heavy")
            en_actions += crop_data["high_rain_actions"]
            hi_actions += HI_ADVISORY_RULE_ACTIONS["high_rain"]

        if crop_name == "rice" and p_onset < 55:
            rule_keys.append("early_rice_low_onset")
            en_actions += crop_data["low_onset_actions"]
            hi_actions += HI_ADVISORY_RULE_ACTIONS["low_onset"]
        elif crop_name == "rice" and not rule_keys:
            rule_keys.append("early_rice_normal")
            en_actions += [
                "Use the forecast window to plan rice sowing or transplanting",
                "Maintain nursery and field moisture appropriately",
                "Keep drainage ready for heavy-rain spells",
                "Monitor local rainfall",
            ]
            hi_actions += [
                "पूर्वानुमान की अनुकूल अवधि में धान की बुवाई/रोपाई की योजना बनाएं",
                "नर्सरी और खेत में उचित नमी बनाए रखें",
                "भारी वर्षा के लिए जल निकासी तैयार रखें",
                "स्थानीय वर्षा की निगरानी करें",
            ]
        elif crop_name == "maize" and not rule_keys:
            rule_keys.append("early_maize_normal")
            en_actions += [
                "Plan maize sowing within the suitable field-moisture window",
                "Apply field preparation and early crop operations on time",
                "Keep drainage ready for heavy-rain spells",
                "Monitor local rainfall",
            ]
            hi_actions += [
                "उचित मिट्टी की नमी वाली अवधि में मक्का की बुवाई की योजना बनाएं",
                "खेत की तैयारी और शुरुआती फसल प्रबंधन समय पर करें",
                "भारी वर्षा के लिए जल निकासी तैयार रखें",
                "स्थानीय वर्षा की निगरानी करें",
            ]

    elif stage_key == "mid_monsoon":
        if is_heavy:
            rule_keys.append("mid_heavy")
            en_actions += crop_data["high_rain_actions"]
            hi_actions += HI_ADVISORY_RULE_ACTIONS["high_rain"]
        elif is_break or anomaly_negative:
            rule_keys.append(f"mid_dry_{crop_name}_{horizon}d")
            if crop_name == "rice":
                if horizon_band == "short":
                    en_actions += [
                        "Check rice-field moisture and irrigate only where crop stress is observed",
                        "Avoid unnecessary irrigation when rain is possible",
                        "Monitor the standing crop and current growth stage",
                        "Recheck the next short-range rainfall update",
                    ]
                elif horizon_band == "medium":
                    en_actions += [
                        "Plan supplemental irrigation for rice only where soil moisture is becoming limiting",
                        "Keep water available for the next 1-2 weeks if possible",
                        "Monitor crop moisture and growth stage",
                        "Review the next rainfall update before irrigating",
                    ]
                else:
                    en_actions += [
                        "Plan a water-availability contingency for the remaining rice season",
                        "Use irrigation only where crop moisture demand requires it",
                        "Monitor crop stage and signs of moisture stress",
                        "Reassess the extended outlook with each new forecast",
                    ]
                hi_actions += [
                    "धान में फसल तनाव दिखने पर ही जरूरत के अनुसार सिंचाई करें",
                    "बारिश की संभावना होने पर अनावश्यक सिंचाई से बचें",
                    "खड़ी फसल और उसकी वर्तमान अवस्था देखें",
                    "अगले निकट अवधि के वर्षा अपडेट को फिर जांचें",
                ]
            else:
                if horizon_band == "short":
                    en_actions += [
                        "Check maize-field moisture and irrigate only when the crop needs it",
                        "Avoid unnecessary irrigation when rain is possible",
                        "Monitor the current maize growth stage and moisture stress",
                        "Recheck the next short-range rainfall update",
                    ]
                elif horizon_band == "medium":
                    en_actions += [
                        "Plan supplemental irrigation for maize only where soil moisture is limiting",
                        "Keep water available for the next 1-2 weeks if possible",
                        "Monitor crop stage and moisture stress",
                        "Review the next rainfall update before irrigating",
                    ]
                else:
                    en_actions += [
                        "Plan a water-availability contingency for the remaining maize season",
                        "Irrigate only when crop moisture demand requires it",
                        "Monitor the current crop stage and stress",
                        "Reassess the extended outlook with each new forecast",
                    ]
                hi_actions += [
                    "मक्का में फसल की जरूरत होने पर ही सिंचाई करें",
                    "बारिश की संभावना होने पर अनावश्यक सिंचाई से बचें",
                    "फसल की वर्तमान अवस्था और नमी तनाव देखें",
                    "अगले वर्षा अपडेट की फिर से जांच करें",
                ]
        else:
            rule_keys.append(f"mid_normal_{crop_name}_{horizon}d")
            if crop_name == "rice":
                en_actions += [
                    "Monitor the standing rice crop regularly",
                    "Maintain field drainage and balanced soil moisture",
                    "Carry out crop-management operations according to the current growth stage",
                    "Watch the next rainfall update",
                ]
                hi_actions += HI_ADVISORY_RULE_ACTIONS["mid_normal"]
            else:
                en_actions += [
                    "Monitor the standing maize crop and current growth stage",
                    "Maintain drainage and avoid prolonged waterlogging",
                    "Carry out timely interculture and crop-management operations",
                    "Watch the next rainfall update",
                ]
                hi_actions += [
                    "खड़ी मक्का फसल और उसकी वर्तमान अवस्था की निगरानी करें",
                    "जल निकासी बनाए रखें और लंबे समय तक जलभराव से बचें",
                    "समय पर फसल प्रबंधन और निराई-गुड़ाई करें",
                    "अगले वर्षा अपडेट पर नजर रखें",
                ]

    elif stage_key == "late_monsoon":
        if is_heavy:
            rule_keys.append(f"late_heavy_{crop_name}_{horizon}d")
            if crop_name == "rice":
                en_actions += [
                    "Maintain field drainage and remove blocked outlets before heavy rain",
                    "Avoid unnecessary irrigation while heavy rain risk is elevated",
                    "Protect mature rice from prolonged waterlogging",
                    "Monitor heavy-rain alerts and the next forecast update",
                ]
                hi_actions += [
                    "भारी वर्षा से पहले खेत की जल निकासी और निकास मार्ग साफ रखें",
                    "भारी वर्षा जोखिम अधिक होने पर अनावश्यक सिंचाई न करें",
                    "पकी हुई धान फसल को लंबे समय के जलभराव से बचाएं",
                    "भारी वर्षा अलर्ट और अगले पूर्वानुमान पर नजर रखें",
                ]
            else:
                en_actions += [
                    "Ensure maize-field drainage and clear blocked outlets before heavy rain",
                    "Avoid unnecessary irrigation while heavy rain risk is elevated",
                    "Protect ears and mature plants from prolonged waterlogging",
                    "Monitor heavy-rain alerts and the next forecast update",
                ]
                hi_actions += [
                    "भारी वर्षा से पहले मक्का खेत की जल निकासी और निकास मार्ग साफ रखें",
                    "भारी वर्षा जोखिम अधिक होने पर अनावश्यक सिंचाई न करें",
                    "बालियों/पौधों को लंबे समय के जलभराव से बचाएं",
                    "भारी वर्षा अलर्ट और अगले पूर्वानुमान पर नजर रखें",
                ]
        elif is_break or anomaly_negative:
            rule_keys.append(f"late_dry_{crop_name}_{horizon}d")
            if crop_name == "rice":
                if horizon_band == "short":
                    en_actions += [
                        "For standing rice, irrigate only where soil moisture stress is observed and water is available",
                        "Avoid unnecessary irrigation when the next few days may still receive rain",
                        "Monitor the rice crop at its current growth or grain-filling stage",
                        "Use the next short-range forecast before the next irrigation decision",
                    ]
                    hi_actions += [
                        "खड़ी धान में मिट्टी की नमी का तनाव दिखने पर उपलब्ध सिंचाई से ही सिंचाई करें",
                        "अगले कुछ दिनों में बारिश संभव हो तो अनावश्यक सिंचाई से बचें",
                        "धान की वर्तमान अवस्था/दाना भरने की अवस्था की निगरानी करें",
                        "अगली सिंचाई से पहले निकट अवधि का वर्षा पूर्वानुमान देखें",
                    ]
                elif horizon_band == "medium":
                    en_actions += [
                        "Plan supplemental irrigation for standing rice only where moisture is becoming limiting",
                        "Keep a practical water source available for the next 1-2 weeks if possible",
                        "Monitor grain filling or the crop's current growth stage",
                        "Review the next forecast before each irrigation decision",
                    ]
                    hi_actions += [
                        "खड़ी धान में नमी सीमित होने पर ही पूरक सिंचाई की योजना बनाएं",
                        "संभव हो तो अगले 1-2 सप्ताह के लिए व्यावहारिक जल स्रोत उपलब्ध रखें",
                        "दाना भरने या वर्तमान फसल अवस्था की निगरानी करें",
                        "हर सिंचाई निर्णय से पहले अगला पूर्वानुमान देखें",
                    ]
                else:
                    en_actions += [
                        "Plan a water-availability contingency for standing rice through the remaining crop period",
                        "Use irrigation only when crop moisture demand justifies it",
                        "Monitor crop maturity and moisture stress",
                        "Reassess the longer-range outlook with each forecast update",
                    ]
                    hi_actions += [
                        "बाकी फसल अवधि के लिए खड़ी धान हेतु जल उपलब्धता की वैकल्पिक योजना रखें",
                        "फसल की नमी जरूरत के अनुसार ही सिंचाई करें",
                        "फसल की परिपक्वता और नमी तनाव की निगरानी करें",
                        "हर नए पूर्वानुमान के साथ लंबी अवधि के संकेतों की फिर समीक्षा करें",
                    ]
            else:
                if horizon_band == "short":
                    en_actions += [
                        "For standing maize, irrigate only where soil moisture stress is observed and water is available",
                        "Avoid unnecessary irrigation when the next few days may still receive rain",
                        "Monitor the maize crop's current growth or grain-filling stage",
                        "Use the next short-range forecast before the next irrigation decision",
                    ]
                    hi_actions += [
                        "खड़ी मक्का में मिट्टी की नमी का तनाव दिखने पर उपलब्ध सिंचाई से ही सिंचाई करें",
                        "अगले कुछ दिनों में बारिश संभव हो तो अनावश्यक सिंचाई से बचें",
                        "मक्का की वर्तमान अवस्था/दाना भरने की अवस्था की निगरानी करें",
                        "अगली सिंचाई से पहले निकट अवधि का वर्षा पूर्वानुमान देखें",
                    ]
                elif horizon_band == "medium":
                    en_actions += [
                        "Plan supplemental irrigation for standing maize only where moisture is becoming limiting",
                        "Keep a practical water source available for the next 1-2 weeks if possible",
                        "Monitor the current maize growth or grain-filling stage",
                        "Review the next forecast before each irrigation decision",
                    ]
                    hi_actions += [
                        "खड़ी मक्का में नमी सीमित होने पर ही पूरक सिंचाई की योजना बनाएं",
                        "संभव हो तो अगले 1-2 सप्ताह के लिए व्यावहारिक जल स्रोत उपलब्ध रखें",
                        "मक्का की वर्तमान अवस्था या दाना भरने की अवस्था की निगरानी करें",
                        "हर सिंचाई निर्णय से पहले अगला पूर्वानुमान देखें",
                    ]
                else:
                    en_actions += [
                        "Plan a water-availability contingency for standing maize through the remaining crop period",
                        "Irrigate only when crop moisture demand justifies it",
                        "Monitor crop maturity and moisture stress",
                        "Reassess the longer-range outlook with each forecast update",
                    ]
                    hi_actions += [
                        "बाकी फसल अवधि के लिए खड़ी मक्का हेतु जल उपलब्धता की वैकल्पिक योजना रखें",
                        "फसल की नमी जरूरत के अनुसार ही सिंचाई करें",
                        "फसल की परिपक्वता और नमी तनाव की निगरानी करें",
                        "हर नए पूर्वानुमान के साथ लंबी अवधि के संकेतों की फिर समीक्षा करें",
                    ]
        else:
            rule_keys.append(f"late_normal_{crop_name}_{horizon}d")
            if crop_name == "rice":
                en_actions += [
                    "Monitor standing rice and its maturity stage",
                    "Avoid both waterlogging and unnecessary irrigation",
                    "Do not begin a new Kharif rice crop late without locally appropriate agricultural guidance",
                    "Watch the next weather update before field operations",
                ]
                hi_actions += [
                    "खड़ी धान और उसकी पकने की अवस्था की निगरानी करें",
                    "जलभराव और अनावश्यक सिंचाई दोनों से बचें",
                    "स्थानीय कृषि सलाह के बिना देर से नई खरीफ धान की बुवाई शुरू न करें",
                    "खेत के काम से पहले अगला मौसम अपडेट देखें",
                ]
            else:
                en_actions += [
                    "Monitor standing maize and its maturity stage",
                    "Avoid prolonged waterlogging and unnecessary irrigation",
                    "Do not begin a new late Kharif maize crop without locally appropriate agricultural guidance",
                    "Plan field operations around the next weather update",
                ]
                hi_actions += [
                    "खड़ी मक्का और उसकी पकने की अवस्था की निगरानी करें",
                    "लंबे समय के जलभराव और अनावश्यक सिंचाई से बचें",
                    "स्थानीय कृषि सलाह के बिना देर से नई खरीफ मक्का फसल शुरू न करें",
                    "अगले मौसम अपडेट के अनुसार खेत के काम की योजना बनाएं",
                ]

    else:
        rule_keys.append(f"post_monsoon_{crop_name}")
        if crop_name == "rice":
            en_actions += [
                "Manage standing rice according to its current maturity stage",
                "Check weather before harvest or drying operations",
                "Avoid unnecessary irrigation",
                "Plan the next crop according to local agricultural guidance",
            ]
            hi_actions += [
                "खड़ी धान का वर्तमान पकने के चरण के अनुसार प्रबंधन करें",
                "कटाई या सुखाने से पहले मौसम अपडेट देखें",
                "अनावश्यक सिंचाई से बचें",
                "स्थानीय कृषि सलाह के अनुसार अगली फसल की योजना बनाएं",
            ]
        else:
            en_actions += [
                "Manage standing maize according to its current maturity stage",
                "Check weather before harvest and post-harvest field operations",
                "Avoid unnecessary irrigation",
                "Plan the next crop according to local agricultural guidance",
            ]
            hi_actions += [
                "खड़ी मक्का का वर्तमान पकने के चरण के अनुसार प्रबंधन करें",
                "कटाई और कटाई के बाद के खेत कार्यों से पहले मौसम अपडेट देखें",
                "अनावश्यक सिंचाई से बचें",
                "स्थानीय कृषि सलाह के अनुसार अगली फसल की योजना बनाएं",
            ]

    return {
        "stage_key": stage_key,
        "stage_label": stage_label,
        "stage_label_hi": stage_label_hi,
        "horizon_band": horizon_band,
        "rule_keys": rule_keys,
        "actions": _dedupe_actions(en_actions, 4),
        "actions_hi": _dedupe_actions(hi_actions, 4),
    }


@app.get("/advisory/{location_id}/{crop}")
def advisory(
    location_id: str,
    crop: str,
    horizon: int = 14,
    lang: str = "en",
):
    # Uses the exact current calibrated values already resolved by /forecast.
    if horizon not in HORIZONS:
        raise HTTPException(
            status_code=400,
            detail="Horizon must be 7, 14, 21 or 30 days",
        )

    resolved = _resolve_forecast(location_id, horizon)

    crop_name = crop.lower()
    crop_data = DATA["advisories"].get(
        crop_name,
        DATA["advisories"]["rice"],
    )

    result = resolved["forecast"]
    p_onset = float(result.get("onset_probability", 0))
    p_break = float(result.get("break_probability", 0))
    p_heavy = float(result.get("heavy_rain_probability", 0))

    anomaly_mm = result.get("rainfall_anomaly_mm")
    if anomaly_mm not in (None, ""):
        try:
            anomaly_mm = float(anomaly_mm)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid rainfall anomaly in current live forecast: {anomaly_mm}",
            ) from exc
    else:
        anomaly_mm = None

    prediction_source = (
        "current_live_ml_calibrated"
        if not resolved.get("prototype", False)
        else "legacy_forecast"
    )
    anomaly_source = result.get("rainfall_anomaly_source")

    reference_date_text = str(result.get("issue_date") or "").strip()
    if reference_date_text:
        try:
            from datetime import date
            reference_date = date.fromisoformat(reference_date_text[:10])
        except ValueError:
            reference_date = _advisory_reference_date()
    else:
        reference_date = _advisory_reference_date()

    built = _build_season_aware_advisory(
        crop_name,
        crop_data,
        p_onset,
        p_break,
        p_heavy,
        anomaly_mm,
        reference_date,
        horizon,
    )

    stage = built["stage_label"]
    stage_hi = built["stage_label_hi"]

    anomaly_text = ""
    if anomaly_mm is not None:
        anomaly_text = f" Rainfall anomaly is {anomaly_mm:+.1f} mm."

    en = {
        "title": f"{stage} {crop_data['title']}",
        "summary": (
            f"Advisory stage: {stage}. Heavy-rain probability is {p_heavy}%, "
            f"break-phase probability is {p_break}%, and onset probability is {p_onset}% "
            f"over the selected {horizon}-day horizon.{anomaly_text}"
        ),
        "actions": built["actions"],
    }

    hi_anomaly_text = ""
    if anomaly_mm is not None:
        hi_anomaly_text = f" वर्षा असामान्यता {anomaly_mm:+.1f} मिमी है।"

    hi = {
        "title": f"{stage_hi} • {crop_data['title_hi']}",
        "summary": (
            f"सलाह का चरण: {stage_hi}। चयनित {horizon} दिनों में भारी वर्षा की संभावना "
            f"{p_heavy}%, मानसून विराम की संभावना {p_break}% और मानसून आरंभ की संभावना "
            f"{p_onset}% है।{hi_anomaly_text}"
        ),
        "actions": built["actions_hi"],
    }

    return {
        "location": resolved["location"],
        "horizon": horizon,
        "crop": crop,
        "language": lang,
        "monsoon_stage": built["stage_key"],
        "monsoon_stage_label": stage,
        "reference_date": reference_date.isoformat(),
        "rule_keys": built["rule_keys"],
        "forecast_source": prediction_source,
        "rainfall_anomaly_source": (
            "data_test/processed/current_climate_2026/mausam_current_rainfall_anomaly.csv"
            if anomaly_source
            else None
        ),
        "advisory": hi if lang == "hi" else en,
    }


# ============================================================
# ML ENDPOINTS
# ============================================================

@app.get("/ml/status")
def ml_status():
    current = _load_current_state_manifest()

    return {
        "status": (
            "ready"
            if SPATIAL_PREDICTIONS_PATH.exists()
            else "waiting_for_refresh"
        ),
        "location": "Bilaspur, Chhattisgarh",
        "model_family": "XGBoost calibrated classifiers",
        "feature_set": "102-feature observation + climate-signal snapshot",
        "supported_horizons": list(HORIZONS),
        "indicators": [
            "heavy_rain_panchayat_mean",
            "heavy_rain_grid",
            "onset",
            "break_proxy",
            "active",
        ],
        "probability_types": ["raw", "calibrated"],
        "current_issue_date": current.get(
            "last_successful_issue_date"
        ),
        "data_mode": "current_live_published",
        "current_prediction_file": "data_test/processed/current_climate_2026/mausam_current_ml_predictions.csv",
        "current_anomaly_file": "data_test/processed/current_climate_2026/mausam_current_rainfall_anomaly.csv",
    }


@app.get("/ml-forecast-demo")
def ml_forecast_demo(
    as_of: str = "2025-09-01",
    horizon: int = 14,
):
    """
    Historical hindcast/demo inference.

    This endpoint intentionally does NOT present the result as a
    current operational forecast. It replays a historical as-of date
    using the saved 2024-2025 feature dataset and trained models.
    """

    if horizon not in HORIZONS:
        raise HTTPException(
            status_code=400,
            detail="Horizon must be 7, 14, 21 or 30 days.",
        )

    try:
        return run_demo_forecast(
            as_of=as_of,
            horizon=horizon,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"ML inference failed: {exc}",
        )
