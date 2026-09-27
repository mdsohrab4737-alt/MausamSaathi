from __future__ import annotations

"""
MAUSAMSAATHI SIH 2026
STEP 15 - DAILY CURRENT DATA REFRESH + AUTOMATIC ML INFERENCE

Purpose
-------
This is the production-oriented Bilaspur pilot refresh orchestrator.

Daily flow:
    1. Fetch new IMD GPM daily rainfall files.
    2. Fetch new Copernicus CDS ERA5T atmospheric data.
    3. Fetch new NOAA PSL CPC blended OLR daily slices.
    4. Rebuild the project's current OMI/RMM signal using the existing
       Wheeler-Kiladis method + original NOAA OMI EOFs.
    5. Resolve the latest fully joinable issue date.
    6. Run the EXISTING 20 calibrated ML classifiers.
    7. Run rainfall-anomaly inference.
    8. Rebuild the current spatial GeoJSON.
    9. Publish a small current_state.json manifest atomically.

Important
---------
- This does NOT retrain the historical ML models every day.
- Training/retraining remains a separate future controlled job.
- Existing trained model weights and calibrators are reused for daily
  inference.
- Failed/newly unavailable data never overwrite the last successful
  published state.
- Data is keyed by local_body_code and date.
- Bilaspur remains the only active pilot district in this file.

This script intentionally reuses already-audited project modules instead of
reimplementing the ML feature logic. Date-dependent constants are overridden
in memory before calling those modules.
"""

from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from contextlib import contextmanager
import asyncio
import csv
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import traceback
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

# ============================================================
# STEP 17 PERSISTENCE INTEGRATION
# ============================================================
try:
    from persistence import (
        record_refresh_run,
        sha256_file,
        upsert_asset,
    )
except ImportError:
    from backend.persistence import (
        record_refresh_run,
        sha256_file,
        upsert_asset,
    )


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data_test" / "processed"
CURRENT_DIR = PROC / "current_climate_2026"
RAW_GPM_DIR = CURRENT_DIR / "raw_gpm"
RAW_ERA5_DIR = ROOT / "data_test" / "historical" / "era5t_atmosphere_2026"
CURRENT_OMI_DIR = PROC / "current_omi"
RAW_OLR_DIR = CURRENT_OMI_DIR / "refresh_olr_daily"
SPATIAL_DIR = PROC / "current_spatial_2026"

WEIGHTS_PATH = PROC / "bilaspur_panchayat_grid_weights.csv"
SEED_RAINFALL = PROC / "bilaspur_panchayat_rainfall_2026-08-09_2026-09-23.csv"
SEED_ATMOSPHERE = PROC / "bilaspur_panchayat_era5t_daily_20260809_20260919.csv"
TELEGRAM_GPM_DIR = Path(os.environ.get("MS_GPM_IMPORT_DIR", r"C:\Users\mdsoh\Downloads\Telegram Desktop"))

RAIN_CUMULATIVE = CURRENT_DIR / "bilaspur_current_rainfall.csv"
ATM_CUMULATIVE = CURRENT_DIR / "bilaspur_current_atmosphere.csv"
OMI_CURRENT = CURRENT_OMI_DIR / "current_omi_pcs_current.csv"

FEATURE_CURRENT = CURRENT_DIR / "mausam_current_ml_features.csv"
PREDICTIONS_CURRENT = CURRENT_DIR / "mausam_current_ml_predictions.csv"
CLIMATE_CURRENT = CURRENT_DIR / "bilaspur_current_climate_signals.csv"
INFERENCE_AUDIT_CURRENT = CURRENT_DIR / "mausam_current_ml_inference_audit.json"
ANOMALY_CURRENT = CURRENT_DIR / "mausam_current_rainfall_anomaly.csv"
ANOMALY_AUDIT_CURRENT = CURRENT_DIR / "mausam_current_rainfall_anomaly_audit.json"

SPATIAL_CURRENT = SPATIAL_DIR / "mausam_current_spatial_current.geojson"
SPATIAL_LAYER_DIR = SPATIAL_DIR / "layers_current"

STATE_PATH = CURRENT_DIR / "current_state.json"
PERSISTENCE_DB = PROC / "persistence" / "mausamsaathi_local.sqlite3"
LOCK_PATH = CURRENT_DIR / "refresh.lock"
LOG_PATH = CURRENT_DIR / "refresh.log"

# Existing audited modules in the project.
V4_INFERENCE = ROOT / "run_current_ml_inference_20260919_v4.py"
ANOMALY_MODULE = ROOT / "step9c_infer_current_rainfall_anomaly_20260919_fixed.py"
SPATIAL_MODULE = ROOT / "step9a_build_current_spatial_20260919.py"

# Canonical source pages.
IMD_GPM_PAGE = (
    "https://imdpune.gov.in/cmpg/Realtimedata/gpm/Rain_Download.html"
)
NOAA_OLR_NCSS = (
    "https://psl.noaa.gov/thredds/ncss/grid/"
    "Datasets/cpc_blended_olr-2.5deg/olr.day.mean.nc"
)

# IMD GPM file geometry already audited in this project.
GPM_NX = 241
GPM_NY = 281
GPM_X0 = 50.0
GPM_Y0 = -30.0
GPM_DX = 0.25
GPM_DY = 0.25
GPM_UNDEF = -999.0
GPM_EXPECTED_BYTES = GPM_NX * GPM_NY * 4

EXPECTED_BODIES = 486
CONTEXT_DAYS = 42
OMI_CONTEXT_START = np.datetime64("2024-01-01")
OMI_OUTPUT_START = np.datetime64("2026-01-01")


def log(message: str) -> None:
    line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
    print(line)
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        prefix=path.stem + "_",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
        newline="",
    ) as f:
        tmp = Path(f.name)
        frame.to_csv(f, index=False)
    tmp.replace(path)


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {
            "status": "never_run",
            "last_successful_issue_date": None,
            "updated_at": None,
        }
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "status": "state_read_error",
            "last_successful_issue_date": None,
            "updated_at": None,
        }


@contextmanager
def single_run_lock():
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        # Treat a very old lock as stale.
        try:
            age_seconds = datetime.now().timestamp() - LOCK_PATH.stat().st_mtime
            if age_seconds > 6 * 60 * 60:
                LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass

    if LOCK_PATH.exists():
        raise RuntimeError(
            f"Another refresh is already running or lock exists: {LOCK_PATH}"
        )

    LOCK_PATH.write_text(
        json.dumps({"pid": os.getpid(), "started_at": datetime.now().astimezone().isoformat()}),
        encoding="utf-8",
    )
    try:
        yield
    finally:
        LOCK_PATH.unlink(missing_ok=True)


def import_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Required project module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_project_module(filename: str) -> Path:
    """Resolve an audited project helper without assuming it lives at ROOT."""
    direct_candidates = [
        ROOT / filename,
        ROOT / "backend" / filename,
        ROOT / "scripts" / filename,
        ROOT / "tools" / filename,
    ]
    for candidate in direct_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    excluded = {".git", ".venv", "node_modules", "__pycache__"}
    matches = []
    for candidate in ROOT.rglob(filename):
        try:
            rel_parts = set(candidate.relative_to(ROOT).parts)
        except ValueError:
            continue
        if rel_parts & excluded:
            continue
        if candidate.is_file():
            matches.append(candidate)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        joined = "\n".join(f"  - {m}" for m in matches[:20])
        raise FileNotFoundError(
            f"Multiple project modules named {filename} were found.\n{joined}"
        )

    raise FileNotFoundError(
        f"Required project module not found anywhere under {ROOT}: {filename}"
    )


def safe_date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# ---------------------------------------------------------------------------
# Shared geography / rainfall helpers
# ---------------------------------------------------------------------------

def load_weights() -> pd.DataFrame:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weight file not found: {WEIGHTS_PATH}")

    w = pd.read_csv(WEIGHTS_PATH)
    required = [
        "local_body_code",
        "local_body_name",
        "grid_lat",
        "grid_lon",
        "weight",
    ]
    missing = [c for c in required if c not in w.columns]
    if missing:
        raise RuntimeError(f"Weight file missing columns: {missing}")

    w["local_body_code"] = pd.to_numeric(
        w["local_body_code"], errors="raise"
    ).astype("int64")
    w["grid_lat"] = pd.to_numeric(w["grid_lat"], errors="raise")
    w["grid_lon"] = pd.to_numeric(w["grid_lon"], errors="raise")
    w["weight"] = pd.to_numeric(w["weight"], errors="raise")

    if w["local_body_code"].nunique() != EXPECTED_BODIES:
        raise RuntimeError(
            f"Expected {EXPECTED_BODIES} local bodies in weights, "
            f"got {w['local_body_code'].nunique()}"
        )

    def coord_to_index(lat: float, lon: float) -> tuple[int, int]:
        row = int(round((lat - GPM_Y0) / GPM_DY))
        col = int(round((lon - GPM_X0) / GPM_DX))
        if not (0 <= row < GPM_NY and 0 <= col < GPM_NX):
            raise RuntimeError(
                f"Grid coordinate out of range: lat={lat}, lon={lon}"
            )
        lat2 = GPM_Y0 + row * GPM_DY
        lon2 = GPM_X0 + col * GPM_DX
        if abs(lat2 - lat) > 1e-6 or abs(lon2 - lon) > 1e-6:
            raise RuntimeError(
                f"Grid coordinate mismatch: ({lat}, {lon}) -> ({lat2}, {lon2})"
            )
        return row, col

    rows_cols = [
        coord_to_index(lat, lon)
        for lat, lon in zip(w["grid_lat"], w["grid_lon"])
    ]
    w["row"] = [x[0] for x in rows_cols]
    w["col"] = [x[1] for x in rows_cols]

    weight_sum = w.groupby("local_body_code")["weight"].sum()
    bad = weight_sum[(weight_sum - 1.0).abs() > 1e-6]
    if len(bad):
        raise RuntimeError(
            "Weight sums not equal to 1, sample: "
            + str(bad.head(10))
        )

    return w


def valid_grd(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size == GPM_EXPECTED_BYTES


def aggregate_grd_one(path: Path, w: pd.DataFrame, d: date) -> pd.DataFrame:
    if not valid_grd(path):
        raise RuntimeError(
            f"Invalid GRD: {path} "
            f"(size={path.stat().st_size if path.exists() else 'missing'})"
        )

    raw = np.fromfile(path, dtype="<f4")
    if raw.size != GPM_NX * GPM_NY:
        raise RuntimeError(
            f"{path.name}: {raw.size} values, expected {GPM_NX * GPM_NY}"
        )

    grid = raw.reshape(GPM_NY, GPM_NX).astype("float64")
    grid[np.isclose(grid, GPM_UNDEF)] = np.nan

    vals = grid[w["row"].to_numpy(), w["col"].to_numpy()]
    temp = w[
        ["local_body_code", "local_body_name", "weight"]
    ].copy()
    temp["rain_mm"] = vals
    temp["valid_weight"] = np.where(
        temp["rain_mm"].notna(), temp["weight"], 0.0
    )
    temp["weighted_rain"] = (
        temp["rain_mm"].fillna(0.0) * temp["weight"]
    )

    grp = temp.groupby(
        ["local_body_code", "local_body_name"],
        sort=True,
    )

    out = grp.agg(
        weighted_rain_sum=("weighted_rain", "sum"),
        effective_weight=("valid_weight", "sum"),
        rainfall_max_grid_mm=("rain_mm", "max"),
        contributing_grid_cells=("rain_mm", lambda s: int(s.notna().sum())),
    ).reset_index()

    out["rainfall_mean_mm"] = np.where(
        out["effective_weight"] > 0,
        out["weighted_rain_sum"] / out["effective_weight"],
        np.nan,
    )

    if len(out) != EXPECTED_BODIES:
        raise RuntimeError(
            f"{path.name}: aggregated rows={len(out)}, expected {EXPECTED_BODIES}"
        )

    if out["rainfall_mean_mm"].isna().any():
        raise RuntimeError(
            f"{path.name}: one or more Panchayats have missing rainfall mean"
        )

    if (
        (out["effective_weight"] <= 0)
        | ((out["effective_weight"] - 1.0).abs() > 1e-6)
    ).any():
        raise RuntimeError(
            f"{path.name}: effective weights are invalid"
        )

    out.insert(0, "date", pd.Timestamp(d))
    return out[
        [
            "date",
            "local_body_code",
            "local_body_name",
            "rainfall_mean_mm",
            "rainfall_max_grid_mm",
            "contributing_grid_cells",
            "effective_weight",
        ]
    ]


def merge_cumulative(
    existing_path: Path,
    new_frames: list[pd.DataFrame],
    *,
    expected_columns: list[str],
    key_columns: list[str],
) -> pd.DataFrame:
    parts = []

    if existing_path.exists():
        parts.append(pd.read_csv(existing_path))

    parts.extend(new_frames)

    if not parts:
        raise RuntimeError(f"No data available for {existing_path}")

    frame = pd.concat(parts, ignore_index=True)

    missing = [c for c in expected_columns if c not in frame.columns]
    if missing:
        raise RuntimeError(
            f"{existing_path.name}: missing columns {missing}"
        )

    frame = frame[expected_columns].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["local_body_code"] = (
        pd.to_numeric(frame["local_body_code"], errors="raise").astype("int64")
    )

    frame = (
        frame.sort_values(key_columns)
        .drop_duplicates(key_columns, keep="last")
        .reset_index(drop=True)
    )

    return frame


def continuous_panchayat_end(
    frame: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
) -> pd.Timestamp | None:
    if frame.empty:
        return None

    counts = (
        frame.groupby("date")["local_body_code"]
        .nunique()
        .sort_index()
    )

    expected = pd.date_range(
        start_date.normalize(),
        counts.index.max().normalize(),
        freq="D",
    )

    for d in expected:
        if d not in counts.index:
            break
        if int(counts.loc[d]) != EXPECTED_BODIES:
            break
    else:
        return expected[-1]

    good = []
    for d in expected:
        if d not in counts.index or int(counts.loc[d]) != EXPECTED_BODIES:
            break
        good.append(d)

    return good[-1] if good else None


# ---------------------------------------------------------------------------
# 1. IMD GPM daily download
# ---------------------------------------------------------------------------

def _playwright_download_gpm_one_sync(target_date: date, output_path: Path) -> bool:
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    except Exception as exc:
        raise RuntimeError(
            "Playwright is required for IMD GPM refresh. "
            "Install with the same backend venv used by the project."
        ) from exc

    iso_date = target_date.strftime("%Y-%m-%d")
    alt_date = target_date.strftime("%d/%m/%Y")
    expected_name = target_date.strftime("%d%m%Y") + ".grd"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(accept_downloads=True)
        try:
            page.goto(
                IMD_GPM_PAGE,
                wait_until="domcontentloaded",
                timeout=60000,
            )

            selectors = [
                "input[type='date']",
                "input[name*='date' i]",
                "input[id*='date' i]",
                "input[type='text']",
            ]

            date_control = None
            for sel in selectors:
                loc = page.locator(sel)
                if loc.count() > 0:
                    # Pick the first visible control.
                    for i in range(min(loc.count(), 5)):
                        candidate = loc.nth(i)
                        try:
                            if candidate.is_visible():
                                date_control = candidate
                                break
                        except Exception:
                            continue
                if date_control is not None:
                    break

            if date_control is None:
                raise RuntimeError("Could not locate IMD GPM date control.")

            try:
                date_control.fill(iso_date)
            except Exception:
                date_control.fill(alt_date)

            submit_selectors = [
                "input[type='submit']",
                "button[type='submit']",
                "input[value='Input']",
                "input[value='Input i']",
                "button:has-text('Input')",
            ]

            submitted = False

            for sel in submit_selectors:
                loc = page.locator(sel)
                if loc.count() == 0:
                    continue
                for i in range(min(loc.count(), 5)):
                    candidate = loc.nth(i)
                    try:
                        if not candidate.is_visible():
                            continue
                        try:
                            with page.expect_download(timeout=20000) as dl_info:
                                candidate.click()
                            download = dl_info.value
                            download.save_as(str(output_path))
                            submitted = True
                            break
                        except PlaywrightTimeout:
                            candidate.click()
                            page.wait_for_timeout(2500)
                            submitted = True
                            break
                    except Exception:
                        continue
                if submitted:
                    break

            if not submitted:
                # A small fallback: inspect newly available .grd links after the form action.
                links = page.locator("a[href*='.grd'], a[href$='.GRD']")
                hrefs = []
                for i in range(min(links.count(), 20)):
                    try:
                        href = links.nth(i).get_attribute("href")
                        if href:
                            hrefs.append(href)
                    except Exception:
                        pass

                matching = [
                    h for h in hrefs
                    if expected_name.lower() in h.lower()
                    or target_date.strftime("%d%m%Y") in h
                ]

                if matching:
                    url = urllib.parse.urljoin(IMD_GPM_PAGE, matching[0])
                    req = urllib.request.Request(
                        url,
                        headers={"User-Agent": "MAUSAMSAATHI-SIH26/1.0"},
                    )
                    with urllib.request.urlopen(req, timeout=60000) as response:
                        data = response.read()
                    output_path.write_bytes(data)
                    submitted = True

            if not submitted:
                return False

            if not valid_grd(output_path):
                return False

            return True
        finally:
            browser.close()


def fetch_gpm_missing() -> pd.DataFrame:
    RAW_GPM_DIR.mkdir(parents=True, exist_ok=True)

    # Seed the stable cumulative dataset from the already-audited pilot CSV.
    # This avoids re-aggregating the existing 2026-08-09 -> 2026-09-23 history.
    if not RAIN_CUMULATIVE.exists() and SEED_RAINFALL.exists():
        shutil.copy2(SEED_RAINFALL, RAIN_CUMULATIVE)
        log(f"GPM: seeded cumulative history from {SEED_RAINFALL.name}")

    # Import existing valid GPM files from the location used in the pilot work.
    if TELEGRAM_GPM_DIR.exists():
        for src in TELEGRAM_GPM_DIR.glob("*.grd"):
            try:
                d = datetime.strptime(src.stem, "%d%m%Y").date()
            except ValueError:
                continue
            dst = RAW_GPM_DIR / src.name
            if valid_grd(src) and not valid_grd(dst):
                shutil.copy2(src, dst)

    start = date(2026, 8, 9)
    today = datetime.now().astimezone().date()

    # We retry all dates from the first pilot day through today that are missing.
    new_frames = []
    weights = load_weights()

    for d in safe_date_range(start, today):
        dst = RAW_GPM_DIR / (d.strftime("%d%m%Y") + ".grd")
        if not valid_grd(dst):
            log(f"GPM: fetching {d.isoformat()}")
            try:
                ok = _playwright_download_gpm_one_sync(d, dst)
            except Exception as exc:
                log(f"GPM: {d.isoformat()} fetch error: {exc}")
                ok = False

            if not ok:
                log(f"GPM: {d.isoformat()} unavailable/invalid, will retry later")
                continue

        # Aggregate only dates that are represented in the cumulative output.
        # Existing duplicate dates are de-duplicated in merge_cumulative().
        try:
            new_frames.append(
                aggregate_grd_one(dst, weights, d)
            )
        except Exception as exc:
            log(f"GPM: {d.isoformat()} aggregation error: {exc}")

    cumulative = merge_cumulative(
        RAIN_CUMULATIVE,
        new_frames,
        expected_columns=[
            "date",
            "local_body_code",
            "local_body_name",
            "rainfall_mean_mm",
            "rainfall_max_grid_mm",
            "contributing_grid_cells",
            "effective_weight",
        ],
        key_columns=["date", "local_body_code"],
    )

    # Do not publish a broken dataset.
    by_date = cumulative.groupby("date")["local_body_code"].nunique()
    bad_dates = by_date[by_date != EXPECTED_BODIES]
    if len(bad_dates):
        cumulative = cumulative.loc[
            ~cumulative["date"].isin(bad_dates.index)
        ].copy()

    cumulative = cumulative.sort_values(
        ["date", "local_body_code"]
    ).reset_index(drop=True)

    if cumulative.empty:
        raise RuntimeError("No valid cumulative rainfall data is available.")

    atomic_write_csv(RAIN_CUMULATIVE, cumulative)

    latest_continuous = continuous_panchayat_end(
        cumulative,
        start_date=pd.Timestamp(start),
    )
    if latest_continuous is None:
        raise RuntimeError(
            "Rainfall has no continuous 486-Panchayat coverage from pilot start."
        )

    log(
        "GPM: cumulative continuous coverage "
        f"2026-08-09 -> {latest_continuous.date()}"
    )
    return cumulative


# ---------------------------------------------------------------------------
# 2. Copernicus CDS ERA5T daily download + exact existing aggregation
# ---------------------------------------------------------------------------

def request_era5_one_day(target_date: date, output_path: Path) -> bool:
    try:
        import cdsapi
    except Exception as exc:
        raise RuntimeError(
            "cdsapi is required for ERA5T refresh in the backend venv."
        ) from exc

    client = cdsapi.Client()

    request = {
        "product_type": "reanalysis",
        "variable": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "surface_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year": f"{target_date.year:04d}",
        "month": f"{target_date.month:02d}",
        "day": [f"{target_date.day:02d}"],
        "time": [
            "00:00",
            "06:00",
            "12:00",
            "18:00",
        ],
        "area": [23.75, 81.25, 21.50, 83.00],
        "grid": [0.25, 0.25],
        "format": "netcdf",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    client.retrieve(
        "reanalysis-era5-single-levels",
        request,
        str(output_path),
    )

    return output_path.exists() and output_path.stat().st_size > 0


def _magnus_rh_percent(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    a = 17.625
    b = 243.04
    gamma_t = (a * t_c) / (b + t_c)
    gamma_td = (a * td_c) / (b + td_c)
    return np.clip(100.0 * np.exp(gamma_td - gamma_t), 0.0, 100.0)


def locate_nc_var(ds, aliases: set[str]) -> str:
    lower = {str(k).lower(): str(k) for k in ds.data_vars}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    raise RuntimeError(f"ERA5 variable missing; aliases={sorted(aliases)}")


def aggregate_era5_one(path: Path, target_date: date, weights: pd.DataFrame) -> pd.DataFrame:
    import xarray as xr

    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"ERA5 file invalid/missing: {path}")

    with xr.open_dataset(path) as ds:
        time_name = "valid_time" if "valid_time" in ds.coords else "time"
        if time_name not in ds:
            raise RuntimeError(
                f"{path.name}: no time/valid_time coordinate"
            )

        times = pd.to_datetime(ds[time_name].values)
        t = np.asarray(
            ds[locate_nc_var(ds, {"t2m", "2t", "2m_temperature"})].values,
            dtype="float64",
        )
        d = np.asarray(
            ds[locate_nc_var(ds, {"d2m", "2d", "2m_dewpoint_temperature"})].values,
            dtype="float64",
        )
        sp = np.asarray(
            ds[locate_nc_var(ds, {"sp", "surface_pressure"})].values,
            dtype="float64",
        )
        u = np.asarray(
            ds[locate_nc_var(ds, {"u10", "10u", "10m_u_component_of_wind"})].values,
            dtype="float64",
        )
        v = np.asarray(
            ds[locate_nc_var(ds, {"v10", "10v", "10m_v_component_of_wind"})].values,
            dtype="float64",
        )
        lat_name = "latitude"
        lon_name = "longitude"
        lat = np.asarray(ds[lat_name].values, dtype=float)
        lon = np.asarray(ds[lon_name].values, dtype=float)

    # Find exact weight cells by nearest coordinate within tolerance.
    cell_lat = weights["grid_lat"].to_numpy()
    cell_lon = weights["grid_lon"].to_numpy()

    lat_idx = []
    lon_idx = []
    for la, lo in zip(cell_lat, cell_lon):
        li = int(np.argmin(np.abs(lat - la)))
        oi = int(np.argmin(np.abs(lon - lo)))
        if abs(float(lat[li]) - la) > 1e-6 or abs(float(lon[oi]) - lo) > 1e-6:
            raise RuntimeError(
                f"ERA5 grid cell mismatch: requested ({la},{lo})"
            )
        lat_idx.append(li)
        lon_idx.append(oi)

    t_c = t[:, lat_idx, lon_idx] - 273.15
    d_c = d[:, lat_idx, lon_idx] - 273.15
    sp_hpa = sp[:, lat_idx, lon_idx] / 100.0
    u10 = u[:, lat_idx, lon_idx]
    v10 = v[:, lat_idx, lon_idx]

    rh = _magnus_rh_percent(t_c, d_c)
    wind = np.sqrt(np.square(u10) + np.square(v10))

    w = weights["weight"].to_numpy(dtype="float64")
    # Area-weight each represented grid cell, then form daily means.
    gp_t = np.sum(t_c * w[None, :], axis=1)
    gp_d = np.sum(d_c * w[None, :], axis=1)
    gp_rh = np.sum(rh * w[None, :], axis=1)
    gp_sp = np.sum(sp_hpa * w[None, :], axis=1)
    gp_u = np.sum(u10 * w[None, :], axis=1)
    gp_v = np.sum(v10 * w[None, :], axis=1)
    gp_wind = np.sum(wind * w[None, :], axis=1)

    obs_count = len(times)
    complete = obs_count == 4
    if obs_count not in (3, 4):
        raise RuntimeError(
            f"{path.name}: unexpected observation count {obs_count}"
        )

    # Build the exact per-Panchayat weighted daily aggregation.
    # Build the full time x cell contribution for each Panchayat.
    codes = weights["local_body_code"].to_numpy(dtype="int64")
    names = weights["local_body_name"].to_numpy()
    # The weights CSV contains multiple grid cells per Panchayat. We need the
    # same group aggregation as the historical audited script.
    temp = weights[[
        "local_body_code",
        "local_body_name",
        "weight",
    ]].copy()

    # The variables were reduced to the cells represented in weights, in the
    # same order as weights rows. Rebuild a [time, rows] representation.
    t_rows = t_c
    d_rows = d_c
    rh_rows = rh
    sp_rows = sp_hpa
    u_rows = u10
    v_rows = v10
    wind_rows = wind

    row_frames = []
    for j, code in enumerate(codes):
        one = {
            "local_body_code": code,
            "local_body_name": names[j],
            "weight": float(weights.iloc[j]["weight"]),
            "temperature": float(np.mean(t_rows[:, j])),
            "dewpoint": float(np.mean(d_rows[:, j])),
            "rh": float(np.mean(rh_rows[:, j])),
            "pressure": float(np.mean(sp_rows[:, j])),
            "u": float(np.mean(u_rows[:, j])),
            "v": float(np.mean(v_rows[:, j])),
            "wind": float(np.mean(wind_rows[:, j])),
        }
        row_frames.append(one)

    cell_daily = pd.DataFrame(row_frames)

    def weighted(col: str):
        return (
            cell_daily.groupby(
                ["local_body_code", "local_body_name"],
                sort=True,
            )
            .apply(
                lambda g: float(np.sum(g[col].to_numpy() * g["weight"].to_numpy()))
            )
            .reset_index(name=col)
        )

    result = weighted("temperature").rename(columns={"temperature": "temperature_c"})
    for source_col, out_col in [
        ("dewpoint", "dewpoint_c"),
        ("rh", "relative_humidity_pct"),
        ("pressure", "pressure_hpa"),
        ("u", "u10_ms"),
        ("v", "v10_ms"),
        ("wind", "wind_speed_ms"),
    ]:
        part = weighted(source_col).rename(columns={source_col: out_col})
        result = result.merge(
            part,
            on=["local_body_code", "local_body_name"],
            how="inner",
            validate="one_to_one",
        )

    result.insert(0, "date", pd.Timestamp(target_date))
    result["era5_observations_per_day"] = obs_count
    result["day_complete"] = complete
    result["source_file"] = path.name
    result["spatial_level"] = "Panchayat"

    if len(result) != EXPECTED_BODIES:
        raise RuntimeError(
            f"{path.name}: ERA5 aggregation produced {len(result)} rows"
        )

    for c in [
        "temperature_c",
        "dewpoint_c",
        "relative_humidity_pct",
        "pressure_hpa",
        "u10_ms",
        "v10_ms",
        "wind_speed_ms",
    ]:
        if not np.isfinite(result[c].to_numpy(float)).all():
            raise RuntimeError(f"{path.name}: non-finite {c}")

    if ((result["relative_humidity_pct"] < 0) |
        (result["relative_humidity_pct"] > 100)).any():
        raise RuntimeError(f"{path.name}: RH outside 0-100%")

    if (result["wind_speed_ms"] < 0).any():
        raise RuntimeError(f"{path.name}: negative wind speed")

    return result[
        [
            "date",
            "local_body_code",
            "local_body_name",
            "temperature_c",
            "dewpoint_c",
            "relative_humidity_pct",
            "pressure_hpa",
            "u10_ms",
            "v10_ms",
            "wind_speed_ms",
            "era5_observations_per_day",
            "day_complete",
            "source_file",
            "spatial_level",
        ]
    ]


def fetch_era5_missing() -> pd.DataFrame:
    RAW_ERA5_DIR.mkdir(parents=True, exist_ok=True)
    weights = load_weights()

    # Seed the stable cumulative dataset from the already-audited pilot CSV.
    # This avoids re-downloading/re-aggregating the existing 42-day history.
    if not ATM_CUMULATIVE.exists() and SEED_ATMOSPHERE.exists():
        shutil.copy2(SEED_ATMOSPHERE, ATM_CUMULATIVE)
        log(f"ERA5T: seeded cumulative history from {SEED_ATMOSPHERE.name}")

    # Existing pilot coverage begins here.
    start = date(2026, 8, 9)
    today = datetime.now().astimezone().date()
    new_frames = []

    download_start = start
    if ATM_CUMULATIVE.exists():
        try:
            existing_tail = pd.to_datetime(
                pd.read_csv(ATM_CUMULATIVE, usecols=["date"])["date"]
            ).dt.normalize().max()
            if pd.notna(existing_tail):
                download_start = max(
                    start,
                    (existing_tail.date() + timedelta(days=1)),
                )
        except Exception:
            download_start = start

    for d in safe_date_range(download_start, today):
        monthly_name = (
            f"era5t_atmosphere_{d:%Y%m%d}.nc"
        )
        nc_path = RAW_ERA5_DIR / monthly_name

        if not nc_path.exists() or nc_path.stat().st_size <= 0:
            log(f"ERA5T: requesting {d.isoformat()}")
            try:
                ok = request_era5_one_day(d, nc_path)
            except Exception as exc:
                log(f"ERA5T: {d.isoformat()} unavailable/error: {exc}")
                ok = False
            if not ok:
                log(f"ERA5T: {d.isoformat()} unavailable, will retry later")
                continue

        try:
            new_frames.append(
                aggregate_era5_one(nc_path, d, weights)
            )
        except Exception as exc:
            log(f"ERA5T: {d.isoformat()} aggregation error: {exc}")

    if ATM_CUMULATIVE.exists():
        existing = pd.read_csv(ATM_CUMULATIVE)
    else:
        existing = pd.DataFrame()

    parts = [existing] if not existing.empty else []
    parts.extend(new_frames)
    if not parts:
        raise RuntimeError("No ERA5T atmosphere data available.")

    cumulative = pd.concat(parts, ignore_index=True)

    expected_columns = [
        "date",
        "local_body_code",
        "local_body_name",
        "temperature_c",
        "dewpoint_c",
        "relative_humidity_pct",
        "pressure_hpa",
        "u10_ms",
        "v10_ms",
        "wind_speed_ms",
        "era5_observations_per_day",
        "day_complete",
        "source_file",
        "spatial_level",
    ]
    cumulative = cumulative[expected_columns].copy()
    cumulative["date"] = pd.to_datetime(cumulative["date"]).dt.normalize()
    cumulative["local_body_code"] = (
        pd.to_numeric(cumulative["local_body_code"], errors="raise").astype("int64")
    )

    cumulative = (
        cumulative.sort_values(["date", "local_body_code"])
        .drop_duplicates(["date", "local_body_code"], keep="last")
        .reset_index(drop=True)
    )

    counts = cumulative.groupby("date")["local_body_code"].nunique()
    bad_dates = counts[counts != EXPECTED_BODIES]
    if len(bad_dates):
        cumulative = cumulative.loc[
            ~cumulative["date"].isin(bad_dates.index)
        ].copy()

    cumulative = cumulative.sort_values(
        ["date", "local_body_code"]
    ).reset_index(drop=True)

    atomic_write_csv(ATM_CUMULATIVE, cumulative)

    latest_continuous = continuous_panchayat_end(
        cumulative,
        start_date=pd.Timestamp(start),
    )
    if latest_continuous is None:
        raise RuntimeError(
            "ERA5T has no continuous 486-Panchayat coverage from pilot start."
        )

    log(
        "ERA5T: cumulative continuous coverage "
        f"2026-08-09 -> {latest_continuous.date()}"
    )
    return cumulative


# ---------------------------------------------------------------------------
# 3. NOAA CPC blended OLR daily refresh + current OMI reconstruction
# ---------------------------------------------------------------------------

def download_olr_day(target_date: date, output_path: Path) -> bool:
    params = urllib.parse.urlencode(
        {
            "var": "olr",
            "time": f"{target_date:%Y-%m-%d}T00:00:00Z",
            "north": "20",
            "south": "-20",
            "west": "0",
            "east": "357.5",
            "horizStride": "1",
            "accept": "netcdf4",
        }
    )
    url = f"{NOAA_OLR_NCSS}?{params}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MAUSAMSAATHI-SIH26/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        data = response.read()

    if not data or len(data) < 1000:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return output_path.stat().st_size > 0


def read_olr_netcdf(path: Path):
    import netCDF4 as nc

    with nc.Dataset(path, "r") as ds:
        if "olr" not in ds.variables:
            raise RuntimeError(f"{path}: 'olr' variable missing")
        lat = np.asarray(ds.variables["lat"][:], dtype=float)
        lon = np.asarray(ds.variables["lon"][:], dtype=float)
        time_var = ds.variables["time"]
        time_values = np.asarray(time_var[:], dtype=float)
        units = time_var.units
        calendar = getattr(time_var, "calendar", "standard")
        dates_obj = nc.num2date(
            time_values,
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=False,
        )
        dates = np.array(
            [np.datetime64(str(d)[:10]) for d in dates_obj],
            dtype="datetime64[D]",
        )
        values = np.asarray(
            ds.variables["olr"][:],
            dtype=np.float64,
        )

    expected_lat = np.arange(-20.0, 20.0001, 2.5)
    expected_lon = np.arange(0.0, 359.9001, 2.5)

    # The audited project OLR file is already on the tropical OMI grid
    # (17 latitude points, -20..20, 2.5 degree spacing). Some single-day
    # NCSS requests can return the full 73-point global latitude grid, so
    # handle that case explicitly by selecting the exact OMI latitude band.
    global_lat = np.arange(90.0, -90.0001, -2.5)

    if lat.size == global_lat.size and np.allclose(
        lat, global_lat, atol=1e-10, rtol=0.0
    ):
        target_indices = [
            int(np.where(np.isclose(lat, wanted, atol=1e-10, rtol=0.0))[0][0])
            for wanted in expected_lat[::-1]
        ]
        # global_lat is north -> south. Selecting expected_lat[::-1]
        # preserves the source ordering for the requested -20..20 band.
        lat = lat[target_indices]
        values = values[:, target_indices, ::]
        lat = lat[::-1]
        values = values[:, ::-1, :]

    elif np.allclose(
        lat, expected_lat[::-1], atol=1e-10, rtol=0.0
    ):
        # Tropical OMI band but north -> south.
        lat = lat[::-1]
        values = values[:, ::-1, :]

    elif np.allclose(
        lat, expected_lat, atol=1e-10, rtol=0.0
    ):
        # Already exactly in OMI order.
        pass

    else:
        raise RuntimeError(
            f"OLR latitude grid does not match OMI grid. "
            f"Received {lat.size} points from {float(lat[0])} to {float(lat[-1])}."
        )

    if lon.size != expected_lon.size or not np.allclose(
        lon, expected_lon, atol=1e-10, rtol=0.0
    ):
        raise RuntimeError("OLR longitude grid does not match OMI grid.")

    if values.ndim != 3:
        raise RuntimeError(
            f"OLR values must be 3-D, got {values.shape}"
        )

    values = values.astype("float64")
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.nan

    return dates, values, lat, lon


def determine_valid_olr_append_dates(base_last: date) -> None:
    RAW_OLR_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().astimezone().date()

    for d in safe_date_range(base_last + timedelta(days=1), today):
        target = RAW_OLR_DIR / f"olr_{d:%Y%m%d}.nc"
        if target.exists() and target.stat().st_size > 0:
            continue
        log(f"NOAA OLR: requesting {d.isoformat()}")
        try:
            ok = download_olr_day(d, target)
        except Exception as exc:
            log(f"NOAA OLR: {d.isoformat()} unavailable/error: {exc}")
            ok = False

        if not ok:
            target.unlink(missing_ok=True)
            log(f"NOAA OLR: {d.isoformat()} unavailable, will retry later")
        else:
            log(f"NOAA OLR: saved {target.name} ({target.stat().st_size:,} bytes)")


def build_current_omi() -> pd.DataFrame:
    import netCDF4 as nc
    import mjoindices.olr_handling as olr_handling

    base = CURRENT_OMI_DIR / "cpc_blended_olr_20260101_20260923_tropical.nc"
    if not base.exists():
        raise FileNotFoundError(
            f"Audited base OLR file not found: {base}"
        )

    base_dates, base_values, lat, lon = read_olr_netcdf(base)
    base_last = date.fromisoformat(str(base_dates[-1]))
    determine_valid_olr_append_dates(base_last)

    extra_paths = sorted(RAW_OLR_DIR.glob("olr_*.nc"))
    extra_dates = []
    extra_values = []

    for path in extra_paths:
        try:
            dts, vals, lat2, lon2 = read_olr_netcdf(path)
            if len(dts) != 1:
                continue
            d = date.fromisoformat(str(dts[0]))
            if d <= base_last:
                continue
            if not np.allclose(lat2, lat, atol=1e-10, rtol=0.0):
                continue
            if not np.allclose(lon2, lon, atol=1e-10, rtol=0.0):
                continue
            if not np.isfinite(vals).all():
                continue
            extra_dates.append(np.datetime64(d))
            extra_values.append(vals[0])
        except Exception as exc:
            log(f"NOAA OLR: ignored bad append {path.name}: {exc}")

    all_dates = list(base_dates)
    all_values = [v for v in base_values]

    if extra_dates:
        order = np.argsort(np.array(extra_dates))
        for idx in order:
            all_dates.append(extra_dates[idx])
            all_values.append(extra_values[idx])

    dates = np.array(all_dates, dtype="datetime64[D]")
    values = np.stack(all_values, axis=0)

    # Keep one row per day and ensure no gap between the audited base and append tail.
    order = np.argsort(dates)
    dates = dates[order]
    values = values[order]

    keep = np.ones(len(dates), dtype=bool)
    if len(dates) > 1:
        dup = dates[1:] == dates[:-1]
        keep[1:] = ~dup
    dates = dates[keep]
    values = values[keep]

    # Validate continuity from the audited base start.
    expected = np.arange(
        dates[0],
        dates[-1] + np.timedelta64(1, "D"),
        np.timedelta64(1, "D"),
    )
    if len(expected) != len(dates) or not np.array_equal(expected, dates):
        # Do not push OMI beyond the first missing day after the base.
        good_end = 0
        for i in range(len(dates)):
            if dates[i] != dates[0] + np.timedelta64(i, "D"):
                break
            good_end = i + 1
        dates = dates[:good_end]
        values = values[:good_end]

    if len(dates) < 30:
        raise RuntimeError("Not enough OLR history to build current OMI.")

    # Reuse the already-audited project's exact OMI method.
    omi_path = resolve_project_module("calculate_current_omi_2026_final_fixed.py")
    log(f"OMI: using audited module {omi_path}")
    omi_mod = import_module(omi_path, "mausam_current_omi_dynamic")

    import mjoindices.empirical_orthogonal_functions as eof

    # Patch the package interpolation exactly as the audited project does.
    def safe_interpolate_spatial_grid(data, target_lat, target_long):
        source_lat = np.asarray(data.lat, dtype=float)
        source_lon = np.asarray(data.long, dtype=float)
        target_lat = np.asarray(target_lat, dtype=float)
        target_long = np.asarray(target_long, dtype=float)
        if (
            source_lat.shape == target_lat.shape
            and np.allclose(source_lat, target_lat, atol=1e-10, rtol=0.0)
            and source_lon.shape == target_long.shape
            and np.allclose(source_lon, target_long, atol=1e-10, rtol=0.0)
        ):
            return olr_handling.OLRData(
                np.array(data.olr, copy=True),
                np.array(data.time, copy=True),
                target_lat.copy(),
                target_long.copy(),
            )
        raise ValueError("Current OLR grid does not exactly match OMI grid.")

    olr_handling.interpolate_spatial_grid = safe_interpolate_spatial_grid

    eof_root = omi_mod.find_original_eof_root()
    eofs = omi_mod.load_original_eofs(eof_root)

    current_olr = olr_handling.OLRData(
        values,
        dates,
        lat,
        lon,
    )

    context_end = dates[-1]
    pcs_extended = omi_mod.omi_calc.calculate_pcs_from_olr(
        current_olr,
        eofs,
        OMI_CONTEXT_START,
        context_end,
        use_quick_temporal_filter=False,
    )

    # Build final current 2026 table using the exact established mapping.
    omi_mod.OUTPUT_START = OMI_OUTPUT_START
    omi_mod.OUTPUT_END = context_end
    final = omi_mod.build_final_output(pcs_extended)

    atomic_write_csv(OMI_CURRENT, final)

    log(
        f"OMI: current output through {str(context_end)} "
        f"({len(final)} rows)"
    )
    return final


# ---------------------------------------------------------------------------
# 4-8. Dynamic inference, anomaly, spatial publish
# ---------------------------------------------------------------------------

def determine_issue_date(
    rainfall: pd.DataFrame,
    atmosphere: pd.DataFrame,
    omi: pd.DataFrame,
) -> pd.Timestamp:
    rain_end = continuous_panchayat_end(
        rainfall,
        start_date=rainfall["date"].min(),
    )
    atm_end = continuous_panchayat_end(
        atmosphere,
        start_date=atmosphere["date"].min(),
    )

    if rain_end is None or atm_end is None:
        raise RuntimeError(
            "Could not determine continuous rainfall/atmosphere coverage."
        )

    omi_dates = pd.to_datetime(omi["date"]).dt.normalize()
    omi_end = omi_dates.max()

    candidates = [
        pd.Timestamp(rain_end),
        pd.Timestamp(atm_end),
        pd.Timestamp(omi_end) + pd.Timedelta(days=1),
    ]

    issue = min(candidates)

    context_start = issue - pd.Timedelta(days=CONTEXT_DAYS - 1)

    for label, frame in [
        ("rainfall", rainfall),
        ("atmosphere", atmosphere),
    ]:
        local = frame[
            (frame["date"] >= context_start)
            & (frame["date"] <= issue)
        ]
        expected_rows = (
            (issue - context_start).days + 1
        ) * EXPECTED_BODIES
        if len(local) != expected_rows:
            raise RuntimeError(
                f"{label} does not provide complete {CONTEXT_DAYS}-day "
                f"context through issue date {issue.date()}"
            )
        if local.duplicated(["date", "local_body_code"]).any():
            raise RuntimeError(
                f"{label} contains duplicate date/local_body_code keys"
            )

    omi_needed = issue - pd.Timedelta(days=1)
    if omi_needed not in set(omi_dates):
        raise RuntimeError(
            f"OMI does not contain required issue-1 date {omi_needed.date()}"
        )

    log(
        f"Resolved fully joinable issue date = {issue.date()} "
        f"(context {context_start.date()} -> {issue.date()})"
    )
    return issue


def run_dynamic_ml_inference(
    issue_date: pd.Timestamp,
    rainfall_path: Path,
    atmosphere_path: Path,
    omi_path: Path,
) -> pd.DataFrame:
    mod = import_module(V4_INFERENCE, "mausam_current_ml_dynamic")

    context_start = issue_date - pd.Timedelta(days=CONTEXT_DAYS - 1)

    mod.ISSUE_DATE = pd.Timestamp(issue_date)
    mod.CONTEXT_START = pd.Timestamp(context_start)

    mod.RAINFALL_FILE = rainfall_path
    mod.ATM_FILE = atmosphere_path
    mod.OMI_FILE = omi_path

    mod.DAILY_CLIM_FILE = (
        PROC / "climatology" / "bilaspur_panchayat_daily_climatology_1951_2023.csv"
    )
    mod.MONTH_CLIM_FILE = (
        PROC / "climatology" / "bilaspur_panchayat_monthly_climatology_1951_2023.csv"
    )

    # Force fresh current-source pages for this run.
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    mod.DMI_CACHE = CURRENT_DIR / f"dmi_cpc_current_{stamp}.html"
    mod.RONI_CACHE = CURRENT_DIR / f"roni_v5_current_{stamp}.html"

    mod.CLIMATE_SIGNALS_OUT = CLIMATE_CURRENT
    mod.FEATURE_OUT = FEATURE_CURRENT
    mod.PRED_OUT = PREDICTIONS_CURRENT
    mod.AUDIT_OUT = INFERENCE_AUDIT_CURRENT

    mod.main()

    predictions = pd.read_csv(PREDICTIONS_CURRENT)
    if len(predictions) != EXPECTED_BODIES:
        raise RuntimeError(
            f"ML prediction rows={len(predictions)}, expected {EXPECTED_BODIES}"
        )

    if predictions["local_body_code"].nunique() != EXPECTED_BODIES:
        raise RuntimeError("ML predictions do not contain 486 unique local bodies.")

    return predictions


def run_dynamic_anomaly() -> pd.DataFrame:
    mod = import_module(ANOMALY_MODULE, "mausam_current_anomaly_dynamic")
    mod.SNAPSHOT = FEATURE_CURRENT
    mod.OUT_DIR = CURRENT_DIR
    mod.OUTPUT_CSV = ANOMALY_CURRENT
    mod.AUDIT_JSON = ANOMALY_AUDIT_CURRENT

    mod.main()

    out = pd.read_csv(ANOMALY_CURRENT)
    if len(out) != EXPECTED_BODIES:
        raise RuntimeError(
            f"Rainfall anomaly rows={len(out)}, expected {EXPECTED_BODIES}"
        )

    # Correct the legacy hard-coded audit date using the actual feature snapshot.
    audit_payload = {}
    if ANOMALY_AUDIT_CURRENT.exists():
        try:
            audit_payload = json.loads(
                ANOMALY_AUDIT_CURRENT.read_text(encoding="utf-8")
            )
        except Exception:
            audit_payload = {}

    feature_dates = pd.to_datetime(out["date"]).dt.normalize()
    issue_date = feature_dates.max()
    audit_payload["issue_date"] = str(issue_date.date())
    audit_payload["status"] = "PASS"
    audit_payload["rows"] = int(len(out))
    audit_payload["local_body_count"] = int(out["local_body_code"].nunique())
    atomic_write_json(ANOMALY_AUDIT_CURRENT, audit_payload)

    return out


def run_dynamic_spatial(issue_date: pd.Timestamp) -> Path:
    mod = import_module(SPATIAL_MODULE, "mausam_current_spatial_dynamic")

    mod.ISSUE_DATE = issue_date.strftime("%Y-%m-%d")
    mod.PREDICTION_PATH = PREDICTIONS_CURRENT
    mod.OUT_DIR = SPATIAL_DIR
    mod.LAYER_DIR = SPATIAL_LAYER_DIR

    code_meta, _ = mod.build_hierarchy()
    predictions = mod.build_prediction_map(code_meta)
    boundaries = mod.load_boundaries(code_meta)

    hierarchy_codes = set(code_meta)
    prediction_codes = set(predictions)
    boundary_codes = {
        mod.as_code(f["properties"]["local_body_code"])
        for f in boundaries
    }

    if not (hierarchy_codes == prediction_codes == boundary_codes):
        raise RuntimeError(
            "Spatial refresh exact local_body_code join failed."
        )

    if not (
        len(hierarchy_codes) == EXPECTED_BODIES
        and len(prediction_codes) == EXPECTED_BODIES
        and len(boundary_codes) == EXPECTED_BODIES
    ):
        raise RuntimeError(
            "Spatial refresh geography/prediction counts are not 486."
        )

    SPATIAL_DIR.mkdir(parents=True, exist_ok=True)
    SPATIAL_LAYER_DIR.mkdir(parents=True, exist_ok=True)

    main_path = mod.build_main_geojson(
        boundaries,
        code_meta,
        predictions,
    )
    layer_paths = mod.build_layer_geojsons(
        boundaries,
        code_meta,
        predictions,
    )

    if len(layer_paths) != 20:
        raise RuntimeError(
            f"Spatial layer count={len(layer_paths)}, expected 20."
        )

    stable_tmp = SPATIAL_CURRENT.with_suffix(".geojson.tmp")
    shutil.copy2(main_path, stable_tmp)
    stable_tmp.replace(SPATIAL_CURRENT)

    log(
        f"Spatial: published {SPATIAL_CURRENT} "
        f"(issue_date={issue_date.date()}, layers={len(layer_paths)})"
    )
    return SPATIAL_CURRENT


def validate_publish_artifacts(issue_date: pd.Timestamp) -> None:
    for path in [
        FEATURE_CURRENT,
        PREDICTIONS_CURRENT,
        CLIMATE_CURRENT,
        INFERENCE_AUDIT_CURRENT,
        ANOMALY_CURRENT,
        ANOMALY_AUDIT_CURRENT,
        SPATIAL_CURRENT,
    ]:
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"Publish artifact missing/empty: {path}")

    pred = pd.read_csv(PREDICTIONS_CURRENT)
    if len(pred) != EXPECTED_BODIES:
        raise RuntimeError("Published prediction row count is not 486.")

    if str(pd.to_datetime(pred["date"]).dt.normalize().iloc[0].date()) != str(issue_date.date()):
        raise RuntimeError(
            "Published predictions do not carry the resolved issue date."
        )

    state = load_state()
    state.update(
        {
            "status": "PASS",
            "pilot_district": "Bilaspur",
            "issue_date": str(issue_date.date()),
            "last_successful_issue_date": str(issue_date.date()),
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "error": None,
            "traceback_tail": None,
            "local_bodies": EXPECTED_BODIES,
            "horizons_days": [7, 14, 21, 30],
            "daily_inference": True,
            "daily_retraining": False,
            "prediction_file": str(PREDICTIONS_CURRENT),
            "feature_file": str(FEATURE_CURRENT),
            "climate_file": str(CLIMATE_CURRENT),
            "rainfall_file": str(RAIN_CUMULATIVE),
            "atmosphere_file": str(ATM_CUMULATIVE),
            "omi_file": str(OMI_CURRENT),
            "anomaly_file": str(ANOMALY_CURRENT),
            "spatial_file": str(SPATIAL_CURRENT),
            "model_source": "existing_calibrated_historical_models",
        }
    )
    atomic_write_json(STATE_PATH, state)



def persist_refresh_success(issue_date: pd.Timestamp) -> None:
    # Record the successful refresh and refresh stable-asset checksums.
    if not PERSISTENCE_DB.exists():
        log(f"Persistence: SQLite not found, skipping DB write: {PERSISTENCE_DB}")
        return

    finished = datetime.now().astimezone().isoformat(timespec="seconds")
    refresh_id = f"refresh-{issue_date.strftime('%Y%m%d')}-{finished.replace(':', '').replace('+', '_')}"
    metadata = {
        "issue_date": str(issue_date.date()),
        "local_body_count": EXPECTED_BODIES,
        "model_source": "existing_calibrated_historical_models",
        "current_state": str(STATE_PATH),
    }

    record_refresh_run(
        refresh_run_id=refresh_id,
        started_at=finished,
        finished_at=finished,
        issue_date=str(issue_date.date()),
        status="PASS",
        pilot_district="Bilaspur",
        local_body_count=EXPECTED_BODIES,
        model_source="existing_calibrated_historical_models",
        manifest_path=str(STATE_PATH),
        error=None,
        metadata=metadata,
        path=PERSISTENCE_DB,
    )

    stable_assets = [
        (FEATURE_CURRENT, "current_ml_features", "current_ml", "published", True),
        (PREDICTIONS_CURRENT, "current_ml_predictions", "current_ml", "published", True),
        (CLIMATE_CURRENT, "current_climate_signals", "current_climate", "published", True),
        (ANOMALY_CURRENT, "current_rainfall_anomaly", "current_ml", "published", True),
        (SPATIAL_CURRENT, "current_spatial_geojson", "spatial", "published", True),
        (STATE_PATH, "current_state", "system", "manifest", True),
        (RAIN_CUMULATIVE, "current_rainfall", "rainfall", "rolling", True),
        (ATM_CUMULATIVE, "current_atmosphere", "era5t", "rolling", True),
        (OMI_CURRENT, "current_omi", "omi", "published", True),
    ]

    updated = 0
    for path, dataset_name, asset_kind, source_name, critical in stable_assets:
        if not path.exists() or path.stat().st_size <= 0:
            continue
        digest = sha256_file(path)
        logical_key = path.relative_to(ROOT).as_posix()
        version_label = f"{issue_date.date()}_{digest[:12]}"
        upsert_asset(
            logical_key=logical_key,
            asset_kind=asset_kind,
            dataset_name=dataset_name,
            source_name=source_name,
            is_critical=critical,
            version_label=version_label,
            issue_date=str(issue_date.date()),
            local_path=str(path),
            object_bucket="mausamsaathi-scientific",
            object_key=f"mausamsaathi/{logical_key}",
            sha256=digest,
            size_bytes=path.stat().st_size,
            status="verified_local",
            verified_at=finished,
            metadata={"registered_by": "refresh_current_bilaspur", "refresh_run_id": refresh_id},
            path=PERSISTENCE_DB,
        )
        updated += 1

    log(f"Persistence: refresh run + {updated} stable assets recorded")


def persist_refresh_failure(exc: Exception) -> None:
    # Record a failed refresh without modifying the last successful artifact.
    if not PERSISTENCE_DB.exists():
        return

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    refresh_id = f"refresh-failed-{now.replace(':', '').replace('+', '_')}"
    try:
        record_refresh_run(
            refresh_run_id=refresh_id,
            started_at=now,
            finished_at=now,
            issue_date=None,
            status="FAILED",
            pilot_district="Bilaspur",
            local_body_count=None,
            model_source="existing_calibrated_historical_models",
            manifest_path=str(STATE_PATH),
            error=str(exc),
            metadata={"recorded_by": "refresh_current_bilaspur", "last_successful_state_preserved": True},
            path=PERSISTENCE_DB,
        )
    except Exception as persistence_exc:
        log(f"Persistence: failed to record refresh failure: {persistence_exc}")

def refresh_once() -> None:
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    CURRENT_OMI_DIR.mkdir(parents=True, exist_ok=True)
    SPATIAL_DIR.mkdir(parents=True, exist_ok=True)

    old_state = load_state()

    rainfall = fetch_gpm_missing()
    atmosphere = fetch_era5_missing()
    omi = build_current_omi()

    issue_date = determine_issue_date(rainfall, atmosphere, omi)

    run_dynamic_ml_inference(
        issue_date,
        RAIN_CUMULATIVE,
        ATM_CUMULATIVE,
        OMI_CURRENT,
    )

    run_dynamic_anomaly()
    run_dynamic_spatial(issue_date)

    validate_publish_artifacts(issue_date)
    persist_refresh_success(issue_date)

    log("REFRESH STATUS: PASS")


def main() -> int:
    log("=" * 78)
    log("MAUSAMSAATHI - DAILY CURRENT REFRESH")
    log("=" * 78)

    try:
        with single_run_lock():
            refresh_once()
        return 0
    except Exception as exc:
        log(f"REFRESH STATUS: FAILED: {exc}")
        traceback.print_exc()

        # Preserve last known good state.
        state = load_state()
        state.update(
            {
                "status": "FAILED",
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "error": str(exc),
                "traceback_tail": traceback.format_exc()[-4000:],
                "last_successful_issue_date": state.get(
                    "last_successful_issue_date"
                ),
            }
        )
        try:
            atomic_write_json(STATE_PATH, state)
        except Exception:
            pass

        persist_refresh_failure(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

