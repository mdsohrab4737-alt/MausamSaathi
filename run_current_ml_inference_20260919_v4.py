from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


# =============================================================================
# MAUSAMSAATHI SIH 2026
# CURRENT 2026 LIVE ML INFERENCE
# =============================================================================
#
# Purpose:
#   Build the same 102-predictor schema used by the trained Step-5/6
#   classifiers using current 2026 observations available through the
#   latest fully joinable date, then run all 20 calibrated classifiers.
#
# Current fully joinable issue date:
#   2026-09-19
#
# Available current inputs:
#   IMD Panchayat rainfall: 2026-08-09 -> 2026-09-23
#   ERA5T Panchayat atmosphere: 2026-08-09 -> 2026-09-19
#   Current OMI: 2026-01-01 -> 2026-09-21
#
# The issue date is therefore fixed at 2026-09-19.
#
# Leakage rules:
#   rainfall/atmosphere predictors use only dates <= issue date
#   rain rolling/lag features are shifted by 1 day
#   atmosphere lag/rolling features are shifted by 1 day
#   DMI uses the previous completed month
#   RONI uses the current source's latest available completed season
#   MJO/OMI uses the previous day's OMI, matching the historical builder
#
# IMPORTANT:
#   The historical model was trained with NOAA/CPC RONI-v5. The current
#   RONI-v5 page may expose a partial 2026 row. This script accepts that
#   partial row, keeps RONI-v5, and uses the latest completed season
#   actually exposed by the source for the requested month. It never
#   substitutes RONI-v6 and records the fallback in the audit/metadata.
#
# No model is retrained.
# =============================================================================


ROOT = Path(__file__).resolve().parent

PROC = ROOT / "data_test" / "processed"
CLIMATE_SRC = PROC / "climate_sources"
MODEL_ROOT = PROC / "models"
CALIBRATOR_ROOT = MODEL_ROOT / "step6a_probability_calibrators"

CURRENT_CLIMATE_OUT = (
    PROC / "current_climate_2026"
)

CURRENT_CLIMATE_OUT.mkdir(
    parents=True,
    exist_ok=True,
)

ISSUE_DATE = pd.Timestamp(
    "2026-09-19"
)

CONTEXT_START = pd.Timestamp(
    "2026-08-09"
)

EXPECTED_BODIES = 486

RAINFALL_FILE = (
    PROC
    / "bilaspur_panchayat_rainfall_2026-08-09_2026-09-23.csv"
)

ATM_FILE = (
    PROC
    / "bilaspur_panchayat_era5t_daily_20260809_20260919.csv"
)

OMI_FILE = (
    PROC
    / "current_omi"
    / "current_omi_pcs_20260101_20260921.csv"
)

DAILY_CLIM_FILE = (
    PROC
    / "climatology"
    / "bilaspur_panchayat_daily_climatology_1951_2023.csv"
)

MONTH_CLIM_FILE = (
    PROC
    / "climatology"
    / "bilaspur_panchayat_monthly_climatology_1951_2023.csv"
)

# Current NOAA CPC monthly DMI source.
# The page exposes a text-version link containing the same monthly data.
# Using the text version avoids an optional pandas/lxml dependency.
DMI_URL = (
    "https://www.cpc.ncep.noaa.gov/products/"
    "international/ocean_monitoring/IODMI/DMI_month.html"
)

DMI_TEXT_URL = (
    "https://www.cpc.ncep.noaa.gov/products/"
    "international/ocean_monitoring/IODMI/"
    "mnth.ersstv5.clim19912020.dmi_current.txt"
)

RONI_V5_URL = (
    "https://cpc.ncep.noaa.gov/products/analysis_monitoring/"
    "enso/roni/v5/"
)

DMI_CACHE = (
    CURRENT_CLIMATE_OUT
    / "dmi_cpc_current_20260925.html"
)

DMI_TEXT_CACHE = (
    CURRENT_CLIMATE_OUT
    / "dmi_cpc_current_20260925.txt"
)

RONI_CACHE = (
    CURRENT_CLIMATE_OUT
    / "roni_v5_current_20260925.html"
)

CLIMATE_SIGNALS_OUT = (
    CURRENT_CLIMATE_OUT
    / "bilaspur_current_climate_signals_20260809_20260919.csv"
)

FEATURE_OUT = (
    CURRENT_CLIMATE_OUT
    / "mausam_current_ml_features_20260919.csv"
)

PRED_OUT = (
    CURRENT_CLIMATE_OUT
    / "mausam_current_ml_predictions_20260919.csv"
)

AUDIT_OUT = (
    CURRENT_CLIMATE_OUT
    / "mausam_current_ml_inference_20260919_audit.json"
)

HORIZONS = [
    7,
    14,
    21,
    30,
]

CLASSIFIER_TARGETS = [
    "heavy_rain_panchayat_mean",
    "heavy_rain_grid",
    "onset",
    "break_proxy",
    "active",
]

ATM_COLS = [
    "temperature_c",
    "dewpoint_c",
    "relative_humidity_pct",
    "pressure_hpa",
    "u10_ms",
    "v10_ms",
    "wind_speed_ms",
]

RONI_SEASON_BY_MONTH = {
    6: "MAM",
    7: "AMJ",
    8: "MJJ",
    9: "JJA",
    10: "JAS",
}


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fail(message: str) -> None:
    raise RuntimeError(
        f"CURRENT ML INFERENCE FAILED: {message}"
    )


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        fail(
            f"{label} not found:\n{path}"
        )


def normalize_date(
    df: pd.DataFrame,
    label: str,
) -> None:
    if "date" not in df.columns:
        fail(
            f"{label}: date column missing"
        )

    df["date"] = (
        pd.to_datetime(
            df["date"],
            errors="coerce",
        )
        .dt.normalize()
    )

    if df["date"].isna().any():
        fail(
            f"{label}: invalid dates found"
        )


def check_unique_key(
    df: pd.DataFrame,
    label: str,
) -> None:
    if not {
        "local_body_code",
        "date",
    }.issubset(df.columns):
        fail(
            f"{label}: required key columns missing"
        )

    dup = df.duplicated(
        [
            "local_body_code",
            "date",
        ],
        keep=False,
    )

    if dup.any():
        sample = df.loc[
            dup,
            [
                "local_body_code",
                "date",
            ],
        ].head(10)

        fail(
            f"{label}: duplicate local_body_code-date keys\n"
            f"{sample.to_string(index=False)}"
        )


def download_text(
    url: str,
    output: Path,
) -> None:
    import urllib.request

    print(
        "Downloading:",
        url,
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                "MAUSAMSAATHI-SIH26/1.0"
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=60,
        ) as response:
            data = response.read()
    except Exception as exc:
        fail(
            f"Download failed for {url}: {exc}"
        )

    if not data:
        fail(
            f"Downloaded 0 bytes from {url}"
        )

    output.write_bytes(data)

    print(
        "Saved:",
        output,
    )
    print(
        "Bytes:",
        len(data),
    )


def _parse_numeric_html_rows(html: str) -> list[list[str]]:
    """Extract simple HTML table rows whose first cell is a 4-digit year."""

    class _RowParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.rows: list[list[str]] = []
            self.in_tr = False
            self.in_cell = False
            self.current_row: list[str] = []
            self.current_cell: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            tag = tag.lower()
            if tag == "tr":
                if self.in_tr:
                    self._finish_row()
                self.in_tr = True
                self.current_row = []
                self.in_cell = False
                self.current_cell = []
            elif tag in {"td", "th"} and self.in_tr:
                if self.in_cell:
                    self._finish_cell()
                self.in_cell = True
                self.current_cell = []

        def handle_endtag(self, tag: str) -> None:
            tag = tag.lower()
            if tag in {"td", "th"} and self.in_cell:
                self._finish_cell()
            elif tag == "tr" and self.in_tr:
                self._finish_row()

        def handle_data(self, data: str) -> None:
            if self.in_tr and self.in_cell:
                self.current_cell.append(data)

        def _finish_cell(self) -> None:
            value = " ".join("".join(self.current_cell).split())
            self.current_row.append(value)
            self.current_cell = []
            self.in_cell = False

        def _finish_row(self) -> None:
            if self.in_cell:
                self._finish_cell()
            if self.current_row:
                self.rows.append(self.current_row)
            self.current_row = []
            self.current_cell = []
            self.in_cell = False
            self.in_tr = False

    parser = _RowParser()
    parser.feed(html)
    parser.close()

    out: list[list[str]] = []
    for row in parser.rows:
        if not row:
            continue
        first = row[0].strip()
        if re.fullmatch(r"\d{4}", first):
            out.append(row)
    return out


def prepare_current_dmi() -> tuple[pd.DataFrame, dict[str, Any]]:
    banner(
        "1. CURRENT DMI"
    )

    # Keep the HTML page as an audit artifact, but parse the source's
    # linked text version so this step does not require pandas/lxml.
    if DMI_CACHE.exists():
        print(
            "Using cached current CPC DMI page:",
            DMI_CACHE,
        )
    else:
        download_text(
            DMI_URL,
            DMI_CACHE,
        )

    if DMI_TEXT_CACHE.exists():
        print(
            "Using cached current CPC DMI text:",
            DMI_TEXT_CACHE,
        )
    else:
        download_text(
            DMI_TEXT_URL,
            DMI_TEXT_CACHE,
        )

    text = DMI_TEXT_CACHE.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    rows = _parse_numeric_text_rows(text)
    current_rows = [
        row
        for row in rows
        if row
        and row[0].strip() == str(ISSUE_DATE.year)
        and len(row) >= 5
    ]

    if not current_rows:
        fail(
            f"Current CPC DMI text source has no {ISSUE_DATE.year} monthly row."
        )

    # Text source layout:
    # Year Month WTIO SETIO DMI
    selected = {}
    for row in current_rows:
        try:
            month = int(row[1])
            value = float(row[4])
        except (ValueError, TypeError, IndexError):
            continue
        if month in {7, 8}:
            selected[month] = value

    missing = [m for m in (7, 8) if m not in selected]
    if missing:
        fail(
            "Current CPC DMI text source is missing required 2026 months: "
            + ", ".join(str(m) for m in missing)
        )

    july_date = pd.Timestamp(
        year=ISSUE_DATE.year,
        month=7,
        day=1,
    )
    august_date = pd.Timestamp(
        year=ISSUE_DATE.year,
        month=8,
        day=1,
    )

    dmi = pd.DataFrame(
        {
            "date": [july_date, august_date],
            "dmi": [selected[7], selected[8]],
        }
    )

    print(
        "Current CPC DMI:",
        july_date.date(),
        "=",
        f"{selected[7]:.6f}",
    )
    print(
        "Current CPC DMI:",
        august_date.date(),
        "=",
        f"{selected[8]:.6f}",
    )
    print(
        "Source dataset:",
        "NOAA CPC IODMI monthly, ERSSTv6",
    )
    print(
        "Historical model DMI source:",
        "NOAA PSL HadISST1.1",
    )
    print(
        "Historical/live DMI source difference:",
        "DOCUMENTED",
    )
    print(
        "DMI alignment:",
        "previous completed month",
    )
    print(
        "DMI current source/audit: PASS"
    )

    return (
        dmi,
        {
            "source_url": DMI_URL,
            "text_source_url": DMI_TEXT_URL,
            "source_file": str(DMI_CACHE),
            "text_source_file": str(DMI_TEXT_CACHE),
            "method":
                "NOAA CPC IODMI monthly text version",
            "source_dataset":
                "ERSSTv6",
            "historical_training_source":
                "NOAA PSL DMI HadISST1.1",
            "source_version_mismatch":
                True,
            "alignment":
                "previous completed month",
            "selected_source_dates": [
                str(july_date.date()),
                str(august_date.date()),
            ],
            "selected_dmi": {
                "2026-07-01": float(selected[7]),
                "2026-08-01": float(selected[8]),
            },
        },
    )


def _parse_numeric_text_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        if not re.fullmatch(r"\d{4}", parts[0]):
            continue

        if not re.fullmatch(r"\d{1,2}", parts[1]):
            continue

        # Validate the numeric fields we need.
        try:
            float(parts[2])
            float(parts[3])
            float(parts[4])
        except ValueError:
            continue

        rows.append(parts)

    return rows

def _flatten_html_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()

    if isinstance(
        out.columns,
        pd.MultiIndex,
    ):
        new_columns = []

        for col in out.columns:
            parts = []

            for item in col:
                value = str(item).strip()

                if value.lower() in {
                    "",
                    "nan",
                    "none",
                }:
                    continue

                parts.append(value)

            new_columns.append(
                " ".join(parts).strip()
            )

        out.columns = new_columns

    else:
        out.columns = [
            str(c).strip()
            for c in out.columns
        ]

    return out


def load_roni_v5() -> tuple[pd.DataFrame, dict[str, Any]]:
    banner(
        "2. CURRENT RONI-v5"
    )

    if RONI_CACHE.exists():
        print(
            "Using cached current RONI-v5:",
            RONI_CACHE,
        )
    else:
        download_text(
            RONI_V5_URL,
            RONI_CACHE,
        )

    html = RONI_CACHE.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    rows = _parse_numeric_html_rows(html)
    if not rows:
        fail(
            "RONI-v5 HTML table contains no numeric year rows."
        )

    season_names = [
        "DJF",
        "JFM",
        "FMA",
        "MAM",
        "AMJ",
        "MJJ",
        "JJA",
        "JAS",
        "ASO",
        "SON",
        "OND",
        "NDJ",
    ]

    parsed_rows = []
    for row in rows:
        # The current RONI-v5 page may expose only completed seasons for
        # the latest year. Therefore a partial year row is valid.
        if len(row) < 2:
            continue

        try:
            year = int(row[0])
        except ValueError:
            continue

        values = []
        for raw_value in row[1:13]:
            try:
                values.append(float(raw_value))
            except (ValueError, TypeError):
                values.append(np.nan)

        for season, value in zip(season_names, values):
            if np.isfinite(value):
                parsed_rows.append(
                    {
                        "year": year,
                        "season": season,
                        "roni": float(value),
                    }
                )

    roni = (
        pd.DataFrame(parsed_rows)
        .sort_values(
            ["year", "season"]
        )
        .drop_duplicates(
            ["year", "season"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if roni.empty:
        fail(
            "RONI-v5 contains no usable rows."
        )

    current_2026 = roni[
        roni["year"] == 2026
    ].copy()

    if current_2026.empty:
        fail(
            "RONI-v5 has no 2026 rows."
        )

    print(
        "2026 RONI-v5 values exposed by source:"
    )

    for _, row in current_2026.iterrows():
        print(
            " ",
            row["season"],
            "=",
            f'{row["roni"]:.6f}',
        )

    # Validate that at least one completed 2026 season is available.
    # Do not require JJA/MJJ specifically. The source can expose a
    # partial 2026 row, so the latest season actually present in v5
    # is the defensible fallback for later months.
    season_order = [
        "DJF",
        "JFM",
        "FMA",
        "MAM",
        "AMJ",
        "MJJ",
        "JJA",
        "JAS",
        "ASO",
        "SON",
        "OND",
        "NDJ",
    ]

    available_2026 = [
        str(s)
        for s in current_2026["season"].tolist()
        if str(s) in season_order
    ]

    if not available_2026:
        fail(
            "RONI-v5 2026 row was detected, but it contains "
            "no usable completed seasons."
        )

    latest_available = max(
        available_2026,
        key=season_order.index,
    )

    latest_value = float(
        current_2026.loc[
            current_2026["season"] == latest_available,
            "roni",
        ].iloc[0]
    )

    expected_season = RONI_SEASON_BY_MONTH.get(
        int(ISSUE_DATE.month)
    )

    if expected_season in available_2026:
        roni_policy = (
            f"historical month-to-season mapping; "
            f"{expected_season} available"
        )
        fallback_used = False
        fallback_season = expected_season
    else:
        roni_policy = (
            f"{expected_season} unavailable in current RONI-v5 "
            f"source; September 2026 uses latest available "
            f"completed v5 season {latest_available}. "
            "RONI-v6 is intentionally NOT substituted."
        )
        fallback_used = True
        fallback_season = latest_available

    print(
        "Latest available 2026 completed RONI-v5 season:",
        latest_available,
        "=",
        f"{latest_value:.6f}",
    )

    print()
    print(
        "RONI policy:",
        roni_policy,
    )
    print(
        "RONI current source/audit: PASS"
    )

    return (
        roni,
        {
            "source_url": RONI_V5_URL,
            "source_file": str(RONI_CACHE),
            "method":
                "NOAA CPC RONI-v5 / ERSSTv5",
            "historical_month_mapping":
                RONI_SEASON_BY_MONTH,
            "jja_available":
                not fallback_used,
            "september_fallback_used":
                fallback_used,
            "september_fallback_season":
                fallback_season,
            "latest_available_2026_season":
                latest_available,
            "latest_available_2026_value":
                latest_value,
            "policy":
                roni_policy,
        },
    )

def load_current_omi() -> pd.DataFrame:
    banner(
        "3. CURRENT OMI"
    )

    require_file(
        OMI_FILE,
        "Current OMI output",
    )

    omi = pd.read_csv(
        OMI_FILE,
        parse_dates=["date"],
    )

    required = {
        "date",
        "pc1",
        "pc2",
        "mjo_rmm1",
        "mjo_rmm2",
        "mjo_phase",
        "amplitude",
    }

    missing = required - set(
        omi.columns
    )

    if missing:
        fail(
            "Current OMI missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    omi["date"] = (
        pd.to_datetime(
            omi["date"],
            errors="coerce",
        )
        .dt.normalize()
    )

    omi = (
        omi.dropna(
            subset=[
                "date",
                "pc1",
                "pc2",
                "mjo_rmm1",
                "mjo_rmm2",
                "mjo_phase",
                "amplitude",
            ]
        )
        .sort_values("date")
        .drop_duplicates(
            "date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    required_mjo_date = (
        ISSUE_DATE
        - pd.Timedelta(days=1)
    )

    if required_mjo_date not in set(
        omi["date"]
    ):
        fail(
            "Current OMI does not contain required "
            f"previous-day MJO date {required_mjo_date.date()}"
        )

    print(
        "Rows:",
        len(omi),
    )
    print(
        "Range:",
        omi["date"].min().date(),
        "->",
        omi["date"].max().date(),
    )

    row = omi[
        omi["date"]
        == required_mjo_date
    ].iloc[0]

    print()
    print(
        "MJO feature source for issue date:",
        required_mjo_date.date(),
    )

    print(
        "  PC1:",
        f'{row["pc1"]:.6f}',
    )
    print(
        "  PC2:",
        f'{row["pc2"]:.6f}',
    )
    print(
        "  RMM1:",
        f'{row["mjo_rmm1"]:.6f}',
    )
    print(
        "  RMM2:",
        f'{row["mjo_rmm2"]:.6f}',
    )
    print(
        "  phase:",
        int(row["mjo_phase"]),
    )
    print(
        "  amplitude:",
        f'{row["amplitude"]:.6f}',
    )

    print(
        "Current OMI/MJO alignment: PASS"
    )

    return omi


def load_rainfall() -> pd.DataFrame:
    banner(
        "4. CURRENT IMD RAINFALL"
    )

    require_file(
        RAINFALL_FILE,
        "Current IMD Panchayat rainfall",
    )

    rain = pd.read_csv(
        RAINFALL_FILE,
        usecols=[
            "date",
            "local_body_code",
            "local_body_name",
            "rainfall_mean_mm",
            "rainfall_max_grid_mm",
        ],
    )

    normalize_date(
        rain,
        "Current rainfall",
    )

    rain = rain[
        (
            rain["date"]
            >= CONTEXT_START
        )
        & (
            rain["date"]
            <= ISSUE_DATE
        )
    ].copy()

    if (
        rain["local_body_code"]
        .nunique()
        != EXPECTED_BODIES
    ):
        fail(
            "Current rainfall local-body count="
            f'{rain["local_body_code"].nunique()}, '
            f"expected {EXPECTED_BODIES}"
        )

    check_unique_key(
        rain,
        "Current rainfall",
    )

    expected_dates = pd.date_range(
        CONTEXT_START,
        ISSUE_DATE,
        freq="D",
    )

    expected_rows = (
        len(expected_dates)
        * EXPECTED_BODIES
    )

    if len(rain) != expected_rows:
        fail(
            "Current rainfall rows="
            f"{len(rain):,}, expected "
            f"{expected_rows:,}"
        )

    actual_dates = pd.DatetimeIndex(
        sorted(
            rain["date"].unique()
        )
    )

    if not actual_dates.equals(
        expected_dates
    ):
        missing = sorted(
            set(expected_dates)
            - set(actual_dates)
        )

        fail(
            "Current rainfall missing dates: "
            + ", ".join(
                str(x.date())
                for x in missing[:20]
            )
        )

    critical = [
        "rainfall_mean_mm",
        "rainfall_max_grid_mm",
    ]

    if rain[critical].isna().any().any():
        fail(
            "Current rainfall contains missing critical values."
        )

    print(
        "Coverage:",
        rain["date"].min().date(),
        "->",
        rain["date"].max().date(),
    )
    print(
        "Rows:",
        len(rain),
    )
    print(
        "Local bodies:",
        rain["local_body_code"].nunique(),
    )
    print(
        "Current rainfall: PASS"
    )

    return rain


def load_atmosphere() -> pd.DataFrame:
    banner(
        "5. CURRENT ERA5T ATMOSPHERE"
    )

    require_file(
        ATM_FILE,
        "Current ERA5T Panchayat atmosphere",
    )

    atm = pd.read_csv(
        ATM_FILE,
        usecols=[
            "date",
            "local_body_code",
        ]
        + ATM_COLS,
    )

    normalize_date(
        atm,
        "Current atmosphere",
    )

    atm = atm[
        (
            atm["date"]
            >= CONTEXT_START
        )
        & (
            atm["date"]
            <= ISSUE_DATE
        )
    ].copy()

    if (
        atm["local_body_code"]
        .nunique()
        != EXPECTED_BODIES
    ):
        fail(
            "Current atmosphere local-body count="
            f'{atm["local_body_code"].nunique()}, '
            f"expected {EXPECTED_BODIES}"
        )

    check_unique_key(
        atm,
        "Current atmosphere",
    )

    expected_dates = pd.date_range(
        CONTEXT_START,
        ISSUE_DATE,
        freq="D",
    )

    expected_rows = (
        len(expected_dates)
        * EXPECTED_BODIES
    )

    if len(atm) != expected_rows:
        fail(
            "Current atmosphere rows="
            f"{len(atm):,}, expected "
            f"{expected_rows:,}"
        )

    actual_dates = pd.DatetimeIndex(
        sorted(
            atm["date"].unique()
        )
    )

    if not actual_dates.equals(
        expected_dates
    ):
        missing = sorted(
            set(expected_dates)
            - set(actual_dates)
        )

        fail(
            "Current atmosphere missing dates: "
            + ", ".join(
                str(x.date())
                for x in missing[:20]
            )
        )

    if atm[ATM_COLS].isna().any().any():
        bad = [
            c
            for c in ATM_COLS
            if atm[c].isna().any()
        ]

        fail(
            "Current atmosphere missing values in: "
            + ", ".join(bad)
        )

    print(
        "Coverage:",
        atm["date"].min().date(),
        "->",
        atm["date"].max().date(),
    )
    print(
        "Rows:",
        len(atm),
    )
    print(
        "Local bodies:",
        atm["local_body_code"].nunique(),
    )
    print(
        "Current ERA5T atmosphere: PASS"
    )

    return atm


def _get_current_roni_value(
    roni: pd.DataFrame,
    month: int,
) -> tuple[float, str]:
    season_order = [
        "DJF",
        "JFM",
        "FMA",
        "MAM",
        "AMJ",
        "MJJ",
        "JJA",
        "JAS",
        "ASO",
        "SON",
        "OND",
        "NDJ",
    ]

    season = RONI_SEASON_BY_MONTH[
        month
    ]

    current = roni[
        roni["year"] == ISSUE_DATE.year
    ].copy()

    exact = current[
        current["season"] == season
    ]

    if not exact.empty:
        return (
            float(
                exact.iloc[0]["roni"]
            ),
            season,
        )

    # The project policy is to use the latest completed 3-month
    # season actually available from the current RONI-v5 source.
    # This is deliberately source-driven: do not require MJJ or
    # substitute RONI-v6 when the page exposes only a partial row.
    available = current[
        current["season"].isin(
            season_order
        )
    ].copy()

    if available.empty:
        fail(
            "No current RONI-v5 value available for "
            f"month={month}."
        )

    # Never select a season later than the season required for the
    # requested month. For September the target is JJA, so if JJA
    # is absent but AMJ is the latest exposed season, AMJ is used.
    target_index = season_order.index(
        season
    )

    available["season_index"] = (
        available["season"].map(
            {
                name: idx
                for idx, name in enumerate(
                    season_order
                )
            }
        )
    )

    eligible = available[
        available["season_index"]
        <= target_index
    ].copy()

    if eligible.empty:
        fail(
            "Current RONI-v5 has no completed season "
            f"available on or before {season} for month={month}."
        )

    selected = eligible.sort_values(
        "season_index"
    ).iloc[-1]

    return (
        float(selected["roni"]),
        str(selected["season"]),
    )


def build_climate_signals(
    dmi: pd.DataFrame,
    roni: pd.DataFrame,
    omi: pd.DataFrame,
) -> pd.DataFrame:
    banner(
        "6. BUILD CURRENT CLIMATE SIGNALS"
    )

    dates = pd.DataFrame(
        {
            "date": pd.date_range(
                CONTEXT_START,
                ISSUE_DATE,
                freq="D",
            )
        }
    )

    dates["year"] = (
        dates["date"].dt.year
    )

    dates["month"] = (
        dates["date"].dt.month
    )

    # DMI = previous completed month.
    dates["dmi_source_date"] = (
        dates["date"]
        .dt.to_period("M")
        .dt.to_timestamp()
        - pd.DateOffset(
            months=1
        )
    )

    dates = dates.merge(
        dmi.rename(
            columns={
                "date":
                    "dmi_source_date"
            }
        ),
        on="dmi_source_date",
        how="left",
        validate="many_to_one",
    )

    if dates["dmi"].isna().any():
        bad = (
            dates.loc[
                dates["dmi"].isna(),
                [
                    "date",
                    "dmi_source_date",
                ],
            ]
            .drop_duplicates()
        )

        fail(
            "Current DMI join missing dates:\n"
            + bad.to_string(
                index=False
            )
        )

    # RONI source by month, using the latest completed v5 season
    # actually exposed by the current source when the target season
    # is unavailable.
    roni_values = []
    roni_source_seasons = []

    for month in dates["month"]:
        value, source_season = (
            _get_current_roni_value(
                roni,
                int(month),
            )
        )

        roni_values.append(
            value
        )
        roni_source_seasons.append(
            source_season
        )

    dates["roni"] = roni_values
    dates["roni_source_season"] = (
        roni_source_seasons
    )

    # MJO = previous day OMI.
    dates["mjo_source_date"] = (
        dates["date"]
        - pd.Timedelta(days=1)
    )

    omi_cols = [
        "date",
        "pc1",
        "pc2",
        "mjo_rmm1",
        "mjo_rmm2",
        "mjo_phase",
        "amplitude",
    ]

    omi_join = omi[
        omi_cols
    ].rename(
        columns={
            "date":
                "mjo_source_date"
        }
    )

    dates = dates.merge(
        omi_join,
        on="mjo_source_date",
        how="left",
        validate="many_to_one",
    )

    climate_cols = [
        "dmi",
        "roni",
        "pc1",
        "pc2",
        "amplitude",
        "mjo_rmm1",
        "mjo_rmm2",
        "mjo_phase",
    ]

    if dates[
        climate_cols
    ].isna().any().any():
        bad_cols = [
            c
            for c in climate_cols
            if dates[c].isna().any()
        ]

        fail(
            "Current climate signal missing values in: "
            + ", ".join(bad_cols)
        )

    out = dates[
        [
            "date",
            "dmi",
            "roni",
            "pc1",
            "pc2",
            "amplitude",
            "mjo_rmm1",
            "mjo_rmm2",
            "mjo_phase",
            "dmi_source_date",
            "roni_source_season",
            "mjo_source_date",
        ]
    ].copy()

    out.to_csv(
        CLIMATE_SIGNALS_OUT,
        index=False,
        float_format="%.10f",
    )

    target_row = out[
        out["date"]
        == ISSUE_DATE
    ].iloc[0]

    print(
        "Climate signal rows:",
        len(out),
    )
    print()
    print(
        "TARGET ISSUE DATE:",
        ISSUE_DATE.date(),
    )

    print(
        "  DMI:",
        f'{target_row["dmi"]:.6f}',
        "(source",
        target_row[
            "dmi_source_date"
        ].date(),
        ")",
    )

    print(
        "  RONI:",
        f'{target_row["roni"]:.6f}',
        "(source season",
        target_row[
            "roni_source_season"
        ],
        ")",
    )

    print(
        "  OMI PC1:",
        f'{target_row["pc1"]:.6f}',
    )

    print(
        "  OMI PC2:",
        f'{target_row["pc2"]:.6f}',
    )

    print(
        "  MJO phase:",
        int(target_row["mjo_phase"]),
    )

    print(
        "  MJO amplitude:",
        f'{target_row["amplitude"]:.6f}',
    )

    print(
        "Climate signal build: PASS"
    )

    return out


def add_rain_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = (
        df.sort_values(
            [
                "local_body_code",
                "date",
            ]
        )
        .copy()
    )

    grouped = df.groupby(
        "local_body_code",
        sort=False,
    )[
        "rainfall_mean_mm"
    ]

    for lag in (
        1,
        2,
        3,
        7,
        14,
        21,
        30,
    ):
        df[
            f"rain_lag_{lag}d_mm"
        ] = grouped.shift(
            lag
        )

    df["_wet"] = (
        df[
            "rainfall_mean_mm"
        ] > 2.5
    ).astype(
        "float32"
    )

    df["_dry"] = (
        df[
            "rainfall_mean_mm"
        ] < 2.5
    ).astype(
        "float32"
    )

    for win in (
        3,
        7,
        14,
        21,
        30,
    ):
        shifted_sum = grouped.shift(
            1
        )

        df[
            f"rain_roll_{win}d_sum_mm"
        ] = (
            shifted_sum
            .groupby(
                df["local_body_code"],
                sort=False,
            )
            .rolling(
                win,
                min_periods=win,
            )
            .sum()
            .reset_index(
                level=0,
                drop=True,
            )
        )

        df[
            f"rain_roll_{win}d_mean_mm"
        ] = (
            shifted_sum
            .groupby(
                df["local_body_code"],
                sort=False,
            )
            .rolling(
                win,
                min_periods=win,
            )
            .mean()
            .reset_index(
                level=0,
                drop=True,
            )
        )

        wet_shift = (
            df.groupby(
                "local_body_code",
                sort=False,
            )["_wet"]
            .shift(1)
        )

        dry_shift = (
            df.groupby(
                "local_body_code",
                sort=False,
            )["_dry"]
            .shift(1)
        )

        df[
            f"rain_roll_{win}d_wet_days"
        ] = (
            wet_shift
            .groupby(
                df["local_body_code"],
                sort=False,
            )
            .rolling(
                win,
                min_periods=win,
            )
            .sum()
            .reset_index(
                level=0,
                drop=True,
            )
        )

        df[
            f"rain_roll_{win}d_dry_days"
        ] = (
            dry_shift
            .groupby(
                df["local_body_code"],
                sort=False,
            )
            .rolling(
                win,
                min_periods=win,
            )
            .sum()
            .reset_index(
                level=0,
                drop=True,
            )
        )

    df.drop(
        columns=[
            "_wet",
            "_dry",
        ],
        inplace=True,
    )

    return df


def add_atm_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = (
        df.sort_values(
            [
                "local_body_code",
                "date",
            ]
        )
        .copy()
    )

    for col in ATM_COLS:
        grouped = df.groupby(
            "local_body_code",
            sort=False,
        )[col]

        for lag in (
            1,
            3,
            7,
        ):
            df[
                f"{col}_lag_{lag}d"
            ] = grouped.shift(
                lag
            )

        shifted = grouped.shift(
            1
        )

        for win in (
            3,
            7,
        ):
            df[
                f"{col}_roll_{win}d_mean"
            ] = (
                shifted
                .groupby(
                    df["local_body_code"],
                    sort=False,
                )
                .rolling(
                    win,
                    min_periods=win,
                )
                .mean()
                .reset_index(
                    level=0,
                    drop=True,
                )
            )

    return df


def merge_daily_climatology(
    base: pd.DataFrame,
) -> pd.DataFrame:
    dclim = pd.read_csv(
        DAILY_CLIM_FILE
    )

    if "local_body_code" not in dclim.columns:
        fail(
            "Daily climatology missing local_body_code"
        )

    if not {
        "month",
        "day",
    }.issubset(
        dclim.columns
    ):
        fail(
            "Daily climatology missing month/day keys"
        )

    dclim["month"] = pd.to_numeric(
        dclim["month"],
        errors="coerce",
    )

    dclim["day"] = pd.to_numeric(
        dclim["day"],
        errors="coerce",
    )

    if dclim[
        ["month", "day"]
    ].isna().any().any():
        fail(
            "Daily climatology has invalid month/day"
        )

    dclim["month"] = (
        dclim["month"]
        .astype("int8")
    )

    dclim["day"] = (
        dclim["day"]
        .astype("int8")
    )

    drop_keys = {
        "local_body_code",
        "local_body_name",
        "month",
        "day",
        "month_day",
    }

    if "date" in dclim.columns:
        drop_keys.add(
            "date"
        )

    value_cols = [
        c
        for c in dclim.columns
        if c not in drop_keys
    ]

    dclim.rename(
        columns={
            c:
            f"clim_daily_{c}"
            for c in value_cols
        },
        inplace=True,
    )

    if dclim.duplicated(
        [
            "local_body_code",
            "month",
            "day",
        ]
    ).any():
        fail(
            "Daily climatology duplicate keys"
        )

    base["month"] = (
        base["date"]
        .dt.month
        .astype("int8")
    )

    base["day"] = (
        base["date"]
        .dt.day
        .astype("int8")
    )

    base = base.merge(
        dclim.drop(
            columns=[
                c
                for c in [
                    "date",
                    "local_body_name",
                ]
                if c in dclim.columns
            ]
        ),
        on=[
            "local_body_code",
            "month",
            "day",
        ],
        how="left",
        validate="many_to_one",
    )

    base.drop(
        columns=[
            "month",
            "day",
        ],
        inplace=True,
    )

    return base


def merge_monthly_climatology(
    base: pd.DataFrame,
) -> pd.DataFrame:
    mclim = pd.read_csv(
        MONTH_CLIM_FILE
    )

    if not {
        "local_body_code",
        "month",
    }.issubset(
        mclim.columns
    ):
        fail(
            "Monthly climatology missing local_body_code/month"
        )

    mclim["month"] = pd.to_numeric(
        mclim["month"],
        errors="coerce",
    )

    if mclim["month"].isna().any():
        fail(
            "Monthly climatology invalid month"
        )

    mclim["month"] = (
        mclim["month"]
        .astype("int8")
    )

    value_cols = [
        c
        for c in mclim.columns
        if c not in {
            "local_body_code",
            "month",
        }
    ]

    mclim.rename(
        columns={
            c:
            f"clim_monthly_{c}"
            for c in value_cols
        },
        inplace=True,
    )

    if mclim.duplicated(
        [
            "local_body_code",
            "month",
        ]
    ).any():
        fail(
            "Monthly climatology duplicate keys"
        )

    base["month"] = (
        base["date"]
        .dt.month
        .astype("int8")
    )

    base = base.merge(
        mclim,
        on=[
            "local_body_code",
            "month",
        ],
        how="left",
        validate="many_to_one",
    )

    base.drop(
        columns=["month"],
        inplace=True,
    )

    return base


def find_seed_model() -> Path:
    pattern = (
        "active_7d_"
        "xgboost_time_split_1979_2015.json"
    )

    matches = sorted(
        MODEL_ROOT.rglob(
            pattern
        )
    )

    if len(matches) != 1:
        fail(
            "Expected exactly one seed active_7d model, "
            f"found {len(matches)}:\n"
            + "\n".join(
                str(x)
                for x in matches
            )
        )

    return matches[0]


def find_model(
    target: str,
    horizon: int,
) -> Path:
    pattern = (
        f"{target}_{horizon}d_"
        "xgboost_time_split_1979_2015.json"
    )

    matches = sorted(
        MODEL_ROOT.rglob(
            pattern
        )
    )

    if len(matches) != 1:
        fail(
            f"Expected exactly one model for "
            f"{target} {horizon}d, found "
            f"{len(matches)}:\n"
            + "\n".join(
                str(x)
                for x in matches
            )
        )

    return matches[0]


def find_calibrator(
    target: str,
    horizon: int,
) -> Path:
    path = (
        CALIBRATOR_ROOT
        / f"{horizon}d_{target}_{horizon}d_calibrator.joblib"
    )

    if not path.exists():
        fail(
            f"Calibrator not found:\n{path}"
        )

    return path


def build_feature_snapshot(
    rain: pd.DataFrame,
    atm: pd.DataFrame,
    climate: pd.DataFrame,
    required_features: list[str],
) -> pd.DataFrame:
    banner(
        "7. BUILD 102-FEATURE CURRENT SNAPSHOT"
    )

    names = (
        rain[
            [
                "local_body_code",
                "local_body_name",
            ]
        ]
        .drop_duplicates(
            "local_body_code"
        )
    )

    base = rain.merge(
        atm,
        on=[
            "local_body_code",
            "date",
        ],
        how="left",
        validate="one_to_one",
    )

    critical = [
        "rainfall_mean_mm",
        "rainfall_max_grid_mm",
    ] + ATM_COLS

    if base[critical].isna().any().any():
        bad_cols = [
            c
            for c in critical
            if base[c].isna().any()
        ]

        fail(
            "Critical current rainfall/atmosphere missing values: "
            + ", ".join(bad_cols)
        )

    climate_keep = [
        "date",
        "dmi",
        "roni",
        "pc1",
        "pc2",
        "amplitude",
        "mjo_rmm1",
        "mjo_rmm2",
        "mjo_phase",
    ]

    base = base.merge(
        climate[
            climate_keep
        ],
        on="date",
        how="left",
        validate="many_to_one",
    )

    if base[
        climate_keep[1:]
    ].isna().any().any():
        bad_cols = [
            c
            for c in climate_keep[1:]
            if base[c].isna().any()
        ]

        fail(
            "Current climate merge contains missing values in: "
            + ", ".join(bad_cols)
        )

    base = merge_daily_climatology(
        base
    )

    base = merge_monthly_climatology(
        base
    )

    base = (
        base.sort_values(
            [
                "local_body_code",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    base["day_of_year"] = (
        base["date"]
        .dt.dayofyear
        .astype("int16")
    )

    base["month"] = (
        base["date"]
        .dt.month
        .astype("int8")
    )

    base["doy_sin"] = np.sin(
        2
        * np.pi
        * base[
            "day_of_year"
        ].to_numpy(
            dtype=float
        )
        / 365.25
    )

    base["doy_cos"] = np.cos(
        2
        * np.pi
        * base[
            "day_of_year"
        ].to_numpy(
            dtype=float
        )
        / 365.25
    )

    base["monsoon_day"] = (
        base["date"].dt.dayofyear
        - pd.to_datetime(
            base["date"].dt.year.astype(str)
            + "-06-01"
        ).dt.dayofyear
        + 1
    ).astype(
        "int16"
    )

    base = add_rain_features(
        base
    )

    base = add_atm_features(
        base
    )

    base[
        "heavy_rain_grid_proxy_today"
    ] = (
        base[
            "rainfall_max_grid_mm"
        ]
        >= 64.5
    ).astype(
        "int8"
    )

    base[
        "dry_day_today"
    ] = (
        base[
            "rainfall_mean_mm"
        ]
        < 2.5
    ).astype(
        "int8"
    )

    clim_candidates = [
        c
        for c in base.columns
        if c.startswith(
            "clim_daily_"
        )
        and any(
            term in c.lower()
            for term in [
                "rainfall_mean_mm",
                "rain_mean_mm",
                "mean_mm",
            ]
        )
    ]

    if not clim_candidates:
        fail(
            "Daily climatology rainfall mean field not found"
        )

    cc = clim_candidates[0]

    base[
        "rainfall_anomaly_mm"
    ] = (
        base[
            "rainfall_mean_mm"
        ]
        - pd.to_numeric(
            base[cc],
            errors="coerce",
        )
    )

    snapshot = base[
        base["date"]
        == ISSUE_DATE
    ].copy()

    if len(snapshot) != EXPECTED_BODIES:
        fail(
            f"Target snapshot rows={len(snapshot):,}, "
            f"expected {EXPECTED_BODIES:,}"
        )

    snapshot = snapshot.merge(
        names,
        on="local_body_code",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_name",
        ),
    )

    if "local_body_name_name" in snapshot.columns:
        snapshot[
            "local_body_name"
        ] = snapshot[
            "local_body_name_name"
        ]

        snapshot.drop(
            columns=[
                "local_body_name_name"
            ],
            inplace=True,
        )

    missing_features = [
        c
        for c in required_features
        if c not in snapshot.columns
    ]

    if missing_features:
        fail(
            "Required model features missing:\n"
            + "\n".join(
                missing_features
            )
        )

    X = snapshot[
        required_features
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    bad = ~np.isfinite(
        X.to_numpy(
            dtype=float
        )
    )

    if bad.any():
        bad_cols = [
            required_features[i]
            for i in np.where(
                bad.any(axis=0)
            )[0]
        ]

        fail(
            "Non-finite target feature values:\n"
            + "\n".join(
                bad_cols
            )
        )

    output = snapshot[
        [
            "date",
            "local_body_code",
            "local_body_name",
        ]
        + required_features
    ].copy()

    output.to_csv(
        FEATURE_OUT,
        index=False,
        float_format="%.10f",
    )

    print(
        "Target issue date:",
        ISSUE_DATE.date(),
    )
    print(
        "Feature rows:",
        len(output),
    )
    print(
        "Feature count:",
        len(required_features),
    )
    print(
        "Local bodies:",
        output[
            "local_body_code"
        ].nunique(),
    )
    print(
        "102-feature snapshot: PASS"
    )

    return output


def predict_model(
    feature_df: pd.DataFrame,
    required_features: list[str],
    model_path: Path,
) -> tuple[
    np.ndarray,
    dict[str, Any],
]:
    metadata_path = (
        model_path.with_suffix(
            ".metadata.json"
        )
    )

    if not metadata_path.exists():
        fail(
            f"Model metadata not found:\n"
            f"{metadata_path}"
        )

    meta = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    if int(
        meta.get(
            "feature_count",
            -1,
        )
    ) != len(
        required_features
    ):
        fail(
            f"{model_path.name}: metadata feature_count "
            f"{meta.get('feature_count')} != {len(required_features)}"
        )

    if list(
        meta.get(
            "features",
            [],
        )
    ) != required_features:
        fail(
            f"{model_path.name}: metadata feature order mismatch"
        )

    booster = xgb.Booster()
    booster.load_model(
        str(model_path)
    )

    if booster.num_features() != len(
        required_features
    ):
        fail(
            f"{model_path.name}: booster features "
            f"{booster.num_features()} != {len(required_features)}"
        )

    X = feature_df[
        required_features
    ].astype(float)

    matrix = xgb.DMatrix(
        X
    )

    best_iteration = meta.get(
        "best_iteration"
    )

    if isinstance(
        best_iteration,
        (int, float),
    ) and int(
        best_iteration
    ) >= 0:
        pred = booster.predict(
            matrix,
            iteration_range=(
                0,
                int(best_iteration) + 1,
            ),
        )
    else:
        pred = booster.predict(
            matrix
        )

    pred = np.asarray(
        pred,
        dtype=float,
    ).reshape(-1)

    if len(pred) != len(
        feature_df
    ):
        fail(
            f"{model_path.name}: prediction row count mismatch"
        )

    if not np.isfinite(
        pred
    ).all():
        fail(
            f"{model_path.name}: non-finite predictions"
        )

    if (
        (pred < 0)
        | (pred > 1)
    ).any():
        fail(
            f"{model_path.name}: predictions outside [0,1]"
        )

    return (
        pred,
        meta,
    )


def calibrate(
    raw: np.ndarray,
    calibrator_path: Path,
) -> tuple[
    np.ndarray,
    str,
]:
    calibrator = joblib.load(
        calibrator_path
    )

    raw2d = raw.reshape(
        -1,
        1,
    )

    if hasattr(
        calibrator,
        "predict_proba",
    ):
        calibrated = calibrator.predict_proba(
            raw2d
        )[:, 1]

    elif hasattr(
        calibrator,
        "predict",
    ):
        calibrated = calibrator.predict(
            raw2d
        )

    else:
        fail(
            "Unsupported calibrator type: "
            f"{type(calibrator).__name__}"
        )

    calibrated = np.asarray(
        calibrated,
        dtype=float,
    ).reshape(-1)

    if (
        not np.isfinite(
            calibrated
        ).all()
        or (
            (calibrated < 0)
            | (calibrated > 1)
        ).any()
    ):
        fail(
            f"{calibrator_path.name}: invalid calibrated probabilities"
        )

    return (
        calibrated,
        type(calibrator).__name__,
    )


def run_models(
    feature_df: pd.DataFrame,
    required_features: list[str],
    dmi_meta: dict[str, Any],
    roni_meta: dict[str, Any],
) -> pd.DataFrame:
    banner(
        "8. RUN 20 CALIBRATED CLASSIFIERS"
    )

    predictions = feature_df[
        [
            "date",
            "local_body_code",
            "local_body_name",
        ]
    ].copy()

    audits = []

    for horizon in HORIZONS:
        print()
        print(
            f"--- Horizon {horizon} days ---"
        )

        for target in CLASSIFIER_TARGETS:
            model_path = find_model(
                target,
                horizon,
            )

            calibrator_path = find_calibrator(
                target,
                horizon,
            )

            raw, meta = predict_model(
                feature_df,
                required_features,
                model_path,
            )

            calibrated, calibrator_type = calibrate(
                raw,
                calibrator_path,
            )

            raw_col = (
                f"{target}_{horizon}d_raw"
            )

            calibrated_col = (
                f"{target}_{horizon}d_calibrated"
            )

            predictions[
                raw_col
            ] = raw

            predictions[
                calibrated_col
            ] = calibrated

            audits.append(
                {
                    "target":
                        target,
                    "horizon_days":
                        horizon,
                    "model":
                        str(model_path),
                    "calibrator":
                        str(calibrator_path),
                    "calibrator_type":
                        calibrator_type,
                    "feature_count":
                        int(meta["feature_count"]),
                    "best_iteration":
                        meta.get(
                            "best_iteration"
                        ),
                    "raw_min":
                        float(raw.min()),
                    "raw_max":
                        float(raw.max()),
                    "calibrated_min":
                        float(calibrated.min()),
                    "calibrated_max":
                        float(calibrated.max()),
                }
            )

            print(
                f"{target:30s}"
                f" | raw {raw.min():.6f}"
                f"-{raw.max():.6f}"
                f" | cal {calibrated.min():.6f}"
                f"-{calibrated.max():.6f}"
            )

    expected_columns = (
        3
        + (
            len(HORIZONS)
            * len(CLASSIFIER_TARGETS)
            * 2
        )
    )

    if len(
        predictions.columns
    ) != expected_columns:
        fail(
            f"Prediction columns={len(predictions.columns)}, "
            f"expected {expected_columns}"
        )

    predictions.to_csv(
        PRED_OUT,
        index=False,
        float_format="%.10f",
    )

    audit = {
        "status":
            "PASS",
        "issue_date":
            str(ISSUE_DATE.date()),
        "context_start":
            str(CONTEXT_START.date()),
        "context_end":
            str(ISSUE_DATE.date()),
        "expected_local_bodies":
            EXPECTED_BODIES,
        "actual_local_bodies":
            int(
                predictions[
                    "local_body_code"
                ].nunique()
            ),
        "feature_count":
            len(required_features),
        "models_run":
            len(audits),
        "prediction_columns":
            len(predictions.columns),
        "rainfall_file":
            str(RAINFALL_FILE),
        "atmosphere_file":
            str(ATM_FILE),
        "current_omi_file":
            str(OMI_FILE),
        "daily_climatology_file":
            str(DAILY_CLIM_FILE),
        "monthly_climatology_file":
            str(MONTH_CLIM_FILE),
        "dmi_source":
            dmi_meta,
        "roni_source":
            roni_meta,
        "live_climate_source_policy":
            {
                "historical_model_dmi_source":
                    "NOAA PSL HadISST1.1",
                "live_dmi_source":
                    "NOAA CPC IODMI monthly ERSSTv6",
                "historical_model_roni_source":
                    "NOAA CPC RONI-v5 / ERSSTv5",
                "live_roni_source":
                    "NOAA CPC RONI-v5 / ERSSTv5",
                "source_mismatch_documented":
                    True,
            },
        "feature_output":
            str(FEATURE_OUT),
        "prediction_output":
            str(PRED_OUT),
        "climate_signal_output":
            str(CLIMATE_SIGNALS_OUT),
        "models":
            audits,
    }

    AUDIT_OUT.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "20 calibrated classifiers: PASS"
    )

    return predictions


def main() -> None:
    banner(
        "MAUSAMSAATHI SIH 2026"
    )
    print(
        "CURRENT 2026 LIVE ML INFERENCE"
    )
    print(
        "Issue date:",
        ISSUE_DATE.date(),
    )
    print(
        "Feature context:",
        CONTEXT_START.date(),
        "->",
        ISSUE_DATE.date(),
    )
    print()

    # 1-3 current climate sources.
    dmi, dmi_meta = (
        prepare_current_dmi()
    )

    roni, roni_meta = (
        load_roni_v5()
    )

    omi = load_current_omi()

    # 4-5 local current observations.
    rain = load_rainfall()
    atm = load_atmosphere()

    # 6 climate signals.
    climate = build_climate_signals(
        dmi,
        roni,
        omi,
    )

    # Model feature schema.
    seed_model = find_seed_model()

    seed_metadata = json.loads(
        seed_model.with_suffix(
            ".metadata.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    required_features = list(
        seed_metadata[
            "features"
        ]
    )

    if len(
        required_features
    ) != 102:
        fail(
            f"Expected 102 predictors, found "
            f"{len(required_features)}"
        )

    # 7 feature snapshot.
    features = build_feature_snapshot(
        rain,
        atm,
        climate,
        required_features,
    )

    # 8 model inference.
    predictions = run_models(
        features,
        required_features,
        dmi_meta,
        roni_meta,
    )

    banner(
        "STEP 8 LIVE ML INFERENCE COMPLETE"
    )

    print(
        "Issue date:",
        ISSUE_DATE.date(),
    )

    print(
        "Local bodies:",
        predictions[
            "local_body_code"
        ].nunique(),
    )

    print(
        "Features:",
        FEATURE_OUT,
    )

    print(
        "Predictions:",
        PRED_OUT,
    )

    print(
        "Climate signals:",
        CLIMATE_SIGNALS_OUT,
    )

    print(
        "Audit:",
        AUDIT_OUT,
    )

    print()
    print(
        "LIVE 2026 ML INFERENCE: PASS"
    )

    print()
    print(
        "NOTE:"
    )
    print(
        "This creates current observation-based inference for "
        "2026-09-19. It does not retrain the historical models."
    )


if __name__ == "__main__":
    main()
