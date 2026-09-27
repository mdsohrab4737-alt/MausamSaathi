 # -*- coding: utf-8 -*-

"""
MAUSAMSAATHI SIH 2026
CURRENT OMI RECONSTRUCTION - EXTENDED FILTER CONTEXT

Purpose
-------
Calculate current 2026 OMI PCs using:

1. Current CPC blended OLR
2. Original NOAA/Kiladis OMI EOF pairs
3. Original Wheeler-Kiladis temporal filter

Important design
----------------
The OLR calculation context is intentionally extended:

    2024-01-01 -> 2026-09-21

Final reported OMI output remains:

    2026-01-01 -> 2026-09-21

This gives the temporal filter historical context before the
2026 output period while keeping the output period unchanged.

The current OLR file was independently audited and promoted in
STEP 6C:

    cpc_blended_olr_20260101_20260923_tropical.nc

Although the filename is retained for continuity, its actual
coverage is now:

    2024-01-01 -> 2026-09-21

Missing days repaired in STEP 6B:
    2024-02-01
    2024-05-29
    2024-05-30
    2024-06-16
    2024-06-22
    2026-02-22
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import mjoindices.olr_handling as olr
import mjoindices.empirical_orthogonal_functions as eof
import mjoindices.omi.omi_calculator as omi_calc


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CURRENT_OMI_DIR = (
    PROJECT_ROOT
    / "data_test"
    / "processed"
    / "current_omi"
)

OLR_FILE = (
    CURRENT_OMI_DIR
    / "cpc_blended_olr_20260101_20260923_tropical.nc"
)

OFFICIAL_OMI_REFERENCE = (
    CURRENT_OMI_DIR
    / "official_omi_reference.txt"
)

OUTPUT_FILE = (
    CURRENT_OMI_DIR
    / "current_omi_pcs_20260101_20260921.csv"
)

VALIDATION_FILE = (
    CURRENT_OMI_DIR
    / "current_omi_official_overlap_validation.csv"
)

VALIDATION_META = (
    CURRENT_OMI_DIR
    / "current_omi_official_overlap_validation.json"
)

# ------------------------------------------------------------
# Extended context for temporal filtering
# ------------------------------------------------------------

CONTEXT_START = np.datetime64("2024-01-01")
CONTEXT_END = np.datetime64("2026-09-21")

# ------------------------------------------------------------
# Final requested output period
# ------------------------------------------------------------

OUTPUT_START = np.datetime64("2026-01-01")
OUTPUT_END = np.datetime64("2026-09-21")

# ------------------------------------------------------------
# Exact original OMI grid
# ------------------------------------------------------------

EXPECTED_LAT = np.arange(
    -20.0,
    20.0001,
    2.5
)

EXPECTED_LON = np.arange(
    0.0,
    359.9001,
    2.5
)


# ============================================================
# BANNER
# ============================================================

def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# SAFE INTERPOLATION PATCH
# ============================================================

def safe_interpolate_spatial_grid(
    data: olr.OLRData,
    target_lat: np.ndarray,
    target_long: np.ndarray,
) -> olr.OLRData:

    source_lat = np.asarray(data.lat, dtype=float)
    source_lon = np.asarray(data.long, dtype=float)

    target_lat = np.asarray(target_lat, dtype=float)
    target_long = np.asarray(target_long, dtype=float)

    if (
        source_lat.shape == target_lat.shape
        and np.allclose(
            source_lat,
            target_lat,
            atol=1e-10,
            rtol=0.0,
        )
        and source_lon.shape == target_long.shape
        and np.allclose(
            source_lon,
            target_long,
            atol=1e-10,
            rtol=0.0,
        )
    ):
        # The input is already exactly on the original OMI grid.
        # No interpolation is needed.
        return olr.OLRData(
            np.array(data.olr, copy=True),
            np.array(data.time, copy=True),
            target_lat.copy(),
            target_long.copy(),
        )

    raise ValueError(
        "Spatial grid mismatch. "
        "Current OLR does not exactly match original OMI EOF grid."
    )


# Replace broken SciPy interp2d based interpolation with exact-grid
# pass-through. This is safe because the current OLR has already
# been audited against the exact OMI grid.
olr.interpolate_spatial_grid = safe_interpolate_spatial_grid


# ============================================================
# FIND ORIGINAL NOAA EOF DIRECTORY
# ============================================================

def find_original_eof_root() -> Path:
    banner("LOCATE 366 ORIGINAL NOAA OMI EOF PAIRS")

    # Project canonical EOF root.
    #
    # This is the root already used by the project's validated
    # current-OMI calculations and by the current-vs-Zenodo EOF audit.
    canonical = (
        PROJECT_ROOT
        / "data_test"
        / "processed"
        / "current_omi"
        / "original_eofs"
    )

    if not canonical.is_dir():
        raise FileNotFoundError(
            "Canonical current OMI EOF root not found:\n"
            f"{canonical}"
        )

    eof1_dir = canonical / "eof1"
    eof2_dir = canonical / "eof2"

    if not eof1_dir.is_dir() or not eof2_dir.is_dir():
        raise RuntimeError(
            "Canonical EOF root exists, but eof1/eof2 directories "
            "are missing.\n"
            f"{canonical}"
        )

    eof1_files = sorted(
        eof1_dir.glob("eof*.txt")
    )
    eof2_files = sorted(
        eof2_dir.glob("eof*.txt")
    )

    if len(eof1_files) != 366 or len(eof2_files) != 366:
        raise RuntimeError(
            "Canonical EOF root does not contain exactly 366 EOF1 "
            "and 366 EOF2 files.\n"
            f"EOF1={len(eof1_files)}, EOF2={len(eof2_files)}\n"
            f"{canonical}"
        )

    required = (
        eof1_dir / "eof001.txt",
        eof1_dir / "eof366.txt",
        eof2_dir / "eof001.txt",
        eof2_dir / "eof366.txt",
    )

    missing = [
        str(p)
        for p in required
        if not p.exists()
    ]

    if missing:
        raise RuntimeError(
            "Canonical EOF root is missing required boundary files:\n"
            + "\n".join(missing)
        )

    print("Canonical validated EOF root:")
    print(canonical)
    print()
    print("EOF1 files:", len(eof1_files))
    print("EOF2 files:", len(eof2_files))
    print("Selection: PASS")

    return canonical


# ============================================================
# LOAD OLR
# ============================================================

def load_current_olr() -> olr.OLRData:

    banner("LOAD EXTENDED CURRENT CPC OLR")

    if not OLR_FILE.exists():
        raise FileNotFoundError(
            f"Current OLR file not found:\n{OLR_FILE}"
        )

    print("Source:", OLR_FILE)

    # Use NetCDF4 directly here so current file metadata is explicit.
    import netCDF4 as nc

    with nc.Dataset(OLR_FILE, "r") as ds:

        if "olr" not in ds.variables:
            raise RuntimeError(
                "Variable 'olr' not found in current OLR file."
            )

        lat = np.asarray(
            ds.variables["lat"][:],
            dtype=float,
        )

        lon = np.asarray(
            ds.variables["lon"][:],
            dtype=float,
        )

        time_var = ds.variables["time"]

        time_values = np.asarray(
            time_var[:],
            dtype=float,
        )

        units = time_var.units
        calendar = getattr(
            time_var,
            "calendar",
            "standard",
        )

        dates = nc.num2date(
            time_values,
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=False,
        )

        np_dates = np.array(
            [
                np.datetime64(str(d)[:10])
                for d in dates
            ],
            dtype="datetime64[D]",
        )

        values = np.asarray(
            ds.variables["olr"][:],
            dtype=np.float64,
        )

    print("Shape:", values.shape)
    print("Dates:", len(np_dates))
    print("First:", np_dates[0])
    print("Last :", np_dates[-1])

    print(
        "Original OLR Lat:",
        float(lat[0]),
        "->",
        float(lat[-1]),
    )

    print(
        "Lon:",
        float(lon[0]),
        "->",
        float(lon[-1]),
    )

    print(
        "Missing cells:",
        int(np.sum(~np.isfinite(values))),
    )

    # --------------------------------------------------------
    # Normalize latitude orientation
    # --------------------------------------------------------
    #
    # The CPC OLR file is stored north -> south:
    #
    #     20.0 -> -20.0
    #
    # The original NOAA OMI EOFs are stored south -> north:
    #
    #     -20.0 -> 20.0
    #
    # The spatial grid is otherwise identical.
    #
    # Reverse the latitude axis ONLY IN MEMORY so that the OLR
    # data and EOF vectors use the same latitude ordering.
    #
    # The NetCDF source file itself is NOT modified.
    # --------------------------------------------------------

    if lat.size != EXPECTED_LAT.size:
        raise RuntimeError(
            f"Unexpected latitude count: {lat.size}"
        )

    if np.allclose(
        lat,
        EXPECTED_LAT[::-1],
        atol=1e-10,
        rtol=0.0,
    ):
        print(
            "Latitude orientation:",
            "REVERSED SOURCE -> NORMALIZING TO OMI ORDER",
        )

        lat = lat[::-1]
        values = values[:, ::-1, :]

    elif np.allclose(
        lat,
        EXPECTED_LAT,
        atol=1e-10,
        rtol=0.0,
    ):
        print(
            "Latitude orientation:",
            "ALREADY IN OMI ORDER",
        )

    else:
        raise RuntimeError(
            "Latitude grid does not match original OMI grid "
            "even after checking reversed orientation."
        )

    # --------------------------------------------------------
    # Exact longitude verification
    # --------------------------------------------------------

    if not np.allclose(
        lon,
        EXPECTED_LON,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Longitude grid does not match original OMI grid."
        )

    # --------------------------------------------------------
    # Final integrity checks
    # --------------------------------------------------------

    if np.sum(~np.isfinite(values)) != 0:
        raise RuntimeError(
            "Current OLR still contains missing values."
        )

    if np_dates[0] != CONTEXT_START:
        raise RuntimeError(
            f"Unexpected context start: {np_dates[0]}"
        )

    if np_dates[-1] != CONTEXT_END:
        raise RuntimeError(
            f"Unexpected context end: {np_dates[-1]}"
        )

    expected_count = (
        CONTEXT_END - CONTEXT_START
    ).astype(int) + 1

    if len(np_dates) != expected_count:
        raise RuntimeError(
            f"Unexpected date count: {len(np_dates)} "
            f"expected {expected_count}"
        )

    # Final exact-grid verification after orientation normalization.
    if not np.allclose(
        lat,
        EXPECTED_LAT,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Normalized latitude grid does not match OMI grid."
        )

    print(
        "Normalized OLR Lat:",
        float(lat[0]),
        "->",
        float(lat[-1]),
    )

    print("Exact OMI latitude grid: PASS")
    print("Exact OMI longitude grid: PASS")
    print("No missing OLR values: PASS")
    print("Extended context coverage: PASS")

    return olr.OLRData(
        values,
        np_dates,
        lat,
        lon,
    )


# ============================================================
# LOAD ORIGINAL EOFs
# ============================================================

def load_original_eofs(
    eof_root: Path,
) -> eof.EOFDataForAllDOYs:

    banner("LOAD 366 ORIGINAL NOAA OMI EOF PAIRS")

    eofs = eof.load_all_original_eofs_from_directory(
        eof_root
    )

    count = len(eofs.eof_list)

    print("EOF count:", count)

    if count != 366:
        raise RuntimeError(
            f"Expected 366 EOF pairs, got {count}"
        )

    print(
        "EOF latitude grid:",
        float(eofs.lat[0]),
        "->",
        float(eofs.lat[-1]),
    )

    print(
        "EOF longitude grid:",
        float(eofs.long[0]),
        "->",
        float(eofs.long[-1]),
    )

    if not np.allclose(
        eofs.lat,
        EXPECTED_LAT,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(
            "EOF latitude grid mismatch."
        )

    if not np.allclose(
        eofs.long,
        EXPECTED_LON,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(
            "EOF longitude grid mismatch."
        )

    # Verify every EOF vector has 17*144 elements.
    expected_elements = 17 * 144

    for i, item in enumerate(eofs.eof_list, start=1):

        if item.eof1vector.size != expected_elements:
            raise RuntimeError(
                f"EOF1 size mismatch at DOY {i}"
            )

        if item.eof2vector.size != expected_elements:
            raise RuntimeError(
                f"EOF2 size mismatch at DOY {i}"
            )

    print("Every EOF pair:", expected_elements, "grid cells")
    print("EOF loading: PASS")

    return eofs


# ============================================================
# CALCULATE EXTENDED PCs
# ============================================================

def calculate_extended_pcs(
    current_olr: olr.OLRData,
    eofs: eof.EOFDataForAllDOYs,
):

    banner("CALCULATE CURRENT OMI PCs WITH EXTENDED FILTER CONTEXT")

    print(
        "Filter context:",
        str(CONTEXT_START),
        "->",
        str(CONTEXT_END),
    )

    print(
        "Final output:",
        str(OUTPUT_START),
        "->",
        str(OUTPUT_END),
    )

    print("Using original NOAA EOFs")
    print("Spatial interpolation: SKIPPED, EXACT GRID MATCH")
    print("Temporal filter: original Wheeler-Kiladis")

    # This calls the package implementation, but because the OLR
    # now contains extended context, the filter operates over the
    # entire 2024-01-01 -> 2026-09-21 interval.

    pcs_extended = omi_calc.calculate_pcs_from_olr(
        current_olr,
        eofs,
        CONTEXT_START,
        CONTEXT_END,
        use_quick_temporal_filter=False,
    )

    print()
    print(
        "Calculated PC rows with context:",
        pcs_extended.time.size
    )

    if pcs_extended.time.size != 995:
        raise RuntimeError(
            f"Expected 995 extended PC rows, "
            f"got {pcs_extended.time.size}"
        )

    return pcs_extended


# ============================================================
# SLICE FINAL 2026 OUTPUT
# ============================================================

def build_final_output(pcs_extended):

    banner("BUILD FINAL 2026 OMI OUTPUT")

    dates = np.asarray(
        pcs_extended.time,
        dtype="datetime64[D]",
    )

    mask = (
        (dates >= OUTPUT_START)
        & (dates <= OUTPUT_END)
    )

    final_dates = dates[mask]

    final_pc1 = np.asarray(
        pcs_extended.pc1[mask],
        dtype=float,
    )

    final_pc2 = np.asarray(
        pcs_extended.pc2[mask],
        dtype=float,
    )

    expected_count = (
        OUTPUT_END - OUTPUT_START
    ).astype(int) + 1

    print("Final rows:", len(final_dates))
    print("Expected rows:", expected_count)
    print("First:", final_dates[0])
    print("Last :", final_dates[-1])

    if len(final_dates) != expected_count:
        raise RuntimeError(
            f"Unexpected final row count: "
            f"{len(final_dates)} expected {expected_count}"
        )

    # --------------------------------------------------------
    # Project's established OMI -> RMM naming
    #
    # mjo_rmm1 = OMI PC2
    # mjo_rmm2 = -OMI PC1
    # --------------------------------------------------------

    mjo_rmm1 = final_pc2
    mjo_rmm2 = -final_pc1

    amplitude = np.sqrt(
        mjo_rmm1 ** 2
        + mjo_rmm2 ** 2
    )

    # Same phase sector logic already used by the project.
    angles = np.arctan2(
        final_pc1,
        final_pc2,
    )

    phases = (
        np.floor(
            (angles + np.pi)
            / (2.0 * np.pi)
            * 8.0
        ).astype(int)
        % 8
    ) + 1

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(
                final_dates.astype(str)
            ),
            "pc1": final_pc1,
            "pc2": final_pc2,
            "mjo_rmm1": mjo_rmm1,
            "mjo_rmm2": mjo_rmm2,
            "mjo_phase": phases,
            "amplitude": amplitude,
        }
    )

    if out["date"].duplicated().any():
        raise RuntimeError(
            "Duplicate dates found in final OMI output."
        )

    if out.isna().any().any():
        raise RuntimeError(
            "NaN values found in final OMI output."
        )

    print()
    print("Final OMI output rows:", len(out))
    print()
    print("PC statistics:")

    print(
        "  PC1 min:",
        float(out["pc1"].min())
    )

    print(
        "  PC1 max:",
        float(out["pc1"].max())
    )

    print(
        "  PC2 min:",
        float(out["pc2"].min())
    )

    print(
        "  PC2 max:",
        float(out["pc2"].max())
    )

    print(
        "  amplitude min:",
        float(out["amplitude"].min())
    )

    print(
        "  amplitude max:",
        float(out["amplitude"].max())
    )

    return out


# ============================================================
# SAVE FINAL OUTPUT
# ============================================================

def save_final_output(out: pd.DataFrame):

    banner("SAVE FINAL CURRENT OMI")

    CURRENT_OMI_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.10f",
    )

    print("Output:", OUTPUT_FILE)
    print("Rows:", len(out))

    print()
    print("Current OMI calculation: PASS")


# ============================================================
# OFFICIAL NOAA REFERENCE
# ============================================================

def load_official_reference():

    banner("LOAD OFFICIAL NOAA OMI REFERENCE")

    if not OFFICIAL_OMI_REFERENCE.exists():
        print(
            "Official reference not found locally."
        )
        print(
            "Skipping official overlap validation."
        )
        return None

    rows = []

    with OFFICIAL_OMI_REFERENCE.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line in f:

            s = line.strip()

            if not s or s.startswith("#"):
                continue

            parts = re.split(
                r"\s+",
                s,
            )

            if len(parts) < 6:
                continue

            try:
                # NOAA current OMI text format:
                # Year Month Day PC1 PC2 Amplitude ...
                # Therefore:
                #   parts[3] = PC1
                #   parts[4] = PC2
                #   parts[5] = Amplitude
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                pc1 = float(parts[3])
                pc2 = float(parts[4])
            except (
                ValueError,
                TypeError,
            ):
                continue

            rows.append(
                {
                    "date": pd.Timestamp(
                        year=year,
                        month=month,
                        day=day,
                    ),
                    "pc1_official": pc1,
                    "pc2_official": pc2,
                }
            )

    official = pd.DataFrame(rows)

    if official.empty:
        raise RuntimeError(
            "Official NOAA OMI reference contained no usable rows."
        )

    official = (
        official
        .sort_values("date")
        .drop_duplicates(
            "date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    print("Rows:", len(official))
    print(
        "Range:",
        official["date"].min(),
        "->",
        official["date"].max()
    )

    print("Official OMI loading: PASS")

    return official


# ============================================================
# VALIDATION
# ============================================================

def validate_against_official(
    current: pd.DataFrame,
    official: pd.DataFrame | None,
):

    banner("CURRENT OMI vs OFFICIAL NOAA OMI")

    if official is None:
        print(
            "Official reference unavailable; "
            "validation skipped."
        )
        return

    merged = current.merge(
        official,
        on="date",
        how="inner",
    )

    print("Overlap rows:", len(merged))

    if len(merged) < 30:
        raise RuntimeError(
            "Too few official overlap rows for validation."
        )

    x1 = merged["pc1"].to_numpy(float)
    y1 = merged["pc1_official"].to_numpy(float)

    x2 = merged["pc2"].to_numpy(float)
    y2 = merged["pc2_official"].to_numpy(float)

    pc1_corr = float(
        np.corrcoef(x1, y1)[0, 1]
    )

    pc2_corr = float(
        np.corrcoef(x2, y2)[0, 1]
    )

    pc1_rmse = float(
        np.sqrt(
            np.mean(
                (x1 - y1) ** 2
            )
        )
    )

    pc2_rmse = float(
        np.sqrt(
            np.mean(
                (x2 - y2) ** 2
            )
        )
    )

    # Direct correlation score:
    # average absolute component agreement.
    direct_score = float(
        (
            abs(pc1_corr)
            + abs(pc2_corr)
        )
        / 2.0
    )

    # Simultaneous sign flip.
    signflip_score = float(
        (
            abs(
                np.corrcoef(
                    -x1,
                    y1,
                )[0, 1]
            )
            +
            abs(
                np.corrcoef(
                    -x2,
                    y2,
                )[0, 1]
            )
        )
        / 2.0
    )

    validation = pd.DataFrame(
        {
            "date": merged["date"],
            "pc1_current": x1,
            "pc1_official": y1,
            "pc2_current": x2,
            "pc2_official": y2,
        }
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False,
        float_format="%.10f",
    )

    metadata = {
        "context_start": str(CONTEXT_START),
        "context_end": str(CONTEXT_END),
        "output_start": str(OUTPUT_START),
        "output_end": str(OUTPUT_END),
        "overlap_rows": int(len(merged)),
        "pc1_correlation": pc1_corr,
        "pc2_correlation": pc2_corr,
        "pc1_rmse": pc1_rmse,
        "pc2_rmse": pc2_rmse,
        "direct_correlation_score": direct_score,
        "simultaneous_sign_flip_score": signflip_score,
        "olr_source": str(OLR_FILE),
        "method": (
            "Original Wheeler-Kiladis filter with "
            "historical filter context"
        ),
        "eof_source": "Original NOAA OMI EOF pairs",
        "spatial_interpolation": "Skipped due exact OMI grid match",
        "repaired_missing_olr_days": [
            "2024-02-01",
            "2024-05-29",
            "2024-05-30",
            "2024-06-16",
            "2024-06-22",
            "2026-02-22",
        ],
    }

    VALIDATION_META.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "PC1 correlation:",
        f"{pc1_corr:.6f}"
    )

    print(
        "PC2 correlation:",
        f"{pc2_corr:.6f}"
    )

    print(
        "PC1 RMSE:",
        f"{pc1_rmse:.6f}"
    )

    print(
        "PC2 RMSE:",
        f"{pc2_rmse:.6f}"
    )

    print()
    print(
        "Direct correlation score:",
        f"{direct_score:.6f}"
    )

    print(
        "Simultaneous sign-flip score:",
        f"{signflip_score:.6f}"
    )

    print()
    print(
        "Validation file:",
        VALIDATION_FILE
    )

    print(
        "Metadata:",
        VALIDATION_META
    )


# ============================================================
# MAIN
# ============================================================

def main():

    banner("MAUSAMSAATHI SIH 2026")
    print("CURRENT OMI RECONSTRUCTION")
    print("EXTENDED FILTER CONTEXT")
    print()

    print(
        "Filter context:",
        str(CONTEXT_START),
        "->",
        str(CONTEXT_END),
    )

    print(
        "Final output:",
        str(OUTPUT_START),
        "->",
        str(OUTPUT_END),
    )

    # --------------------------------------------------------
    # 1. Locate EOFs
    # --------------------------------------------------------

    eof_root = find_original_eof_root()

    # --------------------------------------------------------
    # 2. Load OLR
    # --------------------------------------------------------

    current_olr = load_current_olr()

    # --------------------------------------------------------
    # 3. Load original EOFs
    # --------------------------------------------------------

    original_eofs = load_original_eofs(
        eof_root
    )

    # --------------------------------------------------------
    # 4. Calculate over extended context
    # --------------------------------------------------------

    pcs_extended = calculate_extended_pcs(
        current_olr,
        original_eofs,
    )

    # --------------------------------------------------------
    # 5. Slice final 2026 period
    # --------------------------------------------------------

    final_output = build_final_output(
        pcs_extended
    )

    # --------------------------------------------------------
    # 6. Save
    # --------------------------------------------------------

    save_final_output(
        final_output
    )

    # --------------------------------------------------------
    # 7. Official overlap validation
    # --------------------------------------------------------

    official = load_official_reference()

    validate_against_official(
        final_output,
        official,
    )

    # --------------------------------------------------------
    # 8. Complete
    # --------------------------------------------------------

    banner("STEP 7A COMPLETE")

    print(
        "Filter context used:",
        str(CONTEXT_START),
        "->",
        str(CONTEXT_END)
    )

    print(
        "Final OMI output:",
        str(OUTPUT_START),
        "->",
        str(OUTPUT_END)
    )

    print()
    print(
        "Current OMI reconstruction with "
        "extended filter context: PASS"
    )

    print()
    print(
        "Official-overlap validation uses the corrected NOAA OMI "
        "column positions: PC1=4th field, PC2=5th field."
    )


if __name__ == "__main__":
    main()
