"""Build the (H3 cell, month) training panel for hotspot & severity classification.

This script consumes the accident-grain feature store and the static per-cell
aggregates, and emits ONE ROW PER (h3_cell, year_month) — the training grain
for the RF / XGBoost / Logistic Regression models.

Design principles
-----------------
1. Grain = (h3_cell, year_month). Every cell that appears in any accident
   record gets a row for every month in the study window (2022-01 .. 2023-12),
   including months with zero accidents.

2. Labels are STATIC per cell (following Islam et al. 2022, Al-Omari et al. 2025):
     - is_hotspot          — from ref_h3_cell_aggregates (Poisson-FDR, res 8)
     - dominant_severity   — majority class across full 2022-2023 for the cell,
                             tie-break by severity ladder (Fatal > Serious > ...)

3. Features are MONTH-SPECIFIC — they change across months for the same cell.
   This is what makes the framework spatiotemporal at the prediction level
   even though the labels are static (see thesis §3.3.5.2 / defence notes).

   Feature families:
     a. Causal rolling exposure (as of month start, strictly prior data only)
          same_cell_count_{30,90,180}d, neighbor_count_{30,90,180}d
     b. Cyclical temporal encoding
          month_sin, month_cos (period=12)
     c. Monthly context aggregates
          monthly avg precip / humidity / visibility / cloudcover
          rainy_days_in_month, poor_visibility_days_in_month
          public_holiday_count_in_month, peak_hour_share_prior
          night_share_prior, motorcycle_share_prior
     d. Static cell context
          police_division, historical fatal_rate, historical accident_count,
          first-seen lat/lng centroid of the cell

MongoDB collections
-------------------
Input:
    accidents_spatiotemporal_features  — accident-grain feature store
    ref_h3_cell_aggregates             — static per-cell hotspot labels

Output:
    training_panel_cell_month          — one row per (h3_cell, year_month)

Local output:
    data/processed/panel_cell_month.csv
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import h3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

# ── Environment ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / "config" / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "accident_hotspot")

# ── MongoDB Collection Names ───────────────────────────────────────────────
COL_ACCIDENTS_ST = "accidents_spatiotemporal_features"
COL_CELL_AGGREGATES = "ref_h3_cell_aggregates"
COL_OUTPUT_PANEL = "training_panel_cell_month"

# ── Local output path ──────────────────────────────────────────────────────
CSV_OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "panel_cell_month.csv"

# ── Column config ──────────────────────────────────────────────────────────
DATE_COLUMN = "Date of Accident"
TIME_COLUMN = "Time of Accident"
LAT_COLUMN = "latitude"
LNG_COLUMN = "longitude"
SEVERITY_COLUMN = "Accident Severity"
CELL_COLUMN = "h3_res8"
POLICE_DIVISION_COLUMN = "Police Division"
VEHICLE_TYPES_COLUMN = "Vehicle Types"
LIGHT_COND_COLUMN = "light_condition"
PEAK_HOUR_COLUMN = "Peak Hour Indicator (Derived)"
HOLIDAY_INDICATOR_COLUMN = "Public Holiday Indicator (Derived)"
WEATHER_LABEL_COLUMN = "weather_condition_label"
VISIBILITY_LEVEL_COLUMN = "visibility_level"

# Severity ladder — used for tie-breaking dominant severity
SEVERITY_LADDER = ["Fatal", "Serious", "Minor", "Damage-only"]

# ── Panel config ───────────────────────────────────────────────────────────
STUDY_START = pd.Timestamp("2022-01-01")
STUDY_END = pd.Timestamp("2023-12-31")
ROLLING_WINDOWS_DAYS = [30, 90, 180]
NEIGHBOR_RING_K = 1

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(logging.INFO)
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ══════════════════════════════════════════════════════════════════════════
# MongoDB helpers
# ══════════════════════════════════════════════════════════════════════════

def get_db():
    if not MONGO_URI:
        raise ValueError("MONGO_URI is not set. Check your config/.env file.")
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB], client


def load_accident_features(db) -> pd.DataFrame:
    records = list(db[COL_ACCIDENTS_ST].find({}, {"_id": 0}))
    if not records:
        raise ValueError(
            f"No records found in collection: {COL_ACCIDENTS_ST}. "
            "Run spatiotemporal.py first."
        )
    dataframe = pd.DataFrame(records)

    required = [DATE_COLUMN, CELL_COLUMN, LAT_COLUMN, LNG_COLUMN, SEVERITY_COLUMN]
    missing = [c for c in required if c not in dataframe.columns]
    if missing:
        raise ValueError(f"Accident feature store missing required columns: {missing}")

    dataframe[DATE_COLUMN] = pd.to_datetime(dataframe[DATE_COLUMN], errors="coerce")
    dataframe = dataframe.dropna(subset=[DATE_COLUMN, CELL_COLUMN])
    LOGGER.info("Loaded %d accident records from [%s]", len(dataframe), COL_ACCIDENTS_ST)
    return dataframe


def load_cell_aggregates(db, resolution: int = 8) -> pd.DataFrame:
    records = list(db[COL_CELL_AGGREGATES].find(
        {"h3_resolution": resolution}, {"_id": 0}
    ))
    if not records:
        raise ValueError(
            f"No records found in collection: {COL_CELL_AGGREGATES} at res={resolution}. "
            "Run spatiotemporal.py first."
        )
    dataframe = pd.DataFrame(records)
    LOGGER.info(
        "Loaded %d cell aggregates from [%s] at res=%d",
        len(dataframe), COL_CELL_AGGREGATES, resolution,
    )
    return dataframe


def save_panel(db, dataframe: pd.DataFrame) -> None:
    records = dataframe.where(pd.notna(dataframe), None).to_dict(orient="records")

    operations = [
        UpdateOne(
            {"h3_cell": r["h3_cell"], "year_month": r["year_month"]},
            {"$set": r},
            upsert=True,
        )
        for r in records
    ]
    if not operations:
        LOGGER.warning("Nothing to write to [%s].", COL_OUTPUT_PANEL)
        return

    result = db[COL_OUTPUT_PANEL].bulk_write(operations)
    LOGGER.info(
        "[%s] Upserted: %d | Modified: %d | Total docs: %d",
        COL_OUTPUT_PANEL,
        result.upserted_count,
        result.modified_count,
        db[COL_OUTPUT_PANEL].count_documents({}),
    )


# ══════════════════════════════════════════════════════════════════════════
# Label derivation — dominant severity per cell (static)
# ══════════════════════════════════════════════════════════════════════════

def compute_dominant_severity(accidents: pd.DataFrame) -> pd.DataFrame:
    """Majority severity class per cell, tie-broken by SEVERITY_LADDER (worst wins)."""

    severity_counts = (
        accidents.groupby([CELL_COLUMN, SEVERITY_COLUMN])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure all four columns exist so tie-breaking works even if a class is empty
    for cls in SEVERITY_LADDER:
        if cls not in severity_counts.columns:
            severity_counts[cls] = 0

    # For each cell, find the max count, then pick the worst class among ties
    def pick_dominant(row: pd.Series) -> str:
        max_count = row[SEVERITY_LADDER].max()
        tied = [cls for cls in SEVERITY_LADDER if row[cls] == max_count]
        # SEVERITY_LADDER is ordered worst → least severe, so first tied wins
        return tied[0]

    severity_counts["dominant_severity"] = severity_counts.apply(pick_dominant, axis=1)
    result = severity_counts[["dominant_severity"]].reset_index()
    result = result.rename(columns={CELL_COLUMN: "h3_cell"})
    LOGGER.info("Computed dominant severity for %d cells", len(result))
    return result


# ══════════════════════════════════════════════════════════════════════════
# Static per-cell context features
# ══════════════════════════════════════════════════════════════════════════

def compute_static_cell_context(accidents: pd.DataFrame) -> pd.DataFrame:
    """Per-cell static context: police division (mode), first-seen coords."""

    def mode_or_first(series: pd.Series) -> object:
        m = series.mode()
        return m.iloc[0] if not m.empty else series.iloc[0]

    context = (
        accidents.groupby(CELL_COLUMN)
        .agg(
            police_division=(POLICE_DIVISION_COLUMN, mode_or_first)
            if POLICE_DIVISION_COLUMN in accidents.columns
            else (CELL_COLUMN, "first"),
            cell_lat=(LAT_COLUMN, "mean"),
            cell_lng=(LNG_COLUMN, "mean"),
        )
        .reset_index()
        .rename(columns={CELL_COLUMN: "h3_cell"})
    )
    return context


# ══════════════════════════════════════════════════════════════════════════
# Panel skeleton — every (cell, month) in the study window
# ══════════════════════════════════════════════════════════════════════════

def build_panel_skeleton(cells: List[str]) -> pd.DataFrame:
    """Cross-product of cells × months in the study window."""

    months = pd.date_range(STUDY_START, STUDY_END, freq="MS")
    skeleton = pd.MultiIndex.from_product(
        [cells, months], names=["h3_cell", "month_start"]
    ).to_frame(index=False)
    skeleton["year_month"] = skeleton["month_start"].dt.strftime("%Y-%m")
    skeleton["year"] = skeleton["month_start"].dt.year
    skeleton["month"] = skeleton["month_start"].dt.month
    LOGGER.info(
        "Built panel skeleton: %d cells × %d months = %d rows",
        len(cells), len(months), len(skeleton),
    )
    return skeleton


# ══════════════════════════════════════════════════════════════════════════
# Feature family (a) — causal rolling exposure at month start
# ══════════════════════════════════════════════════════════════════════════

def _neighbor_map(cells: List[str], k: int = NEIGHBOR_RING_K) -> Dict[str, List[str]]:
    return {cell: [c for c in h3.grid_disk(cell, k) if c != cell] for cell in cells}


def compute_rolling_exposure(
    panel: pd.DataFrame,
    accidents: pd.DataFrame,
    windows_days: List[int] = ROLLING_WINDOWS_DAYS,
) -> pd.DataFrame:
    """For each (cell, month_start), count accidents in same cell and neighbor
    ring in the trailing W days STRICTLY BEFORE month_start."""

    unique_cells = panel["h3_cell"].unique().tolist()
    neighbors = _neighbor_map(unique_cells)

    # Daily accident counts per cell — sparse but easy to slice
    daily = (
        accidents.assign(_d=accidents[DATE_COLUMN].dt.normalize())
        .groupby([CELL_COLUMN, "_d"])
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={CELL_COLUMN: "h3_cell", "_d": "date"})
    )
    daily = daily.set_index(["h3_cell", "date"])["count"]

    # Initialize columns
    for w in windows_days:
        panel[f"same_cell_count_{w}d"] = 0
        panel[f"neighbor_count_{w}d"] = 0

    # Build per-cell time series once, slice per month
    cell_series_cache: Dict[str, pd.Series] = {}
    for cell in unique_cells:
        try:
            s = daily.loc[cell].sort_index()
            # Reindex to daily for fast slicing
            full_idx = pd.date_range(
                STUDY_START - pd.Timedelta(days=max(windows_days)),
                STUDY_END, freq="D",
            )
            s = s.reindex(full_idx, fill_value=0)
            cell_series_cache[cell] = s
        except KeyError:
            # Cell exists in panel but has no accidents at all (shouldn't happen
            # given how we derived `cells`, but guard anyway)
            full_idx = pd.date_range(
                STUDY_START - pd.Timedelta(days=max(windows_days)),
                STUDY_END, freq="D",
            )
            cell_series_cache[cell] = pd.Series(0, index=full_idx)

    for cell in unique_cells:
        cell_series = cell_series_cache[cell]
        neighbor_cells = neighbors.get(cell, [])

        # Sum of neighbor daily counts
        if neighbor_cells:
            neighbor_series_parts = [
                cell_series_cache.get(nc) for nc in neighbor_cells
                if nc in cell_series_cache
            ]
            if neighbor_series_parts:
                neighbor_series = sum(neighbor_series_parts)
            else:
                neighbor_series = pd.Series(0, index=cell_series.index)
        else:
            neighbor_series = pd.Series(0, index=cell_series.index)

        # Mask this cell's rows in the panel once
        cell_mask = panel["h3_cell"] == cell
        month_starts = panel.loc[cell_mask, "month_start"]

        for w in windows_days:
            same_col = f"same_cell_count_{w}d"
            neigh_col = f"neighbor_count_{w}d"
            same_vals = []
            neigh_vals = []
            for ms in month_starts:
                window_start = ms - pd.Timedelta(days=w)
                # Strictly before month_start (< ms), so month itself doesn't leak
                window_slice_same = cell_series.loc[
                    (cell_series.index >= window_start) & (cell_series.index < ms)
                ]
                window_slice_neigh = neighbor_series.loc[
                    (neighbor_series.index >= window_start) & (neighbor_series.index < ms)
                ]
                same_vals.append(int(window_slice_same.sum()))
                neigh_vals.append(int(window_slice_neigh.sum()))
            panel.loc[cell_mask, same_col] = same_vals
            panel.loc[cell_mask, neigh_col] = neigh_vals

    LOGGER.info("Computed rolling exposure features for %d cells", len(unique_cells))
    return panel


# ══════════════════════════════════════════════════════════════════════════
# Feature family (b) — cyclical calendar encoding
# ══════════════════════════════════════════════════════════════════════════

def add_cyclical_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    panel["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12)
    return panel


# ══════════════════════════════════════════════════════════════════════════
# Feature family (c) — monthly context aggregates
# ══════════════════════════════════════════════════════════════════════════

def compute_monthly_context(
    panel: pd.DataFrame, accidents: pd.DataFrame
) -> pd.DataFrame:
    """Monthly weather/calendar aggregates per (cell, month).

    NOTE: these are computed from the accidents in that month, so they
    reflect "conditions when accidents happened this month". For months with
    zero accidents, these come out as NaN — filled with dataset-wide means
    at the end so RF/XGB have a value to work with.
    """

    acc = accidents.copy()
    acc["_ym"] = acc[DATE_COLUMN].dt.strftime("%Y-%m")

    numeric_cols = [c for c in ["precip", "humidity", "visibility", "cloudcover"] if c in acc.columns]
    agg_dict = {col: "mean" for col in numeric_cols}

    # Booleans for share features
    if PEAK_HOUR_COLUMN in acc.columns:
        acc["_is_peak"] = (acc[PEAK_HOUR_COLUMN].astype(str) == "Yes").astype(int)
    if LIGHT_COND_COLUMN in acc.columns:
        acc["_is_night"] = (acc[LIGHT_COND_COLUMN].astype(str) == "Night").astype(int)
    if VEHICLE_TYPES_COLUMN in acc.columns:
        acc["_is_motorcycle"] = acc[VEHICLE_TYPES_COLUMN].astype(str).str.lower().str.contains("motorcycle").astype(int)
    if HOLIDAY_INDICATOR_COLUMN in acc.columns:
        acc["_is_holiday"] = (acc[HOLIDAY_INDICATOR_COLUMN].astype(str) == "Yes").astype(int)
    if WEATHER_LABEL_COLUMN in acc.columns:
        acc["_is_rainy"] = acc[WEATHER_LABEL_COLUMN].astype(str).str.contains("Rain").astype(int)
    if VISIBILITY_LEVEL_COLUMN in acc.columns:
        acc["_is_poor_vis"] = (acc[VISIBILITY_LEVEL_COLUMN].astype(str) == "Poor").astype(int)

    share_cols = [c for c in ["_is_peak", "_is_night", "_is_motorcycle", "_is_holiday", "_is_rainy", "_is_poor_vis"] if c in acc.columns]
    for c in share_cols:
        agg_dict[c] = "mean"

    # Accident count that month (used as feature + for divergence checks)
    grouped = acc.groupby([CELL_COLUMN, "_ym"]).agg(agg_dict).reset_index()
    counts = acc.groupby([CELL_COLUMN, "_ym"]).size().rename("accidents_this_month").reset_index()
    monthly = grouped.merge(counts, on=[CELL_COLUMN, "_ym"], how="left")

    # Rename share columns for clarity
    rename_map = {
        "_is_peak": "peak_hour_share",
        "_is_night": "night_share",
        "_is_motorcycle": "motorcycle_share",
        "_is_holiday": "holiday_share",
        "_is_rainy": "rainy_share",
        "_is_poor_vis": "poor_visibility_share",
        "precip": "avg_precip_month",
        "humidity": "avg_humidity_month",
        "visibility": "avg_visibility_month",
        "cloudcover": "avg_cloudcover_month",
        CELL_COLUMN: "h3_cell",
        "_ym": "year_month",
    }
    monthly = monthly.rename(columns=rename_map)

    # Merge onto panel skeleton
    merged = panel.merge(monthly, on=["h3_cell", "year_month"], how="left")

    # Fill NaN for cells with zero accidents in a month:
    #   - accidents_this_month → 0 (no accidents, honest zero)
    #   - shares → 0 (no accidents means no share information to report;
    #                 zero is a defensible default that means "no signal")
    #   - weather means → dataset-wide monthly average (weather still happened
    #                     that month even if no accidents)
    merged["accidents_this_month"] = merged["accidents_this_month"].fillna(0).astype(int)
    for col in ["peak_hour_share", "night_share", "motorcycle_share",
                "holiday_share", "rainy_share", "poor_visibility_share"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
    for col in ["avg_precip_month", "avg_humidity_month",
                "avg_visibility_month", "avg_cloudcover_month"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(merged[col].mean())

    return merged


# ══════════════════════════════════════════════════════════════════════════
# Attach labels + static context
# ══════════════════════════════════════════════════════════════════════════

def attach_labels_and_context(
    panel: pd.DataFrame,
    cell_aggregates: pd.DataFrame,
    dominant_severity: pd.DataFrame,
    static_context: pd.DataFrame,
) -> pd.DataFrame:
    keep_from_aggregates = ["h3_cell", "is_hotspot", "accident_count",
                            "fatal_count", "serious_count", "minor_count",
                            "damage_only_count", "fatal_rate",
                            "hotspot_p_adjusted"]
    keep_from_aggregates = [c for c in keep_from_aggregates if c in cell_aggregates.columns]

    panel = panel.merge(
        cell_aggregates[keep_from_aggregates].rename(
            columns={"accident_count": "historical_accident_count",
                     "fatal_count": "historical_fatal_count",
                     "serious_count": "historical_serious_count",
                     "minor_count": "historical_minor_count",
                     "damage_only_count": "historical_damage_only_count",
                     "fatal_rate": "historical_fatal_rate"}
        ),
        on="h3_cell", how="left",
    )
    panel = panel.merge(dominant_severity, on="h3_cell", how="left")
    panel = panel.merge(static_context, on="h3_cell", how="left")

    # Cells with no aggregate row (shouldn't happen, but guard):
    panel["is_hotspot"] = panel["is_hotspot"].fillna(False)
    return panel


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def build_panel() -> pd.DataFrame:
    configure_logging()
    LOGGER.info("=== Training Panel Build Start ===")

    db, client = get_db()
    try:
        # Step 1 — Load inputs
        LOGGER.info("-- Step 1: Loading accident features + cell aggregates --")
        accidents = load_accident_features(db)
        cell_aggregates = load_cell_aggregates(db, resolution=8)

        # Step 2 — Derive per-cell labels & static context
        LOGGER.info("-- Step 2: Deriving dominant severity + static context --")
        dominant_severity = compute_dominant_severity(accidents)
        static_context = compute_static_cell_context(accidents)

        # Step 3 — Build panel skeleton
        LOGGER.info("-- Step 3: Building panel skeleton --")
        cells = accidents[CELL_COLUMN].dropna().unique().tolist()
        panel = build_panel_skeleton(cells)

        # Step 4 — Attach labels + static context
        LOGGER.info("-- Step 4: Attaching labels + static context --")
        panel = attach_labels_and_context(
            panel, cell_aggregates, dominant_severity, static_context
        )

        # Step 5 — Feature family (a): causal rolling exposure
        LOGGER.info("-- Step 5: Computing causal rolling exposure features --")
        panel = compute_rolling_exposure(panel, accidents, ROLLING_WINDOWS_DAYS)

        # Step 6 — Feature family (b): cyclical calendar
        LOGGER.info("-- Step 6: Adding cyclical calendar features --")
        panel = add_cyclical_calendar(panel)

        # Step 7 — Feature family (c): monthly context aggregates
        LOGGER.info("-- Step 7: Computing monthly context features --")
        panel = compute_monthly_context(panel, accidents)

        # Step 8 — Column ordering + provenance stamp
        LOGGER.info("-- Step 8: Finalizing panel --")
        panel["built_at"] = datetime.utcnow().isoformat()
        panel = panel.drop(columns=["month_start"])  # keep year_month as canonical

        # Step 9 — Persist
        LOGGER.info("-- Step 9: Persisting panel --")
        CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(CSV_OUTPUT_PATH, index=False)
        LOGGER.info("CSV written: %s (%d rows, %d cols)",
                    CSV_OUTPUT_PATH, len(panel), len(panel.columns))

        save_panel(db, panel)

        # Quick sanity summary
        LOGGER.info("--- Panel summary ---")
        LOGGER.info("Rows: %d", len(panel))
        LOGGER.info("Unique cells: %d", panel["h3_cell"].nunique())
        LOGGER.info("Unique months: %d", panel["year_month"].nunique())
        LOGGER.info("Hotspot cells (rows with is_hotspot=True): %d",
                    int(panel["is_hotspot"].sum()))
        LOGGER.info("Dominant severity distribution:")
        LOGGER.info("\n%s", panel.drop_duplicates("h3_cell")["dominant_severity"]
                    .value_counts().to_string())

        LOGGER.info("=== Training Panel Build Complete ===")
        return panel
    finally:
        client.close()


if __name__ == "__main__":
    build_panel()