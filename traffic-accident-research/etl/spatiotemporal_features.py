"""Spatiotemporal feature engineering for the traffic accident research pipeline.

Reads the calendar/holiday-enriched accident dataset from MongoDB
(accidents_feature_engineered collection), assigns each accident to an H3
hexagonal grid cell, and derives two families of spatial features:

1. Static spatial aggregates (per H3 cell, computed over the FULL dataset):
   - total accident count, severity breakdown, fatal rate
   - used to define the hotspot / non-hotspot label (Objective 1-2)

2. Causal spatiotemporal features (per accident record, using ONLY prior
   accidents relative to that record's date):
   - rolling accident count in the same cell over trailing windows
   - rolling accident count in the k=1 neighbor ring (spatial lag)
   - these are leakage-safe and go into the RF/XGBoost feature matrix,
     since static full-dataset counts would leak future information into
     a record's own features.

H3 resolution 8 (~0.7 km^2 cells) is used as the primary grid; resolution 9
is also computed for sensitivity checks given the sparse real-record volume
(~50 real + ~950 synthetic accidents across 220 unique Colombo locations).

MongoDB Collections:
    accidents_feature_engineered       — calendar/holiday enriched accidents (input)
    ref_h3_cell_aggregates             — static per-cell hotspot aggregates (output)
    accidents_spatiotemporal_features  — accidents enriched with H3 + rolling
                                          spatiotemporal features (output)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import h3
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from scipy.stats import poisson
from statsmodels.stats.multitest import multipletests

# ── Environment ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / "config" / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "accident_hotspot")

# ── MongoDB Collection Names ───────────────────────────────────────────────
COL_INPUT_FEATURES = "accidents_feature_engineered"
COL_CELL_AGGREGATES = "ref_h3_cell_aggregates"
COL_OUTPUT_SPATIOTEMPORAL = "accidents_spatiotemporal_features"

# ── Column Config ───────────────────────────────────────────────────────────
DATE_COLUMN = "Date of Accident"
LAT_COLUMN = "latitude"
LNG_COLUMN = "longitude"
SEVERITY_COLUMN = "Accident Severity"  # adjust to match your raw schema if named differently
RECORD_ID_COLUMN = "_lake_id"

# ── H3 / Spatiotemporal Config ──────────────────────────────────────────────
H3_RESOLUTIONS = [8, 9]
PRIMARY_RESOLUTION = 8
NEIGHBOR_RING_K = 1
ROLLING_WINDOWS_DAYS = [30, 90, 180]
HOTSPOT_FDR_ALPHA = 0.05  # Benjamini-Hochberg false discovery rate threshold for hotspot significance

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure a compact logging format for ETL execution."""

    if logging.getLogger().handlers:
        logging.getLogger().setLevel(logging.INFO)
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MongoDB helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_db():
    """Return a MongoDB database handle and client."""

    if not MONGO_URI:
        raise ValueError("MONGO_URI is not set. Check your config/.env file.")
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB], client


def load_source_data(db) -> pd.DataFrame:
    """Load the calendar/holiday-enriched accident dataset from MongoDB."""

    records = list(db[COL_INPUT_FEATURES].find({}, {"_id": 0}))
    if not records:
        raise ValueError(
            f"No records found in collection: {COL_INPUT_FEATURES}. "
            "Run feature_engineering.py first."
        )

    dataframe = pd.DataFrame(records)
    required_columns = [DATE_COLUMN, LAT_COLUMN, LNG_COLUMN]
    missing_columns = [c for c in required_columns if c not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Source dataset is missing required columns: {missing_columns}")

    LOGGER.info(
        "Loaded %d accident records from MongoDB [%s]", len(dataframe), COL_INPUT_FEATURES
    )
    return dataframe


def save_cell_aggregates(db, dataframe: pd.DataFrame, resolution: int) -> None:
    """Replace the static per-cell aggregate collection for a given resolution."""

    records = dataframe.where(pd.notna(dataframe), None).to_dict(orient="records")
    for record in records:
        record["h3_resolution"] = resolution

    operations = [
        UpdateOne(
            {"h3_cell": r["h3_cell"], "h3_resolution": resolution},
            {"$set": r},
            upsert=True,
        )
        for r in records
    ]
    if not operations:
        LOGGER.warning("No cell aggregates to save for resolution %d.", resolution)
        return

    result = db[COL_CELL_AGGREGATES].bulk_write(operations)
    LOGGER.info(
        "[%s res=%d] Upserted: %d | Modified: %d",
        COL_CELL_AGGREGATES, resolution, result.upserted_count, result.modified_count,
    )


def save_spatiotemporal_features(db, dataframe: pd.DataFrame) -> None:
    """Upsert the accident-level spatiotemporal feature dataset into MongoDB."""

    records = dataframe.where(pd.notna(dataframe), None).to_dict(orient="records")

    operations = [
        UpdateOne(
            {RECORD_ID_COLUMN: r[RECORD_ID_COLUMN]},
            {"$set": r},
            upsert=True,
        )
        for r in records
        if r.get(RECORD_ID_COLUMN) is not None
    ]

    if not operations:
        LOGGER.warning(
            "No records had a '%s' — nothing was written to [%s].",
            RECORD_ID_COLUMN, COL_OUTPUT_SPATIOTEMPORAL,
        )
        return

    result = db[COL_OUTPUT_SPATIOTEMPORAL].bulk_write(operations)
    LOGGER.info(
        "[%s] Upserted: %d | Modified: %d | Total: %d",
        COL_OUTPUT_SPATIOTEMPORAL,
        result.upserted_count,
        result.modified_count,
        db[COL_OUTPUT_SPATIOTEMPORAL].count_documents({}),
    )


# ══════════════════════════════════════════════════════════════════════════════
# H3 cell assignment
# ══════════════════════════════════════════════════════════════════════════════

def assign_h3_cells(dataframe: pd.DataFrame, resolutions: List[int] = H3_RESOLUTIONS) -> pd.DataFrame:
    """Assign an H3 cell ID at each configured resolution for every accident."""

    transformed = dataframe.copy()
    valid_coords = transformed[LAT_COLUMN].notna() & transformed[LNG_COLUMN].notna()
    missing_count = (~valid_coords).sum()
    if missing_count:
        LOGGER.warning(
            "%d records missing lat/lng — H3 cells will be null for these rows.",
            missing_count,
        )

    for resolution in resolutions:
        column_name = f"h3_res{resolution}"
        transformed[column_name] = None
        transformed.loc[valid_coords, column_name] = transformed.loc[valid_coords].apply(
            lambda row: h3.latlng_to_cell(row[LAT_COLUMN], row[LNG_COLUMN], resolution),
            axis=1,
        )

    LOGGER.info(
        "Assigned H3 cells at resolutions %s to %d/%d records.",
        resolutions, valid_coords.sum(), len(transformed),
    )
    return transformed


# ══════════════════════════════════════════════════════════════════════════════
# Static per-cell aggregates (defines the hotspot label — full-dataset view)
# ══════════════════════════════════════════════════════════════════════════════

def compute_static_cell_aggregates(dataframe: pd.DataFrame, resolution: int) -> pd.DataFrame:
    """Aggregate accident counts and severity breakdown per H3 cell.

    This is a FULL-dataset view used to define which cells qualify as
    hotspots (Objective 1-2 ground truth). It must NOT be joined back onto
    individual accident records as a predictive feature, since that would
    leak each record's own outcome into its features — use
    compute_causal_rolling_features() for the modeling feature matrix instead.

    Severity classes match the generator's taxonomy exactly:
    Fatal / Serious / Minor / Damage-only.
    """

    cell_column = f"h3_res{resolution}"
    valid = dataframe[dataframe[cell_column].notna()].copy()

    has_severity = SEVERITY_COLUMN in valid.columns
    if not has_severity:
        LOGGER.warning(
            "Column '%s' not found — severity breakdown will be skipped.",
            SEVERITY_COLUMN,
        )

    grouped = valid.groupby(cell_column)
    aggregates = grouped.size().reset_index(name="accident_count")
    aggregates = aggregates.rename(columns={cell_column: "h3_cell"})

    if has_severity:
        severity_lower = valid[SEVERITY_COLUMN].astype(str).str.lower()
        valid = valid.assign(_severity_lower=severity_lower)
        severity_counts = (
            valid.groupby(cell_column)["_severity_lower"]
            .agg(
                fatal_count=lambda s: (s == "fatal").sum(),
                serious_count=lambda s: (s == "serious").sum(),
                minor_count=lambda s: (s == "minor").sum(),
                damage_only_count=lambda s: (s == "damage-only").sum(),
            )
            .reset_index()
            .rename(columns={cell_column: "h3_cell"})
        )
        aggregates = aggregates.merge(severity_counts, on="h3_cell", how="left")
        aggregates["fatal_rate"] = (
            aggregates["fatal_count"] / aggregates["accident_count"]
        ).round(4)

    # Hotspot label: one-sided Poisson test per cell against the CMC-wide mean
    # accident rate, i.e. "is this cell's count significantly higher than
    # expected under complete spatial randomness at the same overall rate,"
    # with Benjamini-Hochberg FDR correction since we're running one test per
    # cell simultaneously (avoids the multiple-testing inflation that a raw
    # per-cell p-value threshold would produce across dozens/hundreds of cells).
    mean_rate = aggregates["accident_count"].mean()
    aggregates["hotspot_p_value"] = aggregates["accident_count"].apply(
        lambda observed_count: poisson.sf(observed_count - 1, mean_rate)
    )
    _, adjusted_p_values, _, _ = multipletests(
        aggregates["hotspot_p_value"], alpha=HOTSPOT_FDR_ALPHA, method="fdr_bh"
    )
    aggregates["hotspot_p_adjusted"] = adjusted_p_values
    aggregates["is_hotspot"] = adjusted_p_values < HOTSPOT_FDR_ALPHA

    aggregates = aggregates.sort_values("accident_count", ascending=False).reset_index(drop=True)

    LOGGER.info(
        "res=%d: %d unique cells | mean rate=%.2f accidents/cell | %d cells flagged hotspot (FDR alpha=%.2f)",
        resolution, len(aggregates), mean_rate, aggregates["is_hotspot"].sum(), HOTSPOT_FDR_ALPHA,
    )
    return aggregates


# ══════════════════════════════════════════════════════════════════════════════
# Causal (leakage-safe) rolling spatiotemporal features
# ══════════════════════════════════════════════════════════════════════════════

def _neighbor_map(cells: List[str], k: int = NEIGHBOR_RING_K) -> dict:
    """Precompute {cell: [neighbor cells excluding self]} for a list of unique cells."""

    return {cell: [c for c in h3.grid_disk(cell, k) if c != cell] for cell in cells}


def compute_causal_rolling_features(
    dataframe: pd.DataFrame,
    resolution: int = PRIMARY_RESOLUTION,
    windows_days: List[int] = ROLLING_WINDOWS_DAYS,
    neighbor_k: int = NEIGHBOR_RING_K,
) -> pd.DataFrame:
    """Add trailing-window accident counts computed using ONLY prior records.

    For each accident, and for each window in `windows_days`, computes:
    - same_cell_count_{w}d: accidents in the same H3 cell in the preceding
      `w` days (strictly before this accident's date)
    - neighbor_count_{w}d: accidents in the k-ring neighboring cells in the
      preceding `w` days

    closed='left' on the time-based rolling window excludes the current
    record's own timestamp, so same-day accidents in the same cell do not
    count towards their own feature value.
    """

    cell_column = f"h3_res{resolution}"
    transformed = dataframe.copy()
    transformed["_date"] = pd.to_datetime(transformed[DATE_COLUMN], errors="coerce")
    transformed = transformed.dropna(subset=["_date", cell_column])
    transformed = transformed.sort_values(["_date"]).reset_index(drop=False)  # keep original index

    unique_cells = transformed[cell_column].dropna().unique().tolist()
    neighbors = _neighbor_map(unique_cells, k=neighbor_k)

    # Per-cell daily accident counts, used to build neighbor rolling sums quickly
    daily_counts = (
        transformed.groupby([cell_column, transformed["_date"].dt.normalize()])
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={cell_column: "h3_cell", "_date": "date"})
    )

    for window in windows_days:
        transformed[f"same_cell_count_{window}d"] = 0
        transformed[f"neighbor_count_{window}d"] = 0

    for cell, group in transformed.groupby(cell_column):
        group_sorted = group.set_index("_date").sort_index()
        for window in windows_days:
            counts = (
                group_sorted[cell_column]
                .rolling(f"{window}D", closed="left", min_periods=0)
                .count()
            )
            transformed.loc[group.index, f"same_cell_count_{window}d"] = counts.fillna(0).values

        neighbor_cells = neighbors.get(cell, [])
        if not neighbor_cells:
            continue

        neighbor_daily = daily_counts[daily_counts["h3_cell"].isin(neighbor_cells)]
        if neighbor_daily.empty:
            continue

        neighbor_series = (
            neighbor_daily.groupby("date")["count"].sum().sort_index()
        )

        for window in windows_days:
            values = []
            for accident_date in group_sorted.index:
                window_start = accident_date - pd.Timedelta(days=window)
                mask = (neighbor_series.index >= window_start) & (neighbor_series.index < accident_date.normalize())
                values.append(neighbor_series[mask].sum())
            transformed.loc[group.index, f"neighbor_count_{window}d"] = values

    transformed = transformed.set_index("index").sort_index()
    transformed = transformed.drop(columns=["_date"])
    return transformed


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the spatiotemporal feature engineering pipeline."""

    configure_logging()
    LOGGER.info("=== Spatiotemporal Feature Engineering Pipeline Start ===")

    db, client = get_db()

    try:
        # Step 1 — Load calendar/holiday-enriched accidents from MongoDB
        LOGGER.info("-- Step 1: Loading accidents_feature_engineered from MongoDB --")
        source_data = load_source_data(db)

        # Step 2 — Assign H3 cells at each configured resolution
        LOGGER.info("-- Step 2: Assigning H3 grid cells --")
        with_cells = assign_h3_cells(source_data, resolutions=H3_RESOLUTIONS)

        # Step 3 — Static per-cell aggregates (hotspot ground-truth label)
        LOGGER.info("-- Step 3: Computing static per-cell aggregates --")
        for resolution in H3_RESOLUTIONS:
            cell_aggregates = compute_static_cell_aggregates(with_cells, resolution)
            save_cell_aggregates(db, cell_aggregates, resolution)

        # Step 4 — Causal rolling spatiotemporal features (leakage-safe, primary res)
        LOGGER.info(
            "-- Step 4: Computing causal rolling features at res=%d --", PRIMARY_RESOLUTION
        )
        enriched = compute_causal_rolling_features(
            with_cells, resolution=PRIMARY_RESOLUTION, windows_days=ROLLING_WINDOWS_DAYS
        )

        # Step 5 — Save accident-level spatiotemporal features
        LOGGER.info("-- Step 5: Saving accidents_spatiotemporal_features to MongoDB --")
        save_spatiotemporal_features(db, enriched)

        LOGGER.info("=== Spatiotemporal Feature Engineering Pipeline Complete ===")

    finally:
        client.close()


if __name__ == "__main__":
    main()