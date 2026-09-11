"""
Build the (H3 cell, month) prediction panel for hotspot & severity classification.

PREDICTION-BASED VERSION
========================

Panel grain:
    one row per (h3_cell, target_month)

For each target month T:

FEATURES
--------
Only information available BEFORE target month T is used.

Example:
    September 2023 row
        -> accident history up to 2023-08-31
        -> previous-month weather (August 2023)
        -> calendar information known in advance

TARGETS
-------
Target-month accidents are used ONLY to construct the labels.

    September 2023:
        -> count September accidents per H3 cell
        -> Poisson test
        -> Benjamini-Hochberg FDR
        -> is_hotspot
        -> target-month severity counts
        -> dominant_severity
        -> is_severe

This replaces the previous static-label design where one hotspot/severity
label was calculated from the 2022-2023 dataset and repeated for every month.

MongoDB collections
-------------------

Input:
    accidents_spatiotemporal_features
        Accident-level H3 + spatiotemporal feature store.

    raw_weather
        Daily Colombo weather from Visual Crossing.

Output:
    training_panel_cell_month
        One row per (h3_cell, year_month).

Local output:
    data/processed/panel_cell_month.csv
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import h3
import numpy as np
import pandas as pd
from dateutil.easter import easter
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from scipy.stats import poisson
from statsmodels.stats.multitest import multipletests


# ============================================================================
# Environment
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

ENV_PATH = ROOT_DIR / "config" / ".env"

load_dotenv(
    dotenv_path=ENV_PATH,
    override=False,
)

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv(
    "MONGO_DB",
    "accident_hotspot",
)


# ============================================================================
# MongoDB collections
# ============================================================================

COL_ACCIDENTS_ST = "accidents_spatiotemporal_features"
COL_RAW_WEATHER = "raw_weather"
COL_OUTPUT_PANEL = "training_panel_cell_month"


# ============================================================================
# Local output
# ============================================================================

CSV_OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "panel_cell_month.csv"
)


# ============================================================================
# Column configuration
# ============================================================================

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


# Worst -> least severe
SEVERITY_LADDER = [
    "Fatal",
    "Serious",
    "Minor",
    "Damage-only",
]


# ============================================================================
# Panel configuration
# ============================================================================

STUDY_START = pd.Timestamp("2022-01-01")
STUDY_END = pd.Timestamp("2023-12-31")

ROLLING_WINDOWS_DAYS = [30, 90, 180]
BEHAVIOURAL_WINDOW_DAYS = 180

NEIGHBOR_RING_K = 1

HOTSPOT_FDR_ALPHA = 0.05

RAIN_PRECIP_THRESHOLD_MM = 0.1


LOGGER = logging.getLogger(__name__)


# ============================================================================
# Logging
# ============================================================================

def configure_logging() -> None:
    """Configure compact logging format."""

    if logging.getLogger().handlers:
        logging.getLogger().setLevel(logging.INFO)
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ============================================================================
# MongoDB helpers
# ============================================================================

def get_db():
    """Return MongoDB database handle and client."""

    if not MONGO_URI:
        raise ValueError(
            "MONGO_URI is not set. Check your config/.env file."
        )

    client = MongoClient(MONGO_URI)

    return client[MONGO_DB], client


def load_accident_features(db) -> pd.DataFrame:
    """Load accident-level H3/spatiotemporal features."""

    records = list(
        db[COL_ACCIDENTS_ST].find(
            {},
            {"_id": 0},
        )
    )

    if not records:
        raise ValueError(
            f"No records found in collection: {COL_ACCIDENTS_ST}. "
            "Run spatiotemporal.py first."
        )

    dataframe = pd.DataFrame(records)

    required = [
        DATE_COLUMN,
        CELL_COLUMN,
        LAT_COLUMN,
        LNG_COLUMN,
        SEVERITY_COLUMN,
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "Accident feature store missing required columns: "
            f"{missing}"
        )

    dataframe[DATE_COLUMN] = pd.to_datetime(
        dataframe[DATE_COLUMN],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=[
            DATE_COLUMN,
            CELL_COLUMN,
        ]
    ).copy()

    dataframe["_date"] = (
        dataframe[DATE_COLUMN]
        .dt.normalize()
    )

    LOGGER.info(
        "Loaded %d accident records from [%s]",
        len(dataframe),
        COL_ACCIDENTS_ST,
    )

    return dataframe


def load_raw_weather(db) -> pd.DataFrame:
    """Load daily Colombo weather from raw_weather."""

    records = list(
        db[COL_RAW_WEATHER].find(
            {},
            {"_id": 0},
        )
    )

    if not records:
        raise ValueError(
            f"No records found in collection: {COL_RAW_WEATHER}."
        )

    weather = pd.DataFrame(records)

    if "datetime" not in weather.columns:
        raise ValueError(
            f"'{COL_RAW_WEATHER}' does not contain 'datetime'."
        )

    weather["_date"] = (
        pd.to_datetime(
            weather["datetime"],
            errors="coerce",
        )
        .dt.normalize()
    )

    weather = (
        weather
        .dropna(subset=["_date"])
        .drop_duplicates(subset=["_date"])
        .copy()
    )

    for column in [
        "precip",
        "humidity",
        "visibility",
        "cloudcover",
    ]:
        if column in weather.columns:
            weather[column] = pd.to_numeric(
                weather[column],
                errors="coerce",
            )

    LOGGER.info(
        "Loaded %d daily weather records from [%s]",
        len(weather),
        COL_RAW_WEATHER,
    )

    return weather


# ============================================================================
# Mongo-safe value helper
# ============================================================================

def _mongo_safe_value(value):
    """
    Convert pandas / NumPy scalar values into MongoDB-safe Python values.

    NaN -> None
    NumPy scalar -> native Python scalar
    """

    if value is None:
        return None

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None

    if isinstance(value, np.generic):
        return value.item()

    return value


# ============================================================================
# Save panel
# ============================================================================

def save_panel(
    db,
    dataframe: pd.DataFrame,
) -> None:
    """
    Replace the prediction panel in MongoDB.

    The collection is rebuilt from scratch so stale fields from the old
    static-label panel cannot remain in documents.
    """

    collection = db[COL_OUTPUT_PANEL]

    deleted = collection.delete_many({})

    LOGGER.info(
        "Cleared %d old documents from [%s]",
        deleted.deleted_count,
        COL_OUTPUT_PANEL,
    )

    raw_records = dataframe.to_dict(
        orient="records"
    )

    records = []

    for raw_record in raw_records:
        clean_record = {
            key: _mongo_safe_value(value)
            for key, value in raw_record.items()
        }

        records.append(clean_record)

    operations = [
        UpdateOne(
            {
                "h3_cell": record["h3_cell"],
                "year_month": record["year_month"],
            },
            {
                "$set": record,
            },
            upsert=True,
        )
        for record in records
    ]

    if not operations:
        LOGGER.warning(
            "Nothing to write to [%s].",
            COL_OUTPUT_PANEL,
        )
        return

    result = collection.bulk_write(
        operations
    )

    collection.create_index(
        [
            ("h3_cell", 1),
            ("year_month", 1),
        ],
        unique=True,
    )

    LOGGER.info(
        "[%s] Upserted: %d | Modified: %d | Total docs: %d",
        COL_OUTPUT_PANEL,
        result.upserted_count,
        result.modified_count,
        collection.count_documents({}),
    )


# ============================================================================
# Panel skeleton
# ============================================================================

def build_panel_skeleton(
    cells: List[str],
) -> pd.DataFrame:
    """
    Build one row for every H3 cell and target month.
    """

    months = pd.date_range(
        STUDY_START,
        STUDY_END,
        freq="MS",
    )

    skeleton = (
        pd.MultiIndex.from_product(
            [
                cells,
                months,
            ],
            names=[
                "h3_cell",
                "month_start",
            ],
        )
        .to_frame(index=False)
    )

    skeleton["year_month"] = (
        skeleton["month_start"]
        .dt.strftime("%Y-%m")
    )

    skeleton["year"] = (
        skeleton["month_start"]
        .dt.year
    )

    skeleton["month"] = (
        skeleton["month_start"]
        .dt.month
    )

    skeleton["feature_cutoff_date"] = (
        skeleton["month_start"]
        - pd.Timedelta(days=1)
    ).dt.strftime("%Y-%m-%d")

    LOGGER.info(
        "Built panel skeleton: %d cells × %d months = %d rows",
        len(cells),
        len(months),
        len(skeleton),
    )

    return skeleton


# ============================================================================
# Static context
# ============================================================================

def compute_static_cell_context(
    accidents: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute static cell information.

    H3 centroid is obtained directly from H3 geometry.

    Police division is treated as static contextual metadata.
    """

    cells = (
        accidents[CELL_COLUMN]
        .dropna()
        .unique()
        .tolist()
    )

    rows = []

    for cell in cells:

        lat, lng = h3.cell_to_latlng(cell)

        cell_accidents = accidents[
            accidents[CELL_COLUMN] == cell
        ]

        if (
            POLICE_DIVISION_COLUMN in accidents.columns
            and not cell_accidents[
                POLICE_DIVISION_COLUMN
            ].dropna().empty
        ):

            mode = (
                cell_accidents[
                    POLICE_DIVISION_COLUMN
                ]
                .dropna()
                .mode()
            )

            police_division = (
                mode.iloc[0]
                if not mode.empty
                else cell_accidents[
                    POLICE_DIVISION_COLUMN
                ]
                .dropna()
                .iloc[0]
            )

        else:
            police_division = None

        rows.append(
            {
                "h3_cell": cell,
                "cell_lat": lat,
                "cell_lng": lng,
                "police_division": police_division,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# Neighbor map
# ============================================================================

def _neighbor_map(
    cells: List[str],
    k: int = NEIGHBOR_RING_K,
) -> Dict[str, List[str]]:
    """
    Return neighboring H3 cells excluding the cell itself.
    """

    return {
        cell: [
            neighbour
            for neighbour in h3.grid_disk(
                cell,
                k,
            )
            if neighbour != cell
        ]
        for cell in cells
    }


# ============================================================================
# Rolling historical accident exposure
# ============================================================================

def compute_rolling_exposure(
    panel: pd.DataFrame,
    accidents: pd.DataFrame,
    windows_days: List[int] = ROLLING_WINDOWS_DAYS,
) -> pd.DataFrame:
    """
    Compute historical accident exposure before each target month.

    For target month T:

        same_cell_count_30d
            accidents in same cell during previous 30 days

        same_cell_count_90d
            accidents in same cell during previous 90 days

        same_cell_count_180d
            accidents in same cell during previous 180 days

    Neighbor counts use the same windows.

    IMPORTANT:
        The target month itself is never included.
    """

    unique_cells = (
        panel["h3_cell"]
        .dropna()
        .unique()
        .tolist()
    )

    neighbors = _neighbor_map(
        unique_cells
    )

    daily = (
        accidents
        .groupby(
            [
                CELL_COLUMN,
                "_date",
            ]
        )
        .size()
        .rename("count")
        .reset_index()
        .rename(
            columns={
                CELL_COLUMN: "h3_cell",
                "_date": "date",
            }
        )
    )

    daily = (
        daily
        .set_index(
            [
                "h3_cell",
                "date",
            ]
        )["count"]
    )

    for window in windows_days:

        panel[
            f"same_cell_count_{window}d"
        ] = 0

        panel[
            f"neighbor_count_{window}d"
        ] = 0

    max_window = max(
        windows_days
    )

    full_idx = pd.date_range(
        STUDY_START
        - pd.Timedelta(
            days=max_window
        ),
        STUDY_END,
        freq="D",
    )

    cell_series_cache: Dict[
        str,
        pd.Series,
    ] = {}

    for cell in unique_cells:

        try:
            series = (
                daily
                .loc[cell]
                .sort_index()
            )

            series = series.reindex(
                full_idx,
                fill_value=0,
            )

        except KeyError:

            series = pd.Series(
                0,
                index=full_idx,
                dtype=int,
            )

        cell_series_cache[cell] = series

    for cell in unique_cells:

        cell_series = (
            cell_series_cache[cell]
        )

        neighbour_cells = neighbors.get(
            cell,
            [],
        )

        if neighbour_cells:

            parts = [
                cell_series_cache[nc]
                for nc in neighbour_cells
                if nc in cell_series_cache
            ]

            if parts:
                neighbour_series = sum(parts)

            else:
                neighbour_series = pd.Series(
                    0,
                    index=cell_series.index,
                )

        else:

            neighbour_series = pd.Series(
                0,
                index=cell_series.index,
            )

        cell_mask = (
            panel["h3_cell"] == cell
        )

        month_starts = panel.loc[
            cell_mask,
            "month_start",
        ]

        for window in windows_days:

            same_values = []
            neighbour_values = []

            for month_start in month_starts:

                window_end = month_start

                window_start = (
                    window_end
                    - pd.Timedelta(
                        days=window
                    )
                )

                same_count = (
                    cell_series.loc[
                        (
                            cell_series.index
                            >= window_start
                        )
                        & (
                            cell_series.index
                            < window_end
                        )
                    ].sum()
                )

                neighbour_count = (
                    neighbour_series.loc[
                        (
                            neighbour_series.index
                            >= window_start
                        )
                        & (
                            neighbour_series.index
                            < window_end
                        )
                    ].sum()
                )

                same_values.append(
                    int(same_count)
                )

                neighbour_values.append(
                    int(neighbour_count)
                )

            panel.loc[
                cell_mask,
                f"same_cell_count_{window}d",
            ] = same_values

            panel.loc[
                cell_mask,
                f"neighbor_count_{window}d",
            ] = neighbour_values

    LOGGER.info(
        "Computed moving historical exposure features for %d cells.",
        len(unique_cells),
    )

    return panel


# ============================================================================
# Historical behavioural features
# ============================================================================

def compute_historical_behavioural_features(
    panel: pd.DataFrame,
    accidents: pd.DataFrame,
    window_days: int = BEHAVIOURAL_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Compute behavioural shares using only accidents before each target month.

    The target month itself is excluded.
    """

    working = accidents.copy()

    # ------------------------------------------------------------------
    # Peak hour
    # ------------------------------------------------------------------

    if PEAK_HOUR_COLUMN in working.columns:

        working["_is_peak"] = (
            working[
                PEAK_HOUR_COLUMN
            ]
            .astype(str)
            .str.strip()
            .eq("Yes")
            .astype(int)
        )

    else:
        working["_is_peak"] = 0

    # ------------------------------------------------------------------
    # Night
    # ------------------------------------------------------------------

    if LIGHT_COND_COLUMN in working.columns:

        working["_is_night"] = (
            working[
                LIGHT_COND_COLUMN
            ]
            .astype(str)
            .str.strip()
            .eq("Night")
            .astype(int)
        )

    else:
        working["_is_night"] = 0

    # ------------------------------------------------------------------
    # Motorcycle
    # ------------------------------------------------------------------

    if VEHICLE_TYPES_COLUMN in working.columns:

        working["_is_motorcycle"] = (
            working[
                VEHICLE_TYPES_COLUMN
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                "motorcycle",
                na=False,
            )
            .astype(int)
        )

    else:
        working["_is_motorcycle"] = 0

    result_columns = [
        "historical_peak_hour_share_180d",
        "historical_night_share_180d",
        "historical_motorcycle_share_180d",
    ]

    for column in result_columns:
        panel[column] = 0.0

    for idx, row in panel.iterrows():

        month_start = row[
            "month_start"
        ]

        window_start = (
            month_start
            - pd.Timedelta(
                days=window_days
            )
        )

        history = working[
            (working["_date"] >= window_start)
            & (working["_date"] < month_start)
            & (
                working[CELL_COLUMN]
                == row["h3_cell"]
            )
        ]

        if history.empty:
            continue

        panel.at[
            idx,
            "historical_peak_hour_share_180d",
        ] = history[
            "_is_peak"
        ].mean()

        panel.at[
            idx,
            "historical_night_share_180d",
        ] = history[
            "_is_night"
        ].mean()

        panel.at[
            idx,
            "historical_motorcycle_share_180d",
        ] = history[
            "_is_motorcycle"
        ].mean()

    LOGGER.info(
        "Computed dynamic %d-day behavioural history features.",
        window_days,
    )

    return panel


# ============================================================================
# Target-month hotspot labels
# ============================================================================

def compute_monthly_hotspot_labels(
    panel: pd.DataFrame,
    accidents: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute hotspot ground truth independently for every target month.

    For each month:

        1. Count accidents in EVERY panel cell.
        2. Calculate one-sided Poisson p-values.
        3. Apply Benjamini-Hochberg FDR.
        4. is_hotspot = adjusted_p < 0.05.

    Zero-accident cells are explicitly included.
    """

    panel["target_accident_count"] = 0

    panel["hotspot_p_value"] = 1.0

    panel["hotspot_p_adjusted"] = 1.0

    panel["is_hotspot"] = pd.Series(
        [False] * len(panel),
        index=panel.index,
        dtype="bool",
    )

    months = (
        panel["month_start"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    for month_start in months:

        next_month = (
            month_start
            + pd.offsets.MonthBegin(1)
        )

        month_mask = (
            panel["month_start"]
            == month_start
        )

        # --------------------------------------------------------------
        # Target-month accidents
        # --------------------------------------------------------------

        month_accidents = accidents[
            (accidents["_date"] >= month_start)
            & (accidents["_date"] < next_month)
        ]

        # --------------------------------------------------------------
        # Count every panel cell
        # --------------------------------------------------------------

        month_cells = (
            panel.loc[
                month_mask,
                "h3_cell",
            ]
            .tolist()
        )

        counts = (
            month_accidents
            .groupby(CELL_COLUMN)
            .size()
            .reindex(
                month_cells,
                fill_value=0,
            )
            .astype(int)
        )

        mean_rate = float(
            counts.mean()
        )

        count_values = counts.to_numpy(
            dtype=int
        )

        panel.loc[
            month_mask,
            "target_accident_count",
        ] = count_values

        # --------------------------------------------------------------
        # No accidents in entire month
        # --------------------------------------------------------------

        if mean_rate <= 0:

            LOGGER.info(
                "%s: no accidents; no hotspot cells.",
                month_start.strftime("%Y-%m"),
            )

            continue

        # --------------------------------------------------------------
        # Poisson p-values
        # --------------------------------------------------------------

        p_values = counts.apply(
            lambda count: poisson.sf(
                count - 1,
                mean_rate,
            )
        )

        # --------------------------------------------------------------
        # Benjamini-Hochberg FDR
        # --------------------------------------------------------------

        _, adjusted, _, _ = multipletests(
            p_values.to_numpy(),
            alpha=HOTSPOT_FDR_ALPHA,
            method="fdr_bh",
        )

        adjusted = np.asarray(
            adjusted,
            dtype=float,
        )

        hotspot = (
            adjusted
            < HOTSPOT_FDR_ALPHA
        )

        # --------------------------------------------------------------
        # Assign values
        # --------------------------------------------------------------

        panel.loc[
            month_mask,
            "hotspot_p_value",
        ] = p_values.to_numpy(
            dtype=float
        )

        panel.loc[
            month_mask,
            "hotspot_p_adjusted",
        ] = adjusted

        panel.loc[
            month_mask,
            "is_hotspot",
        ] = hotspot

        LOGGER.info(
            "%s: target accidents=%d | mean cell rate=%.3f | hotspots=%d",
            month_start.strftime("%Y-%m"),
            len(month_accidents),
            mean_rate,
            int(hotspot.sum()),
        )

    return panel


# ============================================================================
# Target-month severity labels
# ============================================================================

def compute_monthly_severity_labels(
    panel: pd.DataFrame,
    accidents: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute target-month severity labels.

    Only cells with at least one target-month accident receive a severity
    target.

    No-accident cell-months are NOT treated as non-severe.

    This avoids conflating:

        no accident

    with:

        non-severe accident.
    """

    # ------------------------------------------------------------------
    # Numeric target components
    # ------------------------------------------------------------------

    panel["target_fatal_count"] = 0
    panel["target_serious_count"] = 0
    panel["target_minor_count"] = 0
    panel["target_damage_only_count"] = 0

    # ------------------------------------------------------------------
    # String target
    # ------------------------------------------------------------------

    panel["dominant_severity"] = pd.Series(
        [None] * len(panel),
        index=panel.index,
        dtype="object",
    )

    # ------------------------------------------------------------------
    # Binary target
    #
    # object dtype is intentional:
    #
    # True
    # False
    # None
    # ------------------------------------------------------------------

    panel["is_severe"] = pd.Series(
        [None] * len(panel),
        index=panel.index,
        dtype="object",
    )

    # ------------------------------------------------------------------
    # Whether severity target exists
    # ------------------------------------------------------------------

    panel["severity_defined"] = pd.Series(
        [False] * len(panel),
        index=panel.index,
        dtype="bool",
    )

    months = (
        panel["month_start"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    for month_start in months:

        next_month = (
            month_start
            + pd.offsets.MonthBegin(1)
        )

        month_mask = (
            panel["month_start"]
            == month_start
        )

        month_accidents = accidents[
            (accidents["_date"] >= month_start)
            & (accidents["_date"] < next_month)
        ].copy()

        if month_accidents.empty:
            continue

        # --------------------------------------------------------------
        # Normalise severity labels
        # --------------------------------------------------------------

        month_accidents["_severity"] = (
            month_accidents[
                SEVERITY_COLUMN
            ]
            .astype(str)
            .str.strip()
        )

        # --------------------------------------------------------------
        # Count severity per H3 cell
        # --------------------------------------------------------------

        severity_counts = (
            month_accidents
            .groupby(
                [
                    CELL_COLUMN,
                    "_severity",
                ]
            )
            .size()
            .unstack(
                fill_value=0
            )
        )

        # Ensure all expected severity columns exist.

        for severity in SEVERITY_LADDER:

            if severity not in severity_counts.columns:

                severity_counts[
                    severity
                ] = 0

        severity_counts = severity_counts[
            SEVERITY_LADDER
        ]

        # --------------------------------------------------------------
        # Process cells with target-month accidents
        # --------------------------------------------------------------

        month_cells = (
            panel.loc[
                month_mask,
                "h3_cell",
            ]
            .tolist()
        )

        for cell in month_cells:

            if cell not in severity_counts.index:
                continue

            row = severity_counts.loc[cell]

            # ----------------------------------------------------------
            # Make sure there is at least one recognised severity.
            #
            # This protects against a cell containing only unexpected
            # severity strings.
            # ----------------------------------------------------------

            recognised_count = int(
                row[
                    SEVERITY_LADDER
                ].sum()
            )

            if recognised_count <= 0:
                continue

            # ----------------------------------------------------------
            # Determine dominant severity.
            #
            # SEVERITY_LADDER is ordered:
            #
            # Fatal -> Serious -> Minor -> Damage-only
            #
            # If counts tie, the more severe class wins.
            # ----------------------------------------------------------

            max_count = row[
                SEVERITY_LADDER
            ].max()

            tied = [
                severity
                for severity in SEVERITY_LADDER
                if row[severity] == max_count
            ]

            dominant = tied[0]

            cell_mask = (
                month_mask
                & panel["h3_cell"].eq(cell)
            )

            # ----------------------------------------------------------
            # Target severity counts
            # ----------------------------------------------------------

            panel.loc[
                cell_mask,
                "target_fatal_count",
            ] = int(
                row["Fatal"]
            )

            panel.loc[
                cell_mask,
                "target_serious_count",
            ] = int(
                row["Serious"]
            )

            panel.loc[
                cell_mask,
                "target_minor_count",
            ] = int(
                row["Minor"]
            )

            panel.loc[
                cell_mask,
                "target_damage_only_count",
            ] = int(
                row["Damage-only"]
            )

            # ----------------------------------------------------------
            # Dominant severity
            # ----------------------------------------------------------

            panel.loc[
                cell_mask,
                "dominant_severity",
            ] = dominant

            # ----------------------------------------------------------
            # Binary severity
            # ----------------------------------------------------------

            is_severe_value = (
                dominant
                in [
                    "Fatal",
                    "Serious",
                ]
            )

            panel.loc[
                cell_mask,
                "is_severe",
            ] = bool(
                is_severe_value
            )

            # ----------------------------------------------------------
            # Severity is defined
            # ----------------------------------------------------------

            panel.loc[
                cell_mask,
                "severity_defined",
            ] = True

    LOGGER.info(
        "Computed target-month severity labels."
    )

    return panel


# ============================================================================
# Calendar
# ============================================================================

def add_cyclical_calendar(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Add cyclical month representation."""

    panel["month_sin"] = np.sin(
        2 * np.pi * panel["month"] / 12
    )

    panel["month_cos"] = np.cos(
        2 * np.pi * panel["month"] / 12
    )

    return panel


# ============================================================================
# Historical weather
# ============================================================================

def compute_monthly_weather(
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate raw daily weather by calendar month.

    Weather belonging to month M is made available to target month M+1.

    Therefore:

        September 2023 target
            -> August 2023 weather

        October 2023 target
            -> September 2023 weather

    This avoids using future target-month weather.

    The returned dataframe intentionally contains ONLY:

        month_start
        avg_precip_month
        avg_humidity_month
        avg_visibility_month
        avg_cloudcover_month
        rainy_days_share_month

    It does NOT contain year_month because the main panel already owns
    the authoritative target-month year_month column.
    """

    w = weather.copy()

    # ------------------------------------------------------------------
    # Weather source month
    # ------------------------------------------------------------------

    w["weather_year_month"] = (
        w["_date"]
        .dt.to_period("M")
        .astype(str)
    )

    # ------------------------------------------------------------------
    # Rainy day indicator
    # ------------------------------------------------------------------

    if "precip" in w.columns:
        w["_is_rainy_day"] = (
            w["precip"]
            .fillna(0)
            >= RAIN_PRECIP_THRESHOLD_MM
        ).astype(int)
    else:
        w["_is_rainy_day"] = 0

    # ------------------------------------------------------------------
    # Monthly aggregation
    # ------------------------------------------------------------------

    aggregation = {}

    for column in [
        "precip",
        "humidity",
        "visibility",
        "cloudcover",
    ]:
        if column in w.columns:
            aggregation[column] = "mean"

    aggregation["_is_rainy_day"] = "mean"

    monthly = (
        w
        .groupby("weather_year_month")
        .agg(aggregation)
        .reset_index()
    )

    # ------------------------------------------------------------------
    # Rename weather features
    # ------------------------------------------------------------------

    monthly = monthly.rename(
        columns={
            "precip": "avg_precip_month",
            "humidity": "avg_humidity_month",
            "visibility": "avg_visibility_month",
            "cloudcover": "avg_cloudcover_month",
            "_is_rainy_day": "rainy_days_share_month",
        }
    )

    # ------------------------------------------------------------------
    # Convert source weather month to timestamp
    # ------------------------------------------------------------------

    monthly["weather_month"] = pd.to_datetime(
        monthly["weather_year_month"] + "-01"
    )

    # ------------------------------------------------------------------
    # Shift weather forward by one month
    #
    # January weather -> February target
    # August weather  -> September target
    # September weather -> October target
    # ------------------------------------------------------------------

    monthly["month_start"] = (
        monthly["weather_month"]
        + pd.offsets.MonthBegin(1)
    )

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # Remove all source-month helper columns before merging into panel.
    # This prevents year_month_x / year_month_y collisions.
    # ------------------------------------------------------------------

    monthly = monthly.drop(
        columns=[
            "weather_year_month",
            "weather_month",
        ],
        errors="ignore",
    )

    LOGGER.info(
        "Built lagged monthly weather features from raw_weather: %d months.",
        len(monthly),
    )

    return monthly


# ============================================================================
# Sri Lankan holidays
# ============================================================================

def _iter_lk_holidays(
    years: List[int],
) -> Dict[date, List[str]]:
    """Return Sri Lankan holiday dates."""

    holiday_map: Dict[
        date,
        List[str],
    ] = {}

    try:

        import holidays as holidays_module

        for year in years:

            for d, name in (
                holidays_module
                .country_holidays(
                    "LK",
                    years=[year],
                )
                .items()
            ):

                holiday_map.setdefault(
                    d,
                    [],
                ).append(
                    str(name)
                )

    except ImportError:

        LOGGER.warning(
            "'holidays' package unavailable; using fallback dates."
        )

    # --------------------------------------------------------------
    # Fallback / guaranteed dates
    # --------------------------------------------------------------

    for year in years:

        holiday_map.setdefault(
            date(year, 1, 1),
            [],
        ).append(
            "New Year's Day"
        )

        holiday_map.setdefault(
            date(year, 12, 25),
            [],
        ).append(
            "Christmas Day"
        )

        holiday_map.setdefault(
            date(year, 5, 1),
            [],
        ).append(
            "International Workers' Day"
        )

        es = easter(year)

        holiday_map.setdefault(
            es,
            [],
        ).append(
            "Easter Sunday"
        )

        holiday_map.setdefault(
            es - timedelta(days=2),
            [],
        ).append(
            "Good Friday"
        )

        holiday_map.setdefault(
            date(year, 4, 13),
            [],
        ).append(
            "Sinhala and Tamil New Year"
        )

        holiday_map.setdefault(
            date(year, 4, 14),
            [],
        ).append(
            "Sinhala and Tamil New Year"
        )

    return holiday_map


def compute_holidays_in_month(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Count known public holidays in each target month."""

    years = sorted(
        panel["year"]
        .unique()
        .tolist()
    )

    holiday_map = _iter_lk_holidays(
        years
    )

    counts_by_ym: Dict[
        str,
        int,
    ] = {}

    for d in holiday_map.keys():

        ym = (
            f"{d.year:04d}-{d.month:02d}"
        )

        counts_by_ym[ym] = (
            counts_by_ym.get(ym, 0)
            + 1
        )

    return (
        pd.Series(
            counts_by_ym,
            name="holidays_in_month",
        )
        .rename_axis("year_month")
        .reset_index()
    )


# ============================================================================
# Final metadata repair
# ============================================================================

def restore_panel_time_columns(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recreate the core time columns from month_start.

    month_start is the authoritative target-month value.

    This guarantees the final panel contains:

        month_start
        year_month
        year
        month
        feature_cutoff_date

    and does not retain accidental merge-generated columns such as:

        year_month_x
        year_month_y
    """

    if "month_start" not in panel.columns:
        raise ValueError(
            "Panel is missing required 'month_start' column before finalization."
        )

    panel["month_start"] = pd.to_datetime(
        panel["month_start"],
        errors="coerce",
    )

    if panel["month_start"].isna().any():
        raise ValueError(
            "Panel contains invalid/null month_start values."
        )

    # --------------------------------------------------------------
    # Remove accidental merge collision columns
    # --------------------------------------------------------------

    panel = panel.drop(
        columns=[
            "year_month_x",
            "year_month_y",
        ],
        errors="ignore",
    )

    # --------------------------------------------------------------
    # Recreate authoritative target-month metadata
    # --------------------------------------------------------------

    panel["year_month"] = (
        panel["month_start"]
        .dt.strftime("%Y-%m")
    )

    panel["year"] = (
        panel["month_start"]
        .dt.year
        .astype(int)
    )

    panel["month"] = (
        panel["month_start"]
        .dt.month
        .astype(int)
    )

    panel["feature_cutoff_date"] = (
        panel["month_start"]
        - pd.Timedelta(days=1)
    ).dt.strftime("%Y-%m-%d")

    return panel


# ============================================================================
# Sanity checks
# ============================================================================

def run_sanity_checks(
    panel: pd.DataFrame,
) -> None:
    """
    Run final validation checks on the generated prediction panel.
    """

    LOGGER.info(
        "--- Prediction panel sanity checks ---"
    )

    # --------------------------------------------------------------
    # Required structural columns
    # --------------------------------------------------------------

    required_columns = [
        "h3_cell",
        "month_start",
        "year_month",
        "year",
        "month",
        "feature_cutoff_date",
        "target_accident_count",
        "is_hotspot",
        "dominant_severity",
        "is_severe",
        "severity_defined",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in panel.columns
    ]

    if missing_columns:
        raise ValueError(
            "Prediction panel missing required columns: "
            f"{missing_columns}"
        )

    LOGGER.info(
        "✓ Required panel columns present."
    )

    # --------------------------------------------------------------
    # Panel grain
    # --------------------------------------------------------------

    duplicate_count = (
        panel
        .duplicated(
            subset=[
                "h3_cell",
                "year_month",
            ]
        )
        .sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            "(h3_cell, year_month) rows."
        )

    LOGGER.info(
        "✓ No duplicate (h3_cell, year_month) rows."
    )

    # --------------------------------------------------------------
    # Expected panel size
    # --------------------------------------------------------------

    expected_cells = (
        panel["h3_cell"]
        .nunique()
    )

    expected_months = (
        panel["year_month"]
        .nunique()
    )

    expected_rows = (
        expected_cells
        * expected_months
    )

    if len(panel) != expected_rows:
        raise ValueError(
            "Panel is not a complete cell-month grid: "
            f"rows={len(panel)}, "
            f"expected={expected_rows}."
        )

    LOGGER.info(
        "✓ Complete cell-month grid: %d × %d = %d rows.",
        expected_cells,
        expected_months,
        expected_rows,
    )

    # --------------------------------------------------------------
    # Target accident count
    # --------------------------------------------------------------

    target_total = int(
        panel[
            "target_accident_count"
        ].sum()
    )

    LOGGER.info(
        "Total target-month accident labels: %d",
        target_total,
    )

    # --------------------------------------------------------------
    # Hotspots
    # --------------------------------------------------------------

    hotspot_total = int(
        panel[
            "is_hotspot"
        ].sum()
    )

    LOGGER.info(
        "Total hotspot cell-month labels: %d",
        hotspot_total,
    )

    # --------------------------------------------------------------
    # Severity
    # --------------------------------------------------------------

    severity_defined_count = int(
        panel[
            "severity_defined"
        ].sum()
    )

    LOGGER.info(
        "Severity-defined cell-month rows: %d",
        severity_defined_count,
    )

    # --------------------------------------------------------------
    # Dominant severity distribution
    # --------------------------------------------------------------

    LOGGER.info(
        "Target-month dominant severity distribution:\n%s",
        panel[
            "dominant_severity"
        ]
        .value_counts(
            dropna=False
        )
        .to_string(),
    )

    # --------------------------------------------------------------
    # Binary severity distribution
    # --------------------------------------------------------------

    LOGGER.info(
        "Target-month binary severity distribution:\n%s",
        panel.loc[
            panel["severity_defined"],
            "is_severe",
        ]
        .value_counts(
            dropna=False
        )
        .to_string(),
    )

    # --------------------------------------------------------------
    # Weather missingness
    # --------------------------------------------------------------

    weather_columns = [
        "avg_precip_month",
        "avg_humidity_month",
        "avg_visibility_month",
        "avg_cloudcover_month",
        "rainy_days_share_month",
    ]

    existing_weather_columns = [
        column
        for column in weather_columns
        if column in panel.columns
    ]

    if existing_weather_columns:

        LOGGER.info(
            "Weather missing values:"
        )

        for column in existing_weather_columns:

            missing_count = int(
                panel[column]
                .isna()
                .sum()
            )

            LOGGER.info(
                "  %s: %d missing",
                column,
                missing_count,
            )

    # --------------------------------------------------------------
    # Feature cutoff audit
    # --------------------------------------------------------------

    LOGGER.info(
        "Feature cutoff examples:"
    )

    for year_month in [
        "2022-01",
        "2022-02",
        "2023-08",
        "2023-09",
        "2023-10",
        "2023-12",
    ]:

        sample = panel[
            panel["year_month"]
            == year_month
        ]

        if sample.empty:
            continue

        LOGGER.info(
            "  %s -> cutoff %s",
            year_month,
            sample[
                "feature_cutoff_date"
            ].iloc[0],
        )

    LOGGER.info(
        "✓ Sanity checks completed."
    )


# ============================================================================
# Main panel builder
# ============================================================================

def build_panel() -> pd.DataFrame:

    configure_logging()

    LOGGER.info(
        "=== Prediction Panel Build Start ==="
    )

    db, client = get_db()

    try:

        # ==============================================================
        # Step 1 — Load accident + weather sources
        # ==============================================================

        LOGGER.info(
            "-- Step 1: Loading accident features + raw weather --"
        )

        accidents = load_accident_features(
            db
        )

        weather = load_raw_weather(
            db
        )

        # ==============================================================
        # Step 2 — Build panel skeleton
        # ==============================================================

        LOGGER.info(
            "-- Step 2: Building cell-month panel skeleton --"
        )

        cells = (
            accidents[
                CELL_COLUMN
            ]
            .dropna()
            .unique()
            .tolist()
        )

        panel = build_panel_skeleton(
            cells
        )

        # ==============================================================
        # Step 3 — Static spatial/context information
        # ==============================================================

        LOGGER.info(
            "-- Step 3: Adding static H3 context --"
        )

        static_context = (
            compute_static_cell_context(
                accidents
            )
        )

        panel = panel.merge(
            static_context,
            on="h3_cell",
            how="left",
        )

        # ==============================================================
        # Step 4 — Historical accident exposure
        # ==============================================================

        LOGGER.info(
            "-- Step 4: Computing historical rolling exposure --"
        )

        panel = compute_rolling_exposure(
            panel,
            accidents,
            ROLLING_WINDOWS_DAYS,
        )

        # ==============================================================
        # Step 5 — Dynamic behavioural history
        # ==============================================================

        LOGGER.info(
            "-- Step 5: Computing dynamic behavioural history --"
        )

        panel = (
            compute_historical_behavioural_features(
                panel,
                accidents,
                BEHAVIOURAL_WINDOW_DAYS,
            )
        )

        # ==============================================================
        # Step 6 — Calendar features
        # ==============================================================

        LOGGER.info(
            "-- Step 6: Adding calendar features --"
        )

        panel = add_cyclical_calendar(
            panel
        )

        holidays_month = (
            compute_holidays_in_month(
                panel
            )
        )

        panel = panel.merge(
            holidays_month,
            on="year_month",
            how="left",
        )

        panel["holidays_in_month"] = (
            panel[
                "holidays_in_month"
            ]
            .fillna(0)
            .astype(int)
        )

        # ==============================================================
        # Step 7 — Historical weather
        # ==============================================================

        LOGGER.info(
            "-- Step 7: Adding lagged weather from raw_weather --"
        )

        monthly_weather = (
            compute_monthly_weather(
                weather
            )
        )

        panel = panel.merge(
            monthly_weather,
            on="month_start",
            how="left",
            validate="many_to_one",
        )

        # IMPORTANT:
        #
        # Do NOT fill weather using the whole panel.
        #
        # Missing historical weather is left as NaN.
        #
        # The modelling pipeline should fit any imputer using
        # TRAINING data only.

        # ==============================================================
        # Step 8 — Target-month hotspot labels
        # ==============================================================

        LOGGER.info(
            "-- Step 8: Computing monthly Poisson-FDR hotspot labels --"
        )

        panel = (
            compute_monthly_hotspot_labels(
                panel,
                accidents,
            )
        )

        # ==============================================================
        # Step 9 — Target-month severity labels
        # ==============================================================

        LOGGER.info(
            "-- Step 9: Computing monthly severity labels --"
        )

        panel = (
            compute_monthly_severity_labels(
                panel,
                accidents,
            )
        )

        # ==============================================================
        # Step 10 — Final metadata
        # ==============================================================

        LOGGER.info(
            "-- Step 10: Finalizing prediction panel --"
        )

        # --------------------------------------------------------------
        # IMPORTANT FIX:
        #
        # Rebuild all time columns from month_start.
        #
        # This guarantees year_month exists before sorting.
        # --------------------------------------------------------------

        panel = restore_panel_time_columns(
            panel
        )

        # --------------------------------------------------------------
        # Use timezone-aware UTC timestamp.
        #
        # Fixes:
        # datetime.utcnow() deprecation warning.
        # --------------------------------------------------------------

        panel["built_at"] = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        # --------------------------------------------------------------
        # Final sort
        # --------------------------------------------------------------

        panel = (
            panel
            .sort_values(
                [
                    "year_month",
                    "h3_cell",
                ]
            )
            .reset_index(drop=True)
        )

        # ==============================================================
        # Step 11 — Persist local CSV
        # ==============================================================

        LOGGER.info(
            "-- Step 11: Writing and validating prediction panel --"
        )

        CSV_OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        panel.to_csv(
            CSV_OUTPUT_PATH,
            index=False,
        )

        LOGGER.info(
            "CSV written: %s (%d rows, %d columns)",
            CSV_OUTPUT_PATH,
            len(panel),
            len(panel.columns),
        )

        # ==============================================================
        # Step 12 — Sanity checks
        # ==============================================================

        LOGGER.info(
            "-- Step 12: Running sanity checks --"
        )

        run_sanity_checks(
            panel
        )

        # Only save after validation succeeds
        save_panel(
            db,
            panel,
        )

        LOGGER.info(
            "Rows: %d",
            len(panel),
        )

        LOGGER.info(
            "Unique cells: %d",
            panel["h3_cell"].nunique(),
        )

        LOGGER.info(
            "Unique target months: %d",
            panel["year_month"].nunique(),
        )

        LOGGER.info(
            "=== Prediction Panel Build Complete ==="
        )

        return panel

    finally:

        client.close()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    build_panel()