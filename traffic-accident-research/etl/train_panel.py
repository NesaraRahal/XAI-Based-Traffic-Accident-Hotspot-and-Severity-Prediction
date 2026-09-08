"""Build the (H3 cell, month) training panel for hotspot & severity classification.

VERSION 2 — LEAKAGE-FIXED
=========================
This rebuild fixes two label-leakage bugs that made XGBoost trivially hit
100% accuracy in the original panel:

Bug 1: The `is_hotspot` and `dominant_severity` labels were computed from
       the FULL 2022-2023 dataset, then attached to every (cell, month)
       row — including test-window rows. So test-set labels already
       "knew" about test-window accidents that produced them.

Bug 2: `accidents_this_month` was a direct component of the label
       (label = f(sum(accidents across months))), so the model could
       mechanically reconstruct the label from the feature.

Also: monthly weather aggregates and behavioral shares (peak_hour_share,
night_share, motorcycle_share, rainy_share, poor_visibility_share) were
computed from the ACCIDENTS in each month, which subtly ties the feature
value to the label's construction ingredients.

Fixes applied here:
  - Labels are computed using TRAINING-WINDOW data only (<= 2023-08).
    Test-window rows still inherit the same static per-cell label, but
    that label is now genuinely "historical" from the model's perspective.
  - Monthly weather (avg_precip / humidity / visibility / cloudcover,
    rainy_days_share_month) is pulled
    DIRECTLY from raw_weather.
  - Holidays become a calendar count (holidays_in_month), not derived
    from accidents.
  - Behavioural shares become HISTORICAL (per cell, from training window
    only) — historical_peak_hour_share, historical_night_share,
    historical_motorcycle_share.
  - accidents_this_month is DROPPED entirely.
  - historical_accident_count / fatal_count / etc. are still emitted for
    dashboard visibility, but the modelling utils EXCLUDE them from the
    feature list (they're direct label components).

Grain, SHAP story, and multi-source integration are all preserved.

MongoDB collections
-------------------
Input:
    accidents_spatiotemporal_features  — accident-grain feature store
    ref_h3_cell_aggregates             — static per-cell view (kept for reference)
    raw_weather                        — daily Colombo weather (for genuine monthly aggs)

Output:
    training_panel_cell_month          — one row per (h3_cell, year_month)

Local output:
    data/processed/panel_cell_month.csv
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List

import h3
import numpy as np
import pandas as pd
from dateutil.easter import easter
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
COL_RAW_WEATHER = "raw_weather"
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

# Severity ladder — used for tie-breaking dominant severity (worst wins)
SEVERITY_LADDER = ["Fatal", "Serious", "Minor", "Damage-only"]

# ── Panel config ───────────────────────────────────────────────────────────
STUDY_START = pd.Timestamp("2022-01-01")
STUDY_END = pd.Timestamp("2023-12-31")
ROLLING_WINDOWS_DAYS = [30, 90, 180]
NEIGHBOR_RING_K = 1

# ── LEAKAGE FIX: labels computed only from data on or before this date ─────
# Must match TEMPORAL_SPLIT_TRAIN_END in modeling/modelling_utils.py
LABEL_TRAINING_WINDOW_END = pd.Timestamp("2023-08-31")

# ── Poisson-FDR config (mirrors spatiotemporal.py) ─────────────────────────
HOTSPOT_FDR_ALPHA = 0.05

# ── Weather thresholds ─────────────────────────────────────────────────────
# NOTE: A reduced-visibility indicator was constructed and evaluated but
# excluded from the panel. The Visual Crossing feed for 2022–2023 shows
# essentially no days with visibility below 6 km (only 1 day in 730), so
# the feature has no variance for the study window and cannot inform the
# model. The same feed shows substantial visibility variation from 2024
# onwards; extending the framework to more recent data would restore this
# feature. See thesis limitations section.
RAIN_PRECIP_THRESHOLD_MM = 0.1

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


def load_raw_weather(db) -> pd.DataFrame:
    """Load daily Colombo weather from raw_weather (independent of accidents)."""
    records = list(db[COL_RAW_WEATHER].find({}, {"_id": 0}))
    if not records:
        raise ValueError(
            f"No records found in collection: {COL_RAW_WEATHER}. "
            "Weather-based monthly features require the raw weather feed."
        )
    weather = pd.DataFrame(records)

    if "datetime" not in weather.columns:
        raise ValueError(
            f"'{COL_RAW_WEATHER}' missing 'datetime' column. Cannot build "
            "leakage-free monthly weather aggregates."
        )

    weather["_date"] = pd.to_datetime(weather["datetime"], errors="coerce").dt.normalize()
    weather = weather.dropna(subset=["_date"]).drop_duplicates(subset=["_date"])

    for col in ["precip", "humidity", "visibility", "cloudcover"]:
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")

    LOGGER.info("Loaded %d daily weather records from [%s]", len(weather), COL_RAW_WEATHER)
    return weather


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
# LEAKAGE FIX — labels rebuilt from TRAINING-WINDOW ACCIDENTS ONLY
# ══════════════════════════════════════════════════════════════════════════

def compute_training_window_labels(accidents: pd.DataFrame) -> pd.DataFrame:
    """Compute is_hotspot + dominant_severity per cell using only data on or
    before LABEL_TRAINING_WINDOW_END.

    Design note: the label thus becomes 'was this cell historically a hotspot
    based on training-window accidents?' — a static per-cell characterization
    the model tries to recover from monthly features. This is the same
    static-label design as Islam et al. 2022 and Al-Omari et al. 2025, but
    now with an honest label/feature separation across the temporal split.
    """
    from scipy.stats import poisson
    from statsmodels.stats.multitest import multipletests

    train_acc = accidents[accidents[DATE_COLUMN] <= LABEL_TRAINING_WINDOW_END].copy()
    LOGGER.info(
        "Label computation uses %d/%d accidents on or before %s",
        len(train_acc), len(accidents), LABEL_TRAINING_WINDOW_END.date(),
    )

    # ── is_hotspot via Poisson-FDR on training-window counts ──────────────
    per_cell_counts = (
        train_acc.groupby(CELL_COLUMN).size()
        .rename("accident_count").reset_index()
        .rename(columns={CELL_COLUMN: "h3_cell"})
    )

    mean_rate = per_cell_counts["accident_count"].mean()
    per_cell_counts["hotspot_p_value"] = per_cell_counts["accident_count"].apply(
        lambda c: poisson.sf(c - 1, mean_rate)
    )
    _, adjusted, _, _ = multipletests(
        per_cell_counts["hotspot_p_value"], alpha=HOTSPOT_FDR_ALPHA, method="fdr_bh"
    )
    per_cell_counts["hotspot_p_adjusted_train"] = adjusted
    per_cell_counts["is_hotspot"] = adjusted < HOTSPOT_FDR_ALPHA

    LOGGER.info(
        "Training-window hotspot label: %d/%d cells flagged (FDR alpha=%.2f, mean rate=%.2f)",
        int(per_cell_counts["is_hotspot"].sum()), len(per_cell_counts),
        HOTSPOT_FDR_ALPHA, mean_rate,
    )

    # ── dominant_severity via training-window severity counts ─────────────
    sev_counts = (
        train_acc.groupby([CELL_COLUMN, SEVERITY_COLUMN])
        .size().unstack(fill_value=0)
    )
    for cls in SEVERITY_LADDER:
        if cls not in sev_counts.columns:
            sev_counts[cls] = 0

    def pick_dominant(row: pd.Series) -> str:
        max_count = row[SEVERITY_LADDER].max()
        tied = [c for c in SEVERITY_LADDER if row[c] == max_count]
        return tied[0]  # worst-severity wins (SEVERITY_LADDER is worst→least)

    sev_counts["dominant_severity"] = sev_counts.apply(pick_dominant, axis=1)
    train_sev = sev_counts.reset_index().rename(columns={
        CELL_COLUMN: "h3_cell",
        "Fatal": "train_fatal_count",
        "Serious": "train_serious_count",
        "Minor": "train_minor_count",
        "Damage-only": "train_damage_only_count",
    })

    labels = per_cell_counts.merge(train_sev, on="h3_cell", how="left")
    return labels


# ══════════════════════════════════════════════════════════════════════════
# LEAKAGE FIX — behavioural shares from TRAINING WINDOW, per cell (static)
# ══════════════════════════════════════════════════════════════════════════

def compute_historical_behavioural_shares(accidents: pd.DataFrame) -> pd.DataFrame:
    """Per-cell historical shares computed from training-window accidents only.

    These become constant per cell across all months. SHAP still gets to say
    'this cell is dangerous partly because it's a motorcycle-heavy / night-heavy
    cell historically' — the story is intact.
    """
    train_acc = accidents[accidents[DATE_COLUMN] <= LABEL_TRAINING_WINDOW_END].copy()

    if PEAK_HOUR_COLUMN in train_acc.columns:
        train_acc["_is_peak"] = (train_acc[PEAK_HOUR_COLUMN].astype(str) == "Yes").astype(int)
    else:
        train_acc["_is_peak"] = 0
    if LIGHT_COND_COLUMN in train_acc.columns:
        train_acc["_is_night"] = (train_acc[LIGHT_COND_COLUMN].astype(str) == "Night").astype(int)
    else:
        train_acc["_is_night"] = 0
    if VEHICLE_TYPES_COLUMN in train_acc.columns:
        train_acc["_is_motorcycle"] = (
            train_acc[VEHICLE_TYPES_COLUMN].astype(str).str.lower()
            .str.contains("motorcycle").astype(int)
        )
    else:
        train_acc["_is_motorcycle"] = 0

    grouped = train_acc.groupby(CELL_COLUMN).agg(
        historical_peak_hour_share=("_is_peak", "mean"),
        historical_night_share=("_is_night", "mean"),
        historical_motorcycle_share=("_is_motorcycle", "mean"),
    ).reset_index().rename(columns={CELL_COLUMN: "h3_cell"})

    return grouped


# ══════════════════════════════════════════════════════════════════════════
# Static per-cell context (police division, cell centroid) — no leakage
# ══════════════════════════════════════════════════════════════════════════

def compute_static_cell_context(accidents: pd.DataFrame) -> pd.DataFrame:
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
# Panel skeleton
# ══════════════════════════════════════════════════════════════════════════

def build_panel_skeleton(cells: List[str]) -> pd.DataFrame:
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
# Feature family (a) — causal rolling exposure (unchanged; already leakage-safe)
# ══════════════════════════════════════════════════════════════════════════

def _neighbor_map(cells: List[str], k: int = NEIGHBOR_RING_K) -> Dict[str, List[str]]:
    return {cell: [c for c in h3.grid_disk(cell, k) if c != cell] for cell in cells}


def compute_rolling_exposure(
    panel: pd.DataFrame,
    accidents: pd.DataFrame,
    windows_days: List[int] = ROLLING_WINDOWS_DAYS,
) -> pd.DataFrame:
    """same_cell_count_{w}d and neighbor_count_{w}d as of month_start.

    LEAKAGE FIX (v2.1):
    -------------------
    The window end is `min(month_start, LABEL_TRAINING_WINDOW_END + 1 day)`.
    - For TRAINING months (month_start <= LABEL_TRAINING_WINDOW_END): the
      window ends strictly before month_start, as before — causal on the
      full accident history up to that point.
    - For TEST months (month_start > LABEL_TRAINING_WINDOW_END): the window
      ends at LABEL_TRAINING_WINDOW_END + 1 day, so features can only see
      accidents that ALSO went into label construction. Test-window
      accidents never enter test-row features, closing the leakage path
      where rolling counts became a near-perfect proxy for the label.

    This mirrors the label boundary: features and labels see exactly the
    same slice of accident history. Defence framing: 'For test-window
    months, rolling exposure features are truncated at the label training
    window end (2023-08-31) to prevent information from the test window
    entering either the label or the feature construction.'

    Window start is still `window_end - w days`, so the window LENGTH is
    preserved. For test months this means the effective look-back may be
    older than the row itself (e.g. Dec 2023's 30d window covers Aug 2023),
    but it is honest: it is the same info the training rows have access to.
    """
    unique_cells = panel["h3_cell"].unique().tolist()
    neighbors = _neighbor_map(unique_cells)

    daily = (
        accidents.assign(_d=accidents[DATE_COLUMN].dt.normalize())
        .groupby([CELL_COLUMN, "_d"]).size().rename("count").reset_index()
        .rename(columns={CELL_COLUMN: "h3_cell", "_d": "date"})
    )
    daily = daily.set_index(["h3_cell", "date"])["count"]

    for w in windows_days:
        panel[f"same_cell_count_{w}d"] = 0
        panel[f"neighbor_count_{w}d"] = 0

    cell_series_cache: Dict[str, pd.Series] = {}
    full_idx = pd.date_range(
        STUDY_START - pd.Timedelta(days=max(windows_days)),
        STUDY_END, freq="D",
    )
    for cell in unique_cells:
        try:
            s = daily.loc[cell].sort_index()
            s = s.reindex(full_idx, fill_value=0)
        except KeyError:
            s = pd.Series(0, index=full_idx)
        cell_series_cache[cell] = s

    # Feature-construction cutoff: mirrors the label boundary.
    # Test-window rows will use this as their effective window end.
    FEATURE_CUTOFF = LABEL_TRAINING_WINDOW_END + pd.Timedelta(days=1)

    for cell in unique_cells:
        cell_series = cell_series_cache[cell]
        neighbor_cells = neighbors.get(cell, [])

        if neighbor_cells:
            parts = [cell_series_cache.get(nc) for nc in neighbor_cells if nc in cell_series_cache]
            neighbor_series = sum(parts) if parts else pd.Series(0, index=cell_series.index)
        else:
            neighbor_series = pd.Series(0, index=cell_series.index)

        cell_mask = panel["h3_cell"] == cell
        month_starts = panel.loc[cell_mask, "month_start"]

        for w in windows_days:
            same_col = f"same_cell_count_{w}d"
            neigh_col = f"neighbor_count_{w}d"
            same_vals, neigh_vals = [], []
            for ms in month_starts:
                # LEAKAGE FIX: cap the window end at the feature cutoff.
                # Training months are unaffected (ms <= FEATURE_CUTOFF).
                # Test months get capped so they can only see training-window data.
                window_end = min(ms, FEATURE_CUTOFF)
                window_start = window_end - pd.Timedelta(days=w)
                same_vals.append(int(cell_series.loc[
                    (cell_series.index >= window_start) & (cell_series.index < window_end)
                ].sum()))
                neigh_vals.append(int(neighbor_series.loc[
                    (neighbor_series.index >= window_start) & (neighbor_series.index < window_end)
                ].sum()))
            panel.loc[cell_mask, same_col] = same_vals
            panel.loc[cell_mask, neigh_col] = neigh_vals

    LOGGER.info(
        "Computed rolling exposure features for %d cells "
        "(feature cutoff = %s to prevent test-window leakage)",
        len(unique_cells), FEATURE_CUTOFF.date(),
    )
    return panel


# ══════════════════════════════════════════════════════════════════════════
# Feature family (b) — cyclical calendar (no leakage)
# ══════════════════════════════════════════════════════════════════════════

def add_cyclical_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    panel["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12)
    return panel


# ══════════════════════════════════════════════════════════════════════════
# LEAKAGE FIX — monthly weather aggregates from raw_weather (not from accidents)
# ══════════════════════════════════════════════════════════════════════════

def compute_monthly_weather_from_raw(weather: pd.DataFrame) -> pd.DataFrame:
    """One row per year_month with genuine monthly weather aggregates.

    Note: weather in this study is Colombo-wide (single Visual Crossing feed),
    so the same monthly weather values apply to every cell. This is a stated
    project caveat — spatial variation would require per-cell weather feeds.
    """
    w = weather.copy()
    w["year_month"] = w["_date"].dt.strftime("%Y-%m")

    if "precip" in w.columns:
        w["_is_rainy_day"] = (w["precip"].fillna(0) >= RAIN_PRECIP_THRESHOLD_MM).astype(int)
    else:
        w["_is_rainy_day"] = 0
    # Note: reduced-visibility indicator excluded — see comment on weather
    # thresholds at top of file. avg_visibility_month is still emitted below.

    agg_cols = {}
    for col in ["precip", "humidity", "visibility", "cloudcover"]:
        if col in w.columns:
            agg_cols[col] = "mean"
    agg_cols["_is_rainy_day"] = "mean"

    monthly = w.groupby("year_month").agg(agg_cols).reset_index()
    monthly = monthly.rename(columns={
        "precip": "avg_precip_month",
        "humidity": "avg_humidity_month",
        "visibility": "avg_visibility_month",
        "cloudcover": "avg_cloudcover_month",
        "_is_rainy_day": "rainy_days_share_month",
    })

    LOGGER.info(
        "Built monthly weather aggregates from raw_weather: %d months covered",
        len(monthly),
    )
    return monthly


# ══════════════════════════════════════════════════════════════════════════
# Calendar feature — holidays_in_month (no leakage)
# ══════════════════════════════════════════════════════════════════════════

def _iter_lk_holidays(years: List[int]) -> Dict[date, List[str]]:
    """Same LK holiday set feature_engineering.py uses, without needing the
    accident dataframe. Uses `holidays` package if available; augments with
    globally observed and LK-specific dates.
    """
    holiday_map: Dict[date, List[str]] = {}
    try:
        import holidays as holidays_module
        for year in years:
            for d, name in holidays_module.country_holidays("LK", years=[year]).items():
                holiday_map.setdefault(d, []).append(str(name))
    except ImportError:
        LOGGER.warning("`holidays` package unavailable — using globals only.")

    for year in years:
        holiday_map.setdefault(date(year, 1, 1), []).append("New Year's Day")
        holiday_map.setdefault(date(year, 12, 25), []).append("Christmas Day")
        holiday_map.setdefault(date(year, 5, 1), []).append("International Workers' Day")
        es = easter(year)
        holiday_map.setdefault(es, []).append("Easter Sunday")
        holiday_map.setdefault(es - timedelta(days=2), []).append("Good Friday")
        holiday_map.setdefault(date(year, 4, 13), []).append("Sinhala and Tamil New Year")
        holiday_map.setdefault(date(year, 4, 14), []).append("Sinhala and Tamil New Year")

    return holiday_map


def compute_holidays_in_month(panel: pd.DataFrame) -> pd.DataFrame:
    """Count of public holiday dates falling in each year_month."""
    years = sorted(panel["year"].unique().tolist())
    holiday_map = _iter_lk_holidays(years)

    counts_by_ym: Dict[str, int] = {}
    for d in holiday_map.keys():
        ym = f"{d.year:04d}-{d.month:02d}"
        counts_by_ym[ym] = counts_by_ym.get(ym, 0) + 1

    return (
        pd.Series(counts_by_ym, name="holidays_in_month")
        .rename_axis("year_month").reset_index()
    )


# ══════════════════════════════════════════════════════════════════════════
# Attach labels + static context
# ══════════════════════════════════════════════════════════════════════════

def attach_labels_and_context(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    static_context: pd.DataFrame,
    historical_shares: pd.DataFrame,
    cell_aggregates: pd.DataFrame,
) -> pd.DataFrame:
    """Attach labels + static context + full-dataset counts (dashboard-only)."""

    # Training-window labels
    panel = panel.merge(
        labels[["h3_cell", "is_hotspot", "dominant_severity",
                "accident_count", "hotspot_p_adjusted_train",
                "train_fatal_count", "train_serious_count",
                "train_minor_count", "train_damage_only_count"]]
        .rename(columns={"accident_count": "train_window_accident_count"}),
        on="h3_cell", how="left",
    )
    panel["is_hotspot"] = panel["is_hotspot"].fillna(False)

    # Static context (police division, cell centroid)
    panel = panel.merge(static_context, on="h3_cell", how="left")

    # Historical behavioural shares (per cell, from training window)
    panel = panel.merge(historical_shares, on="h3_cell", how="left")
    for col in ["historical_peak_hour_share", "historical_night_share",
                "historical_motorcycle_share"]:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0.0)

    # Full-dataset per-cell counts (dashboard-visibility only — modelling
    # utils EXCLUDES these from the feature list as they're label components)
    keep_from_aggregates = ["h3_cell", "accident_count", "fatal_count",
                            "serious_count", "minor_count", "damage_only_count",
                            "fatal_rate", "hotspot_p_adjusted"]
    keep_from_aggregates = [c for c in keep_from_aggregates if c in cell_aggregates.columns]
    if keep_from_aggregates:
        panel = panel.merge(
            cell_aggregates[keep_from_aggregates].rename(columns={
                "accident_count": "historical_accident_count",
                "fatal_count": "historical_fatal_count",
                "serious_count": "historical_serious_count",
                "minor_count": "historical_minor_count",
                "damage_only_count": "historical_damage_only_count",
                "fatal_rate": "historical_fatal_rate",
            }),
            on="h3_cell", how="left",
        )
    return panel


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def build_panel() -> pd.DataFrame:
    configure_logging()
    LOGGER.info("=== Training Panel Build Start (v2 — leakage-fixed) ===")
    LOGGER.info("LABEL_TRAINING_WINDOW_END = %s", LABEL_TRAINING_WINDOW_END.date())

    db, client = get_db()
    try:
        # Step 1 — Load inputs
        LOGGER.info("-- Step 1: Loading accident features + cell aggregates + raw weather --")
        accidents = load_accident_features(db)
        cell_aggregates_df = pd.DataFrame(list(
            db[COL_CELL_AGGREGATES].find({"h3_resolution": 8}, {"_id": 0})
        ))
        LOGGER.info("Loaded %d cell aggregates for reference (res=8)", len(cell_aggregates_df))
        weather = load_raw_weather(db)

        # Step 2 — LEAKAGE-FIXED label derivation
        LOGGER.info("-- Step 2: Deriving labels from training window only --")
        labels = compute_training_window_labels(accidents)

        # Step 3 — Historical behavioural shares (per-cell, training window)
        LOGGER.info("-- Step 3: Computing historical behavioural shares --")
        historical_shares = compute_historical_behavioural_shares(accidents)

        # Step 4 — Static cell context
        LOGGER.info("-- Step 4: Deriving static cell context --")
        static_context = compute_static_cell_context(accidents)

        # Step 5 — Panel skeleton
        LOGGER.info("-- Step 5: Building panel skeleton --")
        cells = accidents[CELL_COLUMN].dropna().unique().tolist()
        panel = build_panel_skeleton(cells)

        # Step 6 — Attach labels + static + historical + reference counts
        LOGGER.info("-- Step 6: Attaching labels and per-cell context --")
        panel = attach_labels_and_context(
            panel, labels, static_context, historical_shares, cell_aggregates_df
        )

        # Step 7 — Causal rolling exposure (unchanged; already leakage-safe)
        LOGGER.info("-- Step 7: Computing causal rolling exposure features --")
        panel = compute_rolling_exposure(panel, accidents, ROLLING_WINDOWS_DAYS)

        # Step 8 — Cyclical calendar
        LOGGER.info("-- Step 8: Adding cyclical calendar features --")
        panel = add_cyclical_calendar(panel)

        # Step 9 — LEAKAGE-FIXED monthly weather from raw_weather
        LOGGER.info("-- Step 9: Merging monthly weather aggregates from raw_weather --")
        monthly_weather = compute_monthly_weather_from_raw(weather)
        panel = panel.merge(monthly_weather, on="year_month", how="left")
        for col in ["avg_precip_month", "avg_humidity_month",
                    "avg_visibility_month", "avg_cloudcover_month",
                    "rainy_days_share_month"]:
            if col in panel.columns and panel[col].isna().any():
                fill_val = panel[col].mean()
                LOGGER.warning("Filling %d NaNs in %s with dataset mean=%.4f",
                               int(panel[col].isna().sum()), col, fill_val)
                panel[col] = panel[col].fillna(fill_val)

        # Step 10 — Holidays in month
        LOGGER.info("-- Step 10: Adding holidays_in_month --")
        holidays_month = compute_holidays_in_month(panel)
        panel = panel.merge(holidays_month, on="year_month", how="left")
        panel["holidays_in_month"] = panel["holidays_in_month"].fillna(0).astype(int)

        # Step 11 — Finalize
        LOGGER.info("-- Step 11: Finalizing panel --")
        panel["built_at"] = datetime.utcnow().isoformat()
        panel["label_training_window_end"] = LABEL_TRAINING_WINDOW_END.strftime("%Y-%m-%d")
        panel = panel.drop(columns=["month_start"])

        # Step 12 — Persist
        LOGGER.info("-- Step 12: Persisting panel --")
        CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(CSV_OUTPUT_PATH, index=False)
        LOGGER.info("CSV written: %s (%d rows, %d cols)",
                    CSV_OUTPUT_PATH, len(panel), len(panel.columns))

        save_panel(db, panel)

        # ── Sanity summary ────────────────────────────────────────────────
        LOGGER.info("--- Panel summary ---")
        LOGGER.info("Rows: %d | Unique cells: %d | Unique months: %d",
                    len(panel), panel["h3_cell"].nunique(), panel["year_month"].nunique())
        LOGGER.info("Label training window ends at: %s", LABEL_TRAINING_WINDOW_END.date())
        LOGGER.info("Hotspot cells (training-window Poisson-FDR): %d",
                    panel.drop_duplicates("h3_cell")["is_hotspot"].sum())
        LOGGER.info("Dominant severity distribution (per-cell view):")
        LOGGER.info("\n%s",
                    panel.drop_duplicates("h3_cell")["dominant_severity"]
                    .value_counts(dropna=False).to_string())
        LOGGER.info("=== Training Panel Build Complete ===")
        return panel
    finally:
        client.close()


if __name__ == "__main__":
    build_panel()