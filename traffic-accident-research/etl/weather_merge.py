"""Weather extraction and merge for the traffic accident research pipeline.

Reads the raw weather records from MongoDB (raw_weather collection),
derives accident-risk labels and light condition, then merges onto the
geocoded accident dataset (processed_accidents_with_coords collection)
and saves the result to processed_accidents_with_weather collection.

MongoDB Collections:
    raw_weather                      — source Visual Crossing weather records
    processed_accidents_with_coords  — accidents enriched with lat/lng (input)
    processed_accidents_with_weather — accidents enriched with weather (output)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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
COL_RAW_WEATHER = "raw_weather"
COL_INPUT_ACCIDENTS = "processed_accidents_with_coords"
COL_OUTPUT_ACCIDENTS = "processed_accidents_with_weather"

# ── Column Config ──────────────────────────────────────────────────────────
WEATHER_DATE_COLUMN = "datetime"
ACCIDENT_DATE_COLUMN = "Date of Accident"

WEATHER_KEEP_COLUMNS = [
    WEATHER_DATE_COLUMN,
    "conditions",
    "precip",
    "visibility",
    "humidity",
    "cloudcover",
    "sunrise",
    "sunset",
]

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


def load_raw_weather(db) -> pd.DataFrame:
    """Load raw weather records from MongoDB and keep only required columns."""

    records = list(db[COL_RAW_WEATHER].find({}, {"_id": 0}))
    if not records:
        raise ValueError(f"No records found in collection: {COL_RAW_WEATHER}")

    dataframe = pd.DataFrame(records)

    available = [c for c in WEATHER_KEEP_COLUMNS if c in dataframe.columns]
    missing = [c for c in WEATHER_KEEP_COLUMNS if c not in dataframe.columns]
    if missing:
        LOGGER.warning("Weather collection missing columns (will skip): %s", missing)

    weather = dataframe[available].copy()
    weather[WEATHER_DATE_COLUMN] = pd.to_datetime(
        weather[WEATHER_DATE_COLUMN], errors="coerce"
    ).dt.normalize()
    weather = weather.dropna(subset=[WEATHER_DATE_COLUMN]).drop_duplicates(
        subset=[WEATHER_DATE_COLUMN]
    )

    LOGGER.info(
        "Loaded %d weather records from MongoDB [%s]", len(weather), COL_RAW_WEATHER
    )
    return weather


def load_accidents_with_coords(db) -> pd.DataFrame:
    """Load geocoded accident records from MongoDB."""

    records = list(db[COL_INPUT_ACCIDENTS].find({}, {"_id": 0}))
    if not records:
        raise ValueError(
            f"No records found in collection: {COL_INPUT_ACCIDENTS}. "
            "Run geocode_locations.py first."
        )

    dataframe = pd.DataFrame(records)
    if ACCIDENT_DATE_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Accident collection missing required column: {ACCIDENT_DATE_COLUMN}"
        )

    dataframe[ACCIDENT_DATE_COLUMN] = pd.to_datetime(
        dataframe[ACCIDENT_DATE_COLUMN], errors="coerce"
    ).dt.normalize()
    dataframe = dataframe.dropna(subset=[ACCIDENT_DATE_COLUMN])

    LOGGER.info(
        "Loaded %d accident records from MongoDB [%s]",
        len(dataframe),
        COL_INPUT_ACCIDENTS,
    )
    return dataframe


def save_accidents_with_weather(db, dataframe: pd.DataFrame) -> None:
    """Upsert weather-enriched accident records into MongoDB."""

    records = dataframe.where(pd.notna(dataframe), None).to_dict(orient="records")

    operations = [
        UpdateOne(
            {"_lake_id": r["_lake_id"]},
            {"$set": r},
            upsert=True,
        )
        for r in records
        if r.get("_lake_id") is not None
    ]

    result = db[COL_OUTPUT_ACCIDENTS].bulk_write(operations)
    LOGGER.info(
        "[%s] Upserted: %d | Modified: %d | Total: %d",
        COL_OUTPUT_ACCIDENTS,
        result.upserted_count,
        result.modified_count,
        db[COL_OUTPUT_ACCIDENTS].count_documents({}),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Weather feature derivation
# ══════════════════════════════════════════════════════════════════════════════

def derive_weather_condition_label(conditions: object, precip: object) -> str:
    """Map raw conditions + precip amount into an accident-risk label."""

    conditions_text = str(conditions).strip().lower()
    precip_value = pd.to_numeric(precip, errors="coerce")
    precip_value = 0 if pd.isna(precip_value) else precip_value

    # Only classify as rain if it actually rained
    if "rain" in conditions_text and precip_value > 0:
        return "Heavy Rain" if precip_value >= 5 else "Light Rain"

    if "overcast" in conditions_text:
        return "Overcast"

    if "partially cloudy" in conditions_text:
        return "Partially Cloudy"

    if "clear" in conditions_text:
        return "Clear"

    return "Other"


def derive_visibility_level(visibility: object) -> str:
    """Bucket raw visibility (km) into Poor / Moderate / Good."""

    value = pd.to_numeric(visibility, errors="coerce")
    if pd.isna(value):
        return "Unknown"
    if value < 3:
        return "Poor"
    if value <= 6:
        return "Moderate"
    return "Good"


def derive_light_condition(
    accident_time_str: object,
    sunrise_str: object,
    sunset_str: object,
) -> str:
    """Derive Day / Night by comparing accident time against sunrise and sunset.

    Visual Crossing provides exact sunrise/sunset ISO timestamps per day,
    making this more accurate than fixed time-of-day thresholds.
    """

    try:
        acc_time = pd.to_datetime(str(accident_time_str), errors="coerce")
        sunrise = pd.to_datetime(str(sunrise_str), errors="coerce")
        sunset = pd.to_datetime(str(sunset_str), errors="coerce")

        if pd.isna(acc_time) or pd.isna(sunrise) or pd.isna(sunset):
            return "Unknown"

        # Compare time-of-day only — date part is irrelevant
        acc_t = acc_time.time()
        rise_t = sunrise.time()
        set_t = sunset.time()

        return "Day" if rise_t <= acc_t <= set_t else "Night"

    except Exception:
        return "Unknown"


def prepare_weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Derive accident-risk labels from raw weather columns."""

    features = weather.copy()

    features["weather_condition_label"] = features.apply(
        lambda row: derive_weather_condition_label(
            row.get("conditions"), row.get("precip")
        ),
        axis=1,
    )
    features["visibility_level"] = features["visibility"].apply(derive_visibility_level)

    # Rename date column to avoid clash during merge
    features = features.rename(columns={WEATHER_DATE_COLUMN: "weather_date"})
    return features


# ══════════════════════════════════════════════════════════════════════════════
# Merge
# ══════════════════════════════════════════════════════════════════════════════

def merge_weather_with_accidents(
    accidents: pd.DataFrame,
    weather_features: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join daily weather onto the accident dataset by accident date."""

    merged = accidents.merge(
        weather_features,
        how="left",
        left_on=ACCIDENT_DATE_COLUMN,
        right_on="weather_date",
    ).drop(columns=["weather_date"], errors="ignore")

    # Derive light condition now that we have both accident time and sunrise/sunset
    if "sunrise" in merged.columns and "sunset" in merged.columns:
        merged["light_condition"] = merged.apply(
            lambda row: derive_light_condition(
                row.get("Time of Accident"),
                row.get("sunrise"),
                row.get("sunset"),
            ),
            axis=1,
        )
    else:
        LOGGER.warning("sunrise/sunset columns not found — light_condition will be Unknown")
        merged["light_condition"] = "Unknown"

    matched = merged["weather_condition_label"].notna().sum()
    LOGGER.info("Weather matched: %d/%d accident records", matched, len(merged))

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the weather extraction and merge pipeline."""

    configure_logging()
    LOGGER.info("=== Weather Merge Pipeline Start ===")

    db, client = get_db()

    try:
        # Step 1 — Load raw weather from MongoDB
        LOGGER.info("-- Step 1: Loading raw weather from MongoDB --")
        weather_data = load_raw_weather(db)

        # Step 2 — Derive weather feature labels
        LOGGER.info("-- Step 2: Deriving weather features --")
        weather_features = prepare_weather_features(weather_data)

        # Step 3 — Load geocoded accidents from MongoDB
        LOGGER.info("-- Step 3: Loading processed_accidents_with_coords from MongoDB --")
        accidents = load_accidents_with_coords(db)

        # Step 4 — Merge weather onto accidents
        LOGGER.info("-- Step 4: Merging weather onto accidents --")
        merged = merge_weather_with_accidents(accidents, weather_features)

        # Step 5 — Save to MongoDB
        LOGGER.info("-- Step 5: Saving processed_accidents_with_weather to MongoDB --")
        save_accidents_with_weather(db, merged)

        LOGGER.info("=== Weather Merge Pipeline Complete ===")

    finally:
        client.close()


if __name__ == "__main__":
    main()