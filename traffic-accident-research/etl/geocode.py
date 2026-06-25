"""Geocode accident locations and maintain a reusable location master in MongoDB.

This module reads raw accident records from the MongoDB data lake collection,
extracts unique location names, standardizes them for Colombo-area geocoding,
and calls the Google Maps Geocoding API to persist latitude, longitude, and
formatted addresses.

The script is designed to be incremental:
- Existing geocoded locations are loaded from MongoDB and skipped.
- New locations are geocoded and upserted into the location master collection.
- Failures are tracked in a separate MongoDB collection for later review.
- At the end, the enriched dataset (accidents + coordinates) is always saved
  to the processed_accidents_with_coords MongoDB collection.

MongoDB Collections:
    raw_accidents                    — source accident records (data lake)
    ref_location_master              — geocoded location cache (reusable reference)
    ref_failed_locations             — failed geocoding attempts
    processed_accidents_with_coords  — accidents enriched with lat/lng
"""

from __future__ import annotations

import importlib.util
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import requests
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

# ── Environment ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / "config" / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "accident_hotspot")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# ── MongoDB Collection Names ───────────────────────────────────────────────
COL_RAW_ACCIDENTS = "raw_accidents"
COL_LOCATION_MASTER = "ref_location_master"
COL_FAILED_LOCATIONS = "ref_failed_locations"
COL_PROCESSED_WITH_COORDS = "processed_accidents_with_coords"

# ── Geocoding Config ───────────────────────────────────────────────────────
DEFAULT_REQUEST_TIMEOUT_SECONDS = float(os.getenv("GEOCODE_REQUEST_TIMEOUT_SECONDS", "10"))
DEFAULT_REQUEST_DELAY_SECONDS = float(os.getenv("GEOCODE_REQUEST_DELAY_SECONDS", "0.2"))
MAX_GEOCODE_RETRIES = int(os.getenv("GEOCODE_MAX_RETRIES", "3"))

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure a concise, production-friendly logging setup."""

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


def load_raw_accidents(db) -> pd.DataFrame:
    """Load raw accident records from MongoDB into a DataFrame."""

    records = list(db[COL_RAW_ACCIDENTS].find({}, {"_id": 0}))
    if not records:
        raise ValueError(f"No records found in collection: {COL_RAW_ACCIDENTS}")

    dataframe = pd.DataFrame(records)
    if "Location" not in dataframe.columns:
        raise ValueError("raw_accidents collection must contain a 'Location' field.")

    LOGGER.info("Loaded %d accident records from MongoDB [%s]", len(dataframe), COL_RAW_ACCIDENTS)
    return dataframe


def load_existing_location_master(db) -> Dict[str, dict]:
    """Load already-geocoded locations from MongoDB into a dict keyed by location_name."""

    records = list(db[COL_LOCATION_MASTER].find({}, {"_id": 0}))
    master = {r["location_name"]: r for r in records if "location_name" in r}
    LOGGER.info(
        "Loaded %d existing geocoded locations from MongoDB [%s]",
        len(master),
        COL_LOCATION_MASTER,
    )
    return master


def upsert_location_master(db, records: List[dict]) -> None:
    """Upsert geocoded location records into the location master collection."""

    if not records:
        return

    operations = [
        UpdateOne(
            {"location_name": r["location_name"]},
            {"$set": r},
            upsert=True,
        )
        for r in records
    ]
    result = db[COL_LOCATION_MASTER].bulk_write(operations)
    LOGGER.info(
        "[%s] Upserted: %d | Modified: %d",
        COL_LOCATION_MASTER,
        result.upserted_count,
        result.modified_count,
    )


def save_failed_locations(db, records: List[dict]) -> None:
    """Insert failed geocoding attempts into the failed locations collection."""

    if not records:
        return

    db[COL_FAILED_LOCATIONS].insert_many(records)
    LOGGER.info(
        "Saved %d failed geocoding attempts to [%s]",
        len(records),
        COL_FAILED_LOCATIONS,
    )


def save_processed_with_coords(db, dataframe: pd.DataFrame) -> None:
    """Upsert the enriched accident dataset into the processed collection."""

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

    result = db[COL_PROCESSED_WITH_COORDS].bulk_write(operations)
    LOGGER.info(
        "[%s] Upserted: %d | Modified: %d | Total: %d",
        COL_PROCESSED_WITH_COORDS,
        result.upserted_count,
        result.modified_count,
        db[COL_PROCESSED_WITH_COORDS].count_documents({}),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Location standardization
# ══════════════════════════════════════════════════════════════════════════════

def standardize_location(location_name: object) -> Optional[str]:
    """Normalize a raw location name and append Colombo, Sri Lanka context."""

    if pd.isna(location_name):
        return None

    normalized = " ".join(str(location_name).strip().split())
    if not normalized:
        return None

    suffix = ", Colombo, Sri Lanka"
    if normalized.lower().endswith(suffix.lower()):
        return normalized

    return f"{normalized}{suffix}"


def extract_unique_locations(dataframe: pd.DataFrame) -> List[str]:
    """Extract unique standardized locations preserving first-seen order."""

    seen: Set[str] = set()
    unique_locations: List[str] = []

    for raw_location in dataframe["Location"].tolist():
        standardized = standardize_location(raw_location)
        if not standardized or standardized in seen:
            continue
        seen.add(standardized)
        unique_locations.append(standardized)

    return unique_locations


# ══════════════════════════════════════════════════════════════════════════════
# Geocoding
# ══════════════════════════════════════════════════════════════════════════════

def geocode_location(
    location_name: str,
    api_key: str,
    session: requests.Session,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    max_retries: int = MAX_GEOCODE_RETRIES,
) -> dict:
    """Geocode a single standardized location using the Google Maps API."""

    if not location_name:
        raise ValueError("Location name cannot be empty.")
    if not api_key:
        raise ValueError("Google Maps API key is required.")

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": location_name, "key": api_key, "region": "lk"}

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()

            status = payload.get("status")
            if status != "OK":
                error_message = payload.get("error_message") or status or "Unknown API error"
                if status == "ZERO_RESULTS":
                    raise ValueError(f"No geocoding result for location: {location_name}")
                if status in {"OVER_QUERY_LIMIT", "UNKNOWN_ERROR"} and attempt < max_retries:
                    LOGGER.warning(
                        "API returned %s for %s — retrying (%d/%d).",
                        status, location_name, attempt, max_retries,
                    )
                    time.sleep(min(2 ** (attempt - 1), 5))
                    continue
                raise RuntimeError(f"Geocoding API error for {location_name}: {error_message}")

            results = payload.get("results") or []
            if not results:
                raise ValueError(f"Empty results payload for location: {location_name}")

            best_match = results[0]
            geometry = best_match.get("geometry", {}).get("location", {})
            latitude = geometry.get("lat")
            longitude = geometry.get("lng")
            formatted_address = best_match.get("formatted_address")

            if latitude is None or longitude is None or not formatted_address:
                raise ValueError(f"Incomplete geocoding result for location: {location_name}")

            return {
                "location_name": location_name,
                "formatted_address": formatted_address,
                "latitude": latitude,
                "longitude": longitude,
                "geocoded_at": datetime.now(timezone.utc).isoformat(),
            }

        except requests.RequestException as exc:
            last_error = exc
            if attempt < max_retries:
                LOGGER.warning(
                    "Network error geocoding %s on attempt %d/%d: %s",
                    location_name, attempt, max_retries, exc,
                )
                time.sleep(min(2 ** (attempt - 1), 5))
                continue
            raise RuntimeError(f"Network error geocoding {location_name}: {exc}") from exc

        except (ValueError, RuntimeError):
            raise

        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                LOGGER.warning(
                    "Unexpected error geocoding %s on attempt %d/%d: %s",
                    location_name, attempt, max_retries, exc,
                )
                time.sleep(min(2 ** (attempt - 1), 5))
                continue
            raise RuntimeError(
                f"Unexpected geocoding failure for {location_name}: {exc}"
            ) from exc

    raise RuntimeError(
        f"Geocoding failed for {location_name} after {max_retries} attempts: {last_error}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Merge and save enriched dataset
# ══════════════════════════════════════════════════════════════════════════════

def merge_and_save_enriched_dataset(db, accidents: pd.DataFrame) -> None:
    """Merge coords from ref_location_master onto accidents and save to MongoDB."""

    location_master = load_existing_location_master(db)
    if not location_master:
        LOGGER.warning("Location master is empty — skipping merge.")
        return

    master_df = pd.DataFrame(location_master.values())

    accidents = accidents.copy()
    accidents["_location_standardized"] = accidents["Location"].apply(standardize_location)

    enriched = accidents.merge(
        master_df[["location_name", "formatted_address", "latitude", "longitude"]],
        left_on="_location_standardized",
        right_on="location_name",
        how="left",
    ).drop(columns=["_location_standardized", "location_name"], errors="ignore")

    matched = enriched["latitude"].notna().sum()
    LOGGER.info(
        "Coordinates matched: %d/%d records",
        matched,
        len(enriched),
    )

    save_processed_with_coords(db, enriched)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the incremental geocoding workflow."""

    configure_logging()

    if not GOOGLE_MAPS_API_KEY:
        raise ValueError("GOOGLE_MAPS_API_KEY is not set. Check your config/.env file.")

    LOGGER.info("=== Geocoding Pipeline Start ===")
    db, client = get_db()

    try:
        # Step 1 — Load raw accidents from MongoDB
        LOGGER.info("-- Step 1: Loading raw accidents from MongoDB --")
        accident_data = load_raw_accidents(db)

        # Step 2 — Extract unique locations, skip already geocoded
        LOGGER.info("-- Step 2: Checking location master in MongoDB --")
        unique_locations = extract_unique_locations(accident_data)
        LOGGER.info("Found %d unique standardized locations.", len(unique_locations))

        existing_master = load_existing_location_master(db)
        locations_to_geocode = [
            loc for loc in unique_locations if loc not in existing_master
        ]
        LOGGER.info(
            "%d already geocoded | %d new to process",
            len(existing_master),
            len(locations_to_geocode),
        )

        # Step 3 — Geocode new locations
        if not locations_to_geocode:
            LOGGER.info("Location master is up to date — skipping geocoding.")
        else:
            LOGGER.info("-- Step 3: Geocoding new locations --")
            successful_rows: List[dict] = []
            failed_rows: List[dict] = []
            session = requests.Session()

            try:
                for index, location_name in enumerate(locations_to_geocode, start=1):
                    LOGGER.info(
                        "Geocoding %d/%d: %s",
                        index, len(locations_to_geocode), location_name,
                    )
                    try:
                        result = geocode_location(
                            location_name=location_name,
                            api_key=GOOGLE_MAPS_API_KEY,
                            session=session,
                            timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
                            max_retries=MAX_GEOCODE_RETRIES,
                        )
                        successful_rows.append(result)
                        LOGGER.info("  -> (%s, %s)", result["latitude"], result["longitude"])
                    except Exception as exc:
                        failed_rows.append({
                            "location_name": location_name,
                            "error_message": str(exc),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        LOGGER.warning("  Failed: %s", exc)

                    if index < len(locations_to_geocode) and DEFAULT_REQUEST_DELAY_SECONDS > 0:
                        time.sleep(DEFAULT_REQUEST_DELAY_SECONDS)
            finally:
                session.close()

            # Step 4 — Save to MongoDB
            LOGGER.info("-- Step 4: Saving results to MongoDB --")
            upsert_location_master(db, successful_rows)
            save_failed_locations(db, failed_rows)
            LOGGER.info(
                "Geocoding done — %d succeeded | %d failed",
                len(successful_rows),
                len(failed_rows),
            )

        # Step 5 — Merge and save processed collection
        LOGGER.info("-- Step 5: Saving processed_accidents_with_coords to MongoDB --")
        merge_and_save_enriched_dataset(db, accident_data)

        LOGGER.info("=== Geocoding Pipeline Complete ===")

    finally:
        client.close()


if __name__ == "__main__":
    main() 