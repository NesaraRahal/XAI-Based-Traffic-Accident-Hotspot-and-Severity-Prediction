"""Ingest raw data  files into MongoDB Atlas as separate raw collections to create a data lake.

Data Lake structure:
    data/lake/accidents/  ← raw accident JSON (from CSV, no processing)
    data/lake/weather/    ← raw weather JSON (from Visual Crossing CSV)

MongoDB Collections:
    raw_accidents   ← untouched accident records
    raw_weather     ← untouched Visual Crossing records
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
LAKE_ACCIDENTS_DIR = ROOT_DIR / "data" / "lake" / "accidents"
LAKE_WEATHER_DIR = ROOT_DIR / "data" / "lake" / "weather"

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "accident_hotspot")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


# ── Step 1: CSV → JSON into data lake ──────────────────────────────────────

def csv_to_lake(csv_path: Path, lake_dir: Path, filename: str) -> list:
    """Convert a raw CSV into a JSON file in the data lake. No processing."""
    lake_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    # NaN → None so JSON serializes cleanly
    records = df.where(pd.notna(df), None).to_dict(orient="records")

    out_path = lake_dir / filename
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    LOGGER.info("Lake file written: %s (%d records)", out_path, len(records))
    return records


# ── Step 2: JSON lake → MongoDB ────────────────────────────────────────────

def upsert_collection(db, collection_name: str, records: list, id_field: str) -> None:
    """Upsert records into a MongoDB collection using a unique ID field."""
    collection = db[collection_name]

    operations = [
        UpdateOne(
            {id_field: r[id_field]},
            {"$set": r},
            upsert=True,
        )
        for r in records
        if r.get(id_field) is not None
    ]

    result = collection.bulk_write(operations)
    LOGGER.info(
        "[%s] Upserted: %d | Modified: %d | Total in collection: %d",
        collection_name,
        result.upserted_count,
        result.modified_count,
        collection.count_documents({}),
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    LOGGER.info("=== Data Lake → MongoDB Ingest Start ===")

    # Paths to your raw CSVs
    accidents_csv = ROOT_DIR / "data" / "raw" / "synthetic_cma_accidents_2022_2023.csv"
    weather_csv   = ROOT_DIR / "data" / "raw" / "Colombo_weather_2022_to_2026.csv"

    # Step 1 — Write raw CSVs into the data lake as JSON (no changes to data)
    LOGGER.info("-- Writing accidents to data lake --")
    accident_records = csv_to_lake(
        accidents_csv,
        LAKE_ACCIDENTS_DIR,
        "traffic_accidents_colombo.json"
    )

    LOGGER.info("-- Writing weather to data lake --")
    weather_records = csv_to_lake(
        weather_csv,
        LAKE_WEATHER_DIR,
        "colombo_weather_2022_2026.json"
    )

    # Step 2 — Load from lake into MongoDB
    LOGGER.info("-- Connecting to MongoDB Atlas --")
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]

    LOGGER.info("-- Upserting accidents --")
    # Using Location + Date + Time as a composite key via a generated record_id
    for i, r in enumerate(accident_records):
        r["_lake_id"] = f"ACC-{i+1:05d}"

    upsert_collection(db, "raw_accidents", accident_records, "_lake_id")

    LOGGER.info("-- Upserting weather --")
    upsert_collection(db, "raw_weather", weather_records, "datetime")

    client.close()
    LOGGER.info("=== Done! ===")


if __name__ == "__main__":
    main()