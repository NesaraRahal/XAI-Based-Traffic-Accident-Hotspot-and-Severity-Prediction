"""Geocode accident locations and maintain a reusable location master table.

This module reads the accident dataset, extracts unique location names,
standardizes them for Colombo-area geocoding, and calls the Google Maps
Geocoding API to persist latitude, longitude, and formatted addresses.

The script is designed to be incremental:
- Existing geocoded locations are loaded and skipped.
- New locations are geocoded and appended to the master table.
- Failures are tracked separately for later review.
- At the end, the enriched dataset (accidents + coordinates) is always saved.
"""

from __future__ import annotations

import logging
import importlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

_dotenv_module = importlib.import_module("dotenv") if importlib.util.find_spec("dotenv") else None


def load_dotenv(*args, **kwargs):
	"""Load environment variables from a .env file when python-dotenv is available."""

	if _dotenv_module is None:
		return False
	return _dotenv_module.load_dotenv(*args, **kwargs)


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "accidents_colombo_expanded.csv"
REFERENCE_DIR = ROOT_DIR / "data" / "reference"
LOCATION_MASTER_PATH = REFERENCE_DIR / "location_master.csv"
FAILED_LOCATIONS_PATH = REFERENCE_DIR / "failed_locations.csv"
ENV_PATH = ROOT_DIR / "config" / ".env"

LOCATION_MASTER_COLUMNS = ["location_name", "formatted_address", "latitude", "longitude"]
FAILED_LOCATION_COLUMNS = ["location_name", "error_message", "timestamp"]

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


def load_accident_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
	"""Load the source accident dataset from disk."""

	if not file_path.exists():
		raise FileNotFoundError(f"Accident dataset not found: {file_path}")

	dataframe = pd.read_csv(file_path)
	if dataframe.empty:
		raise ValueError(f"Accident dataset is empty: {file_path}")

	if "Location" not in dataframe.columns:
		raise ValueError("Accident dataset must contain a 'Location' column.")

	return dataframe


def load_existing_location_master(file_path: Path = LOCATION_MASTER_PATH) -> pd.DataFrame:
	"""Load any existing geocoded locations from the master table.

	Returns an empty frame with the expected schema when the file does not exist.
	"""

	if not file_path.exists():
		return pd.DataFrame(columns=LOCATION_MASTER_COLUMNS)

	dataframe = pd.read_csv(file_path)
	missing_columns = [column for column in LOCATION_MASTER_COLUMNS if column not in dataframe.columns]
	if missing_columns:
		raise ValueError(
			f"Existing location master is missing required columns: {missing_columns}"
		)

	return dataframe[LOCATION_MASTER_COLUMNS].copy()


def load_existing_failed_locations(file_path: Path = FAILED_LOCATIONS_PATH) -> pd.DataFrame:
	"""Load previously failed geocoding attempts if they exist."""

	if not file_path.exists():
		return pd.DataFrame(columns=FAILED_LOCATION_COLUMNS)

	dataframe = pd.read_csv(file_path)
	missing_columns = [column for column in FAILED_LOCATION_COLUMNS if column not in dataframe.columns]
	if missing_columns:
		raise ValueError(
			f"Existing failed locations file is missing required columns: {missing_columns}"
		)

	return dataframe[FAILED_LOCATION_COLUMNS].copy()


def standardize_location(location_name: object) -> Optional[str]:
	"""Normalize a raw location name before geocoding.

	The function trims extra whitespace and appends the Colombo, Sri Lanka
	context when it is not already present.
	"""

	if pd.isna(location_name):
		return None

	normalized = " ".join(str(location_name).strip().split())
	if not normalized:
		return None

	suffix = ", Colombo, Sri Lanka"
	if normalized.lower().endswith(suffix.lower()):
		return normalized

	return f"{normalized}{suffix}"


def geocode_location(
	location_name: str,
	api_key: str,
	session: Optional[requests.Session] = None,
	timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
	max_retries: int = MAX_GEOCODE_RETRIES,
) -> dict:
	"""Geocode a single standardized location using the Google Maps API."""

	if not location_name:
		raise ValueError("Location name cannot be empty.")
	if not api_key:
		raise ValueError("Google Maps API key is required.")

	http_session = session or requests.Session()
	url = "https://maps.googleapis.com/maps/api/geocode/json"
	params = {"address": location_name, "key": api_key, "region": "lk"}

	last_error: Optional[Exception] = None
	for attempt in range(1, max_retries + 1):
		try:
			response = http_session.get(url, params=params, timeout=timeout_seconds)
			response.raise_for_status()
			payload = response.json()

			status = payload.get("status")
			if status != "OK":
				error_message = payload.get("error_message") or status or "Unknown API error"
				if status == "ZERO_RESULTS":
					raise ValueError(f"No geocoding result for location: {location_name}")
				if status in {"OVER_QUERY_LIMIT", "UNKNOWN_ERROR"} and attempt < max_retries:
					LOGGER.warning(
						"Geocoding API returned %s for %s. Retrying (%s/%s).",
						status,
						location_name,
						attempt,
						max_retries,
					)
					time.sleep(min(2 ** (attempt - 1), 5))
					continue
				raise RuntimeError(f"Geocoding API error for {location_name}: {error_message}")

			results = payload.get("results") or []
			if not results:
				raise ValueError(f"No geocoding result payload returned for location: {location_name}")

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
			}
		except requests.RequestException as exc:
			last_error = exc
			if attempt < max_retries:
				LOGGER.warning(
					"Network error geocoding %s on attempt %s/%s: %s",
					location_name,
					attempt,
					max_retries,
					exc,
				)
				time.sleep(min(2 ** (attempt - 1), 5))
				continue
			raise RuntimeError(f"Network error geocoding {location_name}: {exc}") from exc
		except (ValueError, RuntimeError):
			raise
		except Exception as exc:  # pragma: no cover - defensive fallback
			last_error = exc
			if attempt < max_retries:
				LOGGER.warning(
					"Unexpected error geocoding %s on attempt %s/%s: %s",
					location_name,
					attempt,
					max_retries,
					exc,
				)
				time.sleep(min(2 ** (attempt - 1), 5))
				continue
			raise RuntimeError(f"Unexpected geocoding failure for {location_name}: {exc}") from exc

	raise RuntimeError(f"Geocoding failed for {location_name}: {last_error}")


def save_location_master(dataframe: pd.DataFrame, file_path: Path = LOCATION_MASTER_PATH) -> None:
	"""Persist the location master table to disk."""

	file_path.parent.mkdir(parents=True, exist_ok=True)
	dataframe.to_csv(file_path, index=False)


def save_failed_locations(dataframe: pd.DataFrame, file_path: Path = FAILED_LOCATIONS_PATH) -> None:
	"""Persist failed geocoding attempts, appending to any existing file."""

	if dataframe.empty:
		return

	file_path.parent.mkdir(parents=True, exist_ok=True)
	if file_path.exists():
		existing = load_existing_failed_locations(file_path)
		dataframe = pd.concat([existing, dataframe], ignore_index=True)

	dataframe.to_csv(file_path, index=False)


def merge_and_save_enriched_dataset() -> None:
	"""Merge the accident dataset with the location master and save the result.

	This always runs at the end of main(), regardless of how many locations
	were geocoded in the current run.
	"""

	if not LOCATION_MASTER_PATH.exists():
		LOGGER.warning("No location master found — skipping merge.")
		return

	accidents = load_accident_data()
	location_master = load_existing_location_master()

	accidents["location_standardized"] = accidents["Location"].apply(standardize_location)

	enriched = accidents.merge(
		location_master,
		left_on="location_standardized",
		right_on="location_name",
		how="left",
	)

	output_path = ROOT_DIR / "data" / "processed" / "accidents_with_coords.csv"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	enriched.to_csv(output_path, index=False)
	LOGGER.info("Saved enriched dataset to %s", output_path)


def _extract_unique_locations(dataframe: pd.DataFrame) -> List[str]:
	"""Extract unique standardized locations while preserving first-seen order."""

	seen = set()
	unique_locations: List[str] = []

	for raw_location in dataframe["Location"].tolist():
		standardized_location = standardize_location(raw_location)
		if not standardized_location:
			continue
		if standardized_location in seen:
			continue
		seen.add(standardized_location)
		unique_locations.append(standardized_location)

	return unique_locations


def main() -> None:
	"""Run the incremental geocoding workflow."""

	configure_logging()
	load_dotenv(dotenv_path=ENV_PATH, override=False)

	api_key = os.getenv("GOOGLE_MAPS_API_KEY")
	if not api_key:
		LOGGER.error("Missing GOOGLE_MAPS_API_KEY. Set it in config/.env or the environment.")
		raise ValueError("Missing GOOGLE_MAPS_API_KEY.")

	LOGGER.info("Loading accident dataset from %s", RAW_DATA_PATH)
	accident_data = load_accident_data()
	unique_locations = _extract_unique_locations(accident_data)
	LOGGER.info("Found %s unique standardized locations.", len(unique_locations))

	existing_master = load_existing_location_master()
	existing_location_names = set(existing_master["location_name"].astype(str).tolist())
	locations_to_geocode = [location for location in unique_locations if location not in existing_location_names]

	LOGGER.info(
		"%s locations already geocoded; %s new locations to process.",
		len(existing_location_names),
		len(locations_to_geocode),
	)

	if not locations_to_geocode:
		LOGGER.info("No new locations to geocode. Location master is already up to date.")
		merge_and_save_enriched_dataset()
		return

	successful_rows = []
	failed_rows = []
	session = requests.Session()

	try:
		for index, location_name in enumerate(locations_to_geocode, start=1):
			LOGGER.info("Geocoding %s/%s: %s", index, len(locations_to_geocode), location_name)
			try:
				result = geocode_location(
					location_name=location_name,
					api_key=api_key,
					session=session,
					timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
					max_retries=MAX_GEOCODE_RETRIES,
				)
				successful_rows.append(result)
				LOGGER.info(
					"Geocoded %s -> (%s, %s)",
					location_name,
					result["latitude"],
					result["longitude"],
				)
			except Exception as exc:
				timestamp = datetime.now(timezone.utc).isoformat()
				error_message = str(exc)
				failed_rows.append(
					{
						"location_name": location_name,
						"error_message": error_message,
						"timestamp": timestamp,
					}
				)
				LOGGER.exception("Failed to geocode %s: %s", location_name, error_message)

			if index < len(locations_to_geocode) and DEFAULT_REQUEST_DELAY_SECONDS > 0:
				time.sleep(DEFAULT_REQUEST_DELAY_SECONDS)
	finally:
		session.close()

	successful_df = pd.DataFrame(successful_rows, columns=LOCATION_MASTER_COLUMNS)
	failed_df = pd.DataFrame(failed_rows, columns=FAILED_LOCATION_COLUMNS)

	if not successful_df.empty:
		combined_master = pd.concat([existing_master, successful_df], ignore_index=True)
		combined_master = combined_master.drop_duplicates(subset=["location_name"], keep="first")
		combined_master = combined_master[LOCATION_MASTER_COLUMNS]
		save_location_master(combined_master)
		LOGGER.info("Saved %s total geocoded locations to %s.", len(combined_master), LOCATION_MASTER_PATH)
	else:
		LOGGER.warning("No locations were successfully geocoded.")

	if not failed_df.empty:
		save_failed_locations(failed_df)
		LOGGER.info("Saved %s failed geocoding attempts to %s.", len(failed_df), FAILED_LOCATIONS_PATH)

	# Always merge at the end, regardless of success/failure above
	merge_and_save_enriched_dataset()


if __name__ == "__main__":
	main()