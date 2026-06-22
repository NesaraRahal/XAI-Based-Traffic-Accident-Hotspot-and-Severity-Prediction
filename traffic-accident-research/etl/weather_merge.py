"""Weather extraction and merge for the traffic accident research pipeline.

This module reads the daily Colombo weather dataset, keeps the columns needed
for accident-risk analysis, derives a few interpretable labels, and merges the
result onto the geocoded accident dataset by accident date.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
WEATHER_SOURCE_PATH = ROOT_DIR / "data" / "raw" / "Colombo_weather_2022_to_2026.csv"
ACCIDENTS_SOURCE_PATH = ROOT_DIR / "data" / "processed" / "accidents_with_coords.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "accidents_with_weather.csv"

WEATHER_DATE_COLUMN = "datetime"
ACCIDENT_DATE_COLUMN = "Date of Accident"

KEEP_COLUMNS = [
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


def load_weather_data(source_path: Path = WEATHER_SOURCE_PATH) -> pd.DataFrame:
	"""Load the Colombo weather dataset and keep the required weather columns."""

	if not source_path.exists():
		raise FileNotFoundError(f"Weather dataset not found: {source_path}")

	dataframe = pd.read_csv(source_path)
	missing_columns = [column for column in KEEP_COLUMNS if column not in dataframe.columns]
	if missing_columns:
		raise ValueError(f"Weather dataset is missing required columns: {missing_columns}")

	weather = dataframe[KEEP_COLUMNS].copy()
	weather[WEATHER_DATE_COLUMN] = pd.to_datetime(weather[WEATHER_DATE_COLUMN], errors="coerce").dt.normalize()
	weather = weather.dropna(subset=[WEATHER_DATE_COLUMN]).drop_duplicates(subset=[WEATHER_DATE_COLUMN])
	return weather


def load_accident_data(source_path: Path = ACCIDENTS_SOURCE_PATH) -> pd.DataFrame:
	"""Load the geocoded accident dataset that will receive the weather fields."""

	if not source_path.exists():
		raise FileNotFoundError(f"Accident dataset not found: {source_path}")

	dataframe = pd.read_csv(source_path)
	if ACCIDENT_DATE_COLUMN not in dataframe.columns:
		raise ValueError(f"Accident dataset is missing required column: {ACCIDENT_DATE_COLUMN}")

	accidents = dataframe.copy()
	accidents[ACCIDENT_DATE_COLUMN] = pd.to_datetime(accidents[ACCIDENT_DATE_COLUMN], errors="coerce").dt.normalize()
	accidents = accidents.dropna(subset=[ACCIDENT_DATE_COLUMN])
	return accidents


def derive_weather_condition_label(conditions: object, precip: object) -> str:
    conditions_text = str(conditions).strip().lower()
    precip_value = pd.to_numeric(precip, errors="coerce")
    precip_value = 0 if pd.isna(precip_value) else precip_value

    # Only classify as rain if it actually rained
    if "rain" in conditions_text and precip_value > 0:
        if precip_value >= 5:
            return "Heavy Rain"
        return "Light Rain"

    if "overcast" in conditions_text:
        return "Overcast"

    if "partially cloudy" in conditions_text:
        return "Partially Cloudy"

    if "clear" in conditions_text:
        return "Clear"

    return "Other"

def derive_visibility_level(visibility: object) -> str:
	"""Bucket visibility into a simple accident-risk category."""

	visibility_value = pd.to_numeric(visibility, errors="coerce")
	if pd.isna(visibility_value):
		return "Unknown"
	if visibility_value < 3:
		return "Poor"
	if visibility_value <= 6:
		return "Moderate"
	return "Good"


def prepare_weather_features(weather: pd.DataFrame) -> pd.DataFrame:
	"""Extract and derive the weather columns used by the accident dataset."""

	features = weather.copy()
	features["weather_condition_label"] = features.apply(
		lambda row: derive_weather_condition_label(row["conditions"], row["precip"]),
		axis=1,
	)
	features["visibility_level"] = features["visibility"].apply(derive_visibility_level)
	features = features.rename(columns={WEATHER_DATE_COLUMN: "weather_date"})
	return features


def merge_weather_with_accidents(accidents: pd.DataFrame, weather_features: pd.DataFrame) -> pd.DataFrame:
	"""Merge daily weather data onto the accident dataset using the accident date."""

	merged = accidents.merge(
		weather_features,
		how="left",
		left_on=ACCIDENT_DATE_COLUMN,
		right_on="weather_date",
	)
	return merged.drop(columns=["weather_date"])


def save_merged_data(dataframe: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
	"""Persist the weather-enriched accident dataset to disk."""

	output_path.parent.mkdir(parents=True, exist_ok=True)
	dataframe.to_csv(output_path, index=False)
	LOGGER.info("Saved weather-enriched dataset to %s", output_path)


def main() -> None:
	"""Run the weather extraction and merge pipeline."""

	configure_logging()
	weather_data = load_weather_data()
	weather_features = prepare_weather_features(weather_data)
	accidents = load_accident_data()
	merged = merge_weather_with_accidents(accidents, weather_features)
	save_merged_data(merged)


if __name__ == "__main__":
	main()
