"""Feature engineering for the traffic accident research pipeline.

This module derives calendar-based accident features from the source dataset
or the geocoded/enriched dataset when available. It adds:
- Weekday name
- Month name
- Peak hour indicator
- Public holiday indicator with specific festive labels

Holiday detection is designed for Sri Lanka data, with support for local
holidays such as Poya days and Sinhala/Tamil New Year, plus globally observed
holidays such as Christmas, New Year's Day, Good Friday, and Easter Sunday.
If the `holidays` package is available, it is used to enrich the holiday
coverage for each year in the dataset.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from datetime import date, datetime, time as time_cls, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from dateutil.easter import easter


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "traffic_accidents_colombo.csv"
WEATHER_ENRICHED_DATA_PATH = ROOT_DIR / "data" / "processed" / "accidents_with_weather.csv"
ENRICHED_DATA_PATH = ROOT_DIR / "data" / "processed" / "accidents_with_coords.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "accidents_feature_engineered.csv"

DATE_COLUMN = "Date of Accident"
TIME_COLUMN = "Time of Accident"

OUTPUT_COLUMNS = [
	"Week (Derived)",
	"Month (Derived)",
	"Peak Hour Indicator (Derived)",
	"Public Holiday Indicator (Derived)",
	"Public Holiday Name (Derived)",
	"Public Holiday Category (Derived)",
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


def _load_holidays_module():
	"""Load the optional `holidays` package if it is installed."""

	if importlib.util.find_spec("holidays") is None:
		return None
	return importlib.import_module("holidays")


def load_source_data() -> pd.DataFrame:
	"""Load the best available accident dataset for feature engineering.

	Preference order:
	1. Weather-enriched dataset.
	2. Geocoded/enriched dataset with coordinates.
	3. Raw accident dataset.
	"""

	if WEATHER_ENRICHED_DATA_PATH.exists():
		source_path = WEATHER_ENRICHED_DATA_PATH
	elif ENRICHED_DATA_PATH.exists():
		source_path = ENRICHED_DATA_PATH
	else:
		source_path = RAW_DATA_PATH
	if not source_path.exists():
		raise FileNotFoundError(
			f"No accident dataset found at {WEATHER_ENRICHED_DATA_PATH}, {ENRICHED_DATA_PATH}, or {RAW_DATA_PATH}"
		)

	dataframe = pd.read_csv(source_path)
	if dataframe.empty:
		raise ValueError(f"Source accident dataset is empty: {source_path}")

	required_columns = [DATE_COLUMN, TIME_COLUMN]
	missing_columns = [column for column in required_columns if column not in dataframe.columns]
	if missing_columns:
		raise ValueError(f"Source dataset is missing required columns: {missing_columns}")

	LOGGER.info("Loaded %s rows from %s", len(dataframe), source_path)
	return dataframe


def standardize_date(value: object) -> pd.Timestamp:
	"""Parse a date value into a normalized pandas timestamp."""

	parsed = pd.to_datetime(value, errors="coerce")
	if pd.isna(parsed):
		raise ValueError(f"Invalid accident date: {value!r}")
	return parsed.normalize()


def standardize_time(value: object) -> time_cls:
	"""Parse a time value into a Python time object."""

	parsed = pd.to_datetime(str(value), errors="coerce")
	if pd.isna(parsed):
		raise ValueError(f"Invalid accident time: {value!r}")
	return parsed.time()


def get_weekday_name(accident_date: pd.Timestamp) -> str:
	"""Return the weekday name for the provided date."""

	return accident_date.day_name()


def get_month_name(accident_date: pd.Timestamp) -> str:
	"""Return the month name for the provided date."""

	return accident_date.month_name()


def is_peak_hour(accident_time: time_cls) -> bool:
	"""Check whether the time falls into a configurable traffic peak window.

	Default peak windows are:
	- Morning: 07:00 to 09:00
	- Evening: 16:00 to 19:00
	"""

	morning_start = time_cls(7, 0)
	morning_end = time_cls(9, 0)
	evening_start = time_cls(16, 0)
	evening_end = time_cls(19, 0)

	return (morning_start <= accident_time <= morning_end) or (evening_start <= accident_time <= evening_end)


def _iter_sri_lanka_holidays(years: Iterable[int]) -> Dict[date, List[str]]:
	"""Return holiday labels keyed by date for Sri Lanka and global observances.

	The function uses the `holidays` package when available and augments it with
	a few globally recognized holidays and Easter-based observances.
	"""

	holiday_map: Dict[date, List[str]] = {}
	holidays_module = _load_holidays_module()

	if holidays_module is not None:
		try:
			country_holidays = holidays_module.country_holidays
			for year in years:
				for holiday_date, holiday_name in country_holidays("LK", years=[year]).items():
					holiday_map.setdefault(holiday_date, []).append(str(holiday_name))
		except Exception as exc:  # pragma: no cover - fallback path only
			LOGGER.warning("Failed to load holidays package data: %s", exc)

	for year in years:
		holiday_map.setdefault(date(year, 1, 1), []).append("New Year's Day")
		holiday_map.setdefault(date(year, 12, 25), []).append("Christmas Day")
		holiday_map.setdefault(date(year, 5, 1), []).append("International Workers' Day")

		easter_sunday = easter(year)
		holiday_map.setdefault(easter_sunday, []).append("Easter Sunday")
		holiday_map.setdefault(easter_sunday - timedelta(days=2), []).append("Good Friday")

		# Sri Lankan schools and public life often treat Sinhala and Tamil New Year as
		# a two-day festive period.
		holiday_map.setdefault(date(year, 4, 13), []).append("Sinhala and Tamil New Year")
		holiday_map.setdefault(date(year, 4, 14), []).append("Sinhala and Tamil New Year")

	return holiday_map


def _clean_holiday_labels(labels: Iterable[str]) -> Tuple[str, str, str]:
	"""Normalize holiday labels and derive a simple local/global category."""

	unique_labels: List[str] = []
	seen = set()
	for label in labels:
		normalized = " ".join(str(label).strip().split())
		if not normalized or normalized in seen:
			continue
		seen.add(normalized)
		unique_labels.append(normalized)

	if not unique_labels:
		return "No", "None", "None"

	local_keywords = ("Poya", "Sinhala", "Tamil", "Sri Lanka", "Vesak", "Poson", "Esala", "Unduvap", "Nikini", "Binara", "Il", "Adi", "Madin", "Duruthu", "Navam", "Maha")
	global_keywords = ("Christmas", "Easter", "Good Friday", "New Year's Day", "International Workers' Day", "Ramadan", "Eid")

	categories = set()
	for label in unique_labels:
		if any(keyword.lower() in label.lower() for keyword in local_keywords):
			categories.add("Local")
		if any(keyword.lower() in label.lower() for keyword in global_keywords):
			categories.add("Global")

	if not categories:
		categories.add("Local")

	category_text = ", ".join(sorted(categories))
	return "Yes", "; ".join(unique_labels), category_text


def get_public_holiday_details(accident_date: pd.Timestamp, holiday_map: Dict[date, List[str]]) -> Tuple[str, str, str]:
	"""Return the public holiday indicator and descriptive festive label for a date."""

	labels = holiday_map.get(accident_date.date(), [])
	return _clean_holiday_labels(labels)


def derive_features(dataframe: pd.DataFrame) -> pd.DataFrame:
	"""Add derived calendar and holiday features to the source accident data."""

	transformed = dataframe.copy()
	transformed["_parsed_date"] = transformed[DATE_COLUMN].apply(standardize_date)
	transformed["_parsed_time"] = transformed[TIME_COLUMN].apply(standardize_time)

	years = sorted(transformed["_parsed_date"].dt.year.dropna().astype(int).unique().tolist())
	holiday_map = _iter_sri_lanka_holidays(years)

	transformed["Week (Derived)"] = transformed["_parsed_date"].apply(get_weekday_name)
	transformed["Month (Derived)"] = transformed["_parsed_date"].apply(get_month_name)
	transformed["Peak Hour Indicator (Derived)"] = transformed["_parsed_time"].apply(
		lambda value: "Yes" if is_peak_hour(value) else "No"
	)

	holiday_details = transformed["_parsed_date"].apply(lambda value: get_public_holiday_details(value, holiday_map))
	transformed["Public Holiday Indicator (Derived)"] = holiday_details.apply(lambda item: item[0])
	transformed["Public Holiday Name (Derived)"] = holiday_details.apply(lambda item: item[1])
	transformed["Public Holiday Category (Derived)"] = holiday_details.apply(lambda item: item[2])

	# Sort chronologically using the parsed columns (proper datetime objects, not raw strings)
	transformed = transformed.sort_values(
		by=["_parsed_date", "_parsed_time"]
	).reset_index(drop=True)

	transformed.drop(columns=["_parsed_date", "_parsed_time"], inplace=True)
	return transformed


def save_feature_engineered_data(dataframe: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
	"""Persist the feature engineered dataset to disk."""

	output_path.parent.mkdir(parents=True, exist_ok=True)
	dataframe.to_csv(output_path, index=False)
	LOGGER.info("Saved feature-engineered dataset to %s", output_path)


def main() -> None:
	"""Run the feature engineering pipeline."""

	configure_logging()
	source_data = load_source_data()
	engineered_data = derive_features(source_data)

	preferred_columns = [column for column in OUTPUT_COLUMNS if column in engineered_data.columns]
	ordered_columns = list(engineered_data.columns)
	for column in preferred_columns:
		ordered_columns.remove(column)
	ordered_columns = list(source_data.columns) + preferred_columns + [column for column in ordered_columns if column not in source_data.columns]

	save_feature_engineered_data(engineered_data[ordered_columns])


if __name__ == "__main__":
	main()