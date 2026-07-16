"""Pipeline entrypoint for the traffic accident research project."""

from __future__ import annotations

from etl.geocode import main as geocode_main
from etl.feature_engineering import main as feature_engineering_main
from etl.weather_merge import main as weather_merge_main



def main() -> None:
	"""Run geocoding first, then weather merge, then feature engineering."""

	geocode_main()
	weather_merge_main()
	feature_engineering_main()


if __name__ == "__main__":
	main()
