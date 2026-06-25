"""Pipeline entrypoint for the traffic accident research project."""

from __future__ import annotations

from etl.feature_engineering import main as feature_engineering_main
from etl.weather_merge import main as weather_merge_main



def main() -> None:
	"""Run weather merge first, then feature engineering."""

	weather_merge_main()
	feature_engineering_main()


if __name__ == "__main__":
	main()
