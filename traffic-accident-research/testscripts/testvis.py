from pymongo import MongoClient
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv("config/.env")
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB", "accident_hotspot")]

# 1. Distribution of the new column across ALL panel rows
panel = pd.DataFrame(list(db["training_panel_cell_month"].find(
    {}, {"_id": 0, "year_month": 1, "poor_visibility_days_share_month": 1}
)))
print("=== poor_visibility_days_share_month across panel ===")
print(f"Rows: {len(panel)}")
print(f"Unique values: {panel['poor_visibility_days_share_month'].nunique()}")
print(f"Distribution:\n{panel['poor_visibility_days_share_month'].describe()}")
print(f"Non-zero rows: {(panel['poor_visibility_days_share_month'] > 0).sum()}")

# 2. Which months are non-zero?
print("\n=== Per-month values (dedup) ===")
per_month = panel.drop_duplicates("year_month").sort_values("year_month")
print(per_month.to_string(index=False))

# 3. Verify from raw weather that Jan 2022 has no <6km days
weather = pd.DataFrame(list(db["raw_weather"].find(
    {}, {"_id": 0, "datetime": 1, "visibility": 1}
)))
weather["date"] = pd.to_datetime(weather["datetime"])
weather["ym"] = weather["date"].dt.strftime("%Y-%m")

print("\n=== Days below 6km per month (from raw_weather directly) ===")
below6_by_month = weather.groupby("ym").apply(
    lambda g: (g["visibility"] < 6.0).sum()
).reset_index()
below6_by_month.columns = ["year_month", "days_below_6km"]
print(below6_by_month.to_string(index=False))