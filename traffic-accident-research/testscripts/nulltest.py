from pymongo import MongoClient
import os, pandas as pd
from dotenv import load_dotenv
load_dotenv("config/.env")
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB", "accident_hotspot")]
panel = pd.DataFrame(list(db["training_panel_cell_month"].find({}, {"_id": 0})))
print(f"Total rows: {len(panel)}")
print(f"severity_defined=True: {panel['severity_defined'].sum()}")
print(f"severity_defined=False: {(~panel['severity_defined']).sum()}")
print(f"\nis_hotspot distribution (with True/False/null):")
print(panel['is_hotspot'].value_counts(dropna=False))
print(f"\nis_severe distribution (with True/False/null):")
print(panel['is_severe'].value_counts(dropna=False))
print(f"\ndominant_severity distribution (with True/False/null):")
print(panel['dominant_severity'].value_counts(dropna=False))
print(f"\nUnique cells with any defined severity row: {panel[panel['severity_defined']]['h3_cell'].nunique()}")
print(f"Cells with defined severity across time (count per cell):")
print(panel[panel['severity_defined']].groupby('h3_cell').size().describe())