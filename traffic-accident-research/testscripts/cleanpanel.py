from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv("config/.env")
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB", "accident_hotspot")]

# Drop the whole collection — cleanest way to remove stale fields
db["training_panel_cell_month"].drop()
print("Dropped training_panel_cell_month. Re-run etl/train_panel.py to regenerate.")