# import osmnx as ox
# import geopandas as gpd
# from shapely.geometry import Point


# #Kompannaweediya = Slave Island
# suburbs = [
#     "Kompannaweediya", "Kotahena", "Grandpass", "Borella", "Maradana",
#     "Fort", "Pettah", "Kollupitiya", "Cinnamon Gardens", "Bambalapitiya",
#     "Wellawatta", "Narahenpita", "Kirulapone", "Maligawatta",
#     "Modara", "Mattakkuliya", "Dematagoda", "Bloemendhal"
# ]

# polygons = {}

# for s in suburbs:

#     try:
#         # Try getting actual polygon boundary from OSM
#         gdf = ox.geocode_to_gdf(
#             f"{s}, Colombo, Sri Lanka"
#         )

#         polygons[s] = gdf.geometry.iloc[0]

#         print(f"✓ polygon {s}")


#     except Exception:

#         try:
#             # Fallback: use geocoded point + buffer
#             print(f"⚠ using buffer for {s}")

#             lat, lon = ox.geocode(
#                 f"{s}, Colombo, Sri Lanka"
#             )

#             point = gpd.GeoSeries(
#                 [Point(lon, lat)],
#                 crs="EPSG:4326"
#             )

#             polygons[s] = (
#                 point
#                 .to_crs(3857)
#                 .buffer(750)   # 750m radius for Colombo neighbourhoods
#                 .to_crs(4326)
#                 .iloc[0]
#             )

#         except Exception as e:

#             print(
#                 f"✗ failed {s}: {e}"
#             )

# import osmnx as ox

# # Try the special query
# query = "Maradana, Colombo, Sri Lanka"

# for i in range(1, 8):
#     try:
#         gdf = ox.geocode_to_gdf(query, which_result=i)
#         geom = gdf.geometry.iloc[0]
#         print(f"Result #{i}:")
#         print(f"  type: {geom.geom_type}")
#         print(f"  area: {geom.area}")
#         print(f"  display_name: {gdf.iloc[0].get('display_name', 'N/A')}")
#         print(f"  class/type: {gdf.iloc[0].get('class', 'N/A')} / {gdf.iloc[0].get('type', 'N/A')}")
#         print()
#     except Exception as e:
#         print(f"Result #{i}: FAILED — {e}")
#         break


# import json
# import re

# with open("etl/OSM/verified_streets_by_division.json") as f:
#     vocab = json.load(f)

# # Patterns that Google commonly mis-parses in Sri Lankan road names
# risky_patterns = [
#     (r"\b[A-Z]\.\s*[A-Z]\.", "double initials (S.C., T.B., N.M.)"),
#     (r"^Dr\.?\s", "starts with Dr."),
#     (r"\bMawatha\b.*\bMawatha\b", "duplicated Mawatha"),
#     (r"\b[A-Z]\.\s*[A-Z]\.\s*[A-Z]\.", "triple initials"),
# ]

# print(f"{'Division':<25} {'Total':>7} {'Risky':>7} {'%':>6}")
# print("-" * 50)

# grand_total = 0
# grand_risky = 0
# for div, streets in sorted(vocab.items()):
#     risky = set()
#     for name in streets:
#         for pattern, _ in risky_patterns:
#             if re.search(pattern, name):
#                 risky.add(name)
#                 break
#     grand_total += len(streets)
#     grand_risky += len(risky)
#     if len(streets) > 0:
#         pct = 100 * len(risky) / len(streets)
#         print(f"{div:<25} {len(streets):>7} {len(risky):>7} {pct:>5.1f}%")

# print("-" * 50)
# print(f"{'TOTAL':<25} {grand_total:>7} {grand_risky:>7} "
#       f"{100*grand_risky/grand_total:>5.1f}%")


import pandas as pd
import json
import re

df = pd.read_csv("data/raw/synthetic_cma_accidents_2022_2023.csv")

with open("etl/OSM/verified_streets_by_division.json") as f:
    vocab = json.load(f)

# Build the risky-name set
risky_patterns = [
    r"\b[A-Z]\.\s*[A-Z]\.",
    r"^Dr\.?\s",
    r"\b[A-Z]\.\s*[A-Z]\.\s*[A-Z]\.",
]
combined = re.compile("|".join(risky_patterns))

risky_names = set()
for streets in vocab.values():
    for name in streets:
        if combined.search(name):
            risky_names.add(name)

# How many of your 3,688 records use a risky name?
risky_records = df[df["Location"].isin(risky_names)]
print(f"Records with risky location strings: {len(risky_records)} / {len(df)} "
      f"({100*len(risky_records)/len(df):.1f}%)")

print("\nTop 10 risky names by record count:")
print(risky_records["Location"].value_counts().head(10))