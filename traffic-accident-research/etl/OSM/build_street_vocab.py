"""
Build verified street vocabulary per police division from OSM.
Runs all steps in sequence:
  1. Pull Colombo street network
  2. Build division polygons
  3. Spatial join with proper street-name normalisation
  4. Merge OSM neighbourhoods into Sri Lanka Police jurisdictions
  5. Export JSON with DCS 7.5 division names
"""

import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import re

# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "etl/OSM"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OSM name -> DCS 7.5 name mapping
OSM_TO_DCS = {
    "Kompannaweediya": "Slave Island",
    "Kollupitiya":     "Kollupitiya",
    "Wellawatta":      "Wellawatte",
    "Maligawatta":     "Maligawatte",
    "Modara":          "Modera",
    "Bloemendhal":     "Bluemendhal",
    # unchanged: Kotahena, Grandpass, Borella, Maradana, Fort, Pettah,
    #            Cinnamon Gardens, Bambalapitiya, Narahenpita,
    #            Kirulapone, Mattakkuliya, Dematagoda
}

suburbs = [
    "Kompannaweediya", "Kotahena", "Grandpass", "Borella", "Maradana",
    "Fort", "Pettah", "Kollupitiya", "Cinnamon Gardens", "Bambalapitiya",
    "Wellawatte", "Narahenpita", "Kirulapona", "Maligawatta",
    "Modera", "Mattakkuliya", "Dematagoda", "Bloemendhal",
    "Suduwella", "Hunupitiya"
]

# Special queries for divisions where default query picks wrong first result
SPECIAL_QUERIES = {
    "Maradana": "Maradana, Colombo, Sri Lanka",
}

# ============================================================
# POLICE JURISDICTION MERGE RULES
# ============================================================
# OSM neighbourhood polygons do NOT match Sri Lanka Police jurisdictions
# 1:1 (police boundaries are not publicly available). We use local
# geographic knowledge to merge adjacent OSM neighbourhoods into police
# jurisdictions.
#
# Example: Slave Island Police jurisdiction covers Kompannaweediya +
# Hunupitiya + Suduwella (streets like Union Place, Vauxhall Street,
# T.B. Jayah Mawatha, Justice Akbar Mawatha, Sir James Peiris Mawatha
# all fall under Slave Island Police despite being in different OSM
# neighbourhoods).
#
# Format: OSM neighbourhood (DCS-mapped name) -> target police division
MERGE_INTO_POLICE_DIVISION = {
    "Hunupitiya":  "Slave Island",
    "Suduwella":   "Slave Island",
    # Add more rules here if you identify other OSM-vs-police mismatches
}

# ============================================================
# STEP 1 — Pull Colombo street network
# ============================================================

streets_path = os.path.join(OUTPUT_DIR, "colombo_streets.geojson")

if os.path.exists(streets_path):
    print(f"Streets file already exists, skipping pull: {streets_path}")
else:
    print("Pulling Colombo street network from OSM...")
    G = ox.graph_from_place("Colombo, Sri Lanka", network_type="drive")
    edges = ox.graph_to_gdfs(G, nodes=False)
    edges.to_file(streets_path, driver="GeoJSON")
    print(f"Saved: {streets_path} ({len(edges)} street segments)")

# ============================================================
# STEP 2 — Build polygons
# ============================================================

def get_admin_polygon(query, max_results=10):
    """Walk through Nominatim results until we find a real polygon,
    skipping point features like railway stations."""
    for i in range(1, max_results + 1):
        try:
            gdf = ox.geocode_to_gdf(query, which_result=i)
            geom = gdf.geometry.iloc[0]
            geom_type = geom.geom_type
            area = geom.area
            print(f"    [debug] result #{i}: type={geom_type}, "
                  f"area={area:.2e}")

            if geom_type in ("Polygon", "MultiPolygon") and area > 1e-7:
                print(f"    [debug] -> ACCEPTED result #{i}")
                return geom
        except Exception as e:
            print(f"    [debug] result #{i} failed: {e}")
            break
    return None

print("\nBuilding division polygons...")
polygons = {}

for s in suburbs:
    if s in SPECIAL_QUERIES:
        query = SPECIAL_QUERIES[s]
        print(f"  Searching for {s} using: '{query}'")
        geom = get_admin_polygon(query)
        if geom is not None:
            polygons[s] = geom
            print(f"  ✓ polygon    {s} (special query)")
            continue
        else:
            print(f"  ⚠ special query failed for {s}, falling back...")

    try:
        gdf = ox.geocode_to_gdf(f"{s}, Colombo, Sri Lanka")
        polygons[s] = gdf.geometry.iloc[0]
        print(f"  ✓ polygon    {s}")
    except Exception:
        try:
            print(f"  ⚠ using buffer for {s}")
            lat, lon = ox.geocode(f"{s}, Colombo, Sri Lanka")
            point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
            polygons[s] = (
                point.to_crs(3857).buffer(750).to_crs(4326).iloc[0]
            )
        except Exception as e:
            print(f"  ✗ failed     {s}: {e}")

# Translate to DCS names
polygons_dcs = {OSM_TO_DCS.get(k, k): v for k, v in polygons.items()}
print(f"\nDCS-named divisions: {sorted(polygons_dcs.keys())}")

# ============================================================
# STEP 3 — Preview polygons
# ============================================================

print("\nGenerating polygon preview...")
fig, ax = plt.subplots(figsize=(14, 14))
for name, poly in polygons_dcs.items():
    gs = gpd.GeoSeries([poly], crs="EPSG:4326")
    gs.plot(ax=ax, alpha=0.35, edgecolor="black", linewidth=1)
    c = gs.centroid.iloc[0]
    ax.annotate(name, (c.x, c.y), fontsize=9, ha="center",
                fontweight="bold")
ax.set_title("Colombo Police Division Polygon Approximations (OSM)")
preview_path = os.path.join(OUTPUT_DIR, "division_polygons_preview.png")
plt.savefig(preview_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Preview saved: {preview_path}")

# ============================================================
# STEP 4 — Spatial join
# ============================================================

print("\nLoading streets and running spatial join...")
streets = gpd.read_file(streets_path)


def extract_street_names(value):
    """Extract clean street names from the OSM name field.

    The name field can be:
      - a plain string:                'Union Place'
      - a Python list:                 ['Union Place', 'Sir James Peiris']
      - a numpy array:                 array(['Union Place'])
      - a stringified array:           "['Union Place']" or "['A' 'B']"
      - None / NaN

    Returns a list of clean string names (may be empty).
    """
    # Handle None/NaN
    if value is None:
        return []
    try:
        if isinstance(value, float) and np.isnan(value):
            return []
    except (TypeError, ValueError):
        pass

    # Handle real list or numpy array
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(v).strip() for v in value
                if v is not None and str(v).strip()]

    # Handle string (may be stringified array from GeoJSON round-trip)
    if isinstance(value, str):
        s = value.strip()
        # Stringified array: "['A' 'B']" or "['A' 'B' 'C']"
        if s.startswith("[") and s.endswith("]"):
            # Extract all quoted substrings
            matches = re.findall(r"[\"']([^\"']+)[\"']", s)
            return [m.strip() for m in matches if m.strip()]
        # Plain string
        return [s]

    # Unknown type — best effort
    return [str(value).strip()]


# Explode the streets so each row has a single-string name
streets["name_list"] = streets["name"].apply(extract_street_names)
streets_exploded = streets.explode("name_list").rename(
    columns={"name_list": "street_name"}
)
streets_exploded = streets_exploded[streets_exploded["street_name"].notna()]
streets_exploded = streets_exploded[streets_exploded["street_name"] != ""]

divisions_gdf = gpd.GeoDataFrame(
    {"division": list(polygons_dcs.keys())},
    geometry=list(polygons_dcs.values()),
    crs="EPSG:4326"
)

streets_with_div = gpd.sjoin(
    streets_exploded, divisions_gdf, how="inner", predicate="intersects"
)

# ============================================================
# STEP 5 — Group by division
# ============================================================

street_lookup_raw = (
    streets_with_div
    .groupby("division")["street_name"]
    .apply(lambda x: sorted(set(x)))
    .to_dict()
)

# ============================================================
# STEP 6 — Merge OSM neighbourhoods into police jurisdictions
# ============================================================

print("\nApplying police-jurisdiction merges...")
street_lookup = {}
merge_log = []

for source_division, streets_list in street_lookup_raw.items():
    target = MERGE_INTO_POLICE_DIVISION.get(source_division, source_division)
    if target != source_division:
        merge_log.append((source_division, target, len(streets_list)))
    existing = set(street_lookup.get(target, []))
    existing.update(streets_list)
    street_lookup[target] = sorted(existing)

if merge_log:
    print("\nMerges applied (OSM neighbourhood -> Police jurisdiction):")
    for src, tgt, n in merge_log:
        print(f"  {src:<20} -> {tgt:<20} ({n} streets)")

# ============================================================
# STEP 7 — Summary & export
# ============================================================

print("\nStreet count per police jurisdiction:")
print(f"  {'Division':<25}{'Streets':>10}")
print(f"  {'-'*25}{'-'*10}")
for div, streets_list in sorted(street_lookup.items(),
                                key=lambda x: -len(x[1])):
    print(f"  {div:<25}{len(streets_list):>10}")

# Warn on low counts
LOW_COUNT_THRESHOLD = 5
low_count = [d for d, s in street_lookup.items() if len(s) < LOW_COUNT_THRESHOLD]
if low_count:
    print(f"\n⚠ WARNING: These divisions have <{LOW_COUNT_THRESHOLD} streets:")
    for d in low_count:
        print(f"    {d}: {street_lookup[d]}")

# Slave Island critical-street check
critical_keywords = [
    "union place", "t. b. jayah", "t.b. jayah", "t.b. jaya",
    "vauxhall", "justice akbar", "sir james peiris", "glenie",
]
si_streets_lower = [s.lower() for s in street_lookup.get("Slave Island", [])]
print("\nSlave Island critical-street check "
      "(streets from real 50 records):")
for kw in critical_keywords:
    found = any(kw in s for s in si_streets_lower)
    marker = "✓" if found else "✗"
    print(f"  {marker} {kw}")
print(f"\nSlave Island total streets: "
      f"{len(street_lookup.get('Slave Island', []))}")

# Export
output_json = os.path.join(OUTPUT_DIR, "verified_streets_by_division.json")
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(street_lookup, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {output_json}")
print(f"Total unique streets extracted: "
      f"{sum(len(v) for v in street_lookup.values())}")