import geopandas as gpd

streets = gpd.read_file("colombo_streets.geojson")
divisions_gdf = gpd.GeoDataFrame(
    {"division": list(polygons.keys())},
    geometry=list(polygons.values()),
    crs="EPSG:4326"
)

# Spatial join: which division does each street fall inside?
streets_with_div = gpd.sjoin(streets, divisions_gdf, how="left", predicate="intersects")

# Get unique street names per division
street_lookup = (
    streets_with_div[streets_with_div["name"].notna()]
    .groupby("division")["name"]
    .apply(lambda x: sorted(set(x)))
    .to_dict()
)

import json
with open("verified_streets_by_division.json", "w") as f:
    json.dump(street_lookup, f, indent=2, ensure_ascii=False)