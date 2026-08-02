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

import osmnx as ox

# Try the special query
query = "Maradana, Colombo, Sri Lanka"

for i in range(1, 8):
    try:
        gdf = ox.geocode_to_gdf(query, which_result=i)
        geom = gdf.geometry.iloc[0]
        print(f"Result #{i}:")
        print(f"  type: {geom.geom_type}")
        print(f"  area: {geom.area}")
        print(f"  display_name: {gdf.iloc[0].get('display_name', 'N/A')}")
        print(f"  class/type: {gdf.iloc[0].get('class', 'N/A')} / {gdf.iloc[0].get('type', 'N/A')}")
        print()
    except Exception as e:
        print(f"Result #{i}: FAILED — {e}")
        break