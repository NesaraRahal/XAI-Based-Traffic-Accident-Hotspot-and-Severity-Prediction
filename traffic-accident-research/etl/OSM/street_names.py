import osmnx as ox
import geopandas as gpd

# Pull all streets within Colombo Municipal Area
place = "Colombo, Sri Lanka"
G = ox.graph_from_place(place, network_type="drive")
edges = ox.graph_to_gdfs(G, nodes=False)
edges.to_file("colombo_streets.geojson", driver="GeoJSON")