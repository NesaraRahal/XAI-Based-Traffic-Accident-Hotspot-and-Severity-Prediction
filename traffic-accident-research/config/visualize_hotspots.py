"""Quick sanity-check visualization for H3 hotspot cells.

Reads the static per-cell aggregates (ref_h3_cell_aggregates, produced by
etl/spatiotemporal_features.py) from MongoDB and renders them as hexagon
polygons on a Leaflet map (via folium), colored by accident_count on a
green (low) -> yellow (mid) -> red (high) severity scale, with hotspot
cells outlined in dark red. Saves a standalone HTML file you can open
directly in a browser — no separate frontend needed for this sanity check.

Usage:
    python visualize_hotspots.py --resolution 8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import folium
import h3
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError


def find_env_file(start_dir: Path, filename: str = ".env", max_levels: int = 5) -> Path | None:
    """Search this directory and its parents for config/<filename>, up to max_levels up."""

    current = start_dir.resolve()
    for _ in range(max_levels):
        candidate = current / "config" / filename
        if candidate.exists():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


ENV_PATH = find_env_file(Path(__file__).resolve().parent)
if ENV_PATH:
    print(f"[env] Loading {ENV_PATH}")
    load_dotenv(dotenv_path=ENV_PATH, override=False)
else:
    print(
        "[env] WARNING: could not find config/.env by searching upward from "
        f"{Path(__file__).resolve().parent}. Falling back to already-set "
        "environment variables, if any."
    )

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "accident_hotspot")
COL_CELL_AGGREGATES = "ref_h3_cell_aggregates"

CMA_CENTER = [6.9271, 79.8612]  # Colombo Fort — default map center

# Green -> yellow -> red gradient stops (ratio, (r, g, b))
SEVERITY_GRADIENT = [
    (0.0, (26, 152, 80)),    # green — low severity
    (0.5, (255, 217, 47)),   # yellow — mid severity
    (1.0, (215, 48, 39)),    # red — high severity
]


def load_cell_aggregates(resolution: int) -> pd.DataFrame:
    """Load static per-cell aggregates for a given H3 resolution from MongoDB."""

    if not MONGO_URI:
        print(
            "[error] MONGO_URI is not set. Check that config/.env exists and "
            "contains MONGO_URI=... "
        )
        sys.exit(1)

    print(f"[mongo] Connecting to {MONGO_DB} ...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError as exc:
        print(f"[error] Could not reach MongoDB within 5s: {exc}")
        print("        Check MONGO_URI, network/VPN, and Atlas IP allowlist.")
        sys.exit(1)

    db = client[MONGO_DB]
    print(f"[mongo] Connected. Querying [{COL_CELL_AGGREGATES}] for resolution={resolution} ...")
    records = list(
        db[COL_CELL_AGGREGATES].find({"h3_resolution": resolution}, {"_id": 0})
    )
    client.close()

    if not records:
        print(
            f"[error] No records found for resolution={resolution} in "
            f"[{COL_CELL_AGGREGATES}]. Run etl/spatiotemporal_features.py first — "
            "it populates this collection."
        )
        sys.exit(1)

    print(f"[mongo] Loaded {len(records)} cells.")
    return pd.DataFrame(records)


def color_for_count(count: int, max_count: int) -> str:
    """Return a hex color on a green(low)->yellow(mid)->red(high) severity scale."""

    if max_count <= 0:
        return "#1a9850"
    ratio = min(count / max_count, 1.0)

    for (r0, c0), (r1, c1) in zip(SEVERITY_GRADIENT, SEVERITY_GRADIENT[1:]):
        if r0 <= ratio <= r1:
            t = 0 if r1 == r0 else (ratio - r0) / (r1 - r0)
            rgb = tuple(round(c0[i] + t * (c1[i] - c0[i])) for i in range(3))
            return "#{:02x}{:02x}{:02x}".format(*rgb)

    return "#d73027"


def build_map(cell_aggregates: pd.DataFrame, resolution: int) -> folium.Map:
    """Build a folium map with one hexagon polygon per H3 cell."""

    fmap = folium.Map(location=CMA_CENTER, zoom_start=12, tiles="cartodbpositron")
    max_count = cell_aggregates["accident_count"].max()

    for _, row in cell_aggregates.iterrows():
        boundary = h3.cell_to_boundary(row["h3_cell"])  # [(lat, lng), ...]
        fill_color = color_for_count(row["accident_count"], max_count)
        is_hotspot = bool(row.get("is_hotspot", False))

        popup_lines = [
            f"<b>Cell:</b> {row['h3_cell']}",
            f"<b>Accidents:</b> {row['accident_count']}",
        ]
        if "fatal_count" in row and pd.notna(row.get("fatal_count")):
            popup_lines.append(f"<b>Fatal:</b> {int(row['fatal_count'])}")
        if "fatal_rate" in row and pd.notna(row.get("fatal_rate")):
            popup_lines.append(f"<b>Fatal rate:</b> {row['fatal_rate']:.1%}")
        if "hotspot_p_adjusted" in row and pd.notna(row.get("hotspot_p_adjusted")):
            popup_lines.append(f"<b>Adjusted p-value:</b> {row['hotspot_p_adjusted']:.4f}")
        popup_lines.append(f"<b>Hotspot (FDR-corrected):</b> {'Yes' if is_hotspot else 'No'}")

        folium.Polygon(
            locations=boundary,
            color="#67000d" if is_hotspot else "#999999",
            weight=3 if is_hotspot else 1,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.75 if is_hotspot else 0.45,
            popup=folium.Popup("<br>".join(popup_lines), max_width=250),
        ).add_to(fmap)

    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 6px;
                border: 1px solid #ccc; font-size: 13px;">
      <b>H3 res {resolution} — severity</b><br>
      <span style="background:#1a9850;padding:2px 8px;">&nbsp;</span> low<br>
      <span style="background:#91cf60;padding:2px 8px;">&nbsp;</span><br>
      <span style="background:#ffd92f;padding:2px 8px;">&nbsp;</span> med<br>
      <span style="background:#fc8d59;padding:2px 8px;">&nbsp;</span><br>
      <span style="background:#d73027;padding:2px 8px;color:white;">&nbsp;</span> high<br>
      <span style="border:2px solid #67000d;padding:1px 7px;">&nbsp;</span> hotspot outline
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    return fmap


def main() -> None:
    parser = argparse.ArgumentParser(description="Render H3 hotspot cells on a sanity-check map.")
    parser.add_argument("--resolution", type=int, default=8, help="H3 resolution to visualize (8 or 9)")
    parser.add_argument("--output", type=str, default="hotspot_map.html", help="Output HTML file path")
    args = parser.parse_args()

    cell_aggregates = load_cell_aggregates(args.resolution)
    hotspot_count = int(cell_aggregates.get("is_hotspot", pd.Series(dtype=bool)).sum())
    print(f"[build] Rendering {len(cell_aggregates)} hexagons ({hotspot_count} flagged hotspot) ...")
    fmap = build_map(cell_aggregates, args.resolution)
    fmap.save(args.output)
    output_path = Path(args.output).resolve()
    print(f"[done] Saved to {output_path}")
    print("       Open that file in a browser to view the map.")


if __name__ == "__main__":
    main()