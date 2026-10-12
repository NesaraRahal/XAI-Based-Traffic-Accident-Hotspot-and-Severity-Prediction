from pymongo import MongoClient
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv("config/.env")

client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB", "accident_hotspot")]

panel = pd.DataFrame(
    list(
        db["training_panel_cell_month"].find(
            {},
            {
                "_id": 0,
                "h3_cell": 1,
                "year": 1,
                "month": 1,
                "year_month": 1,
                "target_accident_count": 1,
                "is_hotspot": 1,
                "hotspot_p_value": 1,
                "hotspot_p_adjusted": 1,
            },
        )
    )
)

print(f"Total rows: {len(panel)}")


# =========================================================
# 1. MONTHLY ACCIDENT / HOTSPOT DISTRIBUTION
# =========================================================

monthly = (
    panel.groupby("year_month")
    .agg(
        total_accidents=("target_accident_count", "sum"),
        mean_accidents_per_cell=("target_accident_count", "mean"),
        cells_with_accidents=(
            "target_accident_count",
            lambda x: (x > 0).sum(),
        ),
        number_of_hotspots=(
            "is_hotspot",
            lambda x: (x == True).sum(),
        ),
        max_accidents_in_cell=("target_accident_count", "max"),
    )
    .reset_index()
)

monthly["mean_accidents_per_cell"] = (
    monthly["mean_accidents_per_cell"].round(2)
)

print("\nMonthly Hotspot Distribution")
print("=" * 100)
print(monthly.to_string(index=False))


# =========================================================
# 2. HOTSPOT ACCIDENT COUNT DIAGNOSTIC
# =========================================================

def hotspot_diagnostic(group):

    hotspots = group[group["is_hotspot"] == True]
    non_hotspots = group[group["is_hotspot"] != True]

    return pd.Series({
        "hotspot_count": len(hotspots),

        "hotspot_min_accidents": (
            hotspots["target_accident_count"].min()
            if not hotspots.empty else 0
        ),

        "hotspot_max_accidents": (
            hotspots["target_accident_count"].max()
            if not hotspots.empty else 0
        ),

        "non_hotspot_max_accidents": (
            non_hotspots["target_accident_count"].max()
            if not non_hotspots.empty else 0
        ),
    })


diagnostic = (
    panel
    .groupby("year_month")
    .apply(hotspot_diagnostic, include_groups=False)
    .reset_index()
)

print("\nHotspot Accident Count Diagnostic")
print("=" * 100)
print(diagnostic.to_string(index=False))


# =========================================================
# 3. ACTUAL HOTSPOT CELLS
# =========================================================

print("\nActual Hotspot Cells by Month")
print("=" * 100)

hotspot_cells = (
    panel[panel["is_hotspot"] == True]
    [
        [
            "year_month",
            "h3_cell",
            "target_accident_count",
        ]
    ]
    .sort_values(
        ["year_month", "target_accident_count"],
        ascending=[True, False],
    )
)

print(hotspot_cells.to_string(index=False))


# =========================================================
# 4. POISSON + BH STATISTICAL DIAGNOSTIC
# =========================================================
#
# Check whether hotspot cells have appropriately low
# Poisson p-values and BH-adjusted p-values.
#
# We show:
#
# - minimum / maximum accident count among hotspots
# - largest raw p-value among hotspots
# - largest adjusted p-value among hotspots
# - smallest accident count among non-hotspots
# - smallest raw p-value among non-hotspots
# - smallest adjusted p-value among non-hotspots
#
# This helps identify borderline cases.
# =========================================================

print("\nPoisson + Benjamini-Hochberg Diagnostic")
print("=" * 100)

hotspots = panel[panel["is_hotspot"] == True].copy()
non_hotspots = panel[panel["is_hotspot"] != True].copy()

print("\nHotspot statistical summary")
print("-" * 100)

print(
    f"Number of hotspots: "
    f"{len(hotspots)}"
)

print(
    f"Minimum hotspot accident count: "
    f"{hotspots['target_accident_count'].min()}"
)

print(
    f"Maximum hotspot accident count: "
    f"{hotspots['target_accident_count'].max()}"
)

print(
    f"Largest raw p-value among hotspots: "
    f"{hotspots['hotspot_p_value'].max():.6g}"
)

print(
    f"Largest BH-adjusted p-value among hotspots: "
    f"{hotspots['hotspot_p_adjusted'].max():.6g}"
)

print("\nNon-hotspot statistical summary")
print("-" * 100)

print(
    f"Number of non-hotspots: "
    f"{len(non_hotspots)}"
)

print(
    f"Smallest non-hotspot accident count: "
    f"{non_hotspots['target_accident_count'].min()}"
)

print(
    f"Largest non-hotspot accident count: "
    f"{non_hotspots['target_accident_count'].max()}"
)

print(
    f"Smallest raw p-value among non-hotspots: "
    f"{non_hotspots['hotspot_p_value'].min():.6g}"
)

print(
    f"Smallest BH-adjusted p-value among non-hotspots: "
    f"{non_hotspots['hotspot_p_adjusted'].min():.6g}"
)


# =========================================================
# 5. BORDERLINE CASES
# =========================================================
#
# Show the strongest non-hotspots and weakest hotspots.
# These are especially useful for checking the BH boundary.
# =========================================================

print("\nWeakest Hotspots")
print("=" * 100)

weakest_hotspots = (
    hotspots[
        [
            "year_month",
            "h3_cell",
            "target_accident_count",
            "hotspot_p_value",
            "hotspot_p_adjusted",
            "is_hotspot",
        ]
    ]
    .sort_values("hotspot_p_adjusted", ascending=False)
    .head(15)
)

print(weakest_hotspots.to_string(index=False))


print("\nStrongest Non-Hotspots")
print("=" * 100)

strongest_non_hotspots = (
    non_hotspots[
        [
            "year_month",
            "h3_cell",
            "target_accident_count",
            "hotspot_p_value",
            "hotspot_p_adjusted",
            "is_hotspot",
        ]
    ]
    .sort_values("hotspot_p_adjusted", ascending=True)
    .head(15)
)

print(strongest_non_hotspots.to_string(index=False))


# =========================================================
# 6. CHECK BH THRESHOLD
# =========================================================

print("\nAdjusted P-Value Boundary Check")
print("=" * 100)

print(
    "Largest adjusted p-value among hotspots: "
    f"{hotspots['hotspot_p_adjusted'].max():.6g}"
)

print(
    "Smallest adjusted p-value among non-hotspots: "
    f"{non_hotspots['hotspot_p_adjusted'].min():.6g}"
)


# =========================================================
# 7. FINAL SUMMARY
# =========================================================

print("\nHotspot Summary")
print("=" * 100)

print(
    f"Total hotspot cell-month rows: "
    f"{len(hotspots)}"
)

print(
    f"Months with zero hotspots: "
    f"{(monthly['number_of_hotspots'] == 0).sum()}"
)

print(
    f"Maximum hotspots in one month: "
    f"{monthly['number_of_hotspots'].max()}"
)

print(
    f"Average hotspots per month: "
    f"{monthly['number_of_hotspots'].mean():.2f}"
)

client.close()