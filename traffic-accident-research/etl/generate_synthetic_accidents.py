"""
================================================================================
SYNTHETIC ROAD TRAFFIC ACCIDENT DATA GENERATOR
Colombo Municipal Area (CMA), Sri Lanka — 2022 & 2023
================================================================================

Version 3 — Uses script-relative paths so it works regardless of the CWD
when the script is run.

PURPOSE
-------
Generate a synthetic accident dataset that (a) matches published Sri Lankan
statistical distributions at the marginal level, (b) preserves plausible joint
dependencies via conditional rules, and (c) can be validated against a
held-out real sample of ~50 Slave Island Police Station records (2022).

KEY PRINCIPLE
-------------
Every marginal distribution used by this generator is drawn from a real,
published source. Only the individual record *instances* are synthetic.
The generator NEVER sees or uses the 50 real records — those are held out
strictly for validation (avoiding leakage).

Street names are drawn from OpenStreetMap via a documented spatial-join
extraction (see build_street_vocab.py) and merged into police jurisdictions
using documented rules (Hunupitiya + Suduwella + Kompannaweediya -> Slave
Island).

EVALUATION APPROACH
-------------------
TSTR (Train-on-Synthetic, Test-on-Real) — following the protocol used in
GReaT (Borisov et al., ICLR 2023) for LLM-tabular data generation.

SOURCES ("ANCHORS")
-------------------
[A1] DCS Table 7.5  -- Colombo per-police-area accident counts, 2019-2023
[A2] Summary Report on Road Traffic Accidents 2025 (Sri Lanka Police)
[A3] Thilakarathna et al. 2025 (SLJFMSL Vol 16(1))
[A4] Jayasinghe, Withanage & Mishra 2025 (Matara DSD, 2023)
[A5] OpenStreetMap via osmnx (street vocabulary per police jurisdiction)
[A6] Real Slave Island location format samples (structural format only,
     not the 50 validation records)

CAVEATS (STATED OPENLY, PER DECISIONS.MD)
-----------------------------------------
See METHODOLOGY.md for the full caveats list. Key caveats:
1. Scale/conditioning mismatch across anchors
2. Some fields have no SL empirical distribution (labelled as assumptions)
3. Sri Lanka Police jurisdiction boundaries are not publicly available;
   OSM neighbourhood polygons are used with documented merge rules to
   approximate jurisdictions
4. 6 of 22 CMA police divisions were excluded due to lack of OSM polygons;
   these represent ~5% of DCS 7.5 accident volume
================================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json
import os

# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ============================================================================
# ROBUST PATH CONFIG — resolved relative to this script's location,
# not the current working directory. Assumes the project layout:
#
#   project_root/
#   ├── etl/
#   │   ├── generate_synthetic_accidents.py   <-- this file
#   │   └── OSM/
#   │       └── verified_streets_by_division.json
#   └── data/
#       ├── raw/
#       └── reference/
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # etl/ -> project root

OSM_VOCAB_PATH = os.path.join(
    SCRIPT_DIR, "OSM", "verified_streets_by_division.json"
)
OUTPUT_CSV_PATH = os.path.join(
    PROJECT_ROOT, "data", "raw",
    "synthetic_cma_accidents_2022_2023.csv"
)
OUTPUT_SUMMARY_PATH = os.path.join(
    PROJECT_ROOT, "data", "reference",
    "generation_summary.json"
)

# ============================================================================
# ANCHOR 1: SPATIAL DISTRIBUTION — DCS Table 7.5
# ============================================================================

DCS_7_5_COLOMBO = {
    "Modera":            (26, 18),
    "Grandpass":         (212, 186),
    "Kotahena":          (41, 67),
    "Dematagoda":        (69, 67),
    "Borella":           (277, 258),
    "Maradana":          (66, 77),
    "Fort":              (112, 96),
    "Pettah":            (38, 54),
    "Slave Island":      (44, 47),   # validation target
    "Colpetty":          (154, 151),
    "Cinnamon Gardens":  (283, 308),
    "Bambalapitiya":     (106, 95),
    "Wellawatte":        (68, 45),
    "Narahenpita":       (67, 78),
    "Foreshore Ports":   (0, 54),
    "Kirulapone":        (63, 50),
    "Harbour":           (23, 15),
    "Maligawatte":       (57, 50),
    "Keselwatte":        (16, 25),
    "Bluemendhal":       (1, 5),
    "Dam Street":        (12, 14),
    "Mattakkuliya":      (17, 18),
    "Wolfendhal":        (1, 1),
}

# ============================================================================
# DIVISIONS EXCLUDED FROM GENERATION
# These lack reliable OSM administrative polygons; together ~5% of volume.
# Documented in METHODOLOGY.md caveats.
# ============================================================================

EXCLUDED_DIVISIONS = {
    "Foreshore Ports",   # no OSM polygon
    "Harbour",           # port area, no clean polygon
    "Dam Street",        # street-level name, no polygon
    "Wolfendhal",        # too small in OSM
    "Keselwatte",        # spelling/polygon issue
    "Pettah",            # dropped from suburbs list in build_street_vocab
}

# ============================================================================
# JSON-KEY -> DCS-NAME mapping
# The OSM extraction step may output division names that differ slightly
# from DCS 7.5 spelling. This dict normalises them.
# ============================================================================

JSON_TO_DCS = {
    "Kollupitiya": "Kollupitiya",     # OSM/postal name -> DCS name
    "Kirulapona":  "Kirulapone",   # OSM spelling -> DCS spelling
    "Bluemendhal": "Bluemendhal",  # already matches
    # Everything else in the JSON already uses DCS names
    # (Slave Island, Modera, Wellawatte, Maligawatte etc.)
}

SLAVE_ISLAND_TARGET_TOTAL = 500

# ============================================================================
# ANCHOR 2: SEVERITY MIX — Summary Report 2025
# ============================================================================

SEVERITY_MIX = {
    "Fatal":       2583 / 22153,
    "Serious":     5276 / 22153,
    "Minor":       10067 / 22153,
    "Damage-only": 4227 / 22153,
}

# ============================================================================
# ANCHOR 3: TIME OF ACCIDENT — SEVERITY CONDITIONED
# ============================================================================

TIME_BANDS_FATAL = {
    "06:00-11:59": 0.269,
    "12:00-17:59": 0.247,
    "18:00-23:59": 0.392,
    "00:00-05:59": 0.093,
}

TIME_BANDS_NONFATAL = {
    "06:00-10:59": 0.22,
    "11:00-16:59": 0.38,
    "17:00-19:59": 0.20,
    "20:00-23:59": 0.12,
    "00:00-05:59": 0.08,
}

# ============================================================================
# ANCHOR 4: CAUSE OF ACCIDENT — SEVERITY CONDITIONED
# ============================================================================

CAUSE_FATAL = {
    "Breaking Road Regulations":  0.273,
    "Poor Judgement":             0.252,
    "Intoxication":               0.121,
    "Overspeeding":               0.071,
    "Vision/Hearing Impairment":  0.020,
    "Other Disability":           0.040,
    "Other":                      0.081,
    "Details Not Available":      0.081,
    "None Identified":            0.061,
}

CAUSE_NONFATAL = {
    "Driver Negligence":          0.57,
    "Aggressive Driving":         0.14,
    "Influence of Alcohol":       0.08,
    "Improper Overtaking":        0.06,
    "Out of Control":             0.05,
    "Wrong-side Driving":         0.04,
    "Slippery Road":              0.03,
    "Vehicle Issues":             0.02,
    "Pedestrian Unconsciousness": 0.01,
}

COLLISION_TYPES = {
    "Side Impact":         0.414,
    "Head-on Collision":   0.283,
    "Rear-end Collision":  0.182,
    "Other":               0.121,
}

DRIVER_AGE_GROUPS = {
    "<21":    0.06,
    "21-30":  0.162,
    "31-40":  0.14,
    "41-50":  0.162,
    "51-60":  0.192,
    "61-70":  0.182,
    ">70":    0.10,
}

VEHICLE_BASE_WEIGHTS = {
    "Motorcycle":     0.32,
    "Car":            0.22,
    "Three Wheeler":  0.16,
    "Van":            0.09,
    "Bus":            0.07,
    "Lorry":          0.06,
    "SUV":            0.04,
    "Bicycle":        0.02,
    "Pedestrian":     0.02,
}

# ============================================================================
# LOAD OSM-VERIFIED STREET VOCABULARY
# ============================================================================

def load_osm_vocab(path=OSM_VOCAB_PATH):
    """Load the OSM street vocabulary and normalise keys to DCS names."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"OSM vocabulary not found at {path}. "
            f"Run etl/OSM/build_street_vocab.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    normalised = {}
    for k, v in raw.items():
        dcs_name = JSON_TO_DCS.get(k, k)
        normalised[dcs_name] = v

    return normalised

LOCATIONS_OSM = load_osm_vocab()

# Report vocabulary coverage
print("OSM vocabulary loaded:")
for div, streets in sorted(LOCATIONS_OSM.items()):
    print(f"  {div:<25}{len(streets):>5} streets")
print()

# ============================================================================
# SLAVE ISLAND EXTRAS — junctions & landmarks matching real 50-record format
# The OSM vocab gives us clean street names; we supplement with the
# junction/landmark format seen in the real 6 samples.
# ============================================================================

SLAVE_ISLAND_JUNCTIONS = [
    "Ibbanwela Junction",
    "Union Place Junction",
    "Vauxhall Street Junction",
    "T.B. Jayah Junction",
    "Sir James Peiris Junction",
    "Justice Akbar Mawatha Junction",
]

SLAVE_ISLAND_LANDMARKS = [
    "RHQ SLEME",
    "near Beira Lake",
    "near Nippon Hotel",
    "near Slave Island Railway Station",
    "fuel station, Union Place",
    "near Gangaramaya area",
    "near Sambodhi Chaitya",
    "near Race Course",
]

# ============================================================================
# HELPERS
# ============================================================================

def sample_categorical(dist):
    keys = list(dist.keys())
    probs = np.array(list(dist.values()), dtype=float)
    probs = probs / probs.sum()
    return np.random.choice(keys, p=probs)


def generate_location(division):
    """Generate a location string. Slave Island uses mixed messy format
    (real OSM streets + junctions + landmarks) matching the real 50-record
    format. Other divisions use clean OSM street names only."""
    if division == "Slave Island":
        r = np.random.random()
        if r < 0.55:
            streets = LOCATIONS_OSM.get("Slave Island", ["Slave Island"])
            return random.choice(streets)
        elif r < 0.80:
            return random.choice(SLAVE_ISLAND_JUNCTIONS)
        else:
            return random.choice(SLAVE_ISLAND_LANDMARKS)
    else:
        streets = LOCATIONS_OSM.get(division)
        if not streets:
            return division
        return random.choice(streets)


def generate_time_from_band(band):
    start, end = band.split("-")
    sh, sm = map(int, start.split(":"))
    eh, em = map(int, end.split(":"))
    start_min = sh * 60 + sm
    end_min = eh * 60 + em
    if end_min <= start_min:
        end_min += 24 * 60
    t = np.random.randint(start_min, end_min + 1)
    hh = (t // 60) % 24
    mm = t % 60
    return f"{hh:02d}:{mm:02d}"


def generate_date(year):
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    delta_days = (end - start).days
    d = start + timedelta(days=np.random.randint(0, delta_days + 1))
    return d.strftime("%Y-%m-%d")


def is_night_hour(time_str):
    hh = int(time_str.split(":")[0])
    return hh >= 18 or hh < 6


def apply_severity_conditional_upshift(base_severity, factors):
    """Rules R1-R3."""
    severity_ladder = ["Damage-only", "Minor", "Serious", "Fatal"]
    idx = severity_ladder.index(base_severity)
    upshift_prob = 0.0

    if factors.get("night") and factors.get("main_road"):
        upshift_prob += 0.10
    if factors.get("motorcycle") or factors.get("pedestrian"):
        upshift_prob += 0.15
    if factors.get("alcohol"):
        upshift_prob += 0.20

    upshift_prob = min(upshift_prob, 0.5)
    if np.random.random() < upshift_prob and idx < len(severity_ladder) - 1:
        idx += 1
    return severity_ladder[idx]


def derive_casualties(severity):
    if severity == "Damage-only":
        return 0, 0
    if severity == "Minor":
        return 0, np.random.choice([1, 2], p=[0.7, 0.3])
    if severity == "Serious":
        return 0, np.random.choice([2, 3, 4], p=[0.5, 0.3, 0.2])
    if severity == "Fatal":
        fatalities = np.random.choice([1, 2], p=[0.85, 0.15])
        injuries = np.random.choice([0, 1, 2, 3], p=[0.35, 0.35, 0.2, 0.1])
        return fatalities, injuries


def derive_property_damage(severity, num_vehicles):
    if severity == "Damage-only":
        base = np.random.choice(["Low", "Medium", "High"], p=[0.5, 0.4, 0.1])
    elif severity == "Minor":
        base = np.random.choice(["Low", "Medium", "High"], p=[0.4, 0.5, 0.1])
    elif severity == "Serious":
        base = np.random.choice(["Low", "Medium", "High"], p=[0.15, 0.5, 0.35])
    else:
        base = np.random.choice(["Medium", "High"], p=[0.35, 0.65])

    if num_vehicles >= 3 and base != "High":
        if np.random.random() < 0.4:
            ladder = ["Low", "Medium", "High"]
            base = ladder[min(ladder.index(base) + 1, 2)]
    return base


def sample_vehicles():
    n = np.random.choice([1, 2, 3, 4], p=[0.15, 0.55, 0.22, 0.08])
    types = list(VEHICLE_BASE_WEIGHTS.keys())
    probs = np.array([VEHICLE_BASE_WEIGHTS[t] for t in types])
    probs /= probs.sum()
    chosen = list(np.random.choice(types, size=n, replace=True, p=probs))
    return n, "; ".join(chosen), chosen


def generate_record(division, year):
    base_severity = sample_categorical(SEVERITY_MIX)
    num_vehicles, vehicle_types_str, vehicle_list = sample_vehicles()

    if base_severity == "Fatal":
        cause = sample_categorical(CAUSE_FATAL)
        time_band = sample_categorical(TIME_BANDS_FATAL)
    else:
        cause = sample_categorical(CAUSE_NONFATAL)
        time_band = sample_categorical(TIME_BANDS_NONFATAL)

    time_str = generate_time_from_band(time_band)
    collision = sample_categorical(COLLISION_TYPES)
    age_group = sample_categorical(DRIVER_AGE_GROUPS)
    location = generate_location(division)
    date_str = generate_date(year)

    factors = {
        "night": is_night_hour(time_str),
        "main_road": np.random.random() < 0.727,
        "motorcycle": "Motorcycle" in vehicle_list,
        "pedestrian": "Pedestrian" in vehicle_list,
        "alcohol": cause in ("Intoxication", "Influence of Alcohol"),
    }

    final_severity = apply_severity_conditional_upshift(base_severity, factors)
    fatalities, injuries = derive_casualties(final_severity)
    property_damage = derive_property_damage(final_severity, num_vehicles)

    return {
        "Accident Severity":     final_severity,
        "Cause of Accident":     cause,
        "Collision Type":        collision,
        "Date of Accident":      date_str,
        "Driver Age Group":      age_group,
        "Location":              location,
        "Number of Fatalities":  int(fatalities),
        "Number of Injuries":    int(injuries),
        "Number of Vehicles":    int(num_vehicles),
        "Property Damage Level": property_damage,
        "Time of Accident":      time_str,
        "Vehicle Types":         vehicle_types_str,
        "Police Division":       division,
        "Year":                  year,
    }


def compute_target_counts():
    """Return {division: {2022: n, 2023: n}} skipping excluded divisions."""
    si_split_2022 = int(round(SLAVE_ISLAND_TARGET_TOTAL * 44 / (44 + 47)))
    si_split_2023 = SLAVE_ISLAND_TARGET_TOTAL - si_split_2022

    counts = {}
    for div, (n22, n23) in DCS_7_5_COLOMBO.items():
        if div in EXCLUDED_DIVISIONS:
            continue
        if div not in LOCATIONS_OSM and div != "Slave Island":
            print(f"  ⚠ SKIPPING {div}: no OSM vocabulary available")
            continue

        if div == "Slave Island":
            counts[div] = {2022: si_split_2022, 2023: si_split_2023}
        else:
            counts[div] = {2022: n22, 2023: n23}
    return counts


def generate_dataset(output_csv=OUTPUT_CSV_PATH,
                     summary_json=OUTPUT_SUMMARY_PATH):
    """Full generation pipeline."""
    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(summary_json), exist_ok=True)

    targets = compute_target_counts()
    records = []

    print("Generating synthetic accident records...")
    print(f"{'Division':<25}{'2022':>8}{'2023':>8}{'Total':>8}")
    print("-" * 49)

    grand_total = 0
    for div, yc in targets.items():
        div_total = 0
        for year, n in yc.items():
            for _ in range(n):
                records.append(generate_record(div, year))
            div_total += n
        grand_total += div_total
        print(f"{div:<25}{yc[2022]:>8}{yc[2023]:>8}{div_total:>8}")

    print("-" * 49)
    print(f"{'TOTAL':<25}{sum(t[2022] for t in targets.values()):>8}"
          f"{sum(t[2023] for t in targets.values()):>8}{grand_total:>8}")
    print()

    if EXCLUDED_DIVISIONS:
        excluded_volume = sum(
            n22 + n23 for div, (n22, n23) in DCS_7_5_COLOMBO.items()
            if div in EXCLUDED_DIVISIONS
        )
        total_dcs_volume = sum(
            v[0] + v[1] for v in DCS_7_5_COLOMBO.values()
        )
        print(f"Divisions excluded ({len(EXCLUDED_DIVISIONS)}): "
              f"{sorted(EXCLUDED_DIVISIONS)}")
        print(f"Excluded volume: {excluded_volume} records "
              f"({100*excluded_volume/total_dcs_volume:.1f}% "
              f"of DCS 7.5 total)")
        print()

    df = pd.DataFrame(records)

    schema_order = [
        "Accident Severity", "Cause of Accident", "Collision Type",
        "Date of Accident", "Driver Age Group", "Location",
        "Number of Fatalities", "Number of Injuries", "Number of Vehicles",
        "Property Damage Level", "Time of Accident", "Vehicle Types",
        "Police Division", "Year",
    ]
    df = df[schema_order]
    df.to_csv(output_csv, index=False)
    print(f"CSV written: {output_csv}  ({len(df)} rows)")

    summary = {
        "generation_timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "total_records": len(df),
        "records_2022": int((df["Year"] == 2022).sum()),
        "records_2023": int((df["Year"] == 2023).sum()),
        "slave_island_records":
            int((df["Police Division"] == "Slave Island").sum()),
        "divisions_covered": df["Police Division"].nunique(),
        "divisions_excluded": sorted(EXCLUDED_DIVISIONS),
        "severity_distribution": df["Accident Severity"]
            .value_counts(normalize=True).round(4).to_dict(),
        "location_vocabulary_source": OSM_VOCAB_PATH,
        "location_vocabulary_stats": {
            div: len(streets) for div, streets in LOCATIONS_OSM.items()
        },
        "anchors_used": {
            "spatial":  "DCS Table 7.5 (Colombo, 2022 & 2023)",
            "severity": "Summary Report on Road Traffic Accidents 2025",
            "time_fatal": "Thilakarathna et al. 2025 (fatal, Ragama)",
            "time_nonfatal": "Jayasinghe et al. 2025 (all-severity, Matara)",
            "cause_fatal": "Thilakarathna et al. 2025",
            "cause_nonfatal": "Jayasinghe et al. 2025",
            "collision":
                "Thilakarathna et al. 2025 (fatal, applied uniformly)",
            "age":
                "Thilakarathna et al. 2025 (victim age used as driver proxy)",
            "locations":
                "OpenStreetMap via osmnx (see build_street_vocab.py)",
        },
        "conditional_rules_applied": [
            "R1: Night + main road -> severity upshift +10%",
            "R2: Motorcycle or pedestrian -> severity upshift +15%",
            "R3: Alcohol cause -> severity upshift +20%",
        ],
        "slave_island_oversampling_note":
            f"Slave Island records total {SLAVE_ISLAND_TARGET_TOTAL} "
            "(vs 91 by strict DCS 7.5) to provide TSTR validation power "
            "against 50 held-out real records.",
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written: {summary_json}")

    return df


if __name__ == "__main__":
    df = generate_dataset()
    print("\nSample rows:")
    print(df.head(5).to_string(index=False))
    print("\nSeverity distribution:")
    print(df["Accident Severity"].value_counts(normalize=True).round(3))
    print("\nDivision distribution:")
    print(df["Police Division"].value_counts())
    print("\nSample Slave Island locations:")
    si_locs = (df[df["Police Division"] == "Slave Island"]["Location"]
               .sample(10, random_state=SEED))
    for loc in si_locs:
        print(f"  - {loc}")