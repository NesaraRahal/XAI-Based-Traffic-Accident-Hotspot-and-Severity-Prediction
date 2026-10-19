# Traffic Accident Research — Hotspot & Severity Prediction (Colombo)

The research and modelling half of a BSc Honours dissertation on **explainable, spatiotemporal machine learning for traffic accident hotspot prediction and severity classification** in the Colombo Municipal Area (CMA), Sri Lanka.

This repository builds the dataset, engineers leakage-safe features on an H3 hexagon grid, trains and evaluates the classifiers, explains them with SHAP, and exports predictions to MongoDB. The companion **Traffic Risk Dashboard** repository reads those predictions and shows them on an interactive map.

> **Data note:** the accident records used for training are **synthetic**. They are generated from published Sri Lankan statistics (see [Synthetic data](#1-synthetic-accident-data)). The contribution is the working, explainable pipeline, not predictions about real Colombo roads.

---

## Contents

- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Pipeline, step by step](#pipeline-step-by-step)
- [Prediction tasks and features](#prediction-tasks-and-features)
- [Evaluation design](#evaluation-design)
- [Results](#results)
- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
- [MongoDB collections](#mongodb-collections)
- [Limitations](#limitations)
- [Housekeeping notes](#housekeeping-notes)

---

## Architecture

```
 Published SL statistics          OpenStreetMap (osmnx)        Visual Crossing weather
 (DCS 7.5, Police 2025 report,    street names per             (daily, Colombo 2022–2026)
  peer-reviewed studies)          police division                        │
          │                              │                               │
          └──────────────┬───────────────┘                               │
                         ▼                                               │
          generate_synthetic_accidents.py  ──►  3,688 synthetic records  │
                         │                                               │
                         ▼                                               ▼
                 data_lake/ingestion.py  ───────────►  MongoDB  raw_accidents / raw_weather
                                                          │
            geocode.py ─► weather_merge.py ─► feature_engineering.py      (python main.py)
                                                          │
                                       spatiotemporal_features.py  (H3 cells + rolling counts)
                                                          │
                                       train_panel.py  (one row per H3 cell × month)
                                                          │
                   train_hotspot_classifier.py   train_severity_classifier.py
                        (LR / RF / XGBoost, rolling temporal validation, SHAP)
                                                          │
                                  generate_dashboard_predictions.py
                                                          │
                                                          ▼
                              MongoDB  dashboard_predictions_cell_month
                                                          │
                                                          ▼
                                    Traffic Risk Dashboard (separate repo)
```

---

## Project structure

```
traffic-accident-research/
├── main.py                         # Runs geocode → weather merge → feature engineering
├── requirements.txt                # Python dependencies (see Setup for extras)
│
├── etl/
│   ├── OSM/
│   │   ├── build_street_vocab.py   # OSM streets → police-division street vocabulary
│   │   ├── verified_streets_by_division.json
│   │   ├── colombo_streets.geojson
│   │   └── division_polygons_preview.png
│   ├── generate_synthetic_accidents.py   # Synthetic accident generator (seed 42)
│   ├── validate_synthetic.py       # Chi-square checks against the source distributions
│   ├── geocode.py                  # Google Maps geocoding with a reusable location cache
│   ├── weather_merge.py            # Joins daily weather; derives weather/visibility/light labels
│   ├── feature_engineering.py      # Calendar, peak-hour and Sri Lankan holiday features
│   ├── spatiotemporal_features.py  # H3 cells (res 8/9), causal rolling counts, cell aggregates
│   └── train_panel.py              # Builds the (H3 cell, month) prediction panel + labels
│
├── data_lake/
│   └── ingestion.py                # Raw CSVs → JSON data lake → MongoDB raw collections
│
├── modeling/
│   ├── modelling_utils.py          # Shared: features, preprocessing, splits, metrics, saving
│   ├── train_hotspot_classifier.py # Task 1: is_hotspot (binary)
│   ├── train_severity_classifier.py# Task 2: dominant_severity (4-class) or is_severe (binary)
│   ├── generate_dashboard_predictions.py  # Scores Sep–Dec 2023 and writes to MongoDB
│   ├── plot_hotspot_temporal.py    # Thesis figures, temporal split (hotspot)
│   ├── plot_severity_temporal.py   # Thesis figures, temporal split (severity)
│   ├── plot_hotspot_diagnostics.py # Older grouped-split diagnostics (hotspot)
│   ├── plot_severity_diagnostics.py# Older grouped-split diagnostics (severity)
│   └── plotting_utils.py
│
├── config/
│   ├── .env                        # MONGO_URI, MONGO_DB, GOOGLE_MAPS_API_KEY (not committed)
│   └── visualize_hotspots.py       # Folium sanity-check map → hotspot_map.html
│
├── data/
│   ├── raw/                        # synthetic_cma_accidents_2022_2023.csv, weather CSV, older drafts
│   ├── lake/                       # JSON copies of the raw files (data lake layer)
│   ├── processed/                  # panel_cell_month.csv and earlier intermediate CSVs
│   └── reference/                  # generation_summary.json, location_master.csv
│
├── models/
│   ├── *.joblib                    # Trained model bundles, SHAP artefacts, label encoders
│   ├── *_metrics_report.json       # Full evaluation reports
│   ├── plots/hotspot_temporal/     # Figures used in the thesis
│   ├── plots/severity_temporal/
│   └── diagnostics/
│
├── testscripts/                    # Ad-hoc checks on the MongoDB panel (see below)
├── cache/                          # osmnx / Nominatim HTTP cache
└── hotspot_map.html                # Output of config/visualize_hotspots.py
```

---

## Pipeline, step by step

### 1. Synthetic accident data

`etl/OSM/build_street_vocab.py` pulls the Colombo drive network from OpenStreetMap. It approximates police-division polygons from OSM neighbourhoods and spatially joins streets to divisions. Some neighbourhoods are merged into the police jurisdictions they actually belong to; for example, Hunupitiya and Suduwella go into Slave Island. The result is `verified_streets_by_division.json`.

`etl/generate_synthetic_accidents.py` then generates **3,688 records** (1,861 in 2022 and 1,827 in 2023) across **17 police divisions**. Every *distribution* comes from a published source; only the individual records are synthetic:

| Anchor | Used for |
|---|---|
| DCS Table 7.5 (Colombo, 2022 & 2023) | Accident counts per police division |
| Sri Lanka Police *Summary Report on Road Traffic Accidents 2025* | Severity mix |
| Thilakarathna et al. 2025 | Fatal time-of-day, causes, collision types, age |
| Jayasinghe, Withanage & Mishra 2025 | Non-fatal time-of-day and causes |
| OpenStreetMap via osmnx | Street names per division |

Three conditional rules add realistic dependencies: night on a main road, a motorcycle or pedestrian, and an alcohol cause each shift severity upward. Slave Island is oversampled to 500 records so the generator can later be compared against a held-out sample of real Slave Island records (train-on-synthetic, test-on-real). Six CMA divisions with no usable OSM polygon are excluded (about 5% of DCS accident volume). A run summary is written to `data/reference/generation_summary.json`.

`etl/validate_synthetic.py` checks that the generated marginals match the anchors (chi-square goodness of fit), both overall and for the Slave Island slice.

### 2. Ingestion and enrichment

| Script | What it does | Output collection |
|---|---|---|
| `data_lake/ingestion.py` | Copies raw CSVs to JSON in `data/lake/` and upserts them into MongoDB | `raw_accidents`, `raw_weather` |
| `etl/geocode.py` | Geocodes each unique location with the Google Maps API. Incremental: already-geocoded places are cached and skipped | `ref_location_master`, `ref_failed_locations`, `processed_accidents_with_coords` |
| `etl/weather_merge.py` | Joins daily Visual Crossing weather by date; derives a weather-risk label, visibility level (Poor/Moderate/Good) and Day/Night from sunrise/sunset | `processed_accidents_with_weather` |
| `etl/feature_engineering.py` | Adds weekday, month, peak-hour flag and Sri Lankan public holidays (Poya days, Sinhala/Tamil New Year, etc.) | `accidents_feature_engineered` |

### 3. Spatiotemporal features

`etl/spatiotemporal_features.py` assigns every accident to an **H3 cell** (resolution 8, about 0.7 km², is the primary grid; resolution 9 is kept for sensitivity checks). It computes:

- **Causal rolling counts**: accidents in the same cell and in the ring of neighbouring cells over the previous 30, 90 and 180 days, using **only earlier records**.
- **Static per-cell aggregates** (`ref_h3_cell_aggregates`): used only for the sanity-check map, never as model features.

### 4. Prediction panel

`etl/train_panel.py` builds the modelling table: **one row per (H3 cell, month)**. That is 79 cells × 24 months (Jan 2022 – Dec 2023) = **1,896 rows**.

For a target month *T*, the **features use only information available before *T***: accident history up to the end of the previous month, the previous month's weather, and calendar facts known in advance. The **labels use only month *T***:

- **`is_hotspot`**: count accidents in every cell that month, including cells with zero. Run a one-sided Poisson test of each count against the month's mean cell rate, apply Benjamini–Hochberg FDR correction, and mark the cell a hotspot if adjusted *p* < 0.05. 108 of 1,896 rows (5.7%) are hotspots.
- **`dominant_severity`**: the most frequent severity among that cell-month's accidents (Fatal > Serious > Minor > Damage-only; ties go to the more severe class). It is only defined when the cell had at least one accident (1,112 rows). Months with no accident are **not** treated as non-severe.
- **`is_severe`**: `dominant_severity` is Fatal or Serious.

The panel is saved to MongoDB (`training_panel_cell_month`) and to `data/processed/panel_cell_month.csv`.

### 5. Modelling

`modeling/train_hotspot_classifier.py` and `modeling/train_severity_classifier.py` each train three models: **Logistic Regression** (scaled, class-balanced), **Random Forest** (300 trees, max depth 8, class-balanced) and **XGBoost** (depth 5, learning rate 0.05, class-imbalance weighting, early stopping inside validation folds only). They save the model bundles, a full metrics report and SHAP values for the winning model to `models/`.

Set `SEVERITY_MODE = "multiclass"` or `"binary"` at the top of `train_severity_classifier.py` to choose the severity task.

### 6. Dashboard export

`modeling/generate_dashboard_predictions.py` loads the winning models:

- hotspot: `hotspot_random_forest_winner.joblib`
- severity: `severity_multiclass_xgboost_winner.joblib`

It scores **only the final test window (Sep–Dec 2023)**. Each (cell, month) document gets the predictions, the top 4 SHAP features per task, the 180-day accident history, and the observed outcome (kept separate, for display only). Documents go to `dashboard_predictions_cell_month`. No model is retrained, and observed outcomes are never used as inputs.

---

## Prediction tasks and features

| Task | Target | Rows used | Primary metric |
|---|---|---|---|
| Hotspot | `is_hotspot` (binary, rare class) | all 1,896 | F1 (with ROC-AUC and Average Precision) |
| Severity (main) | `dominant_severity` (4 classes) | 1,112 severity-defined rows | macro-F1 |
| Severity (alt.) | `is_severe` (binary) | 1,112 severity-defined rows | F1 |

**Features (36 after encoding)**, defined in `modeling/modelling_utils.py`:

| Group | Features |
|---|---|
| Location | `cell_lat`, `cell_lng`, `police_division` (one-hot) |
| Accident history | `same_cell_count_{30,90,180}d`, `neighbor_count_{30,90,180}d` |
| Behaviour history (180 d) | `historical_peak_hour_share_180d`, `historical_night_share_180d`, `historical_motorcycle_share_180d` |
| Calendar | `month_sin`, `month_cos`, `holidays_in_month` |
| Weather | `avg_precip_month`, `avg_humidity_month`, `avg_visibility_month`, `avg_cloudcover_month`, `rainy_days_share_month` |

---

## Evaluation design

The pipeline is built to avoid information from the future or from the test set leaking into training:

- **Primary split: temporal.** Train on Jan 2022 – Aug 2023 and test on **Sep – Dec 2023**. This asks: can we predict next month for cells we already know?
- **Secondary split: grouped.** Train on 47 cells and test on 32 unseen cells. This asks: does the model generalise to new places?
- **Model selection** uses expanding-window rolling validation inside the training period: at least 8 training months, 2-month validation windows, step of 2, giving 6 folds. The winner is chosen on validation only.
- **Decision thresholds** (binary tasks) are tuned on pooled validation predictions over 0.10–0.90, then **frozen** before the test set is touched.
- **Imputation and category levels** are fitted on each fold's training part only, and refitted on the full training period for the final model.
- The test window is **never** used for model selection, threshold choice, preprocessing or XGBoost early stopping. Each metrics report records this under `evaluation_rules`.

---

## Results

Temporal test set, Sep–Dec 2023. Full numbers are in `models/*_metrics_report.json`.

**Hotspot** (316 rows, 15 hotspots)

| Model | F1 | Precision | Recall | ROC-AUC | Avg. Precision |
|---|---|---|---|---|---|
| Logistic Regression | 0.522 | 0.387 | 0.800 | 0.940 | 0.478 |
| **Random Forest (selected)** | **0.491** | 0.333 | 0.933 | **0.957** | 0.478 |
| XGBoost | 0.429 | 0.293 | 0.800 | 0.946 | 0.382 |

Random Forest was selected on rolling-validation F1 (0.495), before the test set was seen, which is why it is the winner even though Logistic Regression scores higher on test. On the grouped split (unseen cells), Random Forest reaches F1 0.463 and ROC-AUC 0.917.

**Severity, 4-class** (195 rows)

| Model | Macro-F1 | Accuracy |
|---|---|---|
| Logistic Regression | 0.195 | 0.272 |
| Random Forest | 0.185 | 0.236 |
| **XGBoost (selected)** | **0.236** | **0.328** |

**Severity, binary**: all three models end up predicting "severe" for every row (recall 1.0, precision 0.538, the base rate), with ROC-AUC 0.44–0.47.

### How to read these results

- **Hotspot:** ranking skill is high, but it is almost entirely explained by recent accident history. Ranking cells by `same_cell_count_180d` alone gives ROC-AUC **0.956** and Average Precision **0.48** on the same test set, essentially the same as the Random Forest. The model does not add predictive value beyond that baseline. Its value is in packaging that signal with calibrated thresholds and SHAP explanations.
- **Severity:** performance is at chance level. A uniform random guess over four classes gives macro-F1 of about 0.25, and always predicting "Minor" gives accuracy 0.36. This is expected for synthetic data where severity was sampled largely independently of place and time.

Figures used in the thesis are in `models/plots/hotspot_temporal/` and `models/plots/severity_temporal/`.

---

## Setup

**Requirements:** Python 3.12, a MongoDB Atlas database, and a Google Maps Geocoding API key (only needed to re-run geocoding).

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
# Packages imported by the scripts but not yet listed in requirements.txt:
pip install statsmodels matplotlib folium osmnx geopandas shapely
```

Create **`config/.env`**, which the ETL and modelling scripts read:

```
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>/
MONGO_DB=accident_hotspot
GOOGLE_MAPS_API_KEY=<your key>
```

`data_lake/ingestion.py` instead calls `load_dotenv()` from the working directory, so it also needs a **root `.env`** with `MONGO_URI` and `MONGO_DB` (or run it with those variables set).

---

## Running the pipeline

Run every command from the project root.

```bash
# 0. (optional) rebuild the OSM street vocabulary — needs internet, uses cache/
python etl/OSM/build_street_vocab.py

# 1. Synthetic data + checks
python etl/generate_synthetic_accidents.py
python etl/validate_synthetic.py

# 2. Load raw data into MongoDB
python data_lake/ingestion.py

# 3. Geocode → weather → calendar features
python main.py

# 4. H3 cells and rolling features, then the cell-month panel
python etl/spatiotemporal_features.py
python etl/train_panel.py

# 5. Train and evaluate
python modeling/train_hotspot_classifier.py
python modeling/train_severity_classifier.py     # SEVERITY_MODE at top of file

# 6. Thesis figures (reads data/processed/panel_cell_month.csv; add --source mongo to read MongoDB)
python modeling/plot_hotspot_temporal.py
python modeling/plot_severity_temporal.py

# 7. Export predictions for the dashboard
python modeling/generate_dashboard_predictions.py

# Optional: quick sanity-check map → hotspot_map.html
python config/visualize_hotspots.py --resolution 8
```

Data generation, splits and models use a fixed seed (42). Geocoding depends on the live Google Maps API, so coordinates can shift slightly if it is re-run instead of using the cached `ref_location_master`.

---

## MongoDB collections

| Collection | Written by | Contents |
|---|---|---|
| `raw_accidents` | `data_lake/ingestion.py` | Untouched synthetic accident records |
| `raw_weather` | `data_lake/ingestion.py` | Untouched daily weather records |
| `ref_location_master` | `etl/geocode.py` | Geocoding cache (location → lat/lng) |
| `ref_failed_locations` | `etl/geocode.py` | Locations that failed to geocode |
| `processed_accidents_with_coords` | `etl/geocode.py` | Accidents + coordinates |
| `processed_accidents_with_weather` | `etl/weather_merge.py` | + weather, visibility, light condition |
| `accidents_feature_engineered` | `etl/feature_engineering.py` | + calendar and holiday features |
| `ref_h3_cell_aggregates` | `etl/spatiotemporal_features.py` | Static per-cell counts (sanity map only) |
| `accidents_spatiotemporal_features` | `etl/spatiotemporal_features.py` | + H3 cells and rolling counts |
| `training_panel_cell_month` | `etl/train_panel.py` | The modelling panel |
| `dashboard_predictions_cell_month` | `modeling/generate_dashboard_predictions.py` | Predictions + SHAP for the dashboard |

---

## Limitations

- **Synthetic data.** Records are sampled from published marginal distributions plus three conditional rules, so relationships that were not built in cannot be learned. Findings describe the pipeline, not real Colombo traffic.
- **Real-data validation is not finished.** The train-on-synthetic, test-on-real comparison against held-out real Slave Island records is planned; `validate_synthetic.py` has a placeholder for loading them.
- **Division boundaries are approximate.** Sri Lanka Police jurisdiction boundaries are not public. OSM neighbourhood polygons with documented merge rules stand in for them, and six small divisions are excluded.
- **Small sample.** 79 cells and 24 months; only 15 hotspot cell-months in the test window, so test metrics have wide uncertainty.
- **Severity is not predictable here** (see Results), and the hotspot model matches a simple 180-day-count baseline.

---

## Housekeeping notes

- **Don't commit secrets or local environments.** `.env`, `config/.env`, `venv/` and `__pycache__/` should be listed in `.gitignore`.
- **Large generated files.** `colombo_streets.geojson` (about 6 MB, stored in both the root and `etl/OSM/`), `cache/` and `models/*.joblib` (about 30 MB in total) can be regenerated. Decide whether to version them, or use Git LFS.
- **Legacy and placeholder files:**
  - Empty stubs: `etl/extract.py`, `etl/transform.py`, `etl/load.py`, `warehouse/`, `api/`, `dashboard/` and `notebooks/`. The dashboard lives in its own repository.
  - Superseded scripts: `etl/OSM/street_names.py` and `etl/OSM/join_division.py` (replaced by `build_street_vocab.py`; `join_division.py` will not run on its own). `etl/OSM/cma_division.py` is a one-off analysis of street names that geocode poorly.
  - Old artefacts: `requirement.txt` is an older copy of `requirements.txt`. Files in `data/raw/` and `data/processed/` other than `synthetic_cma_accidents_2022_2023.csv`, the weather CSV and `panel_cell_month.csv` are drafts from earlier pipeline versions.
  - Old model files: in `models/`, the `severity_*.joblib` files without `binary`/`multiclass` in the name and `hotspot_xgboost_winner.joblib` come from earlier runs.
- **`testscripts/`** contains quick diagnostics against MongoDB:
  - `hotspotdist.py`: hotspot counts per month and checks of the BH-adjusted p-value cut-off
  - `nulltest.py`: label null counts
  - `testvis.py`: visibility feature check
  - `cleanpanel.py`: **drops** `dashboard_predictions_cell_month`, so use it with care
- `generate_synthetic_accidents.py` refers to `METHODOLOGY.md` and `DECISIONS.md`, which are not in this repository yet.

---

## Related repository

**Traffic Risk Dashboard**: a FastAPI + React app that reads `dashboard_predictions_cell_month` and shows hotspot and severity predictions with SHAP explanations on an H3 map. It has admin and police-division logins.
