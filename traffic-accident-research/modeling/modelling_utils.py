"""Shared modelling utilities for hotspot and severity classification.

Both training scripts (train_hotspot_classifier.py and
train_severity_classifier.py) import from this module.

VERSION 2 — LEAKAGE-FIXED FEATURE LIST
=======================================
This update aligns with the leakage-fixed panel produced by
etl/train_panel.py (v2). Key changes vs the previous version:

- NUMERIC_FEATURES no longer contains `accidents_this_month` (dropped)
- NUMERIC_FEATURES uses new leakage-safe column names:
    * historical_peak_hour_share / historical_night_share / historical_motorcycle_share
      (replaces same-month peak_hour_share / night_share / motorcycle_share)
    * rainy_days_share_month / poor_visibility_days_share_month
      (replaces rainy_share / poor_visibility_share, now derived from raw weather)
    * holidays_in_month (replaces holiday_share)
- LABEL_LEAKAGE_COLUMNS expanded to guard against reintroduction of
  same-month accident features and full-dataset label components.
- TEMPORAL_SPLIT_TRAIN_END is aligned with the panel's
  LABEL_TRAINING_WINDOW_END so labels are computed only from pre-test data.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

# ── Environment ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / "config" / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "accident_hotspot")

COL_TRAINING_PANEL = "training_panel_cell_month"

MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature Config ─────────────────────────────────────────────────────────
# Everything except target, IDs, and label leakage columns is a feature.
# Explicitly listing this so future column additions to the panel don't
# silently become features without a decision.

ID_COLUMNS = ["h3_cell", "year_month", "built_at", "year", "month",
              "label_training_window_end"]

LABEL_COLUMNS = ["is_hotspot", "dominant_severity", "is_severe"]

# LEAKAGE-FIXED: any column derived from same-month accidents, or that
# directly composes the label, must be excluded from features.
LABEL_LEAKAGE_COLUMNS = [
    # Full-dataset per-cell counts — label components (kept in panel for
    # dashboard visibility only, NEVER used as features)
    "historical_accident_count",
    "historical_fatal_count",
    "historical_serious_count",
    "historical_minor_count",
    "historical_damage_only_count",
    "historical_fatal_rate",
    "hotspot_p_adjusted",

    # Training-window counts (used to derive the labels themselves)
    "train_window_accident_count",
    "hotspot_p_adjusted_train",
    "train_fatal_count",
    "train_serious_count",
    "train_minor_count",
    "train_damage_only_count",

    # Old same-month accident-derived features (must not sneak back in
    # if a stale panel or an older column name is loaded)
    "accidents_this_month",
    "peak_hour_share",
    "night_share",
    "motorcycle_share",
    "holiday_share",
    "rainy_share",
    "poor_visibility_share",
    # Dropped feature — no variance in the 2022–2023 study window's weather
    # feed. Listed here so a stale panel with this column still gets ignored.
    "poor_visibility_days_share_month",
]

# Categorical features → one-hot encoded
CATEGORICAL_FEATURES = ["police_division"]

# Numeric features — passed through (post-leakage-fix column names)
NUMERIC_FEATURES = [
    # Static cell context
    "cell_lat", "cell_lng",

    # Causal rolling exposure (spatiotemporal, already leakage-safe)
    "same_cell_count_30d", "same_cell_count_90d", "same_cell_count_180d",
    "neighbor_count_30d", "neighbor_count_90d", "neighbor_count_180d",

    # Cyclical calendar
    "month_sin", "month_cos",

    # Monthly weather from raw_weather (leakage-safe)
    # NOTE: poor_visibility_days_share_month was constructed and dropped —
    # see train_panel.py comment on weather thresholds. Visual Crossing feed
    # for 2022–2023 shows no meaningful visibility variation, so the feature
    # had no signal. avg_visibility_month is retained (it varies slightly).
    "avg_precip_month", "avg_humidity_month",
    "avg_visibility_month", "avg_cloudcover_month",
    "rainy_days_share_month",

    # Monthly calendar count (leakage-safe)
    "holidays_in_month",

    # Historical per-cell behavioural shares (training window only)
    "historical_peak_hour_share",
    "historical_night_share",
    "historical_motorcycle_share",
]

# Temporal split boundary (must align with LABEL_TRAINING_WINDOW_END in the panel)
TEMPORAL_SPLIT_TRAIN_END = "2023-08"
TEMPORAL_SPLIT_TEST_START = "2023-09"

# Grouped split — fraction of cells held out for test
GROUPED_TEST_SIZE = 0.40
GROUPED_SPLIT_SEED = 42

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(logging.INFO)
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ══════════════════════════════════════════════════════════════════════════
# MongoDB loading
# ══════════════════════════════════════════════════════════════════════════

def get_db():
    if not MONGO_URI:
        raise ValueError("MONGO_URI is not set. Check your config/.env file.")
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB], client


def load_training_panel() -> pd.DataFrame:
    """Load the (cell, month) training panel from MongoDB."""
    db, client = get_db()
    try:
        records = list(db[COL_TRAINING_PANEL].find({}, {"_id": 0}))
    finally:
        client.close()

    if not records:
        raise ValueError(
            f"No records found in collection: {COL_TRAINING_PANEL}. "
            "Run etl/train_panel.py first."
        )

    panel = pd.DataFrame(records)
    LOGGER.info("Loaded %d panel rows from [%s]", len(panel), COL_TRAINING_PANEL)
    LOGGER.info("Unique cells: %d | Unique months: %d",
                panel["h3_cell"].nunique(), panel["year_month"].nunique())

    # Guard: confirm the panel was built by v2 (has the new column names).
    v2_signals = ["historical_peak_hour_share", "rainy_days_share_month",
                  "holidays_in_month", "train_window_accident_count"]
    missing_v2 = [c for c in v2_signals if c not in panel.columns]
    if missing_v2:
        LOGGER.warning(
            "Panel appears to be from an older build — missing v2 columns: %s. "
            "Re-run etl/train_panel.py to produce the leakage-fixed panel.",
            missing_v2,
        )

    return panel


# ══════════════════════════════════════════════════════════════════════════
# Feature encoding
# ══════════════════════════════════════════════════════════════════════════

def encode_features(
    panel: pd.DataFrame,
    fit_encoder: bool = True,
    known_dummies: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """One-hot encode categoricals, keep numerics, drop IDs / labels / leakers.

    Any column listed in LABEL_LEAKAGE_COLUMNS is EXPLICITLY excluded from
    features even if it appears in the panel. This is the guardrail against
    label leakage — if a leakage column sneaks in, the model catches trivial
    100% scores; this list prevents that from ever being possible.
    """
    # Extract numeric features that exist and are not in the leakage list
    numeric_present = [
        c for c in NUMERIC_FEATURES
        if c in panel.columns and c not in LABEL_LEAKAGE_COLUMNS
    ]
    missing_numeric = [
        c for c in NUMERIC_FEATURES
        if c not in panel.columns and c not in LABEL_LEAKAGE_COLUMNS
    ]
    if missing_numeric:
        LOGGER.warning("Expected numeric features missing from panel: %s",
                       missing_numeric)

    numeric_frame = panel[numeric_present].copy()

    # One-hot encode categoricals
    categorical_present = [
        c for c in CATEGORICAL_FEATURES
        if c in panel.columns and c not in LABEL_LEAKAGE_COLUMNS
    ]
    if categorical_present:
        cat_frame = pd.get_dummies(
            panel[categorical_present].fillna("Unknown"),
            columns=categorical_present,
            prefix=categorical_present,
            drop_first=False,
        )
    else:
        cat_frame = pd.DataFrame(index=panel.index)

    X = pd.concat([numeric_frame, cat_frame], axis=1)

    # Sanity guard: assert no leakage column made it through
    leakers_present = [c for c in LABEL_LEAKAGE_COLUMNS if c in X.columns]
    if leakers_present:
        raise ValueError(
            f"LEAKAGE GUARD TRIPPED: {leakers_present} appeared in feature "
            f"matrix. This means a leakage column was added to "
            f"NUMERIC_FEATURES or CATEGORICAL_FEATURES by mistake."
        )

    # NaN check — fill with zero as a last resort (safer than crashing)
    nan_counts = X.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        LOGGER.warning("NaNs remain in features after encoding: %s",
                       nan_cols.to_dict())
        X = X.fillna(0)

    # Alignment mode for inference: reindex to match training columns
    if not fit_encoder and known_dummies is not None:
        for col in known_dummies:
            if col not in X.columns:
                X[col] = 0
        X = X[known_dummies]

    dummy_columns = X.columns.tolist()
    LOGGER.info("Encoded feature matrix: %d rows × %d cols", len(X), len(dummy_columns))
    return X, dummy_columns


# ══════════════════════════════════════════════════════════════════════════
# Splits
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Split:
    """A single train/test split with named indices for provenance."""
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    description: str


def temporal_split(panel: pd.DataFrame) -> Split:
    """Train on months <= 2023-08, test on months >= 2023-09."""
    train_mask = panel["year_month"] <= TEMPORAL_SPLIT_TRAIN_END
    test_mask = panel["year_month"] >= TEMPORAL_SPLIT_TEST_START
    return Split(
        name="temporal",
        train_idx=np.where(train_mask)[0],
        test_idx=np.where(test_mask)[0],
        description=(
            f"Train: year_month <= {TEMPORAL_SPLIT_TRAIN_END} "
            f"({int(train_mask.sum())} rows). "
            f"Test: year_month >= {TEMPORAL_SPLIT_TEST_START} "
            f"({int(test_mask.sum())} rows). "
            "Evaluates 'predict future months for known cells'. "
            "Labels are computed only from training-window accidents "
            "(no future leakage)."
        ),
    )


def grouped_split(panel: pd.DataFrame) -> Split:
    """Train on 60% of cells, test on 40% of cells (all months each)."""
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=GROUPED_TEST_SIZE, random_state=GROUPED_SPLIT_SEED
    )
    train_idx, test_idx = next(splitter.split(
        panel, groups=panel["h3_cell"]
    ))
    train_cells = panel.iloc[train_idx]["h3_cell"].nunique()
    test_cells = panel.iloc[test_idx]["h3_cell"].nunique()
    return Split(
        name="grouped",
        train_idx=train_idx,
        test_idx=test_idx,
        description=(
            f"Train: {train_cells} cells ({len(train_idx)} rows). "
            f"Test: {test_cells} held-out cells ({len(test_idx)} rows). "
            "Evaluates 'predict for unseen cells' (generalisation)."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    task: str = "binary",
    class_labels: Optional[List] = None,
) -> Dict:
    """Compute a consistent metrics dict for both tasks."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
        "test_class_distribution": {
            str(k): int(v) for k, v in pd.Series(y_true).value_counts().to_dict().items()
        },
    }

    if task == "binary":
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        if y_proba is not None:
            if len(np.unique(y_true)) > 1:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            else:
                metrics["roc_auc"] = None
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    else:  # multiclass
        metrics["macro_precision"] = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["macro_recall"] = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["macro_f1"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        per_class_f1 = f1_score(
            y_true, y_pred, average=None, zero_division=0, labels=class_labels
        )
        if class_labels is not None:
            metrics["per_class_f1"] = {
                str(cls): float(score)
                for cls, score in zip(class_labels, per_class_f1)
            }
        metrics["classification_report"] = classification_report(
            y_true, y_pred, zero_division=0, output_dict=True, labels=class_labels
        )
        metrics["confusion_matrix"] = confusion_matrix(
            y_true, y_pred, labels=class_labels
        ).tolist()

    return metrics


# ══════════════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════════════

def save_model_artifacts(
    task: str,
    model_name: str,
    model,
    feature_columns: List[str],
    is_winner: bool = False,
) -> Path:
    """Save a trained model + its feature schema to models/."""
    suffix = "_winner" if is_winner else ""
    filename = f"{task}_{model_name}{suffix}.joblib"
    path = MODELS_DIR / filename
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "task": task,
            "model_name": model_name,
            "trained_at": datetime.utcnow().isoformat(),
        },
        path,
    )
    LOGGER.info("Saved model: %s", path)
    return path


def save_metrics_report(task: str, report: Dict) -> Path:
    """Save the full metrics report (all models, both splits) to JSON."""
    path = MODELS_DIR / f"{task}_metrics_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    LOGGER.info("Saved metrics report: %s", path)
    return path


def save_shap_artifacts(
    task: str,
    explainer,
    shap_values,
    feature_columns: List[str],
    X_sample: pd.DataFrame,
) -> Path:
    """Save the SHAP explainer + values for dashboard consumption."""
    path = MODELS_DIR / f"{task}_shap.joblib"
    joblib.dump(
        {
            "explainer": explainer,
            "shap_values": shap_values,
            "feature_columns": feature_columns,
            "X_sample": X_sample,
            "computed_at": datetime.utcnow().isoformat(),
        },
        path,
    )
    LOGGER.info("Saved SHAP artifacts: %s", path)
    return path


# ══════════════════════════════════════════════════════════════════════════
# Panel + model logging helpers
# ══════════════════════════════════════════════════════════════════════════

def log_split_summary(split: Split, panel: pd.DataFrame, target_col: str) -> None:
    LOGGER.info("--- Split: %s ---", split.name)
    LOGGER.info("%s", split.description)
    train_target = panel.iloc[split.train_idx][target_col]
    test_target = panel.iloc[split.test_idx][target_col]
    LOGGER.info("Train class distribution: %s",
                train_target.value_counts().to_dict())
    LOGGER.info("Test class distribution:  %s",
                test_target.value_counts().to_dict())