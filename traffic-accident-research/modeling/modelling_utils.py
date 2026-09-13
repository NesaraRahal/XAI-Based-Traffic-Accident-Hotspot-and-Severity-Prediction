"""Shared modelling utilities for hotspot and severity classification.

Version 4
---------
The modelling panel uses rolling per-(H3 cell, month) labels.  This module
provides leakage-safe feature preprocessing, temporal/grouped splits, rolling
temporal validation, threshold tuning, metrics, and model persistence.

Important preprocessing rule
----------------------------
Missing numeric values are imputed with medians learned from the fitting
portion only.  Categorical levels are learned from the fitting portion only.
The final model is refit after model/threshold selection using the complete
training portion, and that final preprocessing is persisted with the model.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
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
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


# ============================================================================
# Environment
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / "config" / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "accident_hotspot")
COL_TRAINING_PANEL = "training_panel_cell_month"

MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LOGGER = logging.getLogger(__name__)


# ============================================================================
# Feature configuration
# ============================================================================

ID_COLUMNS = [
    "h3_cell",
    "year_month",
    "built_at",
    "year",
    "month",
    "label_training_window_end",
]

LABEL_COLUMNS = [
    "is_hotspot",
    "dominant_severity",
    "is_severe",
]

LABEL_LEAKAGE_COLUMNS = [
    # Target-month accident counts.
    "target_accident_count",
    "target_fatal_count",
    "target_serious_count",
    "target_minor_count",
    "target_damage_only_count",

    # Hotspot label outputs.
    "hotspot_p_value",
    "hotspot_p_adjusted",

    # Row-level bookkeeping / target definition.
    "severity_defined",
    "feature_cutoff_date",
    "month_start",

    # Legacy v2 columns.
    "historical_accident_count",
    "historical_fatal_count",
    "historical_serious_count",
    "historical_minor_count",
    "historical_damage_only_count",
    "historical_fatal_rate",
    "train_window_accident_count",
    "hotspot_p_adjusted_train",
    "train_fatal_count",
    "train_serious_count",
    "train_minor_count",
    "train_damage_only_count",

    # Old same-month accident-derived features.
    "accidents_this_month",
    "peak_hour_share",
    "night_share",
    "motorcycle_share",
    "holiday_share",
    "rainy_share",
    "poor_visibility_share",
    "poor_visibility_days_share_month",

    # Old static historical shares.
    "historical_peak_hour_share",
    "historical_night_share",
    "historical_motorcycle_share",
]

CATEGORICAL_FEATURES = ["police_division"]

NUMERIC_FEATURES = [
    "cell_lat",
    "cell_lng",
    "same_cell_count_30d",
    "same_cell_count_90d",
    "same_cell_count_180d",
    "neighbor_count_30d",
    "neighbor_count_90d",
    "neighbor_count_180d",
    "month_sin",
    "month_cos",
    "avg_precip_month",
    "avg_humidity_month",
    "avg_visibility_month",
    "avg_cloudcover_month",
    "rainy_days_share_month",
    "holidays_in_month",
    "historical_peak_hour_share_180d",
    "historical_night_share_180d",
    "historical_motorcycle_share_180d",
]


# ============================================================================
# Split configuration
# ============================================================================

TEMPORAL_SPLIT_TRAIN_END = "2023-08"
TEMPORAL_SPLIT_TEST_START = "2023-09"

# Retained for compatibility with older training scripts.
TEMPORAL_VALIDATION_MONTHS = 2

# Rolling expanding-window temporal validation.
ROLLING_VALIDATION_MONTHS = 2
ROLLING_VALIDATION_STEP_MONTHS = 2
ROLLING_MIN_TRAIN_MONTHS = 8

# Validation-only threshold search.
THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.90
THRESHOLD_STEP = 0.01

# Grouped validation/test.
GROUPED_VALIDATION_TEST_SIZE = 0.20
GROUPED_VALIDATION_SEED = 42
GROUPED_TEST_SIZE = 0.40
GROUPED_SPLIT_SEED = 42


# ============================================================================
# Logging
# ============================================================================

def configure_logging() -> None:
    """Configure application-wide logging."""
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(logging.INFO)
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ============================================================================
# MongoDB
# ============================================================================

def get_db():
    """Return MongoDB database and client."""
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

    LOGGER.info(
        "Loaded %d panel rows from [%s]",
        len(panel),
        COL_TRAINING_PANEL,
    )
    LOGGER.info(
        "Unique cells: %d | Unique months: %d",
        panel["h3_cell"].nunique(),
        panel["year_month"].nunique(),
    )

    v3_signals = [
        "historical_peak_hour_share_180d",
        "rainy_days_share_month",
        "holidays_in_month",
        "target_accident_count",
        "severity_defined",
        "feature_cutoff_date",
    ]
    missing_v3 = [c for c in v3_signals if c not in panel.columns]
    if missing_v3:
        LOGGER.warning(
            "Panel appears to be from an older build — missing v3 columns: %s. "
            "Re-run etl/train_panel.py to produce the current panel.",
            missing_v3,
        )

    return panel


# ============================================================================
# Leakage-safe feature preprocessing
# ============================================================================

@dataclass
class FeaturePreprocessor:
    """Fitted feature schema and imputation values.

    All learned values must come from the fitting portion only.
    """

    numeric_features: List[str]
    categorical_features: List[str]
    numeric_medians: Dict[str, float]
    categorical_levels: Dict[str, List[str]]
    feature_columns: List[str] = field(default_factory=list)


def _available_features(panel: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return configured numeric/categorical features present in panel."""
    numeric_present = [
        c for c in NUMERIC_FEATURES
        if c in panel.columns and c not in LABEL_LEAKAGE_COLUMNS
    ]
    categorical_present = [
        c for c in CATEGORICAL_FEATURES
        if c in panel.columns and c not in LABEL_LEAKAGE_COLUMNS
    ]

    missing_numeric = [
        c for c in NUMERIC_FEATURES
        if c not in panel.columns and c not in LABEL_LEAKAGE_COLUMNS
    ]
    if missing_numeric:
        LOGGER.warning("Expected numeric features missing from panel: %s", missing_numeric)

    return numeric_present, categorical_present


def fit_feature_preprocessor(panel: pd.DataFrame) -> FeaturePreprocessor:
    """Fit feature schema and numeric imputation values on *panel only*.

    The caller is responsible for passing only the current fitting portion.
    """
    numeric_present, categorical_present = _available_features(panel)

    numeric_medians: Dict[str, float] = {}
    for column in numeric_present:
        numeric = pd.to_numeric(panel[column], errors="coerce")
        median = numeric.median()
        # A completely missing fitting column has no data-derived median.
        # Zero is only a defensive fallback for that pathological case.
        numeric_medians[column] = float(median) if pd.notna(median) else 0.0

    categorical_levels: Dict[str, List[str]] = {}
    for column in categorical_present:
        values = panel[column].fillna("Unknown").astype(str)
        levels = sorted(values.unique().tolist())
        if "Unknown" not in levels:
            levels.append("Unknown")
        categorical_levels[column] = levels

    # Build the feature schema from the fitting portion only.
    feature_columns = list(numeric_present)
    for column in categorical_present:
        prefix = f"{column}_"
        feature_columns.extend(
            [f"{prefix}{level}" for level in categorical_levels[column]]
        )

    return FeaturePreprocessor(
        numeric_features=numeric_present,
        categorical_features=categorical_present,
        numeric_medians=numeric_medians,
        categorical_levels=categorical_levels,
        feature_columns=feature_columns,
    )


def transform_features(
    panel: pd.DataFrame,
    preprocessor: FeaturePreprocessor,
) -> pd.DataFrame:
    """Transform panel using a previously fitted preprocessor."""
    numeric_frame = pd.DataFrame(index=panel.index)

    for column in preprocessor.numeric_features:
        if column in panel.columns:
            values = pd.to_numeric(panel[column], errors="coerce")
        else:
            values = pd.Series(np.nan, index=panel.index, dtype=float)
        numeric_frame[column] = values.fillna(preprocessor.numeric_medians[column])

    categorical_frames = []
    for column in preprocessor.categorical_features:
        if column in panel.columns:
            values = panel[column].fillna("Unknown").astype(str)
        else:
            values = pd.Series("Unknown", index=panel.index, dtype="object")

        known_levels = preprocessor.categorical_levels[column]
        # Unknown/unseen levels are mapped to Unknown.  If Unknown was not
        # present in the fitted levels, use the first learned level as a
        # defensive fallback (normally unreachable because fit adds Unknown).
        fallback = "Unknown" if "Unknown" in known_levels else known_levels[0]
        values = values.where(values.isin(known_levels), fallback)

        dummy = pd.get_dummies(values, prefix=column, dtype=float)
        expected = [f"{column}_{level}" for level in known_levels]
        dummy = dummy.reindex(columns=expected, fill_value=0.0)
        categorical_frames.append(dummy)

    frames = [numeric_frame] + categorical_frames
    X = pd.concat(frames, axis=1)
    X = X.reindex(columns=preprocessor.feature_columns, fill_value=0.0)

    # Final defensive assertion: preprocessing itself must not create NaNs.
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaNs remain after fitted preprocessing: {bad}")

    return X


def encode_features(
    panel: pd.DataFrame,
    fit_encoder: bool = True,
    known_dummies: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Backward-compatible feature encoding helper.

    IMPORTANT
    ---------
    This function fits its preprocessing state on ``panel`` when
    ``fit_encoder=True``. Therefore it must only be called with a fitting
    subset in a leakage-sensitive experiment. The hotspot trainer below uses
    ``fit_feature_preprocessor`` / ``transform_features`` explicitly so that
    every validation fold and the final test use training-fitted preprocessing.
    """
    preprocessor = fit_feature_preprocessor(panel)
    X = transform_features(panel, preprocessor)

    if not fit_encoder and known_dummies is not None:
        for column in known_dummies:
            if column not in X.columns:
                X[column] = 0.0
        X = X.reindex(columns=known_dummies, fill_value=0.0)
        return X, list(known_dummies)

    return X, preprocessor.feature_columns.copy()


# ============================================================================
# Splits
# ============================================================================

@dataclass
class Split:
    """A single train/test split with named indices."""

    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    description: str


def temporal_split(panel: pd.DataFrame) -> Split:
    """Train through Aug-2023 and evaluate Sep-2023 onward."""
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
            "Evaluates future-month prediction for known cells."
        ),
    )


def grouped_split(panel: pd.DataFrame) -> Split:
    """Train on 60% of cells and test on 40% unseen cells."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=GROUPED_TEST_SIZE,
        random_state=GROUPED_SPLIT_SEED,
    )

    train_idx, test_idx = next(
        splitter.split(panel, groups=panel["h3_cell"])
    )

    train_cells = panel.iloc[train_idx]["h3_cell"].nunique()
    test_cells = panel.iloc[test_idx]["h3_cell"].nunique()

    return Split(
        name="grouped",
        train_idx=train_idx,
        test_idx=test_idx,
        description=(
            f"Train: {train_cells} cells ({len(train_idx)} rows). "
            f"Test: {test_cells} held-out cells ({len(test_idx)} rows). "
            "Evaluates generalisation to unseen cells."
        ),
    )


def make_internal_validation_split(
    panel: pd.DataFrame,
    split: Split,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create one legacy-compatible internal validation split.

    Kept for compatibility with other training scripts. The hotspot trainer
    uses rolling temporal validation for the primary temporal split.
    """
    train_idx = np.asarray(split.train_idx)

    if split.name == "temporal":
        train_panel = panel.iloc[train_idx].copy()
        months = sorted(train_panel["year_month"].drop_duplicates().tolist())
        if len(months) <= TEMPORAL_VALIDATION_MONTHS:
            raise ValueError("Not enough training months for temporal validation.")

        validation_months = months[-TEMPORAL_VALIDATION_MONTHS:]
        validation_mask = train_panel["year_month"].isin(validation_months)
        fit_idx = train_panel.index[~validation_mask].to_numpy()
        validation_idx = train_panel.index[validation_mask].to_numpy()
        return fit_idx, validation_idx

    if split.name == "grouped":
        train_panel = panel.iloc[train_idx].copy()
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=GROUPED_VALIDATION_TEST_SIZE,
            random_state=GROUPED_VALIDATION_SEED,
        )
        relative_fit_idx, relative_validation_idx = next(
            splitter.split(train_panel, groups=train_panel["h3_cell"])
        )
        return (
            train_idx[relative_fit_idx],
            train_idx[relative_validation_idx],
        )

    raise ValueError(f"Unknown split name: {split.name}")


def make_rolling_temporal_validation_splits(
    panel: pd.DataFrame,
    split: Split,
    validation_months: int = ROLLING_VALIDATION_MONTHS,
    step_months: int = ROLLING_VALIDATION_STEP_MONTHS,
    min_train_months: int = ROLLING_MIN_TRAIN_MONTHS,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create expanding-window temporal validation folds inside train only."""
    if split.name != "temporal":
        raise ValueError("Rolling temporal validation requires the temporal split.")

    train_idx = np.asarray(split.train_idx)
    train_panel = panel.iloc[train_idx].copy()
    months = sorted(train_panel["year_month"].drop_duplicates().tolist())

    if len(months) < min_train_months + validation_months:
        raise ValueError("Not enough training months for rolling validation.")

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    validation_start = min_train_months

    while validation_start + validation_months <= len(months):
        fit_months = months[:validation_start]
        valid_months = months[validation_start:validation_start + validation_months]

        fit_mask = train_panel["year_month"].isin(fit_months)
        valid_mask = train_panel["year_month"].isin(valid_months)

        fit_idx = train_panel.index[fit_mask].to_numpy()
        valid_idx = train_panel.index[valid_mask].to_numpy()

        if len(fit_idx) == 0 or len(valid_idx) == 0:
            break

        folds.append((fit_idx, valid_idx))
        LOGGER.info(
            "Rolling validation fold %d: fit=%s..%s | validation=%s..%s | "
            "fit_rows=%d | validation_rows=%d",
            len(folds),
            fit_months[0], fit_months[-1],
            valid_months[0], valid_months[-1],
            len(fit_idx), len(valid_idx),
        )

        validation_start += step_months

    if not folds:
        raise ValueError("No valid rolling temporal validation folds were created.")

    LOGGER.info("Created %d rolling temporal validation folds.", len(folds))
    return folds


# ============================================================================
# Metrics and threshold tuning
# ============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    task: str = "binary",
    class_labels: Optional[List] = None,
) -> Dict:
    """Compute consistent binary or multiclass metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
        "test_class_distribution": {
            str(k): int(v)
            for k, v in pd.Series(y_true).value_counts().to_dict().items()
        },
    }

    if task == "binary":
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

        if y_proba is not None and len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["average_precision"] = float(
                average_precision_score(y_true, y_proba)
            )
        else:
            metrics["roc_auc"] = None
            metrics["average_precision"] = None

        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        return metrics

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
        y_true,
        y_pred,
        zero_division=0,
        output_dict=True,
        labels=class_labels,
    )
    metrics["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=class_labels
    ).tolist()
    return metrics


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_threshold: float = THRESHOLD_MIN,
    max_threshold: float = THRESHOLD_MAX,
    step: float = THRESHOLD_STEP,
) -> Dict:
    """Select the probability threshold that maximises validation F1 only."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)

    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have the same length.")

    thresholds = np.arange(min_threshold, max_threshold + step / 2, step)
    best = None

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        candidate = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        # F1 is primary. On an exact F1 tie, prefer the threshold with better
        # precision, then the one closest to 0.50.
        if best is None or (
            candidate["f1"],
            candidate["precision"],
            -abs(candidate["threshold"] - 0.50),
        ) > (
            best["f1"],
            best["precision"],
            -abs(best["threshold"] - 0.50),
        ):
            best = candidate

    assert best is not None
    LOGGER.info(
        "Best validation threshold: %.2f | F1=%.4f | precision=%.4f | recall=%.4f",
        best["threshold"], best["f1"], best["precision"], best["recall"],
    )
    return best


def aggregate_validation_metrics(fold_metrics: List[Dict]) -> Dict:
    """Aggregate rolling validation metrics with mean and standard deviation."""
    if not fold_metrics:
        raise ValueError("fold_metrics cannot be empty.")

    metric_names = [
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]
    aggregated = {"n_folds": len(fold_metrics)}

    for metric in metric_names:
        values = [
            result[metric]
            for result in fold_metrics
            if result.get(metric) is not None
        ]
        aggregated[f"mean_{metric}"] = float(np.mean(values)) if values else None
        aggregated[f"std_{metric}"] = float(np.std(values)) if values else None

    return aggregated


# ============================================================================
# Persistence
# ============================================================================

def save_model_artifacts(
    task: str,
    model_name: str,
    model,
    feature_columns: List[str],
    scaler=None,
    threshold: Optional[float] = None,
    preprocessor: Optional[FeaturePreprocessor] = None,
    is_winner: bool = False,
) -> Path:
    """Save model, preprocessing, feature schema, and decision threshold."""
    suffix = "_winner" if is_winner else ""
    path = MODELS_DIR / f"{task}_{model_name}{suffix}.joblib"

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "preprocessor": preprocessor,
            "feature_columns": feature_columns,
            "task": task,
            "model_name": model_name,
            "threshold": threshold,
            "trained_at": datetime.utcnow().isoformat(),
        },
        path,
    )

    LOGGER.info("Saved model: %s", path)
    return path


def save_metrics_report(task: str, report: Dict) -> Path:
    """Save the full metrics report."""
    path = MODELS_DIR / f"{task}_metrics_report.json"
    with open(path, "w", encoding="utf-8") as f:
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
    """Save SHAP artifacts for dashboard consumption."""
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


# ============================================================================
# Logging helpers
# ============================================================================

def log_split_summary(
    split: Split,
    panel: pd.DataFrame,
    target_col: str,
) -> None:
    """Log split sizes and target distributions."""
    LOGGER.info("--- Split: %s ---", split.name)
    LOGGER.info("%s", split.description)

    train_target = panel.iloc[split.train_idx][target_col]
    test_target = panel.iloc[split.test_idx][target_col]

    LOGGER.info("Train class distribution: %s", train_target.value_counts().to_dict())
    LOGGER.info("Test class distribution: %s", test_target.value_counts().to_dict())
