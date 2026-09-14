"""Generate dashboard predictions for the final temporal test window.

This script is an OFFLINE artifact-generation step for the dashboard prototype.

It:
1. Loads the final trained hotspot and severity model bundles.
2. Loads the existing training panel from MongoDB.
3. Selects ONLY the final temporal test window: 2023-09 through 2023-12.
4. Generates hotspot and severity predictions.
5. Generates per-prediction SHAP explanations.
6. Adds 180-day historical context available before each target month.
7. Stores observed outcomes separately for evaluation/display.
8. Writes one document per (H3 cell, month) to MongoDB.

Important:
- No model is retrained here.
- No training data is modified.
- Observed outcomes are NEVER used as model inputs.
- This collection is intended for the dashboard prototype/evaluation,
  not as a live production prediction system.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from pymongo import UpdateOne

from modelling_utils import (
    MODELS_DIR,
    configure_logging,
    get_db,
    load_training_panel,
    transform_features,
)

LOGGER = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

TEST_START_YEAR_MONTH = "2023-09"
TEST_END_YEAR_MONTH = "2023-12"

HOTSPOT_BUNDLE_PATH = (
    MODELS_DIR / "hotspot_random_forest_winner.joblib"
)

SEVERITY_BUNDLE_PATH = (
    MODELS_DIR / "severity_multiclass_xgboost_winner.joblib"
)

SEVERITY_ENCODER_PATH = (
    MODELS_DIR / "severity_multiclass_label_encoder.joblib"
)

OUTPUT_COLLECTION = "dashboard_predictions_cell_month"

TOP_SHAP_FEATURES = 4
HISTORY_WINDOW_DAYS = 180


# ============================================================================
# Human-readable feature labels
# ============================================================================

FEATURE_LABELS: Dict[str, str] = {

    # ------------------------------------------------------------------------
    # Location
    # ------------------------------------------------------------------------

    "cell_lat": "Cell latitude",
    "cell_lng": "Cell longitude",

    # ------------------------------------------------------------------------
    # Rolling accident exposure
    # ------------------------------------------------------------------------

    "same_cell_count_30d":
        "Accidents in this cell (last 30 days)",

    "same_cell_count_90d":
        "Accidents in this cell (last 90 days)",

    "same_cell_count_180d":
        "Accidents in this cell (last 180 days)",

    "neighbor_count_30d":
        "Accidents in neighbouring cells (last 30 days)",

    "neighbor_count_90d":
        "Accidents in neighbouring cells (last 90 days)",

    "neighbor_count_180d":
        "Accidents in neighbouring cells (last 180 days)",

    # ------------------------------------------------------------------------
    # Seasonality
    # ------------------------------------------------------------------------

    "month_sin":
        "Time of year (seasonal pattern)",

    "month_cos":
        "Time of year (seasonal pattern)",

    # ------------------------------------------------------------------------
    # Weather
    # These features represent the previous month according to the
    # modelling feature definition.
    # ------------------------------------------------------------------------

    "avg_precip_month":
        "Average rainfall last month",

    "avg_humidity_month":
        "Average humidity last month",

    "avg_visibility_month":
        "Average visibility last month",

    "avg_cloudcover_month":
        "Average cloud cover last month",

    "rainy_days_share_month":
        "Share of rainy days last month",

    # ------------------------------------------------------------------------
    # Calendar
    # ------------------------------------------------------------------------

    # This is target-month calendar information known in advance.
    "holidays_in_month":
        "Public holidays in target month",

    # ------------------------------------------------------------------------
    # Behavioural history
    # ------------------------------------------------------------------------

    "historical_peak_hour_share_180d":
        "Share of peak-hour accidents (last 180 days)",

    "historical_night_share_180d":
        "Share of night-time accidents (last 180 days)",

    "historical_motorcycle_share_180d":
        "Share of motorcycle-involved accidents (last 180 days)",

    # ------------------------------------------------------------------------
    # Police division dummy variables
    # ------------------------------------------------------------------------

    "police_division_Bambalapitiya":
        "Police division: Bambalapitiya",

    "police_division_Borella":
        "Police division: Borella",

    "police_division_Cinnamon Gardens":
        "Police division: Cinnamon Gardens",

    "police_division_Colpetty":
        "Police division: Colpetty",

    "police_division_Dematagoda":
        "Police division: Dematagoda",

    "police_division_Fort":
        "Police division: Fort",

    "police_division_Grandpass":
        "Police division: Grandpass",

    "police_division_Kirulapone":
        "Police division: Kirulapone",

    "police_division_Kotahena":
        "Police division: Kotahena",

    "police_division_Maligawatte":
        "Police division: Maligawatte",

    "police_division_Maradana":
        "Police division: Maradana",

    "police_division_Mattakkuliya":
        "Police division: Mattakkuliya",

    "police_division_Modera":
        "Police division: Modera",

    "police_division_Narahenpita":
        "Police division: Narahenpita",

    "police_division_Slave Island":
        "Police division: Slave Island",

    "police_division_Wellawatte":
        "Police division: Wellawatte",

    "police_division_Unknown":
        "Police division: Unknown",
}


def label_for(feature: str) -> str:
    """Return a human-readable label for a model feature."""
    return FEATURE_LABELS.get(feature, feature)


# ============================================================================
# Bundle loading
# ============================================================================

def load_bundle(path: Path) -> Dict:
    """Load and validate a saved model bundle."""

    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {path}"
        )

    bundle = joblib.load(path)

    LOGGER.info(
        "Loaded bundle: %s | model=%s | trained_at=%s",
        path.name,
        bundle.get("model_name"),
        bundle.get("trained_at"),
    )

    required_keys = {
        "model",
        "preprocessor",
    }

    missing = required_keys - set(bundle.keys())

    if missing:
        raise KeyError(
            f"Model bundle {path.name} is missing required keys: "
            f"{sorted(missing)}"
        )

    return bundle


# ============================================================================
# SHAP helpers
# ============================================================================

def _normalise_shap(
    raw,
    n_samples: int,
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    """Return SHAP values as (n_samples, n_features, n_classes)."""

    if isinstance(raw, np.ndarray):

        if raw.ndim == 3:
            return raw

        if raw.ndim == 2:
            # Binary SHAP output.
            arr = np.zeros(
                (n_samples, n_features, n_classes),
                dtype=float,
            )

            arr[:, :, -1] = raw
            arr[:, :, 0] = -raw

            return arr

        raise ValueError(
            f"Unexpected SHAP array ndim: {raw.ndim}"
        )

    if isinstance(raw, list):
        return np.stack(raw, axis=-1)

    raise TypeError(
        f"Unrecognised SHAP return type: {type(raw)}"
    )


def top_shap_features(
    shap_row: np.ndarray,
    feature_values: pd.Series,
    feature_columns: List[str],
    k: int,
    positive_label: str,
    negative_label: str,
) -> List[Dict]:
    """Return the top-k features by absolute SHAP magnitude."""

    order = np.argsort(-np.abs(shap_row))[:k]

    entries: List[Dict] = []

    for idx in order:

        feature = feature_columns[idx]
        shap_value = float(shap_row[idx])

        raw_value = feature_values.get(feature)

        if isinstance(raw_value, (np.floating, np.integer)):
            raw_value = raw_value.item()

        direction = (
            positive_label
            if shap_value >= 0
            else negative_label
        )

        entries.append(
            {
                "feature": feature,
                "label": label_for(feature),
                "value": raw_value,
                "shap": round(shap_value, 4),
                "direction": direction,
            }
        )

    return entries


# ============================================================================
# Historical context
# ============================================================================

def compute_historical_context(
    panel: pd.DataFrame,
    row: pd.Series,
    window_days: int = HISTORY_WINDOW_DAYS,
) -> Dict:
    """Calculate pre-target historical accident context.

    Only months before the target month are included.

    This is dashboard context only. These values are not passed to the
    prediction models by this function.
    """

    month_start = pd.to_datetime(row["month_start"])

    window_start = (
        month_start - pd.Timedelta(days=window_days)
    )

    history = panel[
        (panel["h3_cell"] == row["h3_cell"])
        & (
            pd.to_datetime(panel["month_start"])
            >= window_start
        )
        & (
            pd.to_datetime(panel["month_start"])
            < month_start
        )
    ]

    total = int(
        history["target_accident_count"].sum()
    )

    if total == 0:
        return {
            "accidents_180d": 0,
            "severity_mix_180d": None,
        }

    fatal = int(
        history["target_fatal_count"].sum()
    )

    serious = int(
        history["target_serious_count"].sum()
    )

    minor = int(
        history["target_minor_count"].sum()
    )

    damage = int(
        history["target_damage_only_count"].sum()
    )

    return {
        "accidents_180d": total,
        "severity_mix_180d": {
            "Fatal": round(fatal / total, 3),
            "Serious": round(serious / total, 3),
            "Minor": round(minor / total, 3),
            "Damage-only": round(damage / total, 3),
        },
    }


# ============================================================================
# Model inference helpers
# ============================================================================

def _apply_scaler(
    X: pd.DataFrame,
    scaler,
) -> pd.DataFrame:
    """Apply the saved scaler, if the model requires one."""

    if scaler is None:
        return X

    return pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )


def score_hotspot(
    bundle: Dict,
    panel_slice: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    pd.DataFrame,
]:
    """Generate hotspot predictions and SHAP values."""

    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    scaler = bundle.get("scaler")

    threshold = bundle.get("threshold")

    if threshold is None:
        threshold = 0.5

    feature_columns = list(
        preprocessor.feature_columns
    )

    X = transform_features(
        panel_slice,
        preprocessor,
    )

    X_scaled = _apply_scaler(
        X,
        scaler,
    )

    probabilities = model.predict_proba(
        X_scaled
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    explainer = shap.TreeExplainer(model)

    raw_shap = explainer.shap_values(
        X_scaled
    )

    shap_3d = _normalise_shap(
        raw_shap,
        len(X_scaled),
        X_scaled.shape[1],
        2,
    )

    shap_positive = shap_3d[:, :, 1]

    return (
        probabilities,
        predictions,
        shap_positive,
        feature_columns,
        X,
    )


def score_severity(
    bundle: Dict,
    encoder_bundle: Dict,
    panel_slice: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    List[str],
    pd.DataFrame,
]:
    """Generate multiclass severity predictions and SHAP values."""

    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    scaler = bundle.get("scaler")

    feature_columns = list(
        preprocessor.feature_columns
    )

    encoder = encoder_bundle["label_encoder"]

    class_names = list(
        encoder.classes_
    )

    X = transform_features(
        panel_slice,
        preprocessor,
    )

    X_scaled = _apply_scaler(
        X,
        scaler,
    )

    probabilities = model.predict_proba(
        X_scaled
    )

    prediction_indices = np.argmax(
        probabilities,
        axis=1,
    )

    explainer = shap.TreeExplainer(model)

    raw_shap = explainer.shap_values(
        X_scaled
    )

    shap_3d = _normalise_shap(
        raw_shap,
        len(X_scaled),
        X_scaled.shape[1],
        len(class_names),
    )

    return (
        probabilities,
        prediction_indices,
        shap_3d,
        feature_columns,
        class_names,
        X,
    )


# ============================================================================
# Observed outcomes
# ============================================================================

def build_observed_outcome(
    row: pd.Series,
) -> Dict:
    """Build evaluation-only observed outcome information.

    These fields describe what actually happened in the held-out test
    window. They are NOT model inputs.
    """

    severity_defined = bool(
        row.get("severity_defined", False)
    )

    dominant_severity = row.get(
        "dominant_severity"
    )

    if pd.isna(dominant_severity):
        dominant_severity = None

    return {
        "accident_count": int(
            row.get("target_accident_count", 0)
        ),
        "is_hotspot": bool(
            row.get("is_hotspot", False)
        ),
        "severity_defined": severity_defined,
        "dominant_severity": dominant_severity,
    }


# ============================================================================
# Document assembly
# ============================================================================

def build_documents(
    panel_slice: pd.DataFrame,
    hotspot_proba: np.ndarray,
    hotspot_pred: np.ndarray,
    hotspot_shap: np.ndarray,
    hotspot_feature_columns: List[str],
    hotspot_X: pd.DataFrame,
    hotspot_threshold: float,
    severity_proba: np.ndarray,
    severity_pred_idx: np.ndarray,
    severity_shap: np.ndarray,
    severity_feature_columns: List[str],
    severity_class_names: List[str],
    severity_X: pd.DataFrame,
    full_panel: pd.DataFrame,
    hotspot_model_name: str,
    severity_model_name: str,
) -> List[Dict]:
    """Assemble MongoDB dashboard documents."""

    documents: List[Dict] = []

    generated_at = datetime.now(
        timezone.utc
    ).isoformat()

    for i, (_, row) in enumerate(
        panel_slice.iterrows()
    ):

        # --------------------------------------------------------------------
        # Hotspot SHAP
        # --------------------------------------------------------------------

        hotspot_top = top_shap_features(
            shap_row=hotspot_shap[i],
            feature_values=hotspot_X.iloc[i],
            feature_columns=hotspot_feature_columns,
            k=TOP_SHAP_FEATURES,
            positive_label="increases hotspot risk",
            negative_label="reduces hotspot risk",
        )

        # --------------------------------------------------------------------
        # Severity prediction
        # --------------------------------------------------------------------

        prediction_index = int(
            severity_pred_idx[i]
        )

        predicted_severity = (
            severity_class_names[prediction_index]
        )

        severity_top = top_shap_features(
            shap_row=severity_shap[
                i,
                :,
                prediction_index,
            ],
            feature_values=severity_X.iloc[i],
            feature_columns=severity_feature_columns,
            k=TOP_SHAP_FEATURES,
            positive_label=(
                f"makes {predicted_severity} more likely"
            ),
            negative_label=(
                f"makes {predicted_severity} less likely"
            ),
        )

        class_probabilities = {
            class_name: round(
                float(
                    severity_proba[i, class_index]
                ),
                4,
            )
            for class_index, class_name
            in enumerate(severity_class_names)
        }

        # --------------------------------------------------------------------
        # Historical context
        # --------------------------------------------------------------------

        historical_context = (
            compute_historical_context(
                full_panel,
                row,
            )
        )

        # --------------------------------------------------------------------
        # Observed test outcome
        # --------------------------------------------------------------------

        observed = build_observed_outcome(
            row
        )

        # --------------------------------------------------------------------
        # Final document
        # --------------------------------------------------------------------

        document = {
            "h3_cell": row["h3_cell"],
            "year_month": row["year_month"],

            "location": {
                "cell_lat": float(
                    row["cell_lat"]
                ),
                "cell_lng": float(
                    row["cell_lng"]
                ),
                "police_division": row.get(
                    "police_division"
                ),
            },

            "hotspot": {
                "predicted": bool(
                    hotspot_pred[i]
                ),
                "probability": round(
                    float(hotspot_proba[i]),
                    4,
                ),
                "threshold": float(
                    hotspot_threshold
                ),
                "top_shap_features": hotspot_top,
            },

            "severity": {
                "predicted_class": predicted_severity,
                "class_probabilities": (
                    class_probabilities
                ),
                "top_shap_features": severity_top,
            },

            "historical_context": historical_context,

            # Evaluation-only information.
            # Never used as model input.
            "observed": observed,

            "model_metadata": {
                "hotspot_model": hotspot_model_name,
                "severity_model": severity_model_name,
                "evaluation_window": (
                    f"{TEST_START_YEAR_MONTH} "
                    f"to {TEST_END_YEAR_MONTH}"
                ),
            },

            "generated_at": generated_at,
        }

        documents.append(document)

    return documents


# ============================================================================
# Save
# ============================================================================

def save_documents(
    documents: List[Dict],
) -> None:
    """Replace the dashboard prediction collection."""

    db, client = get_db()

    try:

        collection = db[
            OUTPUT_COLLECTION
        ]

        deleted = collection.delete_many({})

        LOGGER.info(
            "Cleared %d old documents from [%s]",
            deleted.deleted_count,
            OUTPUT_COLLECTION,
        )

        if not documents:
            LOGGER.warning(
                "No documents to write."
            )
            return

        operations = [
            UpdateOne(
                {
                    "h3_cell": doc["h3_cell"],
                    "year_month": doc["year_month"],
                },
                {
                    "$set": doc
                },
                upsert=True,
            )
            for doc in documents
        ]

        result = collection.bulk_write(
            operations
        )

        collection.create_index(
            [
                ("h3_cell", 1),
                ("year_month", 1),
            ],
            unique=True,
        )

        LOGGER.info(
            "[%s] Upserted: %d | Modified: %d | "
            "Total docs: %d",
            OUTPUT_COLLECTION,
            result.upserted_count,
            result.modified_count,
            collection.count_documents({}),
        )

    finally:
        client.close()


# ============================================================================
# Test-window validation
# ============================================================================

def validate_test_window(
    test_panel: pd.DataFrame,
) -> None:
    """Validate that only the intended test window was selected."""

    if test_panel.empty:
        raise ValueError(
            "Test panel is empty."
        )

    months = sorted(
        test_panel["year_month"]
        .astype(str)
        .unique()
    )

    expected_months = [
        "2023-09",
        "2023-10",
        "2023-11",
        "2023-12",
    ]

    if months != expected_months:
        raise ValueError(
            "Unexpected dashboard test window. "
            f"Found months={months}; "
            f"expected={expected_months}"
        )

    if (
        test_panel["year_month"].astype(str).min()
        < TEST_START_YEAR_MONTH
    ):
        raise ValueError(
            "Rows before the configured test start "
            "were included."
        )

    if (
        test_panel["year_month"].astype(str).max()
        > TEST_END_YEAR_MONTH
    ):
        raise ValueError(
            "Rows after the configured test end "
            "were included."
        )

    LOGGER.info(
        "Validated test window: %s to %s | rows=%d | "
        "months=%s",
        TEST_START_YEAR_MONTH,
        TEST_END_YEAR_MONTH,
        len(test_panel),
        months,
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    configure_logging()

    LOGGER.info(
        "=== Dashboard prediction generation start ==="
    )

    # ------------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------------

    hotspot_bundle = load_bundle(
        HOTSPOT_BUNDLE_PATH
    )

    severity_bundle = load_bundle(
        SEVERITY_BUNDLE_PATH
    )

    if not SEVERITY_ENCODER_PATH.exists():
        raise FileNotFoundError(
            "Severity label encoder not found: "
            f"{SEVERITY_ENCODER_PATH}"
        )

    encoder_bundle = joblib.load(
        SEVERITY_ENCODER_PATH
    )

    # ------------------------------------------------------------------------
    # Load existing modelling panel
    # ------------------------------------------------------------------------

    full_panel = load_training_panel()

    if full_panel.empty:
        raise ValueError(
            "Training panel is empty."
        )

    LOGGER.info(
        "Loaded training panel: %d rows",
        len(full_panel),
    )

    # ------------------------------------------------------------------------
    # Select ONLY final temporal test window
    # ------------------------------------------------------------------------

    year_month = (
        full_panel["year_month"]
        .astype(str)
    )

    test_panel = (
        full_panel[
            (year_month >= TEST_START_YEAR_MONTH)
            & (year_month <= TEST_END_YEAR_MONTH)
        ]
        .sort_values(
            ["year_month", "h3_cell"]
        )
        .reset_index(drop=True)
    )

    validate_test_window(
        test_panel
    )

    # ------------------------------------------------------------------------
    # Hotspot inference
    # ------------------------------------------------------------------------

    (
        hotspot_proba,
        hotspot_pred,
        hotspot_shap,
        hotspot_feature_columns,
        hotspot_X,
    ) = score_hotspot(
        hotspot_bundle,
        test_panel,
    )

    LOGGER.info(
        "Hotspot scored: %d rows | "
        "%d predicted hotspots | threshold=%.4f",
        len(hotspot_proba),
        int(hotspot_pred.sum()),
        float(
            hotspot_bundle.get(
                "threshold",
                0.5,
            )
        ),
    )

    # ------------------------------------------------------------------------
    # Severity inference
    # ------------------------------------------------------------------------

    (
        severity_proba,
        severity_pred_idx,
        severity_shap,
        severity_feature_columns,
        severity_class_names,
        severity_X,
    ) = score_severity(
        severity_bundle,
        encoder_bundle,
        test_panel,
    )

    LOGGER.info(
        "Severity scored: %d rows | classes=%s",
        len(severity_proba),
        severity_class_names,
    )

    # ------------------------------------------------------------------------
    # Build dashboard documents
    # ------------------------------------------------------------------------

    documents = build_documents(
        panel_slice=test_panel,

        hotspot_proba=hotspot_proba,
        hotspot_pred=hotspot_pred,
        hotspot_shap=hotspot_shap,
        hotspot_feature_columns=(
            hotspot_feature_columns
        ),
        hotspot_X=hotspot_X,
        hotspot_threshold=float(
            hotspot_bundle.get(
                "threshold",
                0.5,
            )
        ),

        severity_proba=severity_proba,
        severity_pred_idx=severity_pred_idx,
        severity_shap=severity_shap,
        severity_feature_columns=(
            severity_feature_columns
        ),
        severity_class_names=severity_class_names,
        severity_X=severity_X,

        full_panel=full_panel,

        hotspot_model_name=(
            hotspot_bundle.get(
                "model_name",
                "random_forest",
            )
        ),

        severity_model_name=(
            severity_bundle.get(
                "model_name",
                "xgboost",
            )
        ),
    )

    LOGGER.info(
        "Assembled %d dashboard documents",
        len(documents),
    )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    save_documents(
        documents
    )

    LOGGER.info(
        "=== Dashboard prediction generation complete ==="
    )


if __name__ == "__main__":
    main()