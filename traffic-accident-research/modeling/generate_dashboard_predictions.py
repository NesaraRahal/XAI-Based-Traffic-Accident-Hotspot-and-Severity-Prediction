"""Generate dashboard predictions with SHAP explanations for the test window.

Loads winner artifacts for hotspot (RF) and severity multiclass (XGB),
scores the temporal test window (2023-09 through 2023-12), computes
per-prediction SHAP top features, and writes one document per (cell, month)
to MongoDB for the dashboard.
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

HOTSPOT_BUNDLE_PATH = MODELS_DIR / "hotspot_random_forest_winner.joblib"
SEVERITY_BUNDLE_PATH = MODELS_DIR / "severity_multiclass_xgboost_winner.joblib"
SEVERITY_ENCODER_PATH = MODELS_DIR / "severity_multiclass_label_encoder.joblib"

OUTPUT_COLLECTION = "dashboard_predictions_cell_month"

TOP_SHAP_FEATURES = 4
HISTORY_WINDOW_DAYS = 180


# ============================================================================
# Human-readable feature labels
# ============================================================================

FEATURE_LABELS: Dict[str, str] = {
    # Location
    "cell_lat": "Cell latitude",
    "cell_lng": "Cell longitude",

    # Rolling accident exposure
    "same_cell_count_30d": "Accidents in this cell (last 30 days)",
    "same_cell_count_90d": "Accidents in this cell (last 90 days)",
    "same_cell_count_180d": "Accidents in this cell (last 180 days)",
    "neighbor_count_30d": "Accidents in neighbouring cells (last 30 days)",
    "neighbor_count_90d": "Accidents in neighbouring cells (last 90 days)",
    "neighbor_count_180d": "Accidents in neighbouring cells (last 180 days)",

    # Seasonality
    "month_sin": "Time of year (seasonal pattern)",
    "month_cos": "Time of year (seasonal pattern)",

    # Weather (previous month)
    "avg_precip_month": "Average rainfall last month",
    "avg_humidity_month": "Average humidity last month",
    "avg_visibility_month": "Average visibility last month",
    "avg_cloudcover_month": "Average cloud cover last month",
    "rainy_days_share_month": "Share of rainy days last month",

    # Calendar
    "holidays_in_month": "Public holidays last month",

    # Behavioural history
    "historical_peak_hour_share_180d":
        "Share of peak-hour accidents (last 180 days)",
    "historical_night_share_180d":
        "Share of night-time accidents (last 180 days)",
    "historical_motorcycle_share_180d":
        "Share of motorcycle-involved accidents (last 180 days)",

    # Police division dummies
    "police_division_Bambalapitiya": "Police division: Bambalapitiya",
    "police_division_Borella": "Police division: Borella",
    "police_division_Cinnamon Gardens": "Police division: Cinnamon Gardens",
    "police_division_Colpetty": "Police division: Colpetty",
    "police_division_Dematagoda": "Police division: Dematagoda",
    "police_division_Fort": "Police division: Fort",
    "police_division_Grandpass": "Police division: Grandpass",
    "police_division_Kirulapone": "Police division: Kirulapone",
    "police_division_Kotahena": "Police division: Kotahena",
    "police_division_Maligawatte": "Police division: Maligawatte",
    "police_division_Maradana": "Police division: Maradana",
    "police_division_Mattakkuliya": "Police division: Mattakkuliya",
    "police_division_Modera": "Police division: Modera",
    "police_division_Narahenpita": "Police division: Narahenpita",
    "police_division_Slave Island": "Police division: Slave Island",
    "police_division_Wellawatte": "Police division: Wellawatte",
    "police_division_Unknown": "Police division: Unknown",
}


def label_for(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


# ============================================================================
# Bundle loading
# ============================================================================

def load_bundle(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    bundle = joblib.load(path)
    LOGGER.info(
        "Loaded bundle: %s | model=%s | trained_at=%s",
        path.name, bundle.get("model_name"), bundle.get("trained_at"),
    )
    return bundle


# ============================================================================
# SHAP helpers
# ============================================================================

def _normalise_shap(
    raw, n_samples: int, n_features: int, n_classes: int,
) -> np.ndarray:
    """Return SHAP values as (n_samples, n_features, n_classes)."""
    if isinstance(raw, np.ndarray):
        if raw.ndim == 3:
            return raw
        if raw.ndim == 2:
            # Binary case returned positive-class array only.
            arr = np.zeros((n_samples, n_features, n_classes), dtype=float)
            arr[:, :, -1] = raw
            arr[:, :, 0] = -raw
            return arr
        raise ValueError(f"Unexpected SHAP array ndim: {raw.ndim}")
    if isinstance(raw, list):
        return np.stack(raw, axis=-1)
    raise TypeError(f"Unrecognised SHAP return type: {type(raw)}")


def top_shap_features(
    shap_row: np.ndarray,
    feature_values: pd.Series,
    feature_columns: List[str],
    k: int,
    positive_label: str,
    negative_label: str,
) -> List[Dict]:
    """Top-k features by absolute SHAP magnitude for one prediction."""
    order = np.argsort(-np.abs(shap_row))[:k]
    entries = []
    for idx in order:
        feature = feature_columns[idx]
        shap_value = float(shap_row[idx])
        raw_value = feature_values.get(feature)
        if isinstance(raw_value, (np.floating, np.integer)):
            raw_value = raw_value.item()
        direction = positive_label if shap_value >= 0 else negative_label
        entries.append({
            "feature": feature,
            "label": label_for(feature),
            "value": raw_value,
            "shap": round(shap_value, 4),
            "direction": direction,
        })
    return entries


# ============================================================================
# Historical context (from panel itself)
# ============================================================================

def compute_historical_context(
    panel: pd.DataFrame,
    row: pd.Series,
    window_days: int = HISTORY_WINDOW_DAYS,
) -> Dict:
    """Cell's own accidents in the preceding window, from panel target counts."""
    month_start = pd.to_datetime(row["month_start"])
    window_start = month_start - pd.Timedelta(days=window_days)

    history = panel[
        (panel["h3_cell"] == row["h3_cell"])
        & (pd.to_datetime(panel["month_start"]) >= window_start)
        & (pd.to_datetime(panel["month_start"]) < month_start)
    ]

    total = int(history["target_accident_count"].sum())
    if total == 0:
        return {"accidents_180d": 0, "severity_mix_180d": None}

    fatal = int(history["target_fatal_count"].sum())
    serious = int(history["target_serious_count"].sum())
    minor = int(history["target_minor_count"].sum())
    damage = int(history["target_damage_only_count"].sum())

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
# Inference
# ============================================================================

def _apply_scaler(X: pd.DataFrame, scaler) -> pd.DataFrame:
    if scaler is None:
        return X
    return pd.DataFrame(
        scaler.transform(X), columns=X.columns, index=X.index,
    )


def score_hotspot(
    bundle: Dict, panel_slice: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    scaler = bundle.get("scaler")
    threshold = bundle.get("threshold") or 0.5
    feature_columns = list(preprocessor.feature_columns)

    X = transform_features(panel_slice, preprocessor)
    X_scaled = _apply_scaler(X, scaler)

    proba = model.predict_proba(X_scaled)[:, 1]
    pred = (proba >= threshold).astype(int)

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_scaled)
    shap_3d = _normalise_shap(raw, len(X_scaled), X_scaled.shape[1], 2)
    shap_pos = shap_3d[:, :, 1]

    return proba, pred, shap_pos, feature_columns, X


def score_severity(
    bundle: Dict, encoder_bundle: Dict, panel_slice: pd.DataFrame,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, List[str], List[str], pd.DataFrame
]:
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    scaler = bundle.get("scaler")
    feature_columns = list(preprocessor.feature_columns)

    encoder = encoder_bundle["label_encoder"]
    class_names = list(encoder.classes_)

    X = transform_features(panel_slice, preprocessor)
    X_scaled = _apply_scaler(X, scaler)

    proba = model.predict_proba(X_scaled)
    pred_idx = np.argmax(proba, axis=1)

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_scaled)
    shap_3d = _normalise_shap(
        raw, len(X_scaled), X_scaled.shape[1], len(class_names)
    )

    return proba, pred_idx, shap_3d, feature_columns, class_names, X


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
) -> List[Dict]:
    documents: List[Dict] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for i, (_, row) in enumerate(panel_slice.iterrows()):
        hotspot_top = top_shap_features(
            shap_row=hotspot_shap[i],
            feature_values=hotspot_X.iloc[i],
            feature_columns=hotspot_feature_columns,
            k=TOP_SHAP_FEATURES,
            positive_label="increases hotspot risk",
            negative_label="reduces hotspot risk",
        )

        pred_idx = int(severity_pred_idx[i])
        pred_class = severity_class_names[pred_idx]
        severity_top = top_shap_features(
            shap_row=severity_shap[i, :, pred_idx],
            feature_values=severity_X.iloc[i],
            feature_columns=severity_feature_columns,
            k=TOP_SHAP_FEATURES,
            positive_label=f"makes {pred_class} more likely",
            negative_label=f"makes {pred_class} less likely",
        )

        class_probs = {
            cls: round(float(severity_proba[i, j]), 4)
            for j, cls in enumerate(severity_class_names)
        }

        history = compute_historical_context(full_panel, row)

        doc = {
            "h3_cell": row["h3_cell"],
            "year_month": row["year_month"],
            "cell_lat": float(row["cell_lat"]),
            "cell_lng": float(row["cell_lng"]),
            "police_division": row.get("police_division"),

            "hotspot": {
                "predicted": bool(hotspot_pred[i]),
                "probability": round(float(hotspot_proba[i]), 4),
                "threshold": float(hotspot_threshold),
                "top_shap_features": hotspot_top,
            },

            "severity": {
                "class_probabilities": class_probs,
                "predicted_class": pred_class,
                "top_shap_features": severity_top,
            },

            "historical_context": history,

            "generated_at": now_iso,
        }
        documents.append(doc)

    return documents


# ============================================================================
# Save
# ============================================================================

def save_documents(documents: List[Dict]) -> None:
    db, client = get_db()
    try:
        collection = db[OUTPUT_COLLECTION]
        deleted = collection.delete_many({})
        LOGGER.info(
            "Cleared %d old documents from [%s]",
            deleted.deleted_count, OUTPUT_COLLECTION,
        )

        if not documents:
            LOGGER.warning("No documents to write.")
            return

        operations = [
            UpdateOne(
                {"h3_cell": doc["h3_cell"], "year_month": doc["year_month"]},
                {"$set": doc},
                upsert=True,
            )
            for doc in documents
        ]
        result = collection.bulk_write(operations)
        collection.create_index(
            [("h3_cell", 1), ("year_month", 1)], unique=True,
        )
        LOGGER.info(
            "[%s] Upserted: %d | Modified: %d | Total docs: %d",
            OUTPUT_COLLECTION,
            result.upserted_count, result.modified_count,
            collection.count_documents({}),
        )
    finally:
        client.close()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    configure_logging()
    LOGGER.info("=== Dashboard prediction generation start ===")

    hotspot_bundle = load_bundle(HOTSPOT_BUNDLE_PATH)
    severity_bundle = load_bundle(SEVERITY_BUNDLE_PATH)
    encoder_bundle = joblib.load(SEVERITY_ENCODER_PATH)

    full_panel = load_training_panel()
    test_panel = (
        full_panel[full_panel["year_month"] >= TEST_START_YEAR_MONTH]
        .sort_values(["year_month", "h3_cell"])
        .reset_index(drop=True)
    )
    LOGGER.info(
        "Test window: %s onward | rows=%d",
        TEST_START_YEAR_MONTH, len(test_panel),
    )

    h_proba, h_pred, h_shap, h_cols, h_X = score_hotspot(
        hotspot_bundle, test_panel
    )
    LOGGER.info(
        "Hotspot scored: %d rows | %d predicted hotspots",
        len(h_proba), int(h_pred.sum()),
    )

    (
        s_proba, s_pred_idx, s_shap, s_cols, s_class_names, s_X
    ) = score_severity(severity_bundle, encoder_bundle, test_panel)
    LOGGER.info(
        "Severity scored: %d rows | classes=%s",
        len(s_proba), s_class_names,
    )

    documents = build_documents(
        panel_slice=test_panel,
        hotspot_proba=h_proba,
        hotspot_pred=h_pred,
        hotspot_shap=h_shap,
        hotspot_feature_columns=h_cols,
        hotspot_X=h_X,
        hotspot_threshold=hotspot_bundle.get("threshold") or 0.5,
        severity_proba=s_proba,
        severity_pred_idx=s_pred_idx,
        severity_shap=s_shap,
        severity_feature_columns=s_cols,
        severity_class_names=s_class_names,
        severity_X=s_X,
        full_panel=full_panel,
    )
    LOGGER.info("Assembled %d documents", len(documents))

    save_documents(documents)
    LOGGER.info("=== Dashboard prediction generation complete ===")


if __name__ == "__main__":
    main()