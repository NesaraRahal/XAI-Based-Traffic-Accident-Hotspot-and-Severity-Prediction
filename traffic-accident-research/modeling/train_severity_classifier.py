"""Task 2 — Severity 4-class classification.

Predicts dominant_severity ∈ {Fatal, Serious, Minor, Damage-only} for each
(H3 cell, month) row in the panel.

Key differences from the hotspot script
---------------------------------------
- 4-class multiclass instead of binary
- Class imbalance is severe (Minor 61% / Serious 24% / Fatal 10% / Damage-only 5%)
  → use class_weight='balanced' for LR/RF and sample_weight for XGB
  → report macro-F1 as primary metric (NOT accuracy, which would just reward
    predicting Minor every time)
  → report per-class F1 alongside so the tiny-class instability is visible
- SHAP for multiclass returns per-class shap values (list of arrays)

Outputs
-------
models/severity_logistic_regression.joblib
models/severity_random_forest.joblib
models/severity_xgboost.joblib
models/severity_<winner>_winner.joblib
models/severity_metrics_report.json
models/severity_shap.joblib
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from modelling_utils import (
    compute_metrics,
    configure_logging,
    encode_features,
    grouped_split,
    load_training_panel,
    log_split_summary,
    save_metrics_report,
    save_model_artifacts,
    save_shap_artifacts,
    temporal_split,
)

LOGGER = logging.getLogger(__name__)
TASK = "severity"
TARGET_COL = "dominant_severity"

# For severity with imbalance, macro_f1 is the honest metric
PRIMARY_METRIC = "macro_f1"
PRIMARY_SPLIT = "temporal"

MODEL_SEED = 42

# Consistent class ordering so per-class metrics are always in the same order
CLASS_ORDER = ["Fatal", "Serious", "Minor", "Damage-only"]


# ══════════════════════════════════════════════════════════════════════════
# Model factories
# ══════════════════════════════════════════════════════════════════════════

def make_logistic_regression() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=MODEL_SEED,
        solver="lbfgs",
        multi_class="auto",
    )


def make_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=MODEL_SEED,
        n_jobs=-1,
    )


def make_xgboost() -> XGBClassifier:
    # For multiclass, XGB uses num_class internally. Class balancing is
    # done via sample_weight passed at fit time (see train_and_evaluate).
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=MODEL_SEED,
        n_jobs=-1,
    )


# ══════════════════════════════════════════════════════════════════════════
# Training + evaluation for a single (model, split)
# ══════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    model_name: str,
    X_train, y_train_encoded, X_test, y_test_encoded,
    label_encoder: LabelEncoder,
    scale_features: bool,
) -> Dict:
    """Train one severity model on one split; return metrics dict + fitted model."""

    # Scale features for LR only
    scaler = None
    X_train_used, X_test_used = X_train, X_test
    if scale_features:
        scaler = StandardScaler()
        X_train_used = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_used = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

    # Build + fit model
    if model_name == "logistic_regression":
        model = make_logistic_regression()
        model.fit(X_train_used, y_train_encoded)
    elif model_name == "random_forest":
        model = make_random_forest()
        model.fit(X_train_used, y_train_encoded)
    elif model_name == "xgboost":
        model = make_xgboost()
        sample_weights = compute_sample_weight(class_weight="balanced",
                                               y=y_train_encoded)
        model.fit(X_train_used, y_train_encoded, sample_weight=sample_weights)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Predict + decode back to string labels for metrics
    y_pred_encoded = model.predict(X_test_used)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    y_test = label_encoder.inverse_transform(y_test_encoded)

    # Restrict class_labels to those present in the encoder (in the ladder order)
    present_classes = [c for c in CLASS_ORDER if c in label_encoder.classes_]
    metrics = compute_metrics(
        y_test, y_pred, y_proba=None,
        task="multiclass", class_labels=present_classes,
    )

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "metrics": metrics,
    }


# ══════════════════════════════════════════════════════════════════════════
# SHAP on winner (multiclass)
# ══════════════════════════════════════════════════════════════════════════

def compute_shap_for_winner(
    winner_name: str, winner_bundle: Dict,
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    feature_columns: List[str],
) -> None:
    """For multiclass models, SHAP returns per-class arrays. We save all of them."""

    model = winner_bundle["model"]
    scaler = winner_bundle.get("scaler")

    X_test_for_shap = X_test.copy()
    if scaler is not None:
        X_test_for_shap = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

    LOGGER.info("Computing SHAP values for winner: %s (multiclass)", winner_name)

    if winner_name in ("random_forest", "xgboost"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_for_shap)
    else:
        # LinearExplainer for multiclass LR — SHAP will handle it
        explainer = shap.LinearExplainer(model, X_test_for_shap)
        shap_values = explainer.shap_values(X_test_for_shap)

    save_shap_artifacts(
        task=TASK,
        explainer=explainer,
        shap_values=shap_values,
        feature_columns=feature_columns,
        X_sample=X_test_for_shap,
    )


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    configure_logging()
    LOGGER.info("=== Severity Classifier Training Start ===")

    # Step 1 — Load panel
    panel = load_training_panel()

    # Drop any rows where dominant_severity is missing (shouldn't happen but guard)
    before = len(panel)
    panel = panel.dropna(subset=[TARGET_COL])
    after = len(panel)
    if after < before:
        LOGGER.warning("Dropped %d rows with missing %s", before - after, TARGET_COL)

    # Step 2 — Encode features
    X, feature_columns = encode_features(panel, fit_encoder=True)
    LOGGER.info("Feature matrix shape: %s | %d features",
                X.shape, len(feature_columns))

    # Step 2b — Encode labels (multiclass models want integer targets)
    label_encoder = LabelEncoder()
    label_encoder.fit(panel[TARGET_COL])
    y_encoded = pd.Series(label_encoder.transform(panel[TARGET_COL]),
                          index=panel.index)
    LOGGER.info("Label classes (encoded order): %s",
                list(label_encoder.classes_))
    LOGGER.info("Target distribution (per-cell view, not per-row):")
    LOGGER.info("%s",
                panel.drop_duplicates("h3_cell")[TARGET_COL]
                .value_counts().to_dict())

    # Step 3 — Both splits
    splits = [temporal_split(panel), grouped_split(panel)]
    for split in splits:
        log_split_summary(split, panel, TARGET_COL)

    # Step 4 — Train all 3 models on both splits
    model_names = ["logistic_regression", "random_forest", "xgboost"]
    results: Dict[str, Dict[str, Dict]] = {}

    for split in splits:
        results[split.name] = {}
        X_train, X_test = X.iloc[split.train_idx], X.iloc[split.test_idx]
        y_train_encoded = y_encoded.iloc[split.train_idx]
        y_test_encoded = y_encoded.iloc[split.test_idx]

        # Flag if test is missing a class (macro-F1 will be under-defined)
        test_labels_present = np.unique(
            label_encoder.inverse_transform(y_test_encoded)
        )
        missing_from_test = set(label_encoder.classes_) - set(test_labels_present)
        if missing_from_test:
            LOGGER.warning(
                "[%s split] test classes present: %s. Missing from test: %s. "
                "Macro-F1 will be affected for missing classes.",
                split.name, list(test_labels_present), sorted(missing_from_test),
            )

        for model_name in model_names:
            LOGGER.info("[%s / %s] training...", split.name, model_name)
            scale = (model_name == "logistic_regression")
            bundle = train_and_evaluate(
                model_name, X_train, y_train_encoded, X_test, y_test_encoded,
                label_encoder, scale_features=scale,
            )
            results[split.name][model_name] = bundle
            m = bundle["metrics"]
            LOGGER.info(
                "[%s / %s] acc=%.3f | macro_p=%.3f | macro_r=%.3f | macro_f1=%.3f",
                split.name, model_name,
                m["accuracy"], m["macro_precision"],
                m["macro_recall"], m["macro_f1"],
            )
            if "per_class_f1" in m:
                LOGGER.info("[%s / %s] per_class_f1: %s",
                            split.name, model_name,
                            {k: round(v, 3) for k, v in m["per_class_f1"].items()})

    # Step 5 — Save all trained models (from primary split fits)
    for model_name in model_names:
        bundle = results[PRIMARY_SPLIT][model_name]
        save_model_artifacts(TASK, model_name, bundle["model"], feature_columns)

    # Step 6 — Pick winner by macro-F1 on primary split
    scores = {
        name: results[PRIMARY_SPLIT][name]["metrics"][PRIMARY_METRIC]
        for name in model_names
    }
    winner_name = max(scores, key=scores.get)
    winner_bundle = results[PRIMARY_SPLIT][winner_name]
    LOGGER.info(
        "Winner (by %s on %s split): %s (score=%.3f)",
        PRIMARY_METRIC, PRIMARY_SPLIT, winner_name, scores[winner_name],
    )
    LOGGER.info("All models on %s split by %s: %s",
                PRIMARY_SPLIT, PRIMARY_METRIC,
                {k: round(v, 3) for k, v in scores.items()})

    save_model_artifacts(
        TASK, winner_name, winner_bundle["model"], feature_columns, is_winner=True,
    )

    # Also save the label encoder — the dashboard needs it to decode predictions
    import joblib
    from modelling_utils import MODELS_DIR
    encoder_path = MODELS_DIR / f"{TASK}_label_encoder.joblib"
    joblib.dump(
        {"label_encoder": label_encoder,
         "class_order": CLASS_ORDER},
        encoder_path,
    )
    LOGGER.info("Saved severity label encoder: %s", encoder_path)

    # Step 7 — Save metrics report
    report = {
        "task": TASK,
        "target": TARGET_COL,
        "primary_metric": PRIMARY_METRIC,
        "primary_split": PRIMARY_SPLIT,
        "winner": winner_name,
        "winner_score": scores[winner_name],
        "class_order": CLASS_ORDER,
        "label_classes_present": list(label_encoder.classes_),
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "splits": {
            split.name: {
                "description": split.description,
                "n_train": len(split.train_idx),
                "n_test": len(split.test_idx),
                "models": {
                    name: results[split.name][name]["metrics"]
                    for name in model_names
                },
            }
            for split in splits
        },
    }
    save_metrics_report(TASK, report)

    # Step 8 — SHAP on winner
    winner_split = splits[0]  # temporal
    X_train_win = X.iloc[winner_split.train_idx]
    X_test_win = X.iloc[winner_split.test_idx]
    compute_shap_for_winner(
        winner_name, winner_bundle, X_train_win, X_test_win, feature_columns,
    )

    LOGGER.info("=== Severity Classifier Training Complete ===")


if __name__ == "__main__":
    main()