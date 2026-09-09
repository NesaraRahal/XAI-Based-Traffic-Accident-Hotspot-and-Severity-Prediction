"""Task 2 — Severity classification (binary or 4-class).

Predicts either:
  - is_severe (binary: Fatal ∪ Serious vs Minor ∪ Damage-only), OR
  - dominant_severity (4-class: Fatal / Serious / Minor / Damage-only)

per (H3 cell, month) row in the panel. Which target is used is controlled
by the SEVERITY_MODE constant below — flip it and rerun to get both sets
of results side-by-side.

Design rationale
----------------
Both modes are supported because:

  * The 4-class version is what the framework originally promised and shows
    that granular severity generalisation to unseen cells is limited by
    data volume (Fatal=8 cells, Damage-only=3 cells) and by the absence
    of infrastructure-level features (road geometry, speed limits) that
    determine crash severity. This is a genuine scientific finding and is
    reported honestly in the thesis.

  * The binary version follows Al-Omari et al. (2025) in reformulating
    severity as an operationally-meaningful Severe/Non-severe distinction
    that aligns with police enforcement prioritisation and is learnable
    from the current feature set.

Both are reported in Chapter 4 as complementary evidence. The binary
model is the one the dashboard uses at inference time; the 4-class result
is a documented limitation.

To run both:
    # Set SEVERITY_MODE = "binary" → run once  → artifacts saved with mode suffix
    # Set SEVERITY_MODE = "multiclass" → run   → artifacts saved with mode suffix

Outputs (per mode):
    models/severity_<mode>_logistic_regression.joblib
    models/severity_<mode>_random_forest.joblib
    models/severity_<mode>_xgboost.joblib
    models/severity_<mode>_<winner>_winner.joblib
    models/severity_<mode>_label_encoder.joblib   (multiclass only)
    models/severity_<mode>_metrics_report.json
    models/severity_<mode>_shap.joblib
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from modelling_utils import (
    MODELS_DIR,
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

# ══════════════════════════════════════════════════════════════════════════
# Mode switch — set this to "binary" or "multiclass" and rerun
# ══════════════════════════════════════════════════════════════════════════
SEVERITY_MODE = "binary"   # "binary"  or  "multiclass"
# ══════════════════════════════════════════════════════════════════════════

if SEVERITY_MODE == "binary":
    TASK = "severity_binary"
    TARGET_COL = "is_severe"
    PRIMARY_METRIC = "f1"
    IS_BINARY = True
elif SEVERITY_MODE == "multiclass":
    TASK = "severity_multiclass"
    TARGET_COL = "dominant_severity"
    PRIMARY_METRIC = "macro_f1"
    IS_BINARY = False
else:
    raise ValueError(f"SEVERITY_MODE must be 'binary' or 'multiclass', got {SEVERITY_MODE}")

PRIMARY_SPLIT = "grouped"
MODEL_SEED = 42

# 4-class ordering (only used when IS_BINARY=False)
CLASS_ORDER = ["Fatal", "Serious", "Minor", "Damage-only"]


# ══════════════════════════════════════════════════════════════════════════
# Model factories
# ══════════════════════════════════════════════════════════════════════════

def make_logistic_regression() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=MODEL_SEED,
        solver="lbfgs" if not IS_BINARY else "liblinear",
    )


def make_random_forest() -> RandomForestClassifier:
    # Same hyperparameter constraints as hotspot classifier — tuned via
    # diagnostic plots to prevent per-tree memorisation on small data.
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=MODEL_SEED,
        n_jobs=-1,
    )


def make_xgboost_binary(y_train: np.ndarray) -> XGBClassifier:
    """Binary XGBoost with scale_pos_weight for class imbalance + early stopping."""
    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=MODEL_SEED,
        n_jobs=-1,
    )


def make_xgboost_multiclass() -> XGBClassifier:
    """Multiclass XGBoost — sample_weight handled at fit time, early stopping enabled."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        random_state=MODEL_SEED,
        n_jobs=-1,
    )


# ══════════════════════════════════════════════════════════════════════════
# Training + evaluation
# ══════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    model_name: str,
    X_train, y_train, X_test, y_test,
    label_encoder,   # None for binary, LabelEncoder for multiclass
    scale_features: bool,
) -> Dict:
    """Train one severity model on one split; return metrics dict + fitted model."""

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

    if model_name == "logistic_regression":
        model = make_logistic_regression()
        model.fit(X_train_used, y_train)

    elif model_name == "random_forest":
        model = make_random_forest()
        model.fit(X_train_used, y_train)

    elif model_name == "xgboost":
        if IS_BINARY:
            y_train_arr = y_train.values if hasattr(y_train, "values") else y_train
            model = make_xgboost_binary(y_train_arr)
            model.fit(
                X_train_used, y_train,
                eval_set=[(X_test_used, y_test)],
                verbose=False,
            )
        else:
            model = make_xgboost_multiclass()
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
            model.fit(
                X_train_used, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_test_used, y_test)],
                verbose=False,
            )
        LOGGER.info("[xgboost] early-stopping halted at round %d/%d (best iteration)",
                    model.best_iteration + 1, model.n_estimators)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    y_pred = model.predict(X_test_used)

    # Metrics differ between binary and multiclass
    if IS_BINARY:
        y_proba = model.predict_proba(X_test_used)[:, 1] if hasattr(model, "predict_proba") else None
        y_true = y_test.values if hasattr(y_test, "values") else y_test
        metrics = compute_metrics(y_true, y_pred, y_proba, task="binary")
    else:
        # For multiclass: decode integer targets back to strings for the metrics report
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        y_test_decoded = label_encoder.inverse_transform(y_test)
        present_classes = [c for c in CLASS_ORDER if c in label_encoder.classes_]
        metrics = compute_metrics(
            y_test_decoded, y_pred_decoded, y_proba=None,
            task="multiclass", class_labels=present_classes,
        )

    return {"model": model, "scaler": scaler, "metrics": metrics}


# ══════════════════════════════════════════════════════════════════════════
# SHAP on winner
# ══════════════════════════════════════════════════════════════════════════

def compute_shap_for_winner(
    winner_name: str, winner_bundle: Dict,
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    feature_columns: List[str],
) -> None:
    """Compute SHAP for the winning model. Works for both binary and multiclass —
    multiclass SHAP returns per-class arrays; we save them all."""

    model = winner_bundle["model"]
    scaler = winner_bundle.get("scaler")

    X_test_for_shap = X_test.copy()
    if scaler is not None:
        X_test_for_shap = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

    LOGGER.info("Computing SHAP values for winner: %s (%s)",
                winner_name, "binary" if IS_BINARY else "multiclass")

    if winner_name in ("random_forest", "xgboost"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_for_shap)
    else:
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
    LOGGER.info("=== Severity Classifier Training Start (mode=%s) ===", SEVERITY_MODE)

    # Step 1 — Load panel
    panel = load_training_panel()

    before = len(panel)
    panel = panel.dropna(subset=[TARGET_COL])
    after = len(panel)
    if after < before:
        LOGGER.warning("Dropped %d rows with missing %s", before - after, TARGET_COL)

    # Step 2 — Encode features
    X, feature_columns = encode_features(panel, fit_encoder=True)
    LOGGER.info("Feature matrix shape: %s | %d features", X.shape, len(feature_columns))

    # Step 2b — Encode targets according to mode
    if IS_BINARY:
        # is_severe is already boolean → convert to int for models
        y_encoded = panel[TARGET_COL].astype(int)
        label_encoder = None
        LOGGER.info("Binary target [%s] distribution (per-row):", TARGET_COL)
        LOGGER.info("%s", y_encoded.value_counts().to_dict())
        LOGGER.info("Binary target (per-cell view):")
        LOGGER.info("%s",
                    panel.drop_duplicates("h3_cell")[TARGET_COL]
                    .value_counts().to_dict())
    else:
        label_encoder = LabelEncoder()
        label_encoder.fit(panel[TARGET_COL])
        y_encoded = pd.Series(label_encoder.transform(panel[TARGET_COL]),
                              index=panel.index)
        LOGGER.info("Label classes (encoded order): %s", list(label_encoder.classes_))
        LOGGER.info("Multiclass target (per-cell view):")
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
        y_train, y_test = y_encoded.iloc[split.train_idx], y_encoded.iloc[split.test_idx]

        # Missing-class warning (multiclass only)
        if not IS_BINARY:
            test_labels_present = np.unique(label_encoder.inverse_transform(y_test))
            missing = set(label_encoder.classes_) - set(test_labels_present)
            if missing:
                LOGGER.warning(
                    "[%s split] test classes present: %s. Missing: %s. "
                    "Macro-F1 will be affected for missing classes.",
                    split.name, list(test_labels_present), sorted(missing),
                )

        for model_name in model_names:
            LOGGER.info("[%s / %s] training...", split.name, model_name)
            scale = (model_name == "logistic_regression")
            bundle = train_and_evaluate(
                model_name, X_train, y_train, X_test, y_test,
                label_encoder=label_encoder, scale_features=scale,
            )
            results[split.name][model_name] = bundle
            m = bundle["metrics"]
            if IS_BINARY:
                LOGGER.info(
                    "[%s / %s] acc=%.3f | precision=%.3f | recall=%.3f | f1=%.3f | roc_auc=%s",
                    split.name, model_name,
                    m["accuracy"], m["precision"], m["recall"], m["f1"],
                    f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "n/a",
                )
            else:
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

    # Step 6 — Pick winner by primary metric on primary split
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

    # Step 6b — Save label encoder (multiclass only)
    if not IS_BINARY:
        encoder_path = MODELS_DIR / f"{TASK}_label_encoder.joblib"
        joblib.dump(
            {"label_encoder": label_encoder, "class_order": CLASS_ORDER},
            encoder_path,
        )
        LOGGER.info("Saved severity label encoder: %s", encoder_path)

    # Step 7 — Save metrics report
    report = {
        "task": TASK,
        "mode": SEVERITY_MODE,
        "target": TARGET_COL,
        "primary_metric": PRIMARY_METRIC,
        "primary_split": PRIMARY_SPLIT,
        "winner": winner_name,
        "winner_score": scores[winner_name],
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
    if not IS_BINARY:
        report["class_order"] = CLASS_ORDER
        report["label_classes_present"] = list(label_encoder.classes_)
    save_metrics_report(TASK, report)

    # Step 8 — SHAP on winner
    winner_split = splits[0]  # temporal (first) — we save SHAP against grouped test below
    # Actually we want SHAP on the primary (grouped) split, which is splits[1].
    primary_split_obj = next(s for s in splits if s.name == PRIMARY_SPLIT)
    X_train_win = X.iloc[primary_split_obj.train_idx]
    X_test_win = X.iloc[primary_split_obj.test_idx]
    compute_shap_for_winner(
        winner_name, winner_bundle, X_train_win, X_test_win, feature_columns,
    )

    LOGGER.info("=== Severity Classifier Training Complete (mode=%s) ===", SEVERITY_MODE)


if __name__ == "__main__":
    main()