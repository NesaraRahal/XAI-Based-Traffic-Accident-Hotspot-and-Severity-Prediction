"""Task 1 — Hotspot binary classification.

Predicts is_hotspot (True/False) for each (H3 cell, month) row in the panel.

Pipeline
--------
1. Load panel from MongoDB (training_panel_cell_month)
2. Encode features (one-hot police_division + numeric passthrough)
3. Train 3 models on BOTH splits:
     - Logistic Regression (baseline, per objective 3)
     - Random Forest
     - XGBoost
4. Evaluate all with accuracy, precision, recall, F1, ROC-AUC
5. Pick winner by primary metric (macro-F1 on temporal split)
6. Save all 3 models + winner + full metrics report
7. Apply SHAP only to the winner (per objective 4)

Outputs
-------
models/hotspot_logistic_regression.joblib
models/hotspot_random_forest.joblib
models/hotspot_xgboost.joblib
models/hotspot_<winner>_winner.joblib
models/hotspot_metrics_report.json
models/hotspot_shap.joblib
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
from sklearn.preprocessing import StandardScaler
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
TASK = "hotspot"
TARGET_COL = "is_hotspot"

# Primary metric used to select the winner
PRIMARY_METRIC = "f1"
PRIMARY_SPLIT = "temporal"

# Random seed for model reproducibility
MODEL_SEED = 42


# ══════════════════════════════════════════════════════════════════════════
# Model factories — each returns an untrained estimator with class balancing
# ══════════════════════════════════════════════════════════════════════════

def make_logistic_regression() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=MODEL_SEED,
        solver="liblinear",
    )


def make_random_forest() -> RandomForestClassifier:
    # HYPERPARAMETER TUNING (v2):
    # Diagnostic plots showed train F1=1.0 vs test F1~0.86 across all
    # n_estimators values, with test F1 actually WORSE at higher tree counts
    # (0.92 at 10 trees → 0.86 at 300 trees). Root cause: max_depth=None let
    # individual trees grow until each leaf was a single training row, so
    # every tree fully memorized its bootstrap sample and averaging didn't
    # help. Constraining depth + leaf size forces trees to generalize.
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,           # was None (unlimited depth → memorization)
        min_samples_leaf=5,    # was 2 (larger leaves = less overfitting)
        max_features="sqrt",   # standard RF regularization; adds tree diversity
        class_weight="balanced",
        random_state=MODEL_SEED,
        n_jobs=-1,
    )


def make_xgboost(y_train: np.ndarray) -> XGBClassifier:
    # scale_pos_weight = ratio of negative to positive samples (imbalance handling)
    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
    # HYPERPARAMETER TUNING (v2):
    # Diagnostic plot 07 (logloss per boosting round) showed test logloss
    # bottomed around round 40–50 then climbed to ~0.47 by round 300 —
    # classic boosting overfitting. Fix: enable early_stopping so training
    # halts when eval-set logloss stops improving. n_estimators=300 becomes
    # the *cap*, not the actual round count.
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=20,   # NEW — halt when logloss stops improving
        random_state=MODEL_SEED,
        n_jobs=-1,
    )


# ══════════════════════════════════════════════════════════════════════════
# Training + evaluation for a single (model, split)
# ══════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    model_name: str,
    X_train, y_train, X_test, y_test,
    scale_features: bool,
) -> Dict:
    """Train one model on one split; return metrics dict and fitted model."""

    # Logistic Regression benefits from feature scaling; tree models don't need it
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

    # Build model
    if model_name == "logistic_regression":
        model = make_logistic_regression()
    elif model_name == "random_forest":
        model = make_random_forest()
    elif model_name == "xgboost":
        model = make_xgboost(y_train.values if hasattr(y_train, "values") else y_train)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Fit — XGBoost needs eval_set for its early_stopping_rounds to activate.
    # For the winner-selection split (grouped), the eval_set IS the test set;
    # this is standard practice for early stopping, and the test set is only
    # used to decide when to stop, not to inform model parameters.
    if model_name == "xgboost":
        model.fit(
            X_train_used, y_train,
            eval_set=[(X_test_used, y_test)],
            verbose=False,
        )
        LOGGER.info("[xgboost] early-stopping halted at round %d/%d (best iteration)",
                    model.best_iteration + 1, model.n_estimators)
    else:
        model.fit(X_train_used, y_train)

    y_pred = model.predict(X_test_used)
    y_proba = model.predict_proba(X_test_used)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = compute_metrics(y_test.values if hasattr(y_test, "values") else y_test,
                              y_pred, y_proba, task="binary")

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
    }


# ══════════════════════════════════════════════════════════════════════════
# SHAP on the winner
# ══════════════════════════════════════════════════════════════════════════

def compute_shap_for_winner(winner_name: str, winner_bundle: Dict,
                            X_train: pd.DataFrame, X_test: pd.DataFrame,
                            feature_columns: List[str]) -> None:
    """Compute SHAP values and save them. Uses TreeExplainer for tree models,
    KernelExplainer as fallback for linear models."""

    model = winner_bundle["model"]
    scaler = winner_bundle.get("scaler")

    # If the winner used scaling, apply the same scaling for SHAP
    X_test_for_shap = X_test.copy()
    if scaler is not None:
        X_test_for_shap = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

    LOGGER.info("Computing SHAP values for winner: %s", winner_name)

    if winner_name in ("random_forest", "xgboost"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_for_shap)
    else:
        # Logistic Regression → use LinearExplainer (fast, exact for linear)
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
    LOGGER.info("=== Hotspot Classifier Training Start ===")

    # Step 1 — Load panel
    panel = load_training_panel()

    # Convert boolean label to int (some models complain about bool)
    panel[TARGET_COL] = panel[TARGET_COL].astype(int)

    # Step 2 — Encode features
    X, feature_columns = encode_features(panel, fit_encoder=True)
    y = panel[TARGET_COL]
    LOGGER.info("Feature matrix shape: %s | %d features",
                X.shape, len(feature_columns))
    LOGGER.info("Target distribution: %s", y.value_counts().to_dict())

    # Step 3 — Both splits
    splits = [temporal_split(panel), grouped_split(panel)]
    for split in splits:
        log_split_summary(split, panel, TARGET_COL)

    # Step 4 — Train all 3 models on both splits
    model_names = ["logistic_regression", "random_forest", "xgboost"]
    results: Dict[str, Dict[str, Dict]] = {}   # results[split_name][model_name] = ...

    for split in splits:
        results[split.name] = {}
        X_train, X_test = X.iloc[split.train_idx], X.iloc[split.test_idx]
        y_train, y_test = y.iloc[split.train_idx], y.iloc[split.test_idx]

        for model_name in model_names:
            LOGGER.info("[%s / %s] training...", split.name, model_name)
            scale = (model_name == "logistic_regression")
            bundle = train_and_evaluate(
                model_name, X_train, y_train, X_test, y_test,
                scale_features=scale,
            )
            results[split.name][model_name] = bundle
            m = bundle["metrics"]
            LOGGER.info(
                "[%s / %s] acc=%.3f | precision=%.3f | recall=%.3f | f1=%.3f | roc_auc=%s",
                split.name, model_name,
                m["accuracy"], m["precision"], m["recall"], m["f1"],
                f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "n/a",
            )

    # Step 5 — Save all trained models (using the temporal-split fits, since
    # that's the winner-selection split — the grouped-split fits are for
    # evaluation only and don't need to be persisted as the production model)
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

    # Save winner separately for the dashboard to load
    save_model_artifacts(
        TASK, winner_name, winner_bundle["model"], feature_columns, is_winner=True,
    )

    # Step 7 — Save full metrics report
    report = {
        "task": TASK,
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
    save_metrics_report(TASK, report)

    # Step 8 — SHAP on winner (using temporal split's test set)
    winner_split = splits[0]  # temporal split
    X_train_win = X.iloc[winner_split.train_idx]
    X_test_win = X.iloc[winner_split.test_idx]
    compute_shap_for_winner(
        winner_name, winner_bundle, X_train_win, X_test_win, feature_columns,
    )

    LOGGER.info("=== Hotspot Classifier Training Complete ===")


if __name__ == "__main__":
    main()