"""Task 1 — Hotspot binary classification.

Predicts ``is_hotspot`` for each (H3 cell, month) row.

Evaluation design
-----------------
1. The final temporal test period is Sep-Dec 2023 and is never used for
   model selection, threshold selection, preprocessing fitting, or XGBoost
   early stopping.
2. Model selection on the temporal split uses expanding rolling validation
   folds inside the training period.
3. A single decision threshold is selected from pooled rolling-validation
   predictions for each model and then frozen before final test evaluation.
4. Numeric imputation and categorical encoding are fitted separately inside
   each validation fold and again on the complete training period for the
   final model.
5. Logistic Regression is scaled; tree models are not.
6. Average Precision is reported as the PR-oriented metric for the rare
   hotspot class, alongside F1, precision, recall and ROC-AUC.
7. SHAP is computed only for the selected temporal winner.
"""

from __future__ import annotations

import logging
from statistics import median
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from modelling_utils import (
    FeaturePreprocessor,
    aggregate_validation_metrics,
    compute_metrics,
    configure_logging,
    find_best_threshold,
    fit_feature_preprocessor,
    grouped_split,
    load_training_panel,
    log_split_summary,
    make_internal_validation_split,
    make_rolling_temporal_validation_splits,
    save_metrics_report,
    save_model_artifacts,
    save_shap_artifacts,
    temporal_split,
    transform_features,
)


LOGGER = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

TASK = "hotspot"
TARGET_COL = "is_hotspot"
PRIMARY_METRIC = "f1"
PRIMARY_SPLIT = "temporal"
MODEL_SEED = 42

THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.90
THRESHOLD_STEP = 0.01

XGB_N_ESTIMATORS = 300
XGB_EARLY_STOPPING_ROUNDS = 20


# ============================================================================
# Model factories
# ============================================================================

def make_logistic_regression() -> LogisticRegression:
    """Create Logistic Regression baseline."""
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=MODEL_SEED,
        solver="liblinear",
    )


def make_random_forest() -> RandomForestClassifier:
    """Create regularised Random Forest."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=MODEL_SEED,
        n_jobs=-1,
    )


def make_xgboost(
    y_train: np.ndarray,
    n_estimators: int = XGB_N_ESTIMATORS,
) -> XGBClassifier:
    """Create XGBoost with class-imbalance handling."""
    y_train = np.asarray(y_train).astype(int)
    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    return XGBClassifier(
        n_estimators=max(1, int(n_estimators)),
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=MODEL_SEED,
        n_jobs=-1,
    )


# ============================================================================
# Scaling
# ============================================================================

def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on training data only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_with_scaler(
    scaler: StandardScaler,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Transform data with an already fitted scaler."""
    return pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )


# ============================================================================
# Predictions and thresholding
# ============================================================================

def get_probability(model, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities."""
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"Model {type(model).__name__} lacks predict_proba().")
    return model.predict_proba(X)[:, 1]


def predictions_from_threshold(
    y_proba: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Convert positive-class probabilities to binary predictions."""
    return (np.asarray(y_proba) >= threshold).astype(int)


def safe_auc_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Tuple[float | None, float | None]:
    """Return ROC-AUC and Average Precision when both classes exist."""
    if len(np.unique(y_true)) < 2:
        return None, None
    return (
        float(roc_auc_score(y_true, y_proba)),
        float(average_precision_score(y_true, y_proba)),
    )


# ============================================================================
# Model fitting helper
# ============================================================================

def fit_validation_model(
    model_name: str,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    scale_features: bool,
) -> Tuple[object, StandardScaler | None, int | None]:
    """Fit one validation-fold model without touching the final test set."""
    scaler = None
    X_fit_used = X_fit
    X_validation_used = X_validation

    if scale_features:
        scaler = fit_scaler(X_fit)
        X_fit_used = transform_with_scaler(scaler, X_fit)
        X_validation_used = transform_with_scaler(scaler, X_validation)

    y_fit_np = np.asarray(y_fit).astype(int)

    if model_name == "logistic_regression":
        model = make_logistic_regression()
        model.fit(X_fit_used, y_fit)
        return model, scaler, None

    if model_name == "random_forest":
        model = make_random_forest()
        model.fit(X_fit_used, y_fit)
        return model, scaler, None

    if model_name == "xgboost":
        model = make_xgboost(y_fit_np, XGB_N_ESTIMATORS)
        model.set_params(early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS)
        model.fit(
            X_fit_used,
            y_fit,
            eval_set=[(X_validation_used, y_validation)],
            verbose=False,
        )

        best_iteration = getattr(model, "best_iteration", None)
        best_n_estimators = (
            int(best_iteration) + 1
            if best_iteration is not None and int(best_iteration) >= 0
            else XGB_N_ESTIMATORS
        )
        return model, scaler, best_n_estimators

    raise ValueError(f"Unknown model_name: {model_name}")


# ============================================================================
# Rolling temporal validation
# ============================================================================

def rolling_validate_model(
    model_name: str,
    panel: pd.DataFrame,
    y: pd.Series,
    temporal_split_obj,
    scale_features: bool,
) -> Dict:
    """Run expanding rolling validation for one model.

    Every fold independently fits preprocessing on its fit period. Validation
    predictions are pooled across folds, then ONE threshold is selected from
    those pooled out-of-fold predictions. This gives a single threshold for
    the final model without using the final test period.
    """
    folds = make_rolling_temporal_validation_splits(panel, temporal_split_obj)

    pooled_true: List[np.ndarray] = []
    pooled_proba: List[np.ndarray] = []
    fold_metrics: List[Dict] = []
    fold_thresholds: List[float] = []
    xgb_rounds: List[int] = []
    fold_details: List[Dict] = []

    for fold_number, (fit_idx, validation_idx) in enumerate(folds, start=1):
        fit_panel = panel.iloc[fit_idx]
        validation_panel = panel.iloc[validation_idx]
        y_fit = y.iloc[fit_idx]
        y_validation = y.iloc[validation_idx]

        # CRITICAL: fit imputation/category schema on this fold's fit period.
        preprocessor = fit_feature_preprocessor(fit_panel)
        X_fit = transform_features(fit_panel, preprocessor)
        X_validation = transform_features(validation_panel, preprocessor)

        model, scaler, best_n_estimators = fit_validation_model(
            model_name=model_name,
            X_fit=X_fit,
            y_fit=y_fit,
            X_validation=X_validation,
            y_validation=y_validation,
            scale_features=scale_features,
        )

        X_validation_used = X_validation
        if scaler is not None:
            X_validation_used = transform_with_scaler(scaler, X_validation)

        validation_proba = get_probability(model, X_validation_used)
        y_validation_np = np.asarray(y_validation).astype(int)

        fold_threshold_result = find_best_threshold(
            y_validation_np,
            validation_proba,
            min_threshold=THRESHOLD_MIN,
            max_threshold=THRESHOLD_MAX,
            step=THRESHOLD_STEP,
        )
        fold_thresholds.append(fold_threshold_result["threshold"])

        fold_pred = predictions_from_threshold(
            validation_proba,
            fold_threshold_result["threshold"],
        )
        fold_metrics_result = compute_metrics(
            y_validation_np,
            fold_pred,
            validation_proba,
            task="binary",
        )
        fold_metrics.append(fold_metrics_result)
        pooled_true.append(y_validation_np)
        pooled_proba.append(validation_proba)

        if best_n_estimators is not None:
            xgb_rounds.append(best_n_estimators)

        valid_months = sorted(validation_panel["year_month"].unique().tolist())
        fold_details.append(
            {
                "fold": fold_number,
                "fit_rows": int(len(fit_idx)),
                "validation_rows": int(len(validation_idx)),
                "fit_end": str(fit_panel["year_month"].max()),
                "validation_start": str(valid_months[0]),
                "validation_end": str(valid_months[-1]),
                "validation_positive_count": int(y_validation_np.sum()),
                "validation_metrics": fold_metrics_result,
                "fold_threshold": fold_threshold_result["threshold"],
                "best_n_estimators": best_n_estimators,
            }
        )

        LOGGER.info(
            "[%s] fold %d: threshold=%.2f | F1=%.3f | AP=%s | positives=%d",
            model_name,
            fold_number,
            fold_threshold_result["threshold"],
            fold_metrics_result["f1"],
            (
                f"{fold_metrics_result['average_precision']:.3f}"
                if fold_metrics_result.get("average_precision") is not None
                else "n/a"
            ),
            int(y_validation_np.sum()),
        )

    pooled_y = np.concatenate(pooled_true)
    pooled_p = np.concatenate(pooled_proba)

    # Select the one threshold that will be frozen for final test evaluation.
    pooled_threshold_result = find_best_threshold(
        pooled_y,
        pooled_p,
        min_threshold=THRESHOLD_MIN,
        max_threshold=THRESHOLD_MAX,
        step=THRESHOLD_STEP,
    )
    pooled_threshold = pooled_threshold_result["threshold"]
    pooled_pred = predictions_from_threshold(pooled_p, pooled_threshold)
    pooled_metrics = compute_metrics(
        pooled_y,
        pooled_pred,
        pooled_p,
        task="binary",
    )

    aggregated = aggregate_validation_metrics(fold_metrics)

    if xgb_rounds:
        final_n_estimators = max(1, int(round(median(xgb_rounds))))
    else:
        final_n_estimators = None

    LOGGER.info(
        "[%s] pooled rolling validation: threshold=%.2f | F1=%.3f | AP=%s",
        model_name,
        pooled_threshold,
        pooled_metrics["f1"],
        (
            f"{pooled_metrics['average_precision']:.3f}"
            if pooled_metrics.get("average_precision") is not None
            else "n/a"
        ),
    )

    return {
        "validation_metrics": pooled_metrics,
        "fold_metrics_summary": aggregated,
        "fold_details": fold_details,
        "best_threshold": pooled_threshold,
        "best_n_estimators": final_n_estimators,
        "pooled_validation_y": pooled_y,
        "pooled_validation_proba": pooled_p,
        "selection_score": pooled_metrics[PRIMARY_METRIC],
    }


# ============================================================================
# Final fitting and test evaluation
# ============================================================================

def fit_final_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_features: bool,
    xgb_n_estimators: int | None,
) -> Tuple[object, StandardScaler | None]:
    """Fit final model on the complete training portion only."""
    scaler = None
    X_train_used = X_train

    if scale_features:
        scaler = fit_scaler(X_train)
        X_train_used = transform_with_scaler(scaler, X_train)

    y_train_np = np.asarray(y_train).astype(int)

    if model_name == "logistic_regression":
        model = make_logistic_regression()
    elif model_name == "random_forest":
        model = make_random_forest()
    elif model_name == "xgboost":
        model = make_xgboost(
            y_train_np,
            n_estimators=xgb_n_estimators or XGB_N_ESTIMATORS,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # No test data and no early stopping are used during final fitting.
    model.fit(X_train_used, y_train)
    return model, scaler


def evaluate_final_test(
    model,
    scaler: StandardScaler | None,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> Dict:
    """Evaluate untouched test data using the frozen validation threshold."""
    X_test_used = X_test
    if scaler is not None:
        X_test_used = transform_with_scaler(scaler, X_test)

    y_test_np = np.asarray(y_test).astype(int)
    test_proba = get_probability(model, X_test_used)
    test_pred = predictions_from_threshold(test_proba, threshold)

    metrics = compute_metrics(
        y_test_np,
        test_pred,
        test_proba,
        task="binary",
    )
    metrics["threshold"] = float(threshold)

    return metrics


# ============================================================================
# SHAP
# ============================================================================

def compute_shap_for_winner(
    winner_name: str,
    winner_bundle: Dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_columns: List[str],
) -> None:
    """Compute SHAP values for the selected winner."""
    model = winner_bundle["model"]
    scaler = winner_bundle.get("scaler")

    X_train_for_shap = X_train.copy()
    X_test_for_shap = X_test.copy()

    if scaler is not None:
        X_train_for_shap = transform_with_scaler(scaler, X_train_for_shap)
        X_test_for_shap = transform_with_scaler(scaler, X_test_for_shap)

    LOGGER.info("Computing SHAP values for winner: %s", winner_name)

    if winner_name in ("random_forest", "xgboost"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_for_shap)
    else:
        explainer = shap.LinearExplainer(model, X_train_for_shap)
        shap_values = explainer.shap_values(X_test_for_shap)

    save_shap_artifacts(
        task=TASK,
        explainer=explainer,
        shap_values=shap_values,
        feature_columns=feature_columns,
        X_sample=X_test_for_shap,
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    configure_logging()
    LOGGER.info("=== Hotspot Classifier Training Start ===")

    # ------------------------------------------------------------------------
    # Step 1 — Load panel and target
    # ------------------------------------------------------------------------
    panel = load_training_panel()

    before = len(panel)
    panel = panel.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    if len(panel) < before:
        LOGGER.warning(
            "Dropped %d rows with null %s",
            before - len(panel), TARGET_COL,
        )

    panel[TARGET_COL] = panel[TARGET_COL].astype(int)
    y = panel[TARGET_COL]

    # ------------------------------------------------------------------------
    # Step 2 — Create final splits BEFORE any feature preprocessing
    # ------------------------------------------------------------------------
    splits = [temporal_split(panel), grouped_split(panel)]
    for split in splits:
        log_split_summary(split, panel, TARGET_COL)

    model_names = [
        "logistic_regression",
        "random_forest",
        "xgboost",
    ]

    results: Dict[str, Dict[str, Dict]] = {
        "temporal": {},
        "grouped": {},
    }

    # ------------------------------------------------------------------------
    # Step 3 — Primary temporal rolling validation + final temporal test
    # ------------------------------------------------------------------------
    temporal = next(s for s in splits if s.name == "temporal")
    temporal_train_panel = panel.iloc[temporal.train_idx]
    temporal_test_panel = panel.iloc[temporal.test_idx]
    y_temporal_train = y.iloc[temporal.train_idx]
    y_temporal_test = y.iloc[temporal.test_idx]

    LOGGER.info("============================================================")
    LOGGER.info("PRIMARY TEMPORAL EVALUATION")
    LOGGER.info("============================================================")

    temporal_preprocessor = fit_feature_preprocessor(temporal_train_panel)
    X_temporal_train = transform_features(
        temporal_train_panel,
        temporal_preprocessor,
    )
    X_temporal_test = transform_features(
        temporal_test_panel,
        temporal_preprocessor,
    )

    LOGGER.info(
        "Final temporal preprocessing fitted on %d training rows only: %d features",
        len(temporal_train_panel),
        len(temporal_preprocessor.feature_columns),
    )

    for model_name in model_names:
        scale = model_name == "logistic_regression"

        validation_result = rolling_validate_model(
            model_name=model_name,
            panel=panel,
            y=y,
            temporal_split_obj=temporal,
            scale_features=scale,
        )

        final_model, final_scaler = fit_final_model(
            model_name=model_name,
            X_train=X_temporal_train,
            y_train=y_temporal_train,
            scale_features=scale,
            xgb_n_estimators=validation_result["best_n_estimators"],
        )

        test_metrics = evaluate_final_test(
            model=final_model,
            scaler=final_scaler,
            X_test=X_temporal_test,
            y_test=y_temporal_test,
            threshold=validation_result["best_threshold"],
        )

        results["temporal"][model_name] = {
            "model": final_model,
            "scaler": final_scaler,
            "preprocessor": temporal_preprocessor,
            "feature_columns": temporal_preprocessor.feature_columns,
            "validation_metrics": validation_result["validation_metrics"],
            "validation_fold_summary": validation_result["fold_metrics_summary"],
            "validation_folds": validation_result["fold_details"],
            "metrics": test_metrics,
            "selection_score": validation_result["selection_score"],
            "best_threshold": validation_result["best_threshold"],
            "best_n_estimators": validation_result["best_n_estimators"],
        }

        LOGGER.info(
            "[temporal / %s] FINAL TEST: threshold=%.2f | precision=%.3f | "
            "recall=%.3f | F1=%.3f | ROC-AUC=%s | AP=%s",
            model_name,
            validation_result["best_threshold"],
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["f1"],
            (
                f"{test_metrics['roc_auc']:.3f}"
                if test_metrics.get("roc_auc") is not None else "n/a"
            ),
            (
                f"{test_metrics['average_precision']:.3f}"
                if test_metrics.get("average_precision") is not None else "n/a"
            ),
        )

    # ------------------------------------------------------------------------
    # Step 4 — Grouped evaluation
    # ------------------------------------------------------------------------
    grouped = next(s for s in splits if s.name == "grouped")
    grouped_fit_idx, grouped_validation_idx = make_internal_validation_split(
        panel, grouped
    )

    grouped_fit_panel = panel.iloc[grouped_fit_idx]
    grouped_validation_panel = panel.iloc[grouped_validation_idx]
    grouped_test_panel = panel.iloc[grouped.test_idx]

    y_grouped_fit = y.iloc[grouped_fit_idx]
    y_grouped_validation = y.iloc[grouped_validation_idx]
    y_grouped_test = y.iloc[grouped.test_idx]

    LOGGER.info("============================================================")
    LOGGER.info("GROUPED UNSEEN-CELL EVALUATION")
    LOGGER.info("============================================================")

    for model_name in model_names:
        scale = model_name == "logistic_regression"

        # Fit preprocessing on grouped fit cells only.
        grouped_preprocessor = fit_feature_preprocessor(grouped_fit_panel)
        X_grouped_fit = transform_features(grouped_fit_panel, grouped_preprocessor)
        X_grouped_validation = transform_features(
            grouped_validation_panel,
            grouped_preprocessor,
        )
        X_grouped_test = transform_features(grouped_test_panel, grouped_preprocessor)

        validation_model, validation_scaler, best_n_estimators = fit_validation_model(
            model_name=model_name,
            X_fit=X_grouped_fit,
            y_fit=y_grouped_fit,
            X_validation=X_grouped_validation,
            y_validation=y_grouped_validation,
            scale_features=scale,
        )

        X_validation_used = X_grouped_validation
        if validation_scaler is not None:
            X_validation_used = transform_with_scaler(
                validation_scaler,
                X_grouped_validation,
            )

        validation_proba = get_probability(validation_model, X_validation_used)
        validation_y_np = np.asarray(y_grouped_validation).astype(int)
        threshold_result = find_best_threshold(
            validation_y_np,
            validation_proba,
            min_threshold=THRESHOLD_MIN,
            max_threshold=THRESHOLD_MAX,
            step=THRESHOLD_STEP,
        )
        validation_pred = predictions_from_threshold(
            validation_proba,
            threshold_result["threshold"],
        )
        validation_metrics = compute_metrics(
            validation_y_np,
            validation_pred,
            validation_proba,
            task="binary",
        )
        validation_metrics["threshold"] = threshold_result["threshold"]

        full_grouped_panel = pd.concat(
            [grouped_fit_panel, grouped_validation_panel], axis=0
        )
        full_grouped_y = pd.concat(
            [y_grouped_fit, y_grouped_validation], axis=0
        )

        grouped_final_preprocessor = fit_feature_preprocessor(full_grouped_panel)
        X_grouped_full = transform_features(
            full_grouped_panel,
            grouped_final_preprocessor,
        )
        X_grouped_test_final = transform_features(
            grouped_test_panel,
            grouped_final_preprocessor,
        )

        final_model, final_scaler = fit_final_model(
            model_name=model_name,
            X_train=X_grouped_full,
            y_train=full_grouped_y,
            scale_features=scale,
            xgb_n_estimators=best_n_estimators,
        )
        grouped_test_metrics = evaluate_final_test(
            model=final_model,
            scaler=final_scaler,
            X_test=X_grouped_test_final,
            y_test=y_grouped_test,
            threshold=threshold_result["threshold"],
        )

        results["grouped"][model_name] = {
            "model": final_model,
            "scaler": final_scaler,
            "preprocessor": grouped_final_preprocessor,
            "feature_columns": grouped_final_preprocessor.feature_columns,
            "validation_metrics": validation_metrics,
            "metrics": grouped_test_metrics,
            "selection_score": validation_metrics[PRIMARY_METRIC],
            "best_threshold": threshold_result["threshold"],
            "best_n_estimators": best_n_estimators,
        }

        LOGGER.info(
            "[grouped / %s] TEST: threshold=%.2f | precision=%.3f | "
            "recall=%.3f | F1=%.3f | ROC-AUC=%s | AP=%s",
            model_name,
            threshold_result["threshold"],
            grouped_test_metrics["precision"],
            grouped_test_metrics["recall"],
            grouped_test_metrics["f1"],
            (
                f"{grouped_test_metrics['roc_auc']:.3f}"
                if grouped_test_metrics.get("roc_auc") is not None else "n/a"
            ),
            (
                f"{grouped_test_metrics['average_precision']:.3f}"
                if grouped_test_metrics.get("average_precision") is not None else "n/a"
            ),
        )

    # ------------------------------------------------------------------------
    # Step 5 — Select winner using pooled rolling validation F1
    # ------------------------------------------------------------------------
    selection_scores = {
        name: results[PRIMARY_SPLIT][name]["selection_score"]
        for name in model_names
    }
    winner_name = max(selection_scores, key=selection_scores.get)
    winner_bundle = results[PRIMARY_SPLIT][winner_name]

    LOGGER.info(
        "Winner selected using pooled rolling-validation %s: %s (score=%.3f)",
        PRIMARY_METRIC,
        winner_name,
        selection_scores[winner_name],
    )
    LOGGER.info(
        "Temporal rolling validation F1 by model: %s",
        {k: round(v, 3) for k, v in selection_scores.items()},
    )

    # ------------------------------------------------------------------------
    # Step 6 — Save all temporal models with frozen thresholds
    # ------------------------------------------------------------------------
    for model_name in model_names:
        bundle = results[PRIMARY_SPLIT][model_name]
        save_model_artifacts(
            task=TASK,
            model_name=model_name,
            model=bundle["model"],
            feature_columns=bundle["feature_columns"],
            scaler=bundle["scaler"],
            preprocessor=bundle["preprocessor"],
            threshold=bundle["best_threshold"],
        )

    save_model_artifacts(
        task=TASK,
        model_name=winner_name,
        model=winner_bundle["model"],
        feature_columns=winner_bundle["feature_columns"],
        scaler=winner_bundle["scaler"],
        preprocessor=winner_bundle["preprocessor"],
        threshold=winner_bundle["best_threshold"],
        is_winner=True,
    )

    # ------------------------------------------------------------------------
    # Step 7 — Metrics report
    # ------------------------------------------------------------------------
    report = {
        "task": TASK,
        "target": TARGET_COL,
        "primary_metric": PRIMARY_METRIC,
        "primary_split": PRIMARY_SPLIT,
        "winner_selection": "pooled_rolling_validation_f1",
        "winner": winner_name,
        "winner_validation_score": selection_scores[winner_name],
        "winner_test_score": winner_bundle["metrics"][PRIMARY_METRIC],
        "winner_threshold": winner_bundle["best_threshold"],
        "winner_test_average_precision": winner_bundle["metrics"].get(
            "average_precision"
        ),
        "winner_test_roc_auc": winner_bundle["metrics"].get("roc_auc"),
        "n_features": len(winner_bundle["feature_columns"]),
        "feature_columns": winner_bundle["feature_columns"],
        "evaluation_rules": {
            "final_temporal_test_start": "2023-09",
            "final_temporal_test_end": "2023-12",
            "test_used_for_model_selection": False,
            "test_used_for_threshold_selection": False,
            "test_used_for_preprocessing_fit": False,
            "test_used_for_xgboost_early_stopping": False,
            "threshold_selection_source": "pooled rolling temporal validation predictions",
            "numeric_imputation": "fit-only median",
            "categorical_schema": "fit-only levels",
            "pr_metric": "average_precision",
        },
        "splits": {},
    }

    for split in splits:
        split_results = {}
        for model_name in model_names:
            bundle = results[split.name][model_name]
            split_results[model_name] = {
                "validation_metrics": bundle["validation_metrics"],
                "test_metrics": bundle["metrics"],
                "selection_score": bundle["selection_score"],
                "best_threshold": bundle["best_threshold"],
                "best_n_estimators": bundle["best_n_estimators"],
            }
            if split.name == "temporal":
                split_results[model_name]["rolling_validation_summary"] = bundle[
                    "validation_fold_summary"
                ]
                split_results[model_name]["rolling_validation_folds"] = bundle[
                    "validation_folds"
                ]

        report["splits"][split.name] = {
            "description": split.description,
            "n_train_total": len(split.train_idx),
            "n_test": len(split.test_idx),
            "models": split_results,
        }

    save_metrics_report(TASK, report)

    # ------------------------------------------------------------------------
    # Step 8 — SHAP for winner
    # ------------------------------------------------------------------------
    X_train_for_shap = transform_features(
        temporal_train_panel,
        winner_bundle["preprocessor"],
    )
    X_test_for_shap = transform_features(
        temporal_test_panel,
        winner_bundle["preprocessor"],
    )

    compute_shap_for_winner(
        winner_name=winner_name,
        winner_bundle=winner_bundle,
        X_train=X_train_for_shap,
        X_test=X_test_for_shap,
        feature_columns=winner_bundle["feature_columns"],
    )

    LOGGER.info("=== Hotspot Classifier Training Complete ===")


if __name__ == "__main__":
    main()
