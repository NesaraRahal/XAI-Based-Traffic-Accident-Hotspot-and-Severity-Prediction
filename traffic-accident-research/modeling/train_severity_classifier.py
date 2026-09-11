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

CHANGE LOG (this revision)
---------------------------
  * XGBoost's early-stopping round count and the binary decision threshold
    were previously selected using X_test (eval_set) and X_train's own
    in-sample predictions respectively — a test-set peek and an in-sample-
    bias problem. Both are now selected on an INNER validation split carved
    out of TRAIN ONLY, grouped by h3_cell so no cell straddles the inner
    train/val boundary. X_test is never touched until final scoring.
  * Once the round count / threshold are chosen on the inner split, the
    PRODUCTION model that actually scores X_test is refit on the full
    training set (all train rows/cells), so no training data is wasted.
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
from sklearn.metrics import f1_score as _f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
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

PRIMARY_SPLIT = "tempral"   # "temporal" or "grouped" — which split to use for winner selection
MODEL_SEED = 42

# Fraction of TRAIN cells (not test cells) held out, grouped by h3_cell, to
# pick XGBoost's early-stopping round count and the binary decision
# threshold. Never touches X_test.
INNER_VAL_FRACTION = 0.2

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
    #
    # AGGRESSIVE CLASS WEIGHTS (binary severity only):
    # class_weight="balanced" alone was insufficient — the model still
    # heavily biased toward predicting non-severe (recall on severe class
    # was only ~31%). Manually up-weighting the severe class by a factor
    # beyond "balanced" (which uses inverse frequency ≈ 2.04) pushes the
    # decision boundary toward better severe-recall at the cost of some
    # precision. F1 improves because recall was the limiting factor.
    if IS_BINARY:
        class_weight = {0: 1.0, 1: 3.0}   # was "balanced"
    else:
        class_weight = "balanced"
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight=class_weight,
        random_state=MODEL_SEED,
        n_jobs=-1,
    )


def make_xgboost_binary(
    y_train: np.ndarray, n_estimators: int = 300, use_early_stopping: bool = True,
) -> XGBClassifier:
    """Binary XGBoost with scale_pos_weight for class imbalance.

    AGGRESSIVE CLASS WEIGHTING: standard scale_pos_weight = neg/pos gives
    us the "balanced" starting point. We multiply by 1.5x to push the
    boundary further toward better severe-recall, matching the RF override
    above.

    Two modes, both used by train_and_evaluate below:
      - SELECTION mode (use_early_stopping=True): capped at n_estimators,
        early stopping picks the actual round count against an INNER
        validation split carved out of train (never X_test).
      - PRODUCTION mode (use_early_stopping=False): n_estimators fixed to
        whatever selection mode found; this is the model that scores X_test.
    """
    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    balanced_ratio = (neg_count / pos_count) if pos_count > 0 else 1.0
    scale_pos_weight = balanced_ratio * 1.5   # was just balanced_ratio
    kwargs = dict(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=MODEL_SEED,
        n_jobs=-1,
    )
    if use_early_stopping:
        kwargs["early_stopping_rounds"] = 20
    return XGBClassifier(**kwargs)


def make_xgboost_multiclass(
    n_estimators: int = 300, use_early_stopping: bool = True,
) -> XGBClassifier:
    """Multiclass XGBoost — sample_weight handled at fit time.

    Same selection/production split as make_xgboost_binary above.
    """
    kwargs = dict(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=MODEL_SEED,
        n_jobs=-1,
    )
    if use_early_stopping:
        kwargs["early_stopping_rounds"] = 20
    return XGBClassifier(**kwargs)


# ══════════════════════════════════════════════════════════════════════════
# Inner (train-only) validation split
# ══════════════════════════════════════════════════════════════════════════

def _inner_group_split(
    X_train: pd.DataFrame, y_train: pd.Series, groups_train: np.ndarray,
    val_frac: float = INNER_VAL_FRACTION, seed: int = MODEL_SEED,
):
    """Carve a validation slice out of TRAIN ONLY, grouped by h3_cell so no
    cell straddles the inner train/val boundary. Used to pick the XGBoost
    early-stopping round count and the binary decision threshold without
    ever looking at the outer test set.

    Falls back to a stratified row-level split if the grouped split happens
    to drop an entire class from one side (can happen with rare classes like
    Fatal/Damage-only, which only have a handful of cells to begin with).
    """
    y_arr = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)

    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    inner_train_pos, inner_val_pos = next(gss.split(X_train, y_arr, groups=groups_train))

    train_labels = set(np.unique(y_arr[inner_train_pos]))
    val_labels = set(np.unique(y_arr[inner_val_pos]))
    all_labels = set(np.unique(y_arr))

    if not all_labels.issubset(train_labels) or not val_labels:
        LOGGER.warning(
            "Grouped inner split dropped a class (inner-train classes=%s, "
            "need=%s) — falling back to a stratified random inner split.",
            sorted(train_labels), sorted(all_labels),
        )
        idx = np.arange(len(y_arr))
        inner_train_pos, inner_val_pos = train_test_split(
            idx, test_size=val_frac, random_state=seed, stratify=y_arr,
        )

    return (
        X_train.iloc[inner_train_pos], y_train.iloc[inner_train_pos],
        X_train.iloc[inner_val_pos], y_train.iloc[inner_val_pos],
    )


# ══════════════════════════════════════════════════════════════════════════
# Training + evaluation
# ══════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    model_name: str,
    X_train, y_train, X_test, y_test,
    label_encoder,   # None for binary, LabelEncoder for multiclass
    scale_features: bool,
    groups_train,    # h3_cell values aligned (positionally) with X_train rows
) -> Dict:
    """Train one severity model on one split; return metrics dict + fitted model.

    Two-stage fit:
      1. SELECTION — fit only on an inner train/val split carved out of
         X_train (grouped by cell). Used to pick XGBoost's round count and
         the binary threshold, both validated on the inner-val slice.
      2. PRODUCTION — refit on the FULL X_train using whatever the
         selection stage found. This is the model that scores X_test.
    """

    X_inner_train, y_inner_train, X_inner_val, y_inner_val = _inner_group_split(
        X_train, y_train, groups_train,
    )

    # ── Stage 1: selection (inner split only, X_test never touched) ────────
    inner_scaler = None
    X_inner_train_s, X_inner_val_s = X_inner_train, X_inner_val
    if scale_features:
        inner_scaler = StandardScaler()
        X_inner_train_s = pd.DataFrame(
            inner_scaler.fit_transform(X_inner_train),
            columns=X_inner_train.columns, index=X_inner_train.index,
        )
        X_inner_val_s = pd.DataFrame(
            inner_scaler.transform(X_inner_val),
            columns=X_inner_val.columns, index=X_inner_val.index,
        )

    best_n = 300  # only overwritten for xgboost below
    if model_name == "logistic_regression":
        selection_model = make_logistic_regression()
        selection_model.fit(X_inner_train_s, y_inner_train)

    elif model_name == "random_forest":
        selection_model = make_random_forest()
        selection_model.fit(X_inner_train_s, y_inner_train)

    elif model_name == "xgboost":
        if IS_BINARY:
            y_inner_arr = y_inner_train.values if hasattr(y_inner_train, "values") else y_inner_train
            selection_model = make_xgboost_binary(y_inner_arr, n_estimators=300, use_early_stopping=True)
            selection_model.fit(
                X_inner_train_s, y_inner_train,
                eval_set=[(X_inner_val_s, y_inner_val)],
                verbose=False,
            )
        else:
            sample_weights_inner = compute_sample_weight(class_weight="balanced", y=y_inner_train)
            selection_model = make_xgboost_multiclass(n_estimators=300, use_early_stopping=True)
            selection_model.fit(
                X_inner_train_s, y_inner_train,
                sample_weight=sample_weights_inner,
                eval_set=[(X_inner_val_s, y_inner_val)],
                verbose=False,
            )
        best_n = selection_model.best_iteration + 1
        LOGGER.info(
            "[%s] inner-val early stopping halted at round %d/300 (best iteration)",
            model_name, best_n,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # ── Stage 2: production (full X_train, no test-set contact) ────────────
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
            model = make_xgboost_binary(y_train_arr, n_estimators=best_n, use_early_stopping=False)
            model.fit(X_train_used, y_train, verbose=False)
        else:
            sample_weights_full = compute_sample_weight(class_weight="balanced", y=y_train)
            model = make_xgboost_multiclass(n_estimators=best_n, use_early_stopping=False)
            model.fit(X_train_used, y_train, sample_weight=sample_weights_full, verbose=False)

    # ── Prediction path differs for binary (threshold tuning) vs multiclass ─
    if IS_BINARY:
        # THRESHOLD TUNING: picked on the INNER VALIDATION split using the
        # SELECTION model (fit only on X_inner_train, so X_inner_val is
        # genuinely held out for it). This replaces the old approach of
        # tuning against X_train predictions from a model fit on X_train
        # itself, which is in-sample and uninformative once a model is
        # close to memorising train (as RF/XGB tend to be here).
        y_inner_val_proba = (
            selection_model.predict_proba(X_inner_val_s)[:, 1]
            if hasattr(selection_model, "predict_proba") else None
        )
        y_proba = model.predict_proba(X_test_used)[:, 1] if hasattr(model, "predict_proba") else None

        threshold = 0.5
        if y_inner_val_proba is not None:
            best_f1, best_thr = -1.0, 0.5
            y_inner_val_arr = y_inner_val.values if hasattr(y_inner_val, "values") else y_inner_val
            for thr in np.linspace(0.1, 0.9, 33):
                preds = (y_inner_val_proba >= thr).astype(int)
                f1 = _f1_score(y_inner_val_arr, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thr = f1, thr
            threshold = best_thr
            LOGGER.info(
                "[%s] F1-optimal threshold on inner-val = %.3f (inner-val F1 at threshold = %.3f)",
                model_name, threshold, best_f1,
            )

        if y_proba is not None:
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X_test_used)

        y_true = y_test.values if hasattr(y_test, "values") else y_test
        metrics = compute_metrics(y_true, y_pred, y_proba, task="binary")
        metrics["tuned_threshold"] = float(threshold)
    else:
        y_pred = model.predict(X_test_used)
        # For multiclass: decode integer targets back to strings for the metrics report
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        y_test_decoded = label_encoder.inverse_transform(y_test)
        present_classes = [c for c in CLASS_ORDER if c in label_encoder.classes_]
        metrics = compute_metrics(
            y_test_decoded, y_pred_decoded, y_proba=None,
            task="multiclass", class_labels=present_classes,
        )

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "tuned_threshold": metrics.get("tuned_threshold", 0.5),
    }


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
        # h3_cell for each TRAIN row, aligned positionally with X_train/y_train.
        # Used only to build the inner (train-only) validation split.
        groups_train = panel["h3_cell"].iloc[split.train_idx].values

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
                groups_train=groups_train,
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

    # Step 6a — Save the tuned threshold alongside the winner (binary only).
    # The dashboard/inference layer must apply this threshold to probability
    # predictions instead of the default 0.5, or F1 will regress.
    if IS_BINARY:
        threshold_path = MODELS_DIR / f"{TASK}_winner_threshold.joblib"
        joblib.dump(
            {
                "tuned_threshold": winner_bundle["tuned_threshold"],
                "winner_model": winner_name,
            },
            threshold_path,
        )
        LOGGER.info("Saved tuned threshold: %s (threshold=%.3f)",
                    threshold_path, winner_bundle["tuned_threshold"])

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

    # Step 8 — SHAP on winner (primary/grouped split)
    primary_split_obj = next(s for s in splits if s.name == PRIMARY_SPLIT)
    X_train_win = X.iloc[primary_split_obj.train_idx]
    X_test_win = X.iloc[primary_split_obj.test_idx]
    compute_shap_for_winner(
        winner_name, winner_bundle, X_train_win, X_test_win, feature_columns,
    )

    LOGGER.info("=== Severity Classifier Training Complete (mode=%s) ===", SEVERITY_MODE)


if __name__ == "__main__":
    main()