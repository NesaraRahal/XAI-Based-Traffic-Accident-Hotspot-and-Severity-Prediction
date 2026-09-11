"""Diagnostic plots for the severity classifier.

Mirrors plot_hotspot_diagnostics.py but adapted for severity's two modes:

  * SEVERITY_MODE = "binary"    → is_severe target (Severe vs Non-severe)
  * SEVERITY_MODE = "multiclass" → dominant_severity target (Fatal/Serious/Minor/Damage-only)

Flip SEVERITY_MODE below and rerun to produce both sets of plots.

Reads the artifacts produced by train_severity_classifier.py and generates:
1. Train vs test F1 (overfitting check)
2. Train vs test accuracy (overfitting check with baseline)
3. Confusion matrices per model
4. ROC curves (binary only — multiclass ROC is one-vs-rest, less standard to show)
5. Precision-Recall curves (binary only, same reason)
6. Feature importance for the winner
7. SHAP summary plot for the winner
8. XGBoost training curves (logloss + accuracy/error per boosting round)
9. Random Forest training curves (accuracy + F1 vs tree count)
10. Logistic Regression learning curve (accuracy + F1 vs training set size)

All plots run on the GROUPED split (the primary evaluation regime).

Run:
    # For binary artifacts:
    python modeling/plot_severity_diagnostics.py
    # (with SEVERITY_MODE = "binary" at top of file)

    # For multiclass artifacts:
    # Flip SEVERITY_MODE to "multiclass", save, rerun.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from modelling_utils import (
    MODELS_DIR,
    configure_logging,
    encode_features,
    grouped_split,
    load_training_panel,
)

LOGGER = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════
# Mode switch — set to "binary" or "multiclass" to match your last training run
# ══════════════════════════════════════════════════════════════════════════
SEVERITY_MODE = "binary"   # "binary"  or  "multiclass"
# ══════════════════════════════════════════════════════════════════════════

if SEVERITY_MODE == "binary":
    TASK = "severity_binary"
    TARGET_COL = "is_severe"
    IS_BINARY = True
elif SEVERITY_MODE == "multiclass":
    TASK = "severity_multiclass"
    TARGET_COL = "dominant_severity"
    IS_BINARY = False
else:
    raise ValueError(f"SEVERITY_MODE must be 'binary' or 'multiclass', got {SEVERITY_MODE}")

PLOTS_DIR = MODELS_DIR / "plots" / TASK
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "logistic_regression": MODELS_DIR / f"{TASK}_logistic_regression.joblib",
    "random_forest": MODELS_DIR / f"{TASK}_random_forest.joblib",
    "xgboost": MODELS_DIR / f"{TASK}_xgboost.joblib",
}
METRICS_REPORT = MODELS_DIR / f"{TASK}_metrics_report.json"
SHAP_ARTIFACT = MODELS_DIR / f"{TASK}_shap.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / f"{TASK}_label_encoder.joblib"

CLASS_ORDER = ["Fatal", "Serious", "Minor", "Damage-only"]

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


# ══════════════════════════════════════════════════════════════════════════
# Load artifacts
# ══════════════════════════════════════════════════════════════════════════

def load_metrics_report() -> Dict:
    with open(METRICS_REPORT) as f:
        return json.load(f)


def load_model_bundle(model_name: str) -> Dict:
    return joblib.load(MODEL_FILES[model_name])


def load_shap_artifact() -> Dict:
    return joblib.load(SHAP_ARTIFACT)


def load_label_encoder() -> Dict:
    """Multiclass only — returns {'label_encoder': LabelEncoder, 'class_order': [...]}."""
    return joblib.load(LABEL_ENCODER_PATH)


# ══════════════════════════════════════════════════════════════════════════
# Model refit helper (needed because plots refit fresh to measure train vs test)
# ══════════════════════════════════════════════════════════════════════════

class _ScaledWrapper:
    """Wraps LR so .predict / .predict_proba scale their input first."""
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
    def _scale(self, X):
        return pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
    def predict(self, X):
        return self.model.predict(self._scale(X))
    def predict_proba(self, X):
        return self.model.predict_proba(self._scale(X))


def _clone_and_refit(fitted_model, X_train, y_train, name):
    """Copy hyperparameters from the persisted fitted model, refit on this X,y.

    XGBoost has early_stopping_rounds set on the persisted model, which
    requires eval_set at fit time. For standalone refits (no eval_set) we
    strip that parameter so the model runs the full n_estimators cap.
    """
    from sklearn.base import clone
    if name == "logistic_regression":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_s = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        m = clone(fitted_model)
        m.fit(X_train_s, y_train)
        return _ScaledWrapper(m, scaler)
    else:
        m = clone(fitted_model)
        if name == "xgboost" and getattr(m, "early_stopping_rounds", None) is not None:
            m.set_params(early_stopping_rounds=None)
        # Multiclass XGB was trained with sample_weight; RF/binary XGB weren't.
        # For a clean refit we skip sample_weight — this is diagnostic only.
        m.fit(X_train, y_train)
        return m


def _encode_y(y_series: pd.Series, label_encoder=None) -> np.ndarray:
    """Return integer-encoded y for model consumption."""
    if IS_BINARY:
        return y_series.astype(int).values
    else:
        return label_encoder.transform(y_series).astype(int)


# ══════════════════════════════════════════════════════════════════════════
# Plot 1 — Train vs test F1
# ══════════════════════════════════════════════════════════════════════════

def plot_train_vs_test_f1(panel: pd.DataFrame, X, y_raw, label_encoder=None) -> None:
    model_names = list(MODEL_FILES.keys())
    split = grouped_split(panel)
    X_train, X_test = X.iloc[split.train_idx], X.iloc[split.test_idx]
    y_train = _encode_y(y_raw.iloc[split.train_idx], label_encoder)
    y_test = _encode_y(y_raw.iloc[split.test_idx], label_encoder)

    metric_kwargs = {"zero_division": 0}
    if not IS_BINARY:
        metric_kwargs["average"] = "macro"

    records = []
    for name in model_names:
        bundle = load_model_bundle(name)
        model = _clone_and_refit(bundle["model"], X_train, y_train, name)
        f1_train = f1_score(y_train, model.predict(X_train), **metric_kwargs)
        f1_test = f1_score(y_test, model.predict(X_test), **metric_kwargs)
        records.append({
            "model": name,
            "F1 (train)": f1_train,
            "F1 (test)": f1_test,
            "gap": f1_train - f1_test,
        })

    df = pd.DataFrame(records)
    LOGGER.info("Train vs test F1 (grouped split):\n%s", df.to_string(index=False))

    metric_label = "F1" if IS_BINARY else "macro-F1"

    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df.set_index("model")[["F1 (train)", "F1 (test)"]]
    sub.plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452"], edgecolor="black")
    ax.set_title(f"Severity ({SEVERITY_MODE}) — grouped split — train vs test {metric_label}\n"
                 "Large gap = overfitting. Grouped = evaluates on held-out cells.")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric_label)
    ax.set_xlabel("")
    ax.set_xticklabels(sub.index, rotation=15)
    for i, model in enumerate(sub.index):
        gap = sub.loc[model, "F1 (train)"] - sub.loc[model, "F1 (test)"]
        ax.annotate(f"gap = {gap:+.3f}",
                    xy=(i, max(sub.loc[model].values) + 0.03),
                    ha="center", fontsize=10,
                    color="crimson" if gap > 0.1 else "darkgreen")
    ax.legend(loc="lower right")

    plt.tight_layout()
    out = PLOTS_DIR / "01_train_vs_test_f1.png"
    plt.savefig(out)
    plt.close()
    LOGGER.info("Saved: %s", out)


# ══════════════════════════════════════════════════════════════════════════
# Plot 2 — Train vs test accuracy
# ══════════════════════════════════════════════════════════════════════════

def plot_train_vs_test_accuracy(panel: pd.DataFrame, X, y_raw, label_encoder=None) -> None:
    model_names = list(MODEL_FILES.keys())
    split = grouped_split(panel)
    X_train, X_test = X.iloc[split.train_idx], X.iloc[split.test_idx]
    y_train = _encode_y(y_raw.iloc[split.train_idx], label_encoder)
    y_test = _encode_y(y_raw.iloc[split.test_idx], label_encoder)

    records = []
    for name in model_names:
        bundle = load_model_bundle(name)
        model = _clone_and_refit(bundle["model"], X_train, y_train, name)
        acc_train = accuracy_score(y_train, model.predict(X_train))
        acc_test = accuracy_score(y_test, model.predict(X_test))
        records.append({
            "model": name,
            "Accuracy (train)": acc_train,
            "Accuracy (test)": acc_test,
            "gap": acc_train - acc_test,
        })
    df = pd.DataFrame(records)
    LOGGER.info("Train vs test accuracy (grouped split):\n%s", df.to_string(index=False))

    # Baseline = majority class rate
    if IS_BINARY:
        y_encoded = y_raw.astype(int)
        majority_baseline = max(y_encoded.mean(), 1 - y_encoded.mean())
    else:
        counts = y_raw.value_counts(normalize=True)
        majority_baseline = counts.max()

    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df.set_index("model")[["Accuracy (train)", "Accuracy (test)"]]
    sub.plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452"], edgecolor="black")
    ax.axhline(majority_baseline, color="black", linestyle="--", alpha=0.5,
               label=f"Majority-class baseline = {majority_baseline:.3f}")
    ax.set_title(f"Severity ({SEVERITY_MODE}) — grouped split — train vs test accuracy\n"
                 "Dashed = 'predict majority class always'; test accuracy must beat it.")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.set_xticklabels(sub.index, rotation=15)
    for i, model in enumerate(sub.index):
        gap = sub.loc[model, "Accuracy (train)"] - sub.loc[model, "Accuracy (test)"]
        ax.annotate(f"gap = {gap:+.3f}",
                    xy=(i, max(sub.loc[model].values) + 0.03),
                    ha="center", fontsize=10,
                    color="crimson" if gap > 0.1 else "darkgreen")
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    out = PLOTS_DIR / "01b_train_vs_test_accuracy.png"
    plt.savefig(out)
    plt.close()
    LOGGER.info("Saved: %s", out)


# ══════════════════════════════════════════════════════════════════════════
# Plot 3 — Confusion matrices
# ══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(panel: pd.DataFrame, X, y_raw, label_encoder=None) -> None:
    model_names = list(MODEL_FILES.keys())
    split = grouped_split(panel)
    X_train, X_test = X.iloc[split.train_idx], X.iloc[split.test_idx]
    y_train = _encode_y(y_raw.iloc[split.train_idx], label_encoder)
    y_test = _encode_y(y_raw.iloc[split.test_idx], label_encoder)

    if IS_BINARY:
        tick_labels = ["Non-severe", "Severe"]
    else:
        # Order the tick labels according to CLASS_ORDER so plots are consistent
        present = [c for c in CLASS_ORDER if c in label_encoder.classes_]
        tick_labels = present

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for col_idx, name in enumerate(model_names):
        bundle = load_model_bundle(name)
        model = _clone_and_refit(bundle["model"], X_train, y_train, name)
        y_pred = model.predict(X_test)

        if IS_BINARY:
            cm = confusion_matrix(y_test, y_pred)
        else:
            # Reorder rows/cols so they match CLASS_ORDER for readability
            present_indices = [list(label_encoder.classes_).index(c) for c in tick_labels]
            cm_raw = confusion_matrix(y_test, y_pred, labels=range(len(label_encoder.classes_)))
            cm = cm_raw[np.ix_(present_indices, present_indices)]

        ax = axes[col_idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=tick_labels, yticklabels=tick_labels,
                    ax=ax, annot_kws={"size": 12})
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual" if col_idx == 0 else "")

    plt.suptitle(f"Severity ({SEVERITY_MODE}) — confusion matrices — grouped split\n"
                 "Rows = actual, columns = predicted. Diagonal = correct.",
                 fontsize=11)
    plt.tight_layout()
    out = PLOTS_DIR / "02_confusion_matrices.png"
    plt.savefig(out)
    plt.close()
    LOGGER.info("Saved: %s", out)


# ══════════════════════════════════════════════════════════════════════════
# Plot 4 — ROC (binary only)
# ══════════════════════════════════════════════════════════════════════════

def plot_roc_grouped(panel: pd.DataFrame, X, y_raw) -> None:
    if not IS_BINARY:
        LOGGER.info("Skipping ROC for multiclass (not straightforward without one-vs-rest).")
        return

    split = grouped_split(panel)
    X_train, X_test = X.iloc[split.train_idx], X.iloc[split.test_idx]
    y_train = _encode_y(y_raw.iloc[split.train_idx])
    y_test = _encode_y(y_raw.iloc[split.test_idx])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random baseline")
    for name in MODEL_FILES:
        bundle = load_model_bundle(name)
        model = _clone_and_refit(bundle["model"], X_train, y_train, name)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Severity ({SEVERITY_MODE}) — ROC — grouped split\nHigher AUC = better discrimination")
    ax.legend(loc="lower right")
    out = PLOTS_DIR / "03_roc_grouped.png"
    plt.savefig(out)
    plt.close()
    LOGGER.info("Saved: %s", out)


# ══════════════════════════════════════════════════════════════════════════
# Plot 5 — Precision-Recall (binary only)
# ══════════════════════════════════════════════════════════════════════════

def plot_pr_grouped(panel: pd.DataFrame, X, y_raw) -> None:
    if not IS_BINARY:
        LOGGER.info("Skipping PR for multiclass.")
        return

    split = grouped_split(panel)
    X_train, X_test = X.iloc[split.train_idx], X.iloc[split.test_idx]
    y_train = _encode_y(y_raw.iloc[split.train_idx])
    y_test = _encode_y(y_raw.iloc[split.test_idx])

    fig, ax = plt.subplots(figsize=(7, 6))
    baseline = y_test.mean()
    ax.axhline(baseline, color="k", linestyle="--", alpha=0.4,
               label=f"Base rate = {baseline:.3f}")
    for name in MODEL_FILES:
        bundle = load_model_bundle(name)
        model = _clone_and_refit(bundle["model"], X_train, y_train, name)
        y_proba = model.predict_proba(X_test)[:, 1]
        p, r, _ = precision_recall_curve(y_test, y_proba)
        ax.plot(r, p, label=f"{name}", linewidth=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Severity ({SEVERITY_MODE}) — Precision-Recall — grouped split\n"
                 "More informative than ROC when the positive class is imbalanced")
    ax.legend(loc="lower left")
    out = PLOTS_DIR / "04_pr_grouped.png"
    plt.savefig(out)
    plt.close()
    LOGGER.info("Saved: %s", out)


# ══════════════════════════════════════════════════════════════════════════
# Plot 6 — Feature importance (winner)
# ══════════════════════════════════════════════════════════════════════════

def plot_feature_importance(report: Dict) -> None:
    winner = report["winner"]
    LOGGER.info("Winner (from metrics report): %s", winner)

    winner_path = MODELS_DIR / f"{TASK}_{winner}_winner.joblib"
    if not winner_path.exists():
        winner_path = MODEL_FILES[winner]
    bundle = joblib.load(winner_path)
    model = bundle["model"]
    features = bundle["feature_columns"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # For multiclass LR, coef_ is (n_classes, n_features) → average magnitude across classes
        importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_).flatten()
    else:
        LOGGER.warning("Winner has no feature_importances_ or coef_; skipping.")
        return

    imp_df = (
        pd.DataFrame({"feature": features, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(data=imp_df, y="feature", x="importance",
                color="#4c72b0", edgecolor="black", ax=ax)
    ax.set_title(f"Severity ({SEVERITY_MODE}) — top 15 features by importance — winner: {winner}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    out = PLOTS_DIR / "05_feature_importance.png"
    plt.savefig(out)
    plt.close()
    LOGGER.info("Saved: %s", out)


# ══════════════════════════════════════════════════════════════════════════
# Plot 7 — SHAP summary (winner)
# ══════════════════════════════════════════════════════════════════════════

def plot_shap_summary(label_encoder=None) -> None:
    if not SHAP_ARTIFACT.exists():
        LOGGER.warning("No SHAP artifact at %s — skipping.", SHAP_ARTIFACT)
        return

    shap_bundle = load_shap_artifact()
    shap_values = shap_bundle["shap_values"]
    X_sample = shap_bundle["X_sample"]

    if IS_BINARY:
        # Normalise to positive-class array
        if isinstance(shap_values, list):
            shap_values_use = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:  # (n_samples, n_features, n_classes)
                shap_values_use = arr[:, :, 1] if arr.shape[-1] > 1 else arr[:, :, 0]
            else:
                shap_values_use = arr

        plt.figure(figsize=(9, 7))
        shap.summary_plot(shap_values_use, X_sample, max_display=15, show=False)
        plt.title(f"Severity ({SEVERITY_MODE}) — SHAP top 15 features (winner, grouped-split test set)")
        out = PLOTS_DIR / "06_shap_summary.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        LOGGER.info("Saved: %s", out)
    else:
        # Multiclass: one plot per class
        if isinstance(shap_values, list):
            per_class = shap_values
            class_names = list(label_encoder.classes_) if label_encoder is not None else [f"class_{i}" for i in range(len(shap_values))]
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                per_class = [arr[:, :, i] for i in range(arr.shape[-1])]
                class_names = list(label_encoder.classes_) if label_encoder is not None else [f"class_{i}" for i in range(arr.shape[-1])]
            else:
                LOGGER.warning("Unexpected shap_values shape for multiclass: %s. Skipping.", arr.shape)
                return

        for cls_name, cls_shap in zip(class_names, per_class):
            plt.figure(figsize=(9, 7))
            shap.summary_plot(cls_shap, X_sample, max_display=15, show=False)
            plt.title(f"Severity (multiclass) — SHAP top 15 features — class: {cls_name}")
            # Sanitise class name for filename (Damage-only → Damage-only, no spaces)
            safe = str(cls_name).replace(" ", "_").replace("/", "_")
            out = PLOTS_DIR / f"06_shap_summary_{safe}.png"
            plt.savefig(out, bbox_inches="tight")
            plt.close()
            LOGGER.info("Saved: %s", out)


# ══════════════════════════════════════════════════════════════════════════
# Plot 8 — XGBoost training curves
# ══════════════════════════════════════════════════════════════════════════

def plot_xgboost_training_curves(panel: pd.DataFrame, X, y_raw, label_encoder=None) -> None:
    from xgboost import XGBClassifier

    split = grouped_split(panel)
    X_train, X_test = X.iloc[split.train_idx], X.iloc[split.test_idx]
    y_train = _encode_y(y_raw.iloc[split.train_idx], label_encoder)
    y_test = _encode_y(y_raw.iloc[split.test_idx], label_encoder)

    bundle = load_model_bundle("xgboost")
    ref = bundle["model"]

    params = ref.get_params()
    tracker_params = {
        k: v for k, v in params.items()
        if k not in ("eval_metric", "early_stopping_rounds")
    }
    # Choose metrics based on mode
    metrics_list = ["logloss", "error"] if IS_BINARY else ["mlogloss", "merror"]
    tracker = XGBClassifier(**tracker_params, eval_metric=metrics_list)
    tracker.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )
    hist = tracker.evals_result()

    logloss_key = "logloss" if IS_BINARY else "mlogloss"
    error_key = "error" if IS_BINARY else "merror"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(hist["validation_0"][logloss_key], label="train", linewidth=2)
    ax.plot(hist["validation_1"][logloss_key], label="test", linewidth=2)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel(logloss_key)
    ax.set_title(f"Severity ({SEVERITY_MODE}) — XGBoost {logloss_key} per round (grouped split)")
    ax.legend()

    ax = axes[1]
    train_acc = [1 - e for e in hist["validation_0"][error_key]]
    test_acc = [1 - e for e in hist["validation_1"][error_key]]
    ax.plot(train_acc, label="train", linewidth=2)
    ax.plot(test_acc, label="test", linewidth=2)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Severity ({SEVERITY_MODE}) — XGBoost accuracy per round (grouped split)")
    ax.set_ylim(0, 1.02)
    ax.legend()

    plt.tight_layout()
    out = PLOTS_DIR / "07_xgboost_training_curves.png"
    plt.savefig(out)
    plt.close()
    LOGGER.info("Saved: %s", out)


# ══════════════════════════════════════════════════════════════════════════
# Plot 9 — Random Forest training curves
# ══════════════════════════════════════════════════════════════════════════

def plot_rf_training_curves(panel: pd.DataFrame, X, y_raw, label_encoder=None) -> None:
    from sklearn.ensemble import RandomForestClassifier

    split = grouped_split(panel)
    X_train, X_test = X.iloc[split.train_idx], X.iloc[split.test_idx]
    y_train = _encode_y(y_raw.iloc[split.train_idx], label_encoder)
    y_test = _encode_y(y_raw.iloc[split.test_idx], label_encoder)

    bundle = load_model_bundle("random_forest")
    ref = bundle["model"]
    ref_params = {k: v for k, v in ref.get_params().items() if k != "n_estimators"}

    n_tree_grid = [10, 25, 50, 100, 150, 200, 250, 300]
    metric_kwargs = {"zero_division": 0}
    if not IS_BINARY:
        metric_kwargs["average"] = "macro"

    records = []
    for n in n_tree_grid:
        m = RandomForestClassifier(**ref_params, n_estimators=n)
        m.fit(X_train, y_train)
        records.append({
            "n_estimators": n,
            "train_acc": accuracy_score(y_train, m.predict(X_train)),
            "test_acc": accuracy_score(y_test, m.predict(X_test)),
            "train_f1": f1_score(y_train, m.predict(X_train), **metric_kwargs),
            "test_f1": f1_score(y_test, m.predict(X_test), **metric_kwargs),
        })
    df = pd.DataFrame(records)
    LOGGER.info("RF training curve:\n%s", df.to_string(index=False))

    f1_label = "F1" if IS_BINARY else "macro-F1"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(df["n_estimators"], df["train_acc"], label="train", linewidth=2, marker="o")
    ax.plot(df["n_estimators"], df["test_acc"], label="test", linewidth=2, marker="o")
    ax.set_xlabel("n_estimators (trees)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Severity ({SEVERITY_MODE}) — Random Forest accuracy vs tree count (grouped split)")
    ax.set_ylim(0, 1.02)
    ax.legend()

    ax = axes[1]
    ax.plot(df["n_estimators"], df["train_f1"], label="train", linewidth=2, marker="o")
    ax.plot(df["n_estimators"], df["test_f1"], label="test", linewidth=2, marker="o")
    ax.set_xlabel("n_estimators (trees)")
    ax.set_ylabel(f1_label)
    ax.set_title(f"Severity ({SEVERITY_MODE}) — Random Forest {f1_label} vs tree count (grouped split)")
    ax.set_ylim(0, 1.02)
    ax.legend()

    plt.tight_layout()
    out = PLOTS_DIR / "08_random_forest_training_curves.png"
    plt.savefig(out)
    plt.close()
    LOGGER.info("Saved: %s", out)


# ══════════════════════════════════════════════════════════════════════════
# Plot 10 — Logistic Regression learning curve
# ══════════════════════════════════════════════════════════════════════════

def plot_lr_learning_curve(panel: pd.DataFrame, X, y_raw, label_encoder=None) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import learning_curve
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    split = grouped_split(panel)
    X_train = X.iloc[split.train_idx]
    y_train = _encode_y(y_raw.iloc[split.train_idx], label_encoder)

    bundle = load_model_bundle("logistic_regression")
    ref = bundle["model"]
    ref_params = ref.get_params()

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(**ref_params)),
    ])

    scoring_f1 = "f1" if IS_BINARY else "f1_macro"
    train_sizes = np.linspace(0.1, 1.0, 8)

    sizes, train_scores_acc, test_scores_acc = learning_curve(
        pipe, X_train, y_train,
        train_sizes=train_sizes, cv=3, scoring="accuracy", n_jobs=-1,
        random_state=42, shuffle=True,
    )
    _, train_scores_f1, test_scores_f1 = learning_curve(
        pipe, X_train, y_train,
        train_sizes=train_sizes, cv=3, scoring=scoring_f1, n_jobs=-1,
        random_state=42, shuffle=True,
    )

    f1_label = "F1" if IS_BINARY else "macro-F1"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(sizes, train_scores_acc.mean(axis=1), label="train", linewidth=2, marker="o")
    ax.plot(sizes, test_scores_acc.mean(axis=1), label="test (3-fold CV)", linewidth=2, marker="o")
    ax.fill_between(sizes,
                    train_scores_acc.mean(axis=1) - train_scores_acc.std(axis=1),
                    train_scores_acc.mean(axis=1) + train_scores_acc.std(axis=1), alpha=0.15)
    ax.fill_between(sizes,
                    test_scores_acc.mean(axis=1) - test_scores_acc.std(axis=1),
                    test_scores_acc.mean(axis=1) + test_scores_acc.std(axis=1), alpha=0.15)
    ax.set_xlabel("training set size")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Severity ({SEVERITY_MODE}) — LR learning curve (accuracy)")
    ax.set_ylim(0, 1.02)
    ax.legend()

    ax = axes[1]
    ax.plot(sizes, train_scores_f1.mean(axis=1), label="train", linewidth=2, marker="o")
    ax.plot(sizes, test_scores_f1.mean(axis=1), label="test (3-fold CV)", linewidth=2, marker="o")
    ax.fill_between(sizes,
                    train_scores_f1.mean(axis=1) - train_scores_f1.std(axis=1),
                    train_scores_f1.mean(axis=1) + train_scores_f1.std(axis=1), alpha=0.15)
    ax.fill_between(sizes,
                    test_scores_f1.mean(axis=1) - test_scores_f1.std(axis=1),
                    test_scores_f1.mean(axis=1) + test_scores_f1.std(axis=1), alpha=0.15)
    ax.set_xlabel("training set size")
    ax.set_ylabel(f1_label)
    ax.set_title(f"Severity ({SEVERITY_MODE}) — LR learning curve ({f1_label})")
    ax.set_ylim(0, 1.02)
    ax.legend()

    plt.tight_layout()
    out = PLOTS_DIR / "09_logistic_regression_learning_curve.png"
    plt.savefig(out)
    plt.close()
    LOGGER.info("Saved: %s", out)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    configure_logging()
    LOGGER.info("=== Severity Diagnostic Plots (mode=%s) ===", SEVERITY_MODE)
    LOGGER.info("Output dir: %s", PLOTS_DIR)

    if not METRICS_REPORT.exists():
        raise FileNotFoundError(
            f"Metrics report not found at {METRICS_REPORT}. "
            f"Run train_severity_classifier.py with SEVERITY_MODE={SEVERITY_MODE!r} first."
        )
    report = load_metrics_report()

    panel = load_training_panel()
    panel = panel.dropna(subset=[TARGET_COL])

    X, _ = encode_features(panel, fit_encoder=True)
    y_raw = panel[TARGET_COL]

    label_encoder = None
    if not IS_BINARY:
        if not LABEL_ENCODER_PATH.exists():
            raise FileNotFoundError(
                f"Multiclass label encoder not found at {LABEL_ENCODER_PATH}. "
                "Was training done in multiclass mode?"
            )
        label_encoder = load_label_encoder()["label_encoder"]

    LOGGER.info("Panel: %d rows, %d features", len(X), X.shape[1])
    LOGGER.info("Target [%s] distribution: %s", TARGET_COL, y_raw.value_counts().to_dict())

    LOGGER.info("-- Plot 1: Train vs test F1 --")
    plot_train_vs_test_f1(panel, X, y_raw, label_encoder)

    LOGGER.info("-- Plot 1b: Train vs test accuracy --")
    plot_train_vs_test_accuracy(panel, X, y_raw, label_encoder)

    LOGGER.info("-- Plot 2: Confusion matrices --")
    plot_confusion_matrices(panel, X, y_raw, label_encoder)

    LOGGER.info("-- Plot 3: ROC (grouped split) --")
    plot_roc_grouped(panel, X, y_raw)

    LOGGER.info("-- Plot 4: Precision-Recall (grouped split) --")
    plot_pr_grouped(panel, X, y_raw)

    LOGGER.info("-- Plot 5: Feature importance (winner) --")
    plot_feature_importance(report)

    LOGGER.info("-- Plot 6: SHAP summary (winner) --")
    plot_shap_summary(label_encoder)

    LOGGER.info("-- Plot 7: XGBoost training curves --")
    plot_xgboost_training_curves(panel, X, y_raw, label_encoder)

    LOGGER.info("-- Plot 8: Random Forest training curves --")
    plot_rf_training_curves(panel, X, y_raw, label_encoder)

    LOGGER.info("-- Plot 9: Logistic Regression learning curve --")
    plot_lr_learning_curve(panel, X, y_raw, label_encoder)

    LOGGER.info("=== Done ===")
    LOGGER.info("All plots in: %s", PLOTS_DIR)


if __name__ == "__main__":
    main()