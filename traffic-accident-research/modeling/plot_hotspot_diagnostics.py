"""Diagnostic plots for the hotspot classifier.

Reads the artifacts produced by train_hotspot_classifier.py and generates a
suite of diagnostic plots that answer three questions:

1. Is any model overfitting? (train vs test F1 per model per split)
2. Are the honest (grouped-split) predictions actually skillful? (ROC, PR,
   confusion matrices)
3. Is the trivial temporal-split performance really cell memorization?
   (per-cell prediction consistency check)

Plus one interpretability plot for the winner:
- Top-15 feature importance (from the tree model directly)
- SHAP summary plot (top 15 features with directional impact)

All figures land in models/plots/hotspot/ as PNGs. Nothing overwrites the
trained artifacts.

Run:
    python modeling/plot_hotspot_diagnostics.py
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
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Reuse the same paths + splits as the training pipeline
from modelling_utils import (
    MODELS_DIR,
    configure_logging,
    encode_features,
    grouped_split,
    load_training_panel,
)

LOGGER = logging.getLogger(__name__)
TASK = "hotspot"
TARGET_COL = "is_hotspot"

PLOTS_DIR = MODELS_DIR / "plots" / TASK
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Model bundle filenames (must match training script)
MODEL_FILES = {
    "logistic_regression": MODELS_DIR / f"{TASK}_logistic_regression.joblib",
    "random_forest": MODELS_DIR / f"{TASK}_random_forest.joblib",
    "xgboost": MODELS_DIR / f"{TASK}_xgboost.joblib",
}
METRICS_REPORT = MODELS_DIR / f"{TASK}_metrics_report.json"
SHAP_ARTIFACT = MODELS_DIR / f"{TASK}_shap.joblib"

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


# ══════════════════════════════════════════════════════════════════════════
# Plot 1 — Train vs test F1 (overfitting check)
# ══════════════════════════════════════════════════════════════════════════

def plot_train_vs_test_f1(panel: pd.DataFrame, X, y) -> None:
    """For each model on the GROUPED (primary) split, compute train F1 and
    test F1. Overfitting = big gap. Grouped split is the only honest
    evaluation regime (temporal split is contaminated by cell-identity
    memorization — see thesis §3.5 limitations)."""
    model_names = list(MODEL_FILES.keys())
    split = grouped_split(panel)
    X_train, X_test = X.iloc[split.train_idx], X.iloc[split.test_idx]
    y_train, y_test = y.iloc[split.train_idx], y.iloc[split.test_idx]

    records = []
    for name in model_names:
        bundle = load_model_bundle(name)
        model = _clone_and_refit(bundle["model"], X_train, y_train, name)
        f1_train = f1_score(y_train, model.predict(X_train), zero_division=0)
        f1_test = f1_score(y_test, model.predict(X_test), zero_division=0)
        records.append({
            "model": name,
            "F1 (train)": f1_train,
            "F1 (test)": f1_test,
            "gap": f1_train - f1_test,
        })

    df = pd.DataFrame(records)
    LOGGER.info("Train vs test F1 (grouped split):\n%s", df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df.set_index("model")[["F1 (train)", "F1 (test)"]]
    sub.plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452"], edgecolor="black")
    ax.set_title("Grouped split — train vs test F1\n"
                 "Large gap = overfitting. Grouped = evaluates on held-out cells.")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
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


def _clone_and_refit(fitted_model, X_train, y_train, name):
    """Copy hyperparameters from the persisted fitted model, then re-fit
    on a fresh (X, y) so we can measure both train and test scores honestly.
    """
    from sklearn.base import clone
    # LR was trained on scaled features; RF and XGB weren't. Match that.
    if name == "logistic_regression":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_s = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        m = clone(fitted_model)
        m.fit(X_train_s, y_train)
        # Wrap so .predict on unscaled input still works
        return _ScaledWrapper(m, scaler)
    else:
        m = clone(fitted_model)
        m.fit(X_train, y_train)
        return m


class _ScaledWrapper:
    """Tiny wrapper so .predict / .predict_proba behave like a normal estimator
    even though LR was fit on scaled features."""
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def _scale(self, X):
        return pd.DataFrame(
            self.scaler.transform(X), columns=X.columns, index=X.index
        )

    def predict(self, X):
        return self.model.predict(self._scale(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._scale(X))


# ══════════════════════════════════════════════════════════════════════════
# Plot 1b — Train vs test ACCURACY (overfitting check, complementary to F1)
# ══════════════════════════════════════════════════════════════════════════

def plot_train_vs_test_accuracy(panel: pd.DataFrame, X, y) -> None:
    """Same overfitting check as F1 but using accuracy. Accuracy is easier to
    intuit ('what fraction did we get right?') but it can be misleading when
    classes are imbalanced: with ~28% hotspots, a model that predicts
    'not hotspot' for everything already scores 72% accuracy.

    So read this ALONGSIDE the F1 plot — F1 is more informative here, but
    accuracy is what a non-technical audience will ask about first.
    """
    model_names = list(MODEL_FILES.keys())
    split = grouped_split(panel)
    X_train, X_test = X.iloc[split.train_idx], X.iloc[split.test_idx]
    y_train, y_test = y.iloc[split.train_idx], y.iloc[split.test_idx]

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

    majority_baseline = max(y.mean(), 1 - y.mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df.set_index("model")[["Accuracy (train)", "Accuracy (test)"]]
    sub.plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452"], edgecolor="black")
    ax.axhline(majority_baseline, color="black", linestyle="--", alpha=0.5,
               label=f"Majority-class baseline = {majority_baseline:.3f}")
    ax.set_title("Grouped split — train vs test accuracy\n"
                 "Dashed line is the 'predict majority class always' baseline "
                 "— test accuracy needs to beat it to be meaningful.")
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
# Plot 2 — Confusion matrices (3 models × 2 splits)
# ══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(panel: pd.DataFrame, X, y) -> None:
    model_names = list(MODEL_FILES.keys())
    split = grouped_split(panel)
    X_train, X_test = X.iloc[split.train_idx], X.iloc[split.test_idx]
    y_train, y_test = y.iloc[split.train_idx], y.iloc[split.test_idx]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for col_idx, name in enumerate(model_names):
        bundle = load_model_bundle(name)
        model = _clone_and_refit(bundle["model"], X_train, y_train, name)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        ax = axes[col_idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Not hotspot", "Hotspot"],
                    yticklabels=["Not hotspot", "Hotspot"],
                    ax=ax, annot_kws={"size": 13})
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual" if col_idx == 0 else "")

    plt.suptitle("Confusion matrices — grouped split (held-out cells)\n"
                 "Rows = actual, columns = predicted. Diagonal = correct.",
                 fontsize=11)
    plt.tight_layout()
    out = PLOTS_DIR / "02_confusion_matrices.png"
    plt.savefig(out)
    plt.close()
    LOGGER.info("Saved: %s", out)


# ══════════════════════════════════════════════════════════════════════════
# Plot 3 — ROC curves on grouped split
# ══════════════════════════════════════════════════════════════════════════

def plot_roc_grouped(panel: pd.DataFrame, X, y) -> None:
    split = grouped_split(panel)
    X_train, X_test = X.iloc[split.train_idx], X.iloc[split.test_idx]
    y_train, y_test = y.iloc[split.train_idx], y.iloc[split.test_idx]

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
    ax.set_title("ROC — grouped split (holdout cells)\nThe honest evaluation")
    ax.legend(loc="lower right")
    out = PLOTS_DIR / "03_roc_grouped.png"
    plt.savefig(out)
    plt.close()
    LOGGER.info("Saved: %s", out)


# ══════════════════════════════════════════════════════════════════════════
# Plot 4 — Precision-Recall curves on grouped split
# ══════════════════════════════════════════════════════════════════════════

def plot_pr_grouped(panel: pd.DataFrame, X, y) -> None:
    split = grouped_split(panel)
    X_train, X_test = X.iloc[split.train_idx], X.iloc[split.test_idx]
    y_train, y_test = y.iloc[split.train_idx], y.iloc[split.test_idx]

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
    ax.set_title("Precision-Recall — grouped split\n"
                 "More informative than ROC when classes are imbalanced")
    ax.legend(loc="lower left")
    out = PLOTS_DIR / "04_pr_grouped.png"
    plt.savefig(out)
    plt.close()
    LOGGER.info("Saved: %s", out)


# ══════════════════════════════════════════════════════════════════════════
# Plot 5 — Feature importance (winner)
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
        importances = np.abs(model.coef_).flatten()
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
    ax.set_title(f"Top 15 features by importance — winner: {winner}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    out = PLOTS_DIR / "05_feature_importance.png"
    plt.savefig(out)
    plt.close()
    LOGGER.info("Saved: %s", out)


# ══════════════════════════════════════════════════════════════════════════
# Plot 6 — SHAP summary (winner)
# ══════════════════════════════════════════════════════════════════════════

def plot_shap_summary() -> None:
    if not SHAP_ARTIFACT.exists():
        LOGGER.warning("No SHAP artifact at %s — skipping.", SHAP_ARTIFACT)
        return

    shap_bundle = load_shap_artifact()
    shap_values = shap_bundle["shap_values"]
    X_sample = shap_bundle["X_sample"]

    # For binary XGB/RF, shap_values can be a list [class0_shap, class1_shap]
    # or a single array. Normalize to the positive-class array.
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:  # (n_samples, n_features, n_classes)
            shap_values_pos = arr[:, :, 1] if arr.shape[-1] > 1 else arr[:, :, 0]
        else:
            shap_values_pos = arr

    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values_pos, X_sample, max_display=15, show=False)
    plt.title("SHAP — top 15 features (winner, grouped-split test set)")
    out = PLOTS_DIR / "06_shap_summary.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved: %s", out)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def plot_xgboost_training_curves(panel: pd.DataFrame, X, y) -> None:
    """XGBoost logloss AND error-rate per boosting round on the grouped split.

    Both are computed on train and test simultaneously via XGBoost's built-in
    evals_result mechanism, so this is the correct diagnostic for boosting
    overfitting — if the test curve turns upwards while train keeps falling,
    you're training too many rounds.
    """
    from xgboost import XGBClassifier

    split = grouped_split(panel)
    X_train, X_test = X.iloc[split.train_idx], X.iloc[split.test_idx]
    y_train, y_test = y.iloc[split.train_idx], y.iloc[split.test_idx]

    bundle = load_model_bundle("xgboost")
    ref = bundle["model"]

    # Fresh model with the SAME hyperparameters as the trained one, but tracking
    # both logloss and error on train + test at every round.
    params = ref.get_params()
    # Keep n_estimators as-is (300 in your setup), request both metrics
    tracker = XGBClassifier(
        **{k: v for k, v in params.items() if k != "eval_metric"},
        eval_metric=["logloss", "error"],
    )
    tracker.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )
    hist = tracker.evals_result()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1 — logloss
    ax = axes[0]
    ax.plot(hist["validation_0"]["logloss"], label="train", linewidth=2)
    ax.plot(hist["validation_1"]["logloss"], label="test", linewidth=2)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("logloss")
    ax.set_title("hotspot — XGBoost logloss per round (grouped split)")
    ax.legend()

    # Panel 2 — accuracy (derived from 'error' rate: accuracy = 1 - error)
    ax = axes[1]
    train_acc = [1 - e for e in hist["validation_0"]["error"]]
    test_acc = [1 - e for e in hist["validation_1"]["error"]]
    ax.plot(train_acc, label="train", linewidth=2)
    ax.plot(test_acc, label="test", linewidth=2)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("accuracy")
    ax.set_title("hotspot — XGBoost accuracy per round (grouped split)")
    ax.set_ylim(0, 1.02)
    ax.legend()

    plt.tight_layout()
    out = PLOTS_DIR / "07_xgboost_training_curves.png"
    plt.savefig(out)
    plt.close()
    LOGGER.info("Saved: %s", out)


def plot_rf_training_curves(panel: pd.DataFrame, X, y) -> None:
    """Random Forest doesn't have 'epochs' — trees are added independently.
    But we can still visualize train vs test as n_estimators grows by fitting
    a series of RFs with increasing tree counts. This is the RF analog of a
    training curve — shows whether more trees help or just add capacity to
    memorize.
    """
    from sklearn.ensemble import RandomForestClassifier

    split = grouped_split(panel)
    X_train, X_test = X.iloc[split.train_idx], X.iloc[split.test_idx]
    y_train, y_test = y.iloc[split.train_idx], y.iloc[split.test_idx]

    bundle = load_model_bundle("random_forest")
    ref = bundle["model"]
    ref_params = {k: v for k, v in ref.get_params().items() if k != "n_estimators"}

    # Sample n_estimators values to plot — coarse enough to be fast, dense
    # enough to show the shape of the curve
    n_tree_grid = [10, 25, 50, 100, 150, 200, 250, 300]
    records = []
    for n in n_tree_grid:
        m = RandomForestClassifier(**ref_params, n_estimators=n)
        m.fit(X_train, y_train)
        records.append({
            "n_estimators": n,
            "train_acc": accuracy_score(y_train, m.predict(X_train)),
            "test_acc": accuracy_score(y_test, m.predict(X_test)),
            "train_f1": f1_score(y_train, m.predict(X_train), zero_division=0),
            "test_f1": f1_score(y_test, m.predict(X_test), zero_division=0),
        })
    df = pd.DataFrame(records)
    LOGGER.info("RF training curve:\n%s", df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(df["n_estimators"], df["train_acc"], label="train", linewidth=2, marker="o")
    ax.plot(df["n_estimators"], df["test_acc"], label="test", linewidth=2, marker="o")
    ax.set_xlabel("n_estimators (trees)")
    ax.set_ylabel("accuracy")
    ax.set_title("hotspot — Random Forest accuracy vs tree count (grouped split)")
    ax.set_ylim(0, 1.02)
    ax.legend()

    ax = axes[1]
    ax.plot(df["n_estimators"], df["train_f1"], label="train", linewidth=2, marker="o")
    ax.plot(df["n_estimators"], df["test_f1"], label="test", linewidth=2, marker="o")
    ax.set_xlabel("n_estimators (trees)")
    ax.set_ylabel("F1")
    ax.set_title("hotspot — Random Forest F1 vs tree count (grouped split)")
    ax.set_ylim(0, 1.02)
    ax.legend()

    plt.tight_layout()
    out = PLOTS_DIR / "08_random_forest_training_curves.png"
    plt.savefig(out)
    plt.close()
    LOGGER.info("Saved: %s", out)


def plot_lr_learning_curve(panel: pd.DataFrame, X, y) -> None:
    """Logistic Regression is not iterative in the same sense — it converges
    in one solve. The equivalent 'training curve' is a LEARNING CURVE: how
    do train / test scores change as you feed the model more data? This is
    what sklearn.model_selection.learning_curve produces.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import learning_curve
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    split = grouped_split(panel)
    X_train = X.iloc[split.train_idx]
    y_train = y.iloc[split.train_idx]

    bundle = load_model_bundle("logistic_regression")
    ref = bundle["model"]
    ref_params = ref.get_params()

    # LR is trained on scaled features; wrap in a Pipeline so learning_curve
    # scales inside each fold instead of leaking.
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(**ref_params)),
    ])

    train_sizes = np.linspace(0.1, 1.0, 8)
    sizes, train_scores_acc, test_scores_acc = learning_curve(
        pipe, X_train, y_train,
        train_sizes=train_sizes, cv=3, scoring="accuracy", n_jobs=-1,
        random_state=42, shuffle=True,
    )
    _, train_scores_f1, test_scores_f1 = learning_curve(
        pipe, X_train, y_train,
        train_sizes=train_sizes, cv=3, scoring="f1", n_jobs=-1,
        random_state=42, shuffle=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(sizes, train_scores_acc.mean(axis=1), label="train", linewidth=2, marker="o")
    ax.plot(sizes, test_scores_acc.mean(axis=1), label="test (3-fold CV)", linewidth=2, marker="o")
    ax.fill_between(sizes,
                    train_scores_acc.mean(axis=1) - train_scores_acc.std(axis=1),
                    train_scores_acc.mean(axis=1) + train_scores_acc.std(axis=1),
                    alpha=0.15)
    ax.fill_between(sizes,
                    test_scores_acc.mean(axis=1) - test_scores_acc.std(axis=1),
                    test_scores_acc.mean(axis=1) + test_scores_acc.std(axis=1),
                    alpha=0.15)
    ax.set_xlabel("training set size")
    ax.set_ylabel("accuracy")
    ax.set_title("hotspot — Logistic Regression learning curve (accuracy)")
    ax.set_ylim(0, 1.02)
    ax.legend()

    ax = axes[1]
    ax.plot(sizes, train_scores_f1.mean(axis=1), label="train", linewidth=2, marker="o")
    ax.plot(sizes, test_scores_f1.mean(axis=1), label="test (3-fold CV)", linewidth=2, marker="o")
    ax.fill_between(sizes,
                    train_scores_f1.mean(axis=1) - train_scores_f1.std(axis=1),
                    train_scores_f1.mean(axis=1) + train_scores_f1.std(axis=1),
                    alpha=0.15)
    ax.fill_between(sizes,
                    test_scores_f1.mean(axis=1) - test_scores_f1.std(axis=1),
                    test_scores_f1.mean(axis=1) + test_scores_f1.std(axis=1),
                    alpha=0.15)
    ax.set_xlabel("training set size")
    ax.set_ylabel("F1")
    ax.set_title("hotspot — Logistic Regression learning curve (F1)")
    ax.set_ylim(0, 1.02)
    ax.legend()

    plt.tight_layout()
    out = PLOTS_DIR / "09_logistic_regression_learning_curve.png"
    plt.savefig(out)
    plt.close()
    LOGGER.info("Saved: %s", out)


def main() -> None:
    configure_logging()
    LOGGER.info("=== Hotspot Diagnostic Plots ===")
    LOGGER.info("Output dir: %s", PLOTS_DIR)

    if not METRICS_REPORT.exists():
        raise FileNotFoundError(
            f"Metrics report not found at {METRICS_REPORT}. "
            "Run train_hotspot_classifier.py first."
        )
    report = load_metrics_report()

    panel = load_training_panel()
    panel[TARGET_COL] = panel[TARGET_COL].astype(int)
    X, _ = encode_features(panel, fit_encoder=True)
    y = panel[TARGET_COL]

    LOGGER.info("Panel: %d rows, %d features", len(X), X.shape[1])

    LOGGER.info("-- Plot 1: Train vs test F1 (overfitting check) --")
    plot_train_vs_test_f1(panel, X, y)

    LOGGER.info("-- Plot 1b: Train vs test accuracy (overfitting check) --")
    plot_train_vs_test_accuracy(panel, X, y)

    LOGGER.info("-- Plot 2: Confusion matrices --")
    plot_confusion_matrices(panel, X, y)

    LOGGER.info("-- Plot 3: ROC (grouped split) --")
    plot_roc_grouped(panel, X, y)

    LOGGER.info("-- Plot 4: Precision-Recall (grouped split) --")
    plot_pr_grouped(panel, X, y)

    LOGGER.info("-- Plot 5: Feature importance (winner) --")
    plot_feature_importance(report)

    LOGGER.info("-- Plot 6: SHAP summary (winner) --")
    plot_shap_summary()

    LOGGER.info("-- Plot 7: XGBoost training curves (logloss + accuracy per round) --")
    plot_xgboost_training_curves(panel, X, y)

    LOGGER.info("-- Plot 8: Random Forest training curves (accuracy + F1 vs tree count) --")
    plot_rf_training_curves(panel, X, y)

    LOGGER.info("-- Plot 9: Logistic Regression learning curve --")
    plot_lr_learning_curve(panel, X, y)

    LOGGER.info("=== Done ===")
    LOGGER.info("All plots in: %s", PLOTS_DIR)


if __name__ == "__main__":
    main()