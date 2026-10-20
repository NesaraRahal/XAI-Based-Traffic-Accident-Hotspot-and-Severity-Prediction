"""Temporal-split diagnostic plots for the hotspot classifier.

This script produces the same set of figures as plot_hotspot_diagnostics.py,
but on the PRIMARY TEMPORAL SPLIT used in the thesis:

    Train: year_month <= 2023-08
    Test:  year_month >= 2023-09  (September-December 2023, 316 rows)

Unlike the grouped-split diagnostics, plots 01-04 do NOT retrain anything.
They load the saved temporal models (models/hotspot_<model>.joblib), apply
each model's own fitted preprocessor, scaler and frozen decision threshold,
and therefore reproduce the temporal test results reported in Chapter 5
exactly (e.g. Random Forest: F1 = 0.491, ROC-AUC = 0.957).

Plots 07-09 (training curves) refit copies of the models on the temporal
training window only, purely for visualisation. The test window is never
used to train or tune anything.

All figures are written to models/plots/hotspot_temporal/ so the original
grouped-split figures in models/plots/hotspot/ are left untouched.

Run (from the project root):
    python modeling/plot_hotspot_temporal.py               # panel from CSV
    python modeling/plot_hotspot_temporal.py --source mongo  # panel from MongoDB
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from modelling_utils import (
    MODELS_DIR,
    ROOT_DIR,
    TEMPORAL_SPLIT_TEST_START,
    TEMPORAL_SPLIT_TRAIN_END,
    configure_logging,
    fit_feature_preprocessor,
    load_training_panel,
    transform_features,
)

LOGGER = logging.getLogger(__name__)

TASK = "hotspot"
TARGET_COL = "is_hotspot"
PANEL_CSV = ROOT_DIR / "data" / "processed" / "panel_cell_month.csv"

PLOTS_DIR = MODELS_DIR / "plots" / "hotspot_temporal"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]
DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
MODEL_FILES = {name: MODELS_DIR / f"{TASK}_{name}.joblib" for name in MODEL_NAMES}
METRICS_REPORT = MODELS_DIR / f"{TASK}_metrics_report.json"
SHAP_ARTIFACT = MODELS_DIR / f"{TASK}_shap.joblib"

SPLIT_LABEL = (
    f"Temporal split (train ≤ {TEMPORAL_SPLIT_TRAIN_END}, "
    f"test {TEMPORAL_SPLIT_TEST_START} to 2023-12)"
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


# ══════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════

def load_panel(source: str) -> pd.DataFrame:
    """Load the (cell, month) panel exactly as the trainer prepares it."""
    if source == "mongo":
        panel = load_training_panel()
    else:
        if not PANEL_CSV.exists():
            raise FileNotFoundError(
                f"{PANEL_CSV} not found. Run with --source mongo instead."
            )
        panel = pd.read_csv(PANEL_CSV)
        LOGGER.info("Loaded %d panel rows from %s", len(panel), PANEL_CSV)

    panel = panel.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    panel[TARGET_COL] = panel[TARGET_COL].astype(int)
    panel["year_month"] = panel["year_month"].astype(str)
    return panel


def temporal_parts(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = panel[panel["year_month"] <= TEMPORAL_SPLIT_TRAIN_END]
    test = panel[panel["year_month"] >= TEMPORAL_SPLIT_TEST_START]
    LOGGER.info(
        "Temporal train: %d rows (%d hotspots) | test: %d rows (%d hotspots)",
        len(train), int(train[TARGET_COL].sum()),
        len(test), int(test[TARGET_COL].sum()),
    )
    return train, test


def load_bundle(name: str) -> Dict:
    return joblib.load(MODEL_FILES[name])


def prepare_X(panel_part: pd.DataFrame, bundle: Dict) -> pd.DataFrame:
    """Apply the bundle's own fitted preprocessor (and scaler, for LR)."""
    X = transform_features(panel_part, bundle["preprocessor"])
    if bundle.get("scaler") is not None:
        X = pd.DataFrame(
            bundle["scaler"].transform(X), columns=X.columns, index=X.index
        )
    return X


def scored(bundle: Dict, panel_part: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, thresholded predictions) for a saved model."""
    X = prepare_X(panel_part, bundle)
    proba = bundle["model"].predict_proba(X)[:, 1]
    pred = (proba >= float(bundle["threshold"])).astype(int)
    return proba, pred


def save(fig_name: str) -> None:
    out = PLOTS_DIR / fig_name
    plt.savefig(out)
    plt.close()
    LOGGER.info("Saved: %s", out)


# ══════════════════════════════════════════════════════════════════════════
# Plot 1 / 1b — Train vs test F1 and accuracy (saved models, frozen thresholds)
# ══════════════════════════════════════════════════════════════════════════

def _train_test_bar(records: pd.DataFrame, metric: str, title: str,
                    out_name: str, baseline: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = records.set_index("model")[[f"{metric} (train)", f"{metric} (test)"]]
    sub.plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452"], edgecolor="black")
    if baseline is not None:
        ax.axhline(baseline, color="black", linestyle="--", alpha=0.5,
                   label=f"Majority-class baseline (test) = {baseline:.3f}")
    ax.set_title(title)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(metric)
    ax.set_xlabel("")
    ax.set_xticklabels(sub.index, rotation=0)
    for i, model in enumerate(sub.index):
        gap = sub.loc[model].iloc[0] - sub.loc[model].iloc[1]
        ax.annotate(f"gap = {gap:+.3f}",
                    xy=(i, max(sub.loc[model].values) + 0.03),
                    ha="center", fontsize=10,
                    color="crimson" if gap > 0.1 else "darkgreen")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    save(out_name)


def plot_train_vs_test(train: pd.DataFrame, test: pd.DataFrame) -> None:
    rows = []
    for name in MODEL_NAMES:
        bundle = load_bundle(name)
        _, pred_train = scored(bundle, train)
        _, pred_test = scored(bundle, test)
        rows.append({
            "model": DISPLAY_NAMES[name],
            "F1 (train)": f1_score(train[TARGET_COL], pred_train, zero_division=0),
            "F1 (test)": f1_score(test[TARGET_COL], pred_test, zero_division=0),
            "Accuracy (train)": accuracy_score(train[TARGET_COL], pred_train),
            "Accuracy (test)": accuracy_score(test[TARGET_COL], pred_test),
        })
    df = pd.DataFrame(rows)
    LOGGER.info("Train vs test (temporal, frozen thresholds):\n%s",
                df.round(3).to_string(index=False))

    _train_test_bar(
        df, "F1",
        f"Train vs test F1-score\n{SPLIT_LABEL}",
        "01_train_vs_test_f1.png",
    )

    y_test = test[TARGET_COL]
    baseline = max(y_test.mean(), 1 - y_test.mean())
    _train_test_bar(
        df, "Accuracy",
        f"Train vs test accuracy\n{SPLIT_LABEL}",
        "01b_train_vs_test_accuracy.png",
        baseline=baseline,
    )


# ══════════════════════════════════════════════════════════════════════════
# Plot 2 — Confusion matrices on the temporal test window
# ══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(test: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, name in zip(axes, MODEL_NAMES):
        bundle = load_bundle(name)
        _, pred = scored(bundle, test)
        cm = confusion_matrix(test[TARGET_COL], pred, labels=[0, 1])
        LOGGER.info("%s confusion matrix: %s", name, cm.tolist())
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Not hotspot", "Hotspot"],
                    yticklabels=["Not hotspot", "Hotspot"],
                    annot_kws={"size": 13})
        ax.set_title(f"{DISPLAY_NAMES[name]} (threshold = {float(bundle['threshold']):.2f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual" if name == MODEL_NAMES[0] else "")
    plt.suptitle(f"Confusion matrices on the temporal test window "
                 f"({TEMPORAL_SPLIT_TEST_START} to 2023-12)", y=1.03)
    plt.tight_layout()
    save("02_confusion_matrices.png")


# ══════════════════════════════════════════════════════════════════════════
# Plot 3 / 4 — ROC and Precision-Recall on the temporal test window
# ══════════════════════════════════════════════════════════════════════════

def plot_roc(test: pd.DataFrame) -> None:
    y = test[TARGET_COL]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random baseline")
    for name in MODEL_NAMES:
        proba, _ = scored(load_bundle(name), test)
        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        ax.plot(fpr, tpr, linewidth=2, label=f"{DISPLAY_NAMES[name]} (AUC = {auc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curves on the temporal test window\n{SPLIT_LABEL}")
    ax.legend(loc="lower right")
    save("03_roc_temporal.png")


def plot_pr(test: pd.DataFrame) -> None:
    y = test[TARGET_COL]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(y.mean(), color="k", linestyle="--", alpha=0.4,
               label=f"Base rate = {y.mean():.3f}")
    for name in MODEL_NAMES:
        proba, _ = scored(load_bundle(name), test)
        p, r, _ = precision_recall_curve(y, proba)
        ap = average_precision_score(y, proba)
        ax.plot(r, p, linewidth=2, label=f"{DISPLAY_NAMES[name]} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Precision-Recall curves on the temporal test window\n{SPLIT_LABEL}")
    ax.legend(loc="upper right")
    save("04_pr_temporal.png")


# ══════════════════════════════════════════════════════════════════════════
# Plot 5 / 6 — Feature importance and SHAP for the winner
# ══════════════════════════════════════════════════════════════════════════

def plot_feature_importance(report: Dict) -> None:
    winner = report["winner"]
    path = MODELS_DIR / f"{TASK}_{winner}_winner.joblib"
    bundle = joblib.load(path if path.exists() else MODEL_FILES[winner])
    model = bundle["model"]
    features = bundle["feature_columns"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        LOGGER.warning("Winner has no feature importances; skipping.")
        return

    imp = (pd.DataFrame({"feature": features, "importance": importances})
           .sort_values("importance", ascending=False).head(15))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(data=imp, y="feature", x="importance",
                color="#4c72b0", edgecolor="black", ax=ax)
    ax.set_title(f"Top 15 features by importance — {DISPLAY_NAMES.get(winner, winner)} "
                 f"(trained on temporal training window)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    save("05_feature_importance.png")


def plot_shap_summary() -> None:
    if not SHAP_ARTIFACT.exists():
        LOGGER.warning("No SHAP artifact at %s; skipping.", SHAP_ARTIFACT)
        return
    art = joblib.load(SHAP_ARTIFACT)
    values = art["shap_values"]
    X_sample = art["X_sample"]

    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    else:
        arr = np.asarray(values)
        values = arr[:, :, 1] if arr.ndim == 3 and arr.shape[-1] > 1 else (
            arr[:, :, 0] if arr.ndim == 3 else arr)

    plt.figure(figsize=(9, 7))
    shap.summary_plot(values, X_sample, max_display=15, show=False)
    plt.title(f"SHAP summary — top 15 features (winner)\n"
              f"Temporal test window ({TEMPORAL_SPLIT_TEST_START} to 2023-12, "
              f"{len(X_sample)} cell-months)")
    save("06_shap_summary.png")


# ══════════════════════════════════════════════════════════════════════════
# Plot 7 — XGBoost log loss / accuracy per boosting round
# ══════════════════════════════════════════════════════════════════════════

def plot_xgboost_training_curves(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Refit a copy of the saved XGBoost with identical hyperparameters on the
    temporal training window, tracking train and test loss at every round.
    The test window is only *monitored*; it does not influence the fit."""
    from xgboost import XGBClassifier

    bundle = load_bundle("xgboost")
    params = {k: v for k, v in bundle["model"].get_params().items()
              if k not in ("eval_metric", "early_stopping_rounds")}
    tracker = XGBClassifier(**params, eval_metric=["logloss", "error"])

    X_train = prepare_X(train, bundle)
    X_test = prepare_X(test, bundle)
    tracker.fit(X_train, train[TARGET_COL],
                eval_set=[(X_train, train[TARGET_COL]), (X_test, test[TARGET_COL])],
                verbose=False)
    hist = tracker.evals_result()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(hist["validation_0"]["logloss"], label="train", linewidth=2)
    ax.plot(hist["validation_1"]["logloss"], label="test", linewidth=2)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Log loss")
    ax.set_title("XGBoost log loss per round (temporal split)")
    ax.legend()

    ax = axes[1]
    ax.plot([1 - e for e in hist["validation_0"]["error"]], label="train", linewidth=2)
    ax.plot([1 - e for e in hist["validation_1"]["error"]], label="test", linewidth=2)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Accuracy (at 0.5 cut-off)")
    ax.set_title("XGBoost accuracy per round (temporal split)")
    ax.set_ylim(0, 1.02)
    ax.legend()

    plt.tight_layout()
    save("07_xgboost_training_curves.png")


# ══════════════════════════════════════════════════════════════════════════
# Plot 8 — Random Forest accuracy / F1 vs number of trees
# ══════════════════════════════════════════════════════════════════════════

def plot_rf_training_curves(train: pd.DataFrame, test: pd.DataFrame) -> None:
    from sklearn.ensemble import RandomForestClassifier

    bundle = load_bundle("random_forest")
    threshold = float(bundle["threshold"])
    params = {k: v for k, v in bundle["model"].get_params().items()
              if k != "n_estimators"}
    X_train = prepare_X(train, bundle)
    X_test = prepare_X(test, bundle)
    y_train, y_test = train[TARGET_COL], test[TARGET_COL]

    rows = []
    for n in [10, 25, 50, 100, 150, 200, 250, 300]:
        m = RandomForestClassifier(**params, n_estimators=n).fit(X_train, y_train)
        p_tr = (m.predict_proba(X_train)[:, 1] >= threshold).astype(int)
        p_te = (m.predict_proba(X_test)[:, 1] >= threshold).astype(int)
        rows.append({
            "n_estimators": n,
            "train_acc": accuracy_score(y_train, p_tr),
            "test_acc": accuracy_score(y_test, p_te),
            "train_f1": f1_score(y_train, p_tr, zero_division=0),
            "test_f1": f1_score(y_test, p_te, zero_division=0),
        })
    df = pd.DataFrame(rows)
    LOGGER.info("RF vs tree count (threshold %.2f):\n%s",
                threshold, df.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, label in [(axes[0], "acc", "Accuracy"), (axes[1], "f1", "F1-score")]:
        ax.plot(df["n_estimators"], df[f"train_{metric}"], label="train", linewidth=2, marker="o")
        ax.plot(df["n_estimators"], df[f"test_{metric}"], label="test", linewidth=2, marker="o")
        ax.set_xlabel("Number of trees (n_estimators)")
        ax.set_ylabel(label)
        ax.set_title(f"Random Forest {label.lower()} vs tree count "
                     f"(temporal split, threshold = {threshold:.2f})")
        ax.set_ylim(0, 1.02)
        ax.legend()
    plt.tight_layout()
    save("08_random_forest_training_curves.png")


# ══════════════════════════════════════════════════════════════════════════
# Plot 9 — Logistic Regression learning curve (expanding monthly window)
# ══════════════════════════════════════════════════════════════════════════

def plot_lr_learning_curve(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Temporal learning curve: train on the first k months of the training
    window (k = 4, 6, ... 20), refitting the preprocessor and scaler on those
    months only, and score on the same fixed temporal test window. This keeps
    time order intact, unlike shuffled k-fold learning curves."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    bundle = load_bundle("logistic_regression")
    threshold = float(bundle["threshold"])
    params = bundle["model"].get_params()
    months = sorted(train["year_month"].unique())
    y_test = test[TARGET_COL]

    rows = []
    for k in range(4, len(months) + 1, 2):
        part = train[train["year_month"].isin(months[:k])]
        if part[TARGET_COL].nunique() < 2:
            continue
        pre = fit_feature_preprocessor(part)
        X_part = transform_features(part, pre)
        X_te = transform_features(test, pre)
        scaler = StandardScaler().fit(X_part)
        m = LogisticRegression(**params).fit(scaler.transform(X_part), part[TARGET_COL])
        p_tr = (m.predict_proba(scaler.transform(X_part))[:, 1] >= threshold).astype(int)
        p_te = (m.predict_proba(scaler.transform(X_te))[:, 1] >= threshold).astype(int)
        rows.append({
            "months": k,
            "rows": len(part),
            "train_acc": accuracy_score(part[TARGET_COL], p_tr),
            "test_acc": accuracy_score(y_test, p_te),
            "train_f1": f1_score(part[TARGET_COL], p_tr, zero_division=0),
            "test_f1": f1_score(y_test, p_te, zero_division=0),
        })
    df = pd.DataFrame(rows)
    LOGGER.info("LR temporal learning curve (threshold %.2f):\n%s",
                threshold, df.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, label in [(axes[0], "acc", "Accuracy"), (axes[1], "f1", "F1-score")]:
        ax.plot(df["rows"], df[f"train_{metric}"], label="train", linewidth=2, marker="o")
        ax.plot(df["rows"], df[f"test_{metric}"], label="test (Sep-Dec 2023)", linewidth=2, marker="o")
        ax.set_xlabel("Training set size (cell-months, earliest months first)")
        ax.set_ylabel(label)
        ax.set_title(f"Logistic Regression learning curve — {label.lower()} (temporal split)")
        ax.set_ylim(0, 1.02)
        ax.legend()
    plt.tight_layout()
    save("09_logistic_regression_learning_curve.png")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=["csv", "mongo"], default="csv",
                        help="Where to load the training panel from (default: csv)")
    args = parser.parse_args()

    configure_logging()
    LOGGER.info("=== Hotspot temporal-split plots -> %s ===", PLOTS_DIR)

    with open(METRICS_REPORT, encoding="utf-8") as f:
        report = json.load(f)

    panel = load_panel(args.source)
    train, test = temporal_parts(panel)

    plot_train_vs_test(train, test)
    plot_confusion_matrices(test)
    plot_roc(test)
    plot_pr(test)
    plot_feature_importance(report)
    try:
        plot_shap_summary()
    except Exception as exc:  # keep the remaining plots running
        LOGGER.warning("SHAP summary skipped: %s", exc)
    plot_xgboost_training_curves(train, test)
    plot_rf_training_curves(train, test)
    plot_lr_learning_curve(train, test)

    LOGGER.info("=== Done. All plots in: %s ===", PLOTS_DIR)


if __name__ == "__main__":
    main()
