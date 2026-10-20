"""Temporal-split diagnostic plots for the multiclass severity classifier.

Companion to plot_hotspot_temporal.py. Produces appendix figures for the
4-class severity task (Fatal / Serious / Minor / Damage-only) on the PRIMARY
TEMPORAL SPLIT used in the thesis:

    Rows:  severity_defined = true cell-months only
    Train: year_month <= 2023-08
    Test:  year_month >= 2023-09  (September-December 2023, 195 rows)

Plots 01-04 do NOT retrain anything. They load the saved temporal models
(models/severity_multiclass_<model>.joblib) with their own fitted
preprocessor and scaler, so they reproduce the Chapter 5 severity results
exactly (e.g. XGBoost: macro-F1 = 0.236, accuracy = 0.328).

Plots 07-09 refit copies of the models on the temporal training window only,
purely for visualisation. The test window never trains or tunes anything.

All figures are written to models/plots/severity_temporal/.

Run (from the project root):
    python modeling/plot_severity_temporal.py                # panel from CSV
    python modeling/plot_severity_temporal.py --source mongo # panel from MongoDB
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, Tuple

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
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_sample_weight

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

TASK = "severity_multiclass"
TARGET_COL = "dominant_severity"
CLASS_ORDER = ["Fatal", "Serious", "Minor", "Damage-only"]
PANEL_CSV = ROOT_DIR / "data" / "processed" / "panel_cell_month.csv"

PLOTS_DIR = MODELS_DIR / "plots" / "severity_temporal"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]
DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
MODEL_FILES = {n: MODELS_DIR / f"{TASK}_{n}.joblib" for n in MODEL_NAMES}
ENCODER_FILE = MODELS_DIR / f"{TASK}_label_encoder.joblib"
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
    """Load the panel and keep only severity-defined rows, as the trainer does."""
    if source == "mongo":
        panel = load_training_panel()
    else:
        if not PANEL_CSV.exists():
            raise FileNotFoundError(f"{PANEL_CSV} not found. Run with --source mongo.")
        panel = pd.read_csv(PANEL_CSV)
        LOGGER.info("Loaded %d panel rows from %s", len(panel), PANEL_CSV)

    if "severity_defined" in panel.columns:
        panel = panel[panel["severity_defined"].astype(bool)].copy()
    panel = (panel.dropna(subset=[TARGET_COL])
             .sort_values(["year_month", "h3_cell"])
             .reset_index(drop=True))
    panel["year_month"] = panel["year_month"].astype(str)
    panel[TARGET_COL] = panel[TARGET_COL].astype(str)
    LOGGER.info("Severity-defined rows: %d", len(panel))
    return panel


def temporal_parts(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = panel[panel["year_month"] <= TEMPORAL_SPLIT_TRAIN_END]
    test = panel[panel["year_month"] >= TEMPORAL_SPLIT_TEST_START]
    LOGGER.info("Temporal train: %d rows | test: %d rows | test classes: %s",
                len(train), len(test), test[TARGET_COL].value_counts().to_dict())
    return train, test


def load_encoder():
    return joblib.load(ENCODER_FILE)["label_encoder"]


def load_bundle(name: str) -> Dict:
    return joblib.load(MODEL_FILES[name])


def prepare_X(panel_part: pd.DataFrame, bundle: Dict) -> pd.DataFrame:
    X = transform_features(panel_part, bundle["preprocessor"])
    if bundle.get("scaler") is not None:
        X = pd.DataFrame(bundle["scaler"].transform(X), columns=X.columns, index=X.index)
    return X


def predict_labels(bundle: Dict, panel_part: pd.DataFrame, encoder) -> np.ndarray:
    """Predicted class names for a saved model."""
    pred = bundle["model"].predict(prepare_X(panel_part, bundle))
    return encoder.inverse_transform(np.asarray(pred).astype(int).ravel())


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)


def save(fig_name: str) -> None:
    out = PLOTS_DIR / fig_name
    plt.savefig(out)
    plt.close()
    LOGGER.info("Saved: %s", out)


# ══════════════════════════════════════════════════════════════════════════
# Plot 1 / 1b — Train vs test macro-F1 and accuracy
# ══════════════════════════════════════════════════════════════════════════

def _bar(df: pd.DataFrame, metric: str, title: str, out: str,
         baseline: float | None = None, baseline_label: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df.set_index("model")[[f"{metric} (train)", f"{metric} (test)"]]
    sub.plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452"], edgecolor="black")
    if baseline is not None:
        ax.axhline(baseline, color="black", linestyle="--", alpha=0.5,
                   label=f"{baseline_label} = {baseline:.3f}")
    ax.set_title(title)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(metric)
    ax.set_xlabel("")
    ax.set_xticklabels(sub.index, rotation=0)
    for i, model in enumerate(sub.index):
        gap = sub.loc[model].iloc[0] - sub.loc[model].iloc[1]
        ax.annotate(f"gap = {gap:+.3f}", xy=(i, max(sub.loc[model].values) + 0.03),
                    ha="center", fontsize=10,
                    color="crimson" if gap > 0.1 else "darkgreen")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save(out)


def plot_train_vs_test(train, test, encoder) -> None:
    rows = []
    for name in MODEL_NAMES:
        b = load_bundle(name)
        p_tr = predict_labels(b, train, encoder)
        p_te = predict_labels(b, test, encoder)
        rows.append({
            "model": DISPLAY_NAMES[name],
            "Macro-F1 (train)": macro_f1(train[TARGET_COL], p_tr),
            "Macro-F1 (test)": macro_f1(test[TARGET_COL], p_te),
            "Accuracy (train)": accuracy_score(train[TARGET_COL], p_tr),
            "Accuracy (test)": accuracy_score(test[TARGET_COL], p_te),
        })
    df = pd.DataFrame(rows)
    LOGGER.info("Train vs test (temporal):\n%s", df.round(3).to_string(index=False))

    _bar(df, "Macro-F1", f"Train vs test macro-F1\n{SPLIT_LABEL}",
         "01_train_vs_test_macro_f1.png",
         baseline=0.25, baseline_label="Random guessing (4 classes)")

    majority = test[TARGET_COL].value_counts(normalize=True).max()
    _bar(df, "Accuracy", f"Train vs test accuracy\n{SPLIT_LABEL}",
         "01b_train_vs_test_accuracy.png",
         baseline=majority, baseline_label="Majority-class baseline (test)")


# ══════════════════════════════════════════════════════════════════════════
# Plot 2 — Confusion matrices
# ══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(test, encoder) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, name in zip(axes, MODEL_NAMES):
        pred = predict_labels(load_bundle(name), test, encoder)
        cm = confusion_matrix(test[TARGET_COL], pred, labels=CLASS_ORDER)
        LOGGER.info("%s confusion matrix (rows=%s): %s", name, CLASS_ORDER, cm.tolist())
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
                    annot_kws={"size": 12})
        ax.set_title(DISPLAY_NAMES[name])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual" if name == MODEL_NAMES[0] else "")
    plt.suptitle(f"Severity confusion matrices on the temporal test window "
                 f"({TEMPORAL_SPLIT_TEST_START} to 2023-12)", y=1.03)
    plt.tight_layout()
    save("02_confusion_matrices.png")


# ══════════════════════════════════════════════════════════════════════════
# Plot 3 — Per-class F1
# ══════════════════════════════════════════════════════════════════════════

def plot_per_class_f1(test, encoder) -> None:
    rows = []
    for name in MODEL_NAMES:
        pred = predict_labels(load_bundle(name), test, encoder)
        scores = f1_score(test[TARGET_COL], pred, labels=CLASS_ORDER,
                          average=None, zero_division=0)
        for cls, s in zip(CLASS_ORDER, scores):
            rows.append({"Model": DISPLAY_NAMES[name], "Class": cls, "F1": s})
    df = pd.DataFrame(rows)
    LOGGER.info("Per-class F1:\n%s", df.pivot(index="Class", columns="Model",
                                              values="F1").round(3).to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df, x="Class", y="F1", hue="Model", order=CLASS_ORDER,
                edgecolor="black", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("F1-score")
    ax.set_title(f"Per-class F1-score on the temporal test window\n{SPLIT_LABEL}")
    ax.legend(title="", loc="upper right")
    plt.tight_layout()
    save("03_per_class_f1.png")


# ══════════════════════════════════════════════════════════════════════════
# Plot 4 — One-vs-rest ROC for the winner
# ══════════════════════════════════════════════════════════════════════════

def plot_ovr_roc(test, encoder, report: Dict) -> None:
    winner = report["winner"]
    b = load_bundle(winner)
    proba = b["model"].predict_proba(prepare_X(test, b))
    classes = list(encoder.classes_)  # column order of predict_proba
    y = test[TARGET_COL].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random baseline")
    for cls in CLASS_ORDER:
        if cls not in classes:
            continue
        y_bin = (y == cls).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        scores = proba[:, classes.index(cls)]
        fpr, tpr, _ = roc_curve(y_bin, scores)
        auc = roc_auc_score(y_bin, scores)
        ax.plot(fpr, tpr, linewidth=2, label=f"{cls} vs rest (AUC = {auc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"One-vs-rest ROC curves — {DISPLAY_NAMES.get(winner, winner)}\n"
                 f"Temporal test window ({TEMPORAL_SPLIT_TEST_START} to 2023-12)")
    ax.legend(loc="lower right")
    save("04_ovr_roc_winner.png")


# ══════════════════════════════════════════════════════════════════════════
# Plot 5 / 6 — Feature importance and SHAP for the winner
# ══════════════════════════════════════════════════════════════════════════

def plot_feature_importance(report: Dict) -> None:
    winner = report["winner"]
    path = MODELS_DIR / f"{TASK}_{winner}_winner.joblib"
    b = joblib.load(path if path.exists() else MODEL_FILES[winner])
    model, features = b["model"], b["feature_columns"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        LOGGER.warning("Winner has no feature importances; skipping.")
        return
    imp = (pd.DataFrame({"feature": features, "importance": importances})
           .sort_values("importance", ascending=False).head(15))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(data=imp, y="feature", x="importance",
                color="#4c72b0", edgecolor="black", ax=ax)
    ax.set_title(f"Top 15 features by importance — {DISPLAY_NAMES.get(winner, winner)} "
                 f"severity classifier")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    save("05_feature_importance.png")


def plot_shap_summary(encoder) -> None:
    if not SHAP_ARTIFACT.exists():
        LOGGER.warning("No SHAP artifact at %s; skipping.", SHAP_ARTIFACT)
        return
    art = joblib.load(SHAP_ARTIFACT)
    values, X_sample = art["shap_values"], art["X_sample"]

    # Normalise to a list of (n_samples, n_features) arrays, one per class.
    if isinstance(values, list):
        per_class = values
    else:
        arr = np.asarray(values)
        per_class = [arr[:, :, k] for k in range(arr.shape[-1])] if arr.ndim == 3 else [arr]

    class_names = list(encoder.classes_)[:len(per_class)]
    plt.figure(figsize=(9, 7))
    shap.summary_plot(per_class, X_sample, plot_type="bar", max_display=15,
                      class_names=class_names, show=False)
    plt.title(f"SHAP mean |value| by severity class — top 15 features (winner)\n"
              f"Temporal test window ({TEMPORAL_SPLIT_TEST_START} to 2023-12, "
              f"{len(X_sample)} cell-months)")
    save("06_shap_summary.png")


# ══════════════════════════════════════════════════════════════════════════
# Plot 7 — XGBoost multiclass log loss per round
# ══════════════════════════════════════════════════════════════════════════

def plot_xgboost_training_curves(train, test, encoder) -> None:
    from xgboost import XGBClassifier

    b = load_bundle("xgboost")
    params = {k: v for k, v in b["model"].get_params().items()
              if k not in ("eval_metric", "early_stopping_rounds")}
    tracker = XGBClassifier(**params, eval_metric=["mlogloss", "merror"])

    X_tr, X_te = prepare_X(train, b), prepare_X(test, b)
    y_tr = encoder.transform(train[TARGET_COL])
    y_te = encoder.transform(test[TARGET_COL])
    tracker.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr),
                eval_set=[(X_tr, y_tr), (X_te, y_te)], verbose=False)
    hist = tracker.evals_result()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(hist["validation_0"]["mlogloss"], label="train", linewidth=2)
    ax.plot(hist["validation_1"]["mlogloss"], label="test", linewidth=2)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Multiclass log loss")
    ax.set_title("XGBoost multiclass log loss per round (temporal split)")
    ax.legend()

    ax = axes[1]
    ax.plot([1 - e for e in hist["validation_0"]["merror"]], label="train", linewidth=2)
    ax.plot([1 - e for e in hist["validation_1"]["merror"]], label="test", linewidth=2)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Accuracy")
    ax.set_title("XGBoost accuracy per round (temporal split)")
    ax.set_ylim(0, 1.02)
    ax.legend()
    plt.tight_layout()
    save("07_xgboost_training_curves.png")


# ══════════════════════════════════════════════════════════════════════════
# Plot 8 — Random Forest macro-F1 vs number of trees
# ══════════════════════════════════════════════════════════════════════════

def plot_rf_training_curves(train, test, encoder) -> None:
    from sklearn.ensemble import RandomForestClassifier

    b = load_bundle("random_forest")
    params = {k: v for k, v in b["model"].get_params().items() if k != "n_estimators"}
    X_tr, X_te = prepare_X(train, b), prepare_X(test, b)
    y_tr = encoder.transform(train[TARGET_COL])

    rows = []
    for n in [10, 25, 50, 100, 150, 200, 250, 300]:
        m = RandomForestClassifier(**params, n_estimators=n).fit(X_tr, y_tr)
        p_tr = encoder.inverse_transform(m.predict(X_tr))
        p_te = encoder.inverse_transform(m.predict(X_te))
        rows.append({
            "n_estimators": n,
            "train_acc": accuracy_score(train[TARGET_COL], p_tr),
            "test_acc": accuracy_score(test[TARGET_COL], p_te),
            "train_f1": macro_f1(train[TARGET_COL], p_tr),
            "test_f1": macro_f1(test[TARGET_COL], p_te),
        })
    df = pd.DataFrame(rows)
    LOGGER.info("RF vs tree count:\n%s", df.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, label in [(axes[0], "acc", "Accuracy"), (axes[1], "f1", "Macro-F1")]:
        ax.plot(df["n_estimators"], df[f"train_{metric}"], label="train", linewidth=2, marker="o")
        ax.plot(df["n_estimators"], df[f"test_{metric}"], label="test", linewidth=2, marker="o")
        ax.set_xlabel("Number of trees (n_estimators)")
        ax.set_ylabel(label)
        ax.set_title(f"Random Forest {label.lower()} vs tree count (temporal split)")
        ax.set_ylim(0, 1.02)
        ax.legend()
    plt.tight_layout()
    save("08_random_forest_training_curves.png")


# ══════════════════════════════════════════════════════════════════════════
# Plot 9 — Logistic Regression learning curve (expanding monthly window)
# ══════════════════════════════════════════════════════════════════════════

def plot_lr_learning_curve(train, test, encoder) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    params = load_bundle("logistic_regression")["model"].get_params()
    months = sorted(train["year_month"].unique())

    rows = []
    for k in range(4, len(months) + 1, 2):
        part = train[train["year_month"].isin(months[:k])]
        if part[TARGET_COL].nunique() < 2:
            continue
        pre = fit_feature_preprocessor(part)
        X_part = transform_features(part, pre)
        X_te = transform_features(test, pre)
        scaler = StandardScaler().fit(X_part)
        y_part = encoder.transform(part[TARGET_COL])
        m = LogisticRegression(**params).fit(scaler.transform(X_part), y_part)
        p_tr = encoder.inverse_transform(m.predict(scaler.transform(X_part)))
        p_te = encoder.inverse_transform(m.predict(scaler.transform(X_te)))
        rows.append({
            "months": k, "rows": len(part),
            "train_acc": accuracy_score(part[TARGET_COL], p_tr),
            "test_acc": accuracy_score(test[TARGET_COL], p_te),
            "train_f1": macro_f1(part[TARGET_COL], p_tr),
            "test_f1": macro_f1(test[TARGET_COL], p_te),
        })
    df = pd.DataFrame(rows)
    LOGGER.info("LR temporal learning curve:\n%s", df.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, label in [(axes[0], "acc", "Accuracy"), (axes[1], "f1", "Macro-F1")]:
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
    LOGGER.info("=== Severity temporal-split plots -> %s ===", PLOTS_DIR)

    with open(METRICS_REPORT, encoding="utf-8") as f:
        report = json.load(f)
    encoder = load_encoder()

    panel = load_panel(args.source)
    train, test = temporal_parts(panel)

    plot_train_vs_test(train, test, encoder)
    plot_confusion_matrices(test, encoder)
    plot_per_class_f1(test, encoder)
    plot_ovr_roc(test, encoder, report)
    plot_feature_importance(report)
    try:
        plot_shap_summary(encoder)
    except Exception as exc:  # keep the remaining plots running
        LOGGER.warning("SHAP summary skipped: %s", exc)
    plot_xgboost_training_curves(train, test, encoder)
    plot_rf_training_curves(train, test, encoder)
    plot_lr_learning_curve(train, test, encoder)

    LOGGER.info("=== Done. All plots in: %s ===", PLOTS_DIR)


if __name__ == "__main__":
    main()
