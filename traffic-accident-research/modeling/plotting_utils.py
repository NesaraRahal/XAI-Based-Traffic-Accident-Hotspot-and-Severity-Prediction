"""Overfitting diagnostics: XGBoost train/eval loss curves + train-vs-test
score bar charts for all models. Import from both train_hotspot_classifier.py
and train_severity_classifier.py.

Why two different plot types:
- XGBoost is the only one of the three models trained iteratively (boosting
  rounds), so it's the only one with a genuine per-round loss curve. Pass
  eval_set at fit time and this reads it back via model.evals_result().
- Random Forest and Logistic Regression aren't trained "over rounds" in a
  way that produces a comparable curve. The honest substitute for all three
  models is comparing the score on the TRAINING data the model was fit on
  vs. the held-out TEST data — a big gap (high train, low test) is the
  standard signature of overfitting regardless of algorithm.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # headless — no display needed to save PNGs
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)


def plot_xgb_loss_curve(
    model, task: str, split_name: str, out_dir: Path, eval_metric: str = "logloss",
) -> Optional[Path]:
    """Plot train vs. eval loss per boosting round for a fitted XGBClassifier
    that was trained with eval_set=[(X_train, y_train), (X_test, y_test)].
    Returns the saved path, or None if evals_result is unavailable (e.g.
    eval_set wasn't passed at fit time).
    """
    results = model.evals_result() if hasattr(model, "evals_result") else {}
    if not results:
        LOGGER.warning(
            "[%s/%s] No evals_result on this XGBoost model — did you pass "
            "eval_set=[(X_train,y_train),(X_test,y_test)] to .fit()?",
            task, split_name,
        )
        return None

    # XGBoost names eval_set entries "validation_0", "validation_1", ...
    set_names = list(results.keys())
    metric_key = eval_metric if eval_metric in results[set_names[0]] else list(results[set_names[0]].keys())[0]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["train", "test"] if len(set_names) == 2 else set_names
    for set_name, label in zip(set_names, labels):
        ax.plot(results[set_name][metric_key], label=label)

    ax.set_xlabel("Boosting round")
    ax.set_ylabel(metric_key)
    ax.set_title(f"{task} — XGBoost {metric_key} per round ({split_name} split)")
    ax.legend()
    ax.grid(alpha=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}_xgboost_{split_name}_loss_curve.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved XGBoost loss curve: %s", out_path)
    return out_path


def plot_train_test_scores(
    task: str, split_name: str, scores: Dict[str, Dict[str, float]],
    metric_label: str, out_dir: Path,
) -> Path:
    """Grouped bar chart: train score vs. test score, one group per model.

    scores = {
        "logistic_regression": {"train": 0.91, "test": 0.62},
        "random_forest":       {"train": 0.99, "test": 0.71},
        "xgboost":              {"train": 0.97, "test": 0.74},
    }
    A big train/test gap for any model = overfitting on that model.
    """
    model_names = list(scores.keys())
    train_vals = [scores[m]["train"] for m in model_names]
    test_vals = [scores[m]["test"] for m in model_names]

    x = range(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - width / 2 for i in x], train_vals, width, label="train", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], test_vals, width, label="test", color="#DD8452")

    ax.set_xticks(list(x))
    ax.set_xticklabels(model_names, rotation=15)
    ax.set_ylabel(metric_label)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{task} — train vs test {metric_label} ({split_name} split)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    for i, (tr, te) in enumerate(zip(train_vals, test_vals)):
        ax.annotate(f"{tr:.2f}", (i - width / 2, tr), ha="center", va="bottom", fontsize=8)
        ax.annotate(f"{te:.2f}", (i + width / 2, te), ha="center", va="bottom", fontsize=8)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}_{split_name}_train_vs_test_{metric_label}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved train/test score chart: %s", out_path)
    return out_path