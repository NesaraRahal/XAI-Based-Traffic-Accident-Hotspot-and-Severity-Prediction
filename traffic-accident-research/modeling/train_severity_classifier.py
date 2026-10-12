"""Leakage-safe severity classifier training.

Modes: binary (is_severe) or multiclass (dominant_severity).
Primary evaluation: temporal future-month prediction for known cells.
Secondary evaluation: grouped held-out-cell generalisation.
"""
from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Dict, List, Optional
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
    MODELS_DIR, aggregate_validation_metrics, compute_metrics, configure_logging,
    fit_feature_preprocessor, grouped_split, load_training_panel, log_split_summary,
    make_internal_validation_split, make_rolling_temporal_validation_splits,
    save_metrics_report, save_model_artifacts, save_shap_artifacts, temporal_split,
    transform_features,
)
from modelling_utils import find_best_threshold

LOGGER = logging.getLogger(__name__)
SEVERITY_MODE = "multiclass"  # change to "multiclass" for the 4-class run
MODEL_SEED = 42
PRIMARY_SPLIT = "temporal"
CLASS_ORDER = ["Fatal", "Serious", "Minor", "Damage-only"]
MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]

if SEVERITY_MODE == "binary":
    TASK, TARGET_COL, PRIMARY_METRIC, IS_BINARY = "severity_binary", "is_severe", "f1", True
elif SEVERITY_MODE == "multiclass":
    TASK, TARGET_COL, PRIMARY_METRIC, IS_BINARY = "severity_multiclass", "dominant_severity", "macro_f1", False
else:
    raise ValueError("SEVERITY_MODE must be 'binary' or 'multiclass'")


def make_lr():
    return LogisticRegression(class_weight="balanced", max_iter=3000, random_state=MODEL_SEED,
                              solver="liblinear" if IS_BINARY else "lbfgs")


def make_rf():
    # Preserve the existing binary severity experiment; multiclass remains balanced.
    weight = {0: 1.0, 1: 3.0} if IS_BINARY else "balanced"
    return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                  max_features="sqrt", class_weight=weight,
                                  random_state=MODEL_SEED, n_jobs=-1)


def make_xgb_binary(y):
    y = np.asarray(y).astype(int)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    ratio = neg / pos if pos else 1.0
    return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=.05,
                         subsample=.8, colsample_bytree=.8,
                         scale_pos_weight=ratio * 1.5, eval_metric="logloss",
                         early_stopping_rounds=20, random_state=MODEL_SEED, n_jobs=-1)


def make_xgb_multi():
    return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=.05,
                         subsample=.8, colsample_bytree=.8, objective="multi:softprob",
                         num_class=len(CLASS_ORDER), eval_metric="mlogloss",
                         early_stopping_rounds=20, random_state=MODEL_SEED, n_jobs=-1)


def xgb_final(n_estimators: int, y=None):
    model = make_xgb_binary(y if y is not None else np.array([0, 1])) if IS_BINARY else make_xgb_multi()
    params = model.get_params()
    params["n_estimators"] = max(1, int(n_estimators))
    params["early_stopping_rounds"] = None
    return XGBClassifier(**params)


def arr(y):
    return y.to_numpy(dtype=int) if isinstance(y, pd.Series) else np.asarray(y, dtype=int)


def multiclass_metrics(y_true, y_pred, encoder):
    yt = encoder.inverse_transform(arr(y_true)); yp = encoder.inverse_transform(arr(y_pred))
    return compute_metrics(yt, yp, task="multiclass", class_labels=CLASS_ORDER)


def log_metrics(scope, name, m):
    if IS_BINARY:
        LOGGER.info("[%s/%s] acc=%.3f p=%.3f r=%.3f f1=%.3f roc_auc=%s AP=%s", scope, name,
                    m["accuracy"], m["precision"], m["recall"], m["f1"],
                    "%.3f" % m["roc_auc"] if m.get("roc_auc") is not None else "n/a",
                    "%.3f" % m["average_precision"] if m.get("average_precision") is not None else "n/a")
    else:
        LOGGER.info("[%s/%s] acc=%.3f macro_p=%.3f macro_r=%.3f macro_f1=%.3f", scope, name,
                    m["accuracy"], m["macro_precision"], m["macro_recall"], m["macro_f1"])
        LOGGER.info("[%s/%s] per_class_f1=%s", scope, name,
                    {k: round(v, 3) for k, v in m.get("per_class_f1", {}).items()})


def fit_model(name, Xtr, ytr, Xval=None, yval=None, final_n=None):
    scaler = None
    Xtr_u, Xval_u = Xtr, Xval
    if name == "logistic_regression":
        scaler = StandardScaler()
        Xtr_u = pd.DataFrame(scaler.fit_transform(Xtr), columns=Xtr.columns, index=Xtr.index)
        if Xval is not None:
            Xval_u = pd.DataFrame(scaler.transform(Xval), columns=Xval.columns, index=Xval.index)
    if name == "logistic_regression":
        model = make_lr(); model.fit(Xtr_u, ytr)
    elif name == "random_forest":
        model = make_rf(); model.fit(Xtr_u, ytr)
    elif name == "xgboost":
        if final_n is not None:
            model = xgb_final(final_n, arr(ytr) if IS_BINARY else None)
            if IS_BINARY:
                model.fit(Xtr_u, ytr, verbose=False)
            else:
                model.fit(Xtr_u, ytr, sample_weight=compute_sample_weight("balanced", ytr), verbose=False)
        else:
            model = make_xgb_binary(arr(ytr)) if IS_BINARY else make_xgb_multi()
            kwargs = {"eval_set": [(Xval_u, yval)], "verbose": False}
            if not IS_BINARY: kwargs["sample_weight"] = compute_sample_weight("balanced", ytr)
            model.fit(Xtr_u, ytr, **kwargs)
    else:
        raise ValueError(name)
    return model, scaler


def rolling_validation(panel, split, y, encoder):
    folds = make_rolling_temporal_validation_splits(panel, split)
    fold_metrics = {n: [] for n in MODEL_NAMES}
    pooled_y, pooled_p = {n: [] for n in MODEL_NAMES}, {n: [] for n in MODEL_NAMES}
    best_iters = {n: [] for n in MODEL_NAMES}

    for i, (fit_idx, val_idx) in enumerate(folds, 1):
        fit_panel, val_panel = panel.loc[fit_idx], panel.loc[val_idx]
        prep = fit_feature_preprocessor(fit_panel)
        Xtr, Xv = transform_features(fit_panel, prep), transform_features(val_panel, prep)
        ytr, yv = y.loc[fit_idx], y.loc[val_idx]
        LOGGER.info("Rolling severity fold %d/%d: fit=%d validation=%d", i, len(folds), len(fit_idx), len(val_idx))
        for name in MODEL_NAMES:
            model, scaler = fit_model(name, Xtr, ytr, Xv, yv)
            Xv_u = scaler.transform(Xv) if scaler is not None else Xv
            if scaler is not None: Xv_u = pd.DataFrame(Xv_u, columns=Xv.columns, index=Xv.index)
            if name == "xgboost": best_iters[name].append(int(model.best_iteration) + 1)
            if IS_BINARY:
                p = model.predict_proba(Xv_u)[:, 1]
                yt = arr(yv); pred = (p >= .5).astype(int)
                m = compute_metrics(yt, pred, p, task="binary")
                pooled_y[name].append(yt); pooled_p[name].append(p)
            else:
                pred = model.predict(Xv_u); m = multiclass_metrics(yv, pred, encoder)
            fold_metrics[name].append(m)
            LOGGER.info("[rolling %d/%s] %s=%.4f", i, name, PRIMARY_METRIC, m[PRIMARY_METRIC])

    summary, thresholds = {}, {}
    for name in MODEL_NAMES:
        if IS_BINARY:
            yt = np.concatenate(pooled_y[name]); p = np.concatenate(pooled_p[name])
            t = find_best_threshold(yt, p)
            pred = (p >= t["threshold"]).astype(int)
            pm = compute_metrics(yt, pred, p, task="binary")
            summary[name] = aggregate_validation_metrics(fold_metrics[name])
            summary[name].update({"selection_metric": "pooled_rolling_validation_f1",
                                  "pooled_f1": pm["f1"], "pooled_precision": pm["precision"],
                                  "pooled_recall": pm["recall"], "pooled_roc_auc": pm["roc_auc"],
                                  "pooled_average_precision": pm["average_precision"],
                                  "selected_threshold": t["threshold"],
                                  "threshold_validation_f1": t["f1"]})
            thresholds[name] = float(t["threshold"])
        else:
            vals = [m["macro_f1"] for m in fold_metrics[name]]
            summary[name] = {"n_folds": len(vals), "mean_macro_f1": float(np.mean(vals)),
                             "std_macro_f1": float(np.std(vals)),
                             "selection_metric": "mean_rolling_validation_macro_f1",
                             "selection_score": float(np.mean(vals))}
            thresholds[name] = None
    medians = {n: (int(round(np.median(best_iters[n]))) if best_iters[n] else None) for n in MODEL_NAMES}
    return summary, thresholds, medians


def final_temporal(name, panel, split, y, encoder, threshold, n_estimators):
    tr_idx, te_idx = np.asarray(split.train_idx), np.asarray(split.test_idx)
    trp, tep = panel.loc[tr_idx], panel.loc[te_idx]
    prep = fit_feature_preprocessor(trp)
    Xtr, Xte = transform_features(trp, prep), transform_features(tep, prep)
    model, scaler = fit_model(name, Xtr, y.loc[tr_idx], final_n=n_estimators if name == "xgboost" else None)
    Xte_u = Xte
    if scaler is not None:
        Xte_u = pd.DataFrame(scaler.transform(Xte), columns=Xte.columns, index=Xte.index)
    yte = y.loc[te_idx]
    if IS_BINARY:
        p = model.predict_proba(Xte_u)[:, 1]
        threshold = .5 if threshold is None else float(threshold)
        pred = (p >= threshold).astype(int)
        m = compute_metrics(arr(yte), pred, p, task="binary"); m["tuned_threshold"] = threshold
    else:
        pred = model.predict(Xte_u); m = multiclass_metrics(yte, pred, encoder)
    return {"model": model, "scaler": scaler, "preprocessor": prep, "metrics": m,
            "threshold": threshold, "X_test": Xte, "X_train": Xtr}


def grouped_secondary(panel, split, y, encoder):
    fit_idx, val_idx = make_internal_validation_split(panel, split)
    fitp, valp = panel.loc[fit_idx], panel.loc[val_idx]
    prep_val = fit_feature_preprocessor(fitp)
    Xf, Xv = transform_features(fitp, prep_val), transform_features(valp, prep_val)
    yf, yv = y.loc[fit_idx], y.loc[val_idx]
    info, iters = {}, {}
    for name in MODEL_NAMES:
        model, scaler = fit_model(name, Xf, yf, Xv, yv)
        Xv_u = Xv if scaler is None else pd.DataFrame(scaler.transform(Xv), columns=Xv.columns, index=Xv.index)
        if name == "xgboost": iters[name] = int(model.best_iteration) + 1
        if IS_BINARY:
            p = model.predict_proba(Xv_u)[:, 1]; t = find_best_threshold(arr(yv), p)
            info[name] = {"threshold": t["threshold"], "validation_f1": t["f1"]}
        else:
            info[name] = {"validation_macro_f1": multiclass_metrics(yv, model.predict(Xv_u), encoder)["macro_f1"]}
    # Refit on all grouped training cells; test cells never enter preprocessing.
    tr_idx, te_idx = np.asarray(split.train_idx), np.asarray(split.test_idx)
    prep = fit_feature_preprocessor(panel.loc[tr_idx])
    Xtr = transform_features(panel.loc[tr_idx], prep); Xte = transform_features(panel.loc[te_idx], prep)
    results = {}
    for name in MODEL_NAMES:
        model, scaler = fit_model(name, Xtr, y.loc[tr_idx], final_n=iters.get(name) if name == "xgboost" else None)
        Xte_u = Xte if scaler is None else pd.DataFrame(scaler.transform(Xte), columns=Xte.columns, index=Xte.index)
        if IS_BINARY:
            p = model.predict_proba(Xte_u)[:, 1]; t = info[name]["threshold"]; pred = (p >= t).astype(int)
            m = compute_metrics(arr(y.loc[te_idx]), pred, p, task="binary"); m["tuned_threshold"] = t
        else:
            m = multiclass_metrics(y.loc[te_idx], model.predict(Xte_u), encoder)
        results[name] = {"validation": info[name], "metrics": m}
    return results


def shap_winner(bundle, name, feature_columns):
    X = bundle["X_test"]
    if bundle["scaler"] is not None:
        X = pd.DataFrame(bundle["scaler"].transform(X), columns=X.columns, index=X.index)
    explainer = shap.TreeExplainer(bundle["model"]) if name in ("random_forest", "xgboost") else shap.LinearExplainer(bundle["model"], X)
    values = explainer.shap_values(X)
    save_shap_artifacts(TASK, explainer, values, feature_columns, X)


def main():
    configure_logging()
    LOGGER.info("=== Severity training start: mode=%s ===", SEVERITY_MODE)
    panel = load_training_panel()
    if "severity_defined" in panel.columns:
        before = len(panel)
        panel = panel[panel["severity_defined"].astype(bool)].copy()
        LOGGER.info("Retained %d/%d severity-defined rows", len(panel), before)
    panel = panel.dropna(subset=[TARGET_COL]).sort_values(["year_month", "h3_cell"]).reset_index(drop=True)
    if panel.empty: raise ValueError("No severity-defined rows remain.")

    if IS_BINARY:
        y, encoder = panel[TARGET_COL].astype(int), None
    else:
        encoder = LabelEncoder(); y = pd.Series(encoder.fit_transform(panel[TARGET_COL].astype(str)), index=panel.index)
        LOGGER.info("Multiclass encoder classes: %s", list(encoder.classes_))

    temporal, grouped = temporal_split(panel), grouped_split(panel)
    log_split_summary(temporal, panel, TARGET_COL); log_split_summary(grouped, panel, TARGET_COL)

    validation, thresholds, medians = rolling_validation(panel, temporal, y, encoder)
    scores = {n: validation[n]["pooled_f1"] if IS_BINARY else validation[n]["selection_score"] for n in MODEL_NAMES}
    winner = max(scores, key=scores.get)
    LOGGER.info("PRIMARY WINNER: %s | validation score=%.4f", winner, scores[winner])

    temporal_results = {}
    for name in MODEL_NAMES:
        temporal_results[name] = final_temporal(name, panel, temporal, y, encoder, thresholds[name], medians[name])
        log_metrics("temporal_test", name, temporal_results[name]["metrics"])

    grouped_results = grouped_secondary(panel, grouped, y, encoder)
    for name in MODEL_NAMES: log_metrics("grouped_test", name, grouped_results[name]["metrics"])

    feature_columns = temporal_results[winner]["preprocessor"].feature_columns
    for name in MODEL_NAMES:
        b = temporal_results[name]
        save_model_artifacts(TASK, name, b["model"], feature_columns, scaler=b["scaler"], threshold=b["threshold"], preprocessor=b["preprocessor"])
    wb = temporal_results[winner]
    save_model_artifacts(TASK, winner, wb["model"], feature_columns, scaler=wb["scaler"], threshold=wb["threshold"], preprocessor=wb["preprocessor"], is_winner=True)

    if IS_BINARY:
        joblib.dump({"tuned_threshold": wb["threshold"], "winner_model": winner,
                     "selection": "pooled_rolling_temporal_validation"}, MODELS_DIR / f"{TASK}_winner_threshold.joblib")
    else:
        joblib.dump({"label_encoder": encoder, "class_order": CLASS_ORDER}, MODELS_DIR / f"{TASK}_label_encoder.joblib")

    report = {
        "task": TASK, "mode": SEVERITY_MODE, "target": TARGET_COL,
        "primary_metric": PRIMARY_METRIC, "primary_split": PRIMARY_SPLIT,
        "winner": winner, "winner_validation_score": float(scores[winner]),
        "n_features": len(feature_columns), "feature_columns": feature_columns,
        "severity_training_definition": "Only severity_defined=true rows with a non-null severity target are used.",
        "evaluation_rules": {
            "temporal_test_period": "2023-09 through 2023-12",
            "temporal_test_used_for_model_selection": False,
            "temporal_test_used_for_threshold_selection": False,
            "temporal_test_used_for_preprocessing_fit": False,
            "temporal_test_used_for_xgboost_early_stopping": False,
            "binary_threshold_selection": "pooled rolling validation only" if IS_BINARY else None,
            "xgboost_final_refit": "median best iteration from rolling validation folds",
            "primary_scope": "future-month prediction for known cells",
            "secondary_scope": "generalisation to held-out unseen cells",
        },
        "temporal_rolling_validation": validation,
        "temporal_test": {"description": temporal.description, "n_train": len(temporal.train_idx), "n_test": len(temporal.test_idx),
                          "models": {n: temporal_results[n]["metrics"] for n in MODEL_NAMES}},
        "grouped_secondary": {"description": grouped.description, "n_train": len(grouped.train_idx), "n_test": len(grouped.test_idx),
                              "models": grouped_results},
    }
    if not IS_BINARY: report.update({"class_order": CLASS_ORDER, "label_classes_present": list(encoder.classes_)})
    save_metrics_report(TASK, report)
    shap_winner(wb, winner, feature_columns)
    LOGGER.info("=== Severity training complete: %s | temporal test %s=%.4f ===", winner, PRIMARY_METRIC, wb["metrics"][PRIMARY_METRIC])


if __name__ == "__main__": main()
