"""Baseline model training script for accident hotspot prediction.

Trains Logistic Regression, Random Forest, and XGBoost on the
feature-engineered accident dataset and prints accuracy scores.
"""

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "accidents_feature_engineered.csv"

# ── Config ─────────────────────────────────────────────────────────────────
HOTSPOT_THRESHOLD = 5  # locations with >= this many accidents = hotspot
TARGET_COLUMN = "Is_Hotspot"
FEATURE_COLS = [
    "Week (Derived)",
    "Month (Derived)",
    "Peak Hour Indicator (Derived)",
    "Public Holiday Indicator (Derived)",
    "Weather_Condition",
    "Visibility_Level",
    "humidity",
    "cloudcover",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    return df


def create_hotspot_label(df: pd.DataFrame) -> pd.DataFrame:
    location_counts = df["Location"].value_counts()
    hotspot_locations = location_counts[location_counts >= HOTSPOT_THRESHOLD].index
    df["Is_Hotspot"] = df["Location"].apply(lambda x: 1 if x in hotspot_locations else 0)
    print(f"\nHotspot distribution:\n{df['Is_Hotspot'].value_counts()}")
    return df


def prepare_features(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"\nWarning: these feature columns not found and skipped: {missing}")

    df_model = df[available + [TARGET_COLUMN]].copy()

    for col in df_model.select_dtypes(include="object").columns:
        df_model[col] = LabelEncoder().fit_transform(df_model[col].astype(str))

    X = df_model[available]
    y = df_model[TARGET_COLUMN]
    return X, y


def train_and_evaluate(X, y) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
    }

    print("\n── Accuracy Scores ───────────────────────────────")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"  {name:<25} {acc:.4f}")
    print("──────────────────────────────────────────────────")


def main():
    df = load_data()
    df = create_hotspot_label(df)
    X, y = prepare_features(df)
    train_and_evaluate(X, y)


if __name__ == "__main__":
    main()