"""
preprocessing.py
----------------
Feature engineering + preprocessing pipeline for the AI4I 2020 Predictive Maintenance Dataset.
Run directly to fit, serialize, and save the pipeline artifact.

Usage:
    python preprocessing.py                         # fits on predictive_maintenance.csv
    python preprocessing.py --input my_data.csv     # custom input file
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

COLUMNS_TO_DROP  = ["UDI", "Process temperature [K]", "Air temperature [K]", "Torque/RPM ratio"]
CATEGORICAL_COLS = ["Product ID", "Type", "Failure Type"]
NUMERICAL_COLS   = [
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    "Angular speed [rad/s]", "Power [kW]", "Delta Temperature [K]", "Stress Index",
]
TARGET_COL       = "Target"
ARTIFACT_DIR     = "artifacts"
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.joblib")


# ──────────────────────────────────────────────
# Step 1 — Feature Engineering
# ──────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds engineered features derived from raw sensor columns."""
    df = df.copy()
    df["Angular speed [rad/s]"] = df["Rotational speed [rpm]"] * (2 * np.pi / 60)
    df["Power [kW]"]            = (df["Torque [Nm]"] * df["Angular speed [rad/s]"]) / 1000
    df["Delta Temperature [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Torque/RPM ratio"]      = df["Torque [Nm]"] / df["Rotational speed [rpm]"]
    df["Stress Index"]          = df["Power [kW]"] * df["Delta Temperature [K]"]
    return df


# ──────────────────────────────────────────────
# Step 2 — Custom Transformers
# ──────────────────────────────────────────────

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.columns if c in X.columns], errors="ignore")


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.encoders_ = {}

    def fit(self, X, y=None):
        for col in self.columns:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.encoders_[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col, le in self.encoders_.items():
            if col in X.columns:
                X[col] = le.transform(X[col].astype(str))
        return X


class NumericalScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.scaler_ = StandardScaler()

    def fit(self, X, y=None):
        cols = [c for c in self.columns if c in X.columns]
        self.scaler_.fit(X[cols])
        self.fitted_columns_ = cols
        return self

    def transform(self, X):
        X = X.copy()
        X[self.fitted_columns_] = self.scaler_.transform(X[self.fitted_columns_])
        return X


# ──────────────────────────────────────────────
# Step 3 — Build Pipeline
# ──────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ("drop_columns",   ColumnDropper(columns=COLUMNS_TO_DROP)),
        ("encode_cats",    CategoricalEncoder(columns=CATEGORICAL_COLS)),
        ("scale_numerics", NumericalScaler(columns=NUMERICAL_COLS)),
    ])


# ──────────────────────────────────────────────
# Public API — used by training & Streamlit app
# ──────────────────────────────────────────────

def preprocess(df: pd.DataFrame, fit: bool = True):
    """
    Full preprocessing: feature engineering → drop → encode → scale.

    Parameters
    ----------
    df  : raw DataFrame (straight from CSV)
    fit : True for training (fits pipeline), False for inference (loads saved pipeline)

    Returns
    -------
    X_out    : pd.DataFrame  — processed features
    y        : pd.Series     — target column
    pipeline : fitted Pipeline
    """
    df = compute_features(df)

    y = df[TARGET_COL].copy()
    X = df.drop(columns=[TARGET_COL])

    if fit:
        pipeline = build_pipeline()
        X_transformed = pipeline.fit_transform(X)
    else:
        pipeline = load_preprocessor()
        X_transformed = pipeline.transform(X)

    output_cols = [c for c in X.columns if c not in COLUMNS_TO_DROP]
    X_out = pd.DataFrame(X_transformed, columns=output_cols)

    return X_out, y, pipeline


# ──────────────────────────────────────────────
# Serialization helpers
# ──────────────────────────────────────────────

def save_preprocessor(pipeline: Pipeline, path: str = PREPROCESSOR_PATH) -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"✅ Preprocessor saved → {path}")


def load_preprocessor(path: str = PREPROCESSOR_PATH) -> Pipeline:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No preprocessor found at: {path}. Run preprocessing.py first.")
    return joblib.load(path)


# ──────────────────────────────────────────────
# Entry point — run to fit & serialize
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="predictive_maintenance.csv", help="Path to raw CSV")
    args = parser.parse_args()

    print(f"📂 Loading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"   Raw shape: {df.shape}")

    X, y, pipeline = preprocess(df, fit=True)

    print(f"\n✅ Processed shape: {X.shape}")
    print(f"   Features: {X.columns.tolist()}")
    print(f"\n   Target distribution:\n{y.value_counts().to_string()}")

    save_preprocessor(pipeline)
