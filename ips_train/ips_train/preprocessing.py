"""
Feature engineering + preprocessing pipeline for the AI4I 2020 Predictive Maintenance Dataset.
Run directly to fit, serialize, and save the pipeline artifact.

Usage:
    python preprocessing.py                         # fits on predictive_maintenance.csv
    python preprocessing.py --input my_data.csv     # custom input file
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .data import (
    AIR_TEMPERATURE_COL,
    ANGULAR_SPEED_COL,
    CATEGORICAL_COLS,
    COLUMNS_TO_DROP,
    DELTA_TEMPERATURE_COL,
    FAILURE_TYPE_COL,
    NUMERICAL_COLS,
    POWER_COL,
    PRODUCT_ID_COL,
    PROCESS_TEMPERATURE_COL,
    PREPROCESSOR_PATH,
    ROTATIONAL_SPEED_COL,
    STRESS_INDEX_COL,
    TARGET_COL,
    TOOL_WEAR_COL,
    TORQUE_COL,
    TORQUE_RPM_RATIO_COL,
    TYPE_COL,
    UDI_COL,
)

# ──────────────────────────────────────────────
# Step 1 — Feature Engineering
# ──────────────────────────────────────────────


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds engineered features derived from raw sensor columns."""
    df = df.copy()
    df[ANGULAR_SPEED_COL] = df[ROTATIONAL_SPEED_COL] * (2 * np.pi / 60)
    df[POWER_COL] = (df[TORQUE_COL] * df[ANGULAR_SPEED_COL]) / 1000
    df[DELTA_TEMPERATURE_COL] = df[PROCESS_TEMPERATURE_COL] - df[AIR_TEMPERATURE_COL]
    df[TORQUE_RPM_RATIO_COL] = df[TORQUE_COL] / df[ROTATIONAL_SPEED_COL]
    df[STRESS_INDEX_COL] = df[POWER_COL] * df[DELTA_TEMPERATURE_COL]
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
    return Pipeline(
        steps=[
            ("drop_columns", ColumnDropper(columns=COLUMNS_TO_DROP)),
            ("encode_cats", CategoricalEncoder(columns=CATEGORICAL_COLS)),
            ("scale_numerics", NumericalScaler(columns=NUMERICAL_COLS)),
        ]
    )


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


def save_preprocessor(pipeline: Pipeline, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"✅ Preprocessor saved → {path}")


def load_preprocessor(path: str = PREPROCESSOR_PATH) -> Pipeline:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No preprocessor found at: {path}. Run preprocessing.py first.")
    return joblib.load(path)
