"""
train.py
--------
This script trains two classification models — LightGBM and Logistic Regression —
on the preprocessed predictive maintenance data. It evaluates both models using
F1 score, ROC-AUC, and cross-validation, then serializes the trained models
to the folder for deployment.

Usage:
    python train.py
    python train.py --input my_data.csv
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

from .preprocessing import preprocess

# ──────────────────────────────────────────────
# Config
# Paths, hyperparameters, and split settings
# in one place — easy to change without touching the logic below.
# ──────────────────────────────────────────────

ARTIFACT_DIR    = "artifacts"
LGBM_PATH       = os.path.join(ARTIFACT_DIR, "lgbm_model.joblib")
LR_PATH         = os.path.join(ARTIFACT_DIR, "lr_model.joblib")

TEST_SIZE       = 0.3     # 30% test, 70% train
RANDOM_STATE    = 42
CV_FOLDS        = 5       # number of cross-validation folds


# ──────────────────────────────────────────────
# Model definitions
# class_weight="balanced" adjusts for the imbalanced dataset
# (~3% failures) so the model doesn't just predict "no failure" always.
# ──────────────────────────────────────────────

def get_models():
    lgbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        importance_type="gain",
        random_state=RANDOM_STATE,
    )

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return {"LightGBM": lgbm, "Logistic Regression": lr}


# ──────────────────────────────────────────────
# Evaluation
# Reports F1, ROC-AUC, and cross-validation score
# for each model on the test set.
# ──────────────────────────────────────────────

def evaluate(model, X_train, X_test, y_train, y_test, name):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1      = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Cross-validation on training data
    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="f1")
    print(f"CV F1   : {cv_scores.round(3)} | Mean: {cv_scores.mean():.3f}")

    return f1, roc_auc


# ──────────────────────────────────────────────
# Serialization
# Saves each trained model to artifacts/ folder.
# ──────────────────────────────────────────────

def save_models(models):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    paths = {"LightGBM": LGBM_PATH, "Logistic Regression": LR_PATH}
    for name, model in models.items():
        joblib.dump(model, paths[name])
        print(f"✅ {name} saved → {paths[name]}")


def load_model(name: str):
    """Load a saved model by name: 'LightGBM' or 'Logistic Regression'."""
    paths = {"LightGBM": LGBM_PATH, "Logistic Regression": LR_PATH}
    path  = paths[name]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run train.py first.")
    return joblib.load(path)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="predictive_maintenance.csv")
    args = parser.parse_args()

    # 1. Load & preprocess
    print(f" Loading: {args.input}")
    df = pd.read_csv(args.input)
    X, y, pipeline = preprocess(df, fit=True)

    # Clean column names (LightGBM dislikes special characters)
    X.columns = X.columns.str.replace(r"[\[\]/]", "_", regex=True).str.strip()

    print(f" Preprocessed shape: {X.shape}")

    # 2. Train/test split — stratified to preserve failure ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train failures: {y_train.sum()} | Test failures: {y_test.sum()}")

    # 3. Train & evaluate both models
    models   = get_models()
    results  = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        f1, roc_auc = evaluate(model, X_train, X_test, y_train, y_test, name)
        results[name] = {"f1": f1, "roc_auc": roc_auc}

    # 4. Summary
    print(f"\n{'='*40}")
    print("  Summary")
    print(f"{'='*40}")
    for name, scores in results.items():
        print(f"{name:25s} | F1: {scores['f1']:.4f} | ROC-AUC: {scores['roc_auc']:.4f}")

    best = max(results, key=lambda k: results[k]["f1"])
    print(f"\n Best model by F1: {best}")

    # 5. Save both models
    save_models(models)
