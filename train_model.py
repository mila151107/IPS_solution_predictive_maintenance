"""
train_model.py
--------------
XGBoost, Random Forest, Logistic Regression
on AI4I 2020 Predictive Maintenance Dataset.

Usage:
    python train_model.py
    python train_model.py --input my_data.csv --threshold 0.25
"""

import argparse
import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler

from preprocessing import preprocess

# ── Config ────────────────────────────────────

ARTIFACT_DIR = "artifacts"
PATHS = {
    "XGBoost":             os.path.join(ARTIFACT_DIR, "xgb_model.joblib"),
    "Random Forest":       os.path.join(ARTIFACT_DIR, "rf_model.joblib"),
    "Logistic Regression": os.path.join(ARTIFACT_DIR, "lr_model.joblib"),
    "mm_scaler":           os.path.join(ARTIFACT_DIR, "mm_scaler.joblib"),
}
TEST_SIZE    = 0.3
RANDOM_STATE = 42
CV_FOLDS     = 5


# ── Models ────────────────────────────────────
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()*0.2

def get_models(spw: float) -> dict:
    return {
        "XGBoost": XGBClassifier(
            n_estimators=100, learning_rate=0.01, max_depth=10,
            scale_pos_weight=spw, eval_metric="aucpr", random_state=42,
        ),
   "Random Forest": RandomForestClassifier(
    n_estimators=300,         
    max_depth=6,              
    min_samples_leaf=20,      
    min_samples_split=10,     
    max_features="sqrt",
    max_samples=0.8,           
    class_weight="balanced_subsample",
    random_state=42
),
        "Logistic Regression": LogisticRegression(
            max_iter=10000, C=0.1, solver="saga", penalty="l2",
            class_weight="balanced", random_state=42,
        ),
    }


# ── Evaluate ──────────────────────────────────

def evaluate(model, X_tr, X_te, y_tr, y_te, name, threshold, cv):
    y_proba   = model.predict_proba(X_te)[:, 1]
    y_pred    = (y_proba >= threshold).astype(int)
    cv_mean   = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="f1").mean()

    print(f"{name:25s} | F1: {f1_score(y_te, y_pred):.4f} | "
          f"AUC: {roc_auc_score(y_te, y_proba):.4f} | "
          f"P: {precision_score(y_te, y_pred):.4f} | "
          f"R: {recall_score(y_te, y_pred):.4f} | "
          f"CV F1: {cv_mean:.4f}")

    return {
        "f1":        f1_score(y_te, y_pred),
        "roc_auc":   roc_auc_score(y_te, y_proba),
        "precision": precision_score(y_te, y_pred),
        "recall":    recall_score(y_te, y_pred),
        "cv_f1":     cv_mean,
    }


# ── Main ──────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     default="predictive_maintenance.csv")
    parser.add_argument("--threshold", default=0.25, type=float)
    args, _ = parser.parse_known_args()

    # 1. Load & preprocess
    print(f"\n Loading: {args.input}")
    df = pd.read_csv(args.input)
    X, y, _ = preprocess(df, fit=True)
    X.columns = X.columns.str.replace(r"[\[\]/]", "_", regex=True).str.strip()
    print(f"   Shape: {X.shape} | Failures: {y.sum()} / {len(y)}")

    # 2. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 3. scale_pos_weight for XGBoost
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"   scale_pos_weight: {spw:.2f}")

    # 4. MinMax scale for Logistic Regression only
    mm = MinMaxScaler()
    X_train_mm = pd.DataFrame(mm.fit_transform(X_train), columns=X_train.columns)
    X_test_mm  = pd.DataFrame(mm.transform(X_test),      columns=X_test.columns)

    # 5. Train & evaluate
    cv      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models  = get_models(spw)
    results = {}

    print(f"\n{'Model':25s} | F1     | AUC    | P      | R      | CV F1")
    print("-" * 75)

    for name, model in models.items():
        X_tr, X_te = (X_train_mm, X_test_mm) if name == "Logistic Regression" \
                     else (X_train, X_test)
        model.fit(X_tr, y_train)
        results[name] = evaluate(model, X_tr, X_te, y_train, y_test,
                                 name, args.threshold, cv)

    # 6. Summary
    best = "Random Forest"
    print(f"\n🏆 Selected model: {best} (fixed choice)")

    # 7. Save
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, PATHS[name])
        print(f"✅ {name} → {PATHS[name]}")
    joblib.dump(mm, PATHS["mm_scaler"])
    print(f"✅ mm_scaler → {PATHS['mm_scaler']}")