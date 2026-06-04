"""
mlflow_train.py
---------------
MLflow experiment tracking wrapper around train_model.py.
Run this instead of train_model.py when you want tracked experiments.

Usage:
    python mlflow_train.py
    mlflow ui
"""

import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from preprocessing import preprocess
from train_model import get_models, evaluate

# ── Config ────────────────────────────────────

EXPERIMENT   = "predictive_maintenance"
TEST_SIZE    = 0.3
RANDOM_STATE = 42
CV_FOLDS     = 5

# ── Main ──────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     default="predictive_maintenance.csv")
    parser.add_argument("--threshold", default=0.70, type=float)
    args, _ = parser.parse_known_args()

    # 1. Load & preprocess
    print(f"\n📂 Loading: {args.input}")
    df = pd.read_csv(args.input)
    X, y, _ = preprocess(df, fit=True)
    X.columns = X.columns.str.replace(r"[\[\]/]", "_", regex=True).str.strip()
    print(f"   Shape: {X.shape} | Failures: {y.sum()} / {len(y)}")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Scale for LR
    mm = MinMaxScaler()
    X_train_mm = pd.DataFrame(mm.fit_transform(X_train), columns=X_train.columns)
    X_test_mm  = pd.DataFrame(mm.transform(X_test),      columns=X_test.columns)

    # 4. scale_pos_weight
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"   scale_pos_weight: {spw:.2f}")

    # 5. Run experiments
    mlflow.set_experiment(EXPERIMENT)
    cv     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models = get_models(spw)

    print(f"\n{'Model':25s} | F1     | AUC    | P      | R      | CV F1")
    print("-" * 75)

    for name, model in models.items():
        X_tr, X_te = (X_train_mm, X_test_mm) if name == "Logistic Regression" \
                     else (X_train, X_test)

        with mlflow.start_run(run_name=name):

            model.fit(X_tr, y_train)
            metrics = evaluate(model, X_tr, X_te, y_train, y_test,
                               name, args.threshold, cv)

            # Log parameters
            try:
                mlflow.log_params(model.get_params())
            except Exception:
                mlflow.log_param("model_type", name)

            # Log metrics
            mlflow.log_metrics({
                "f1":        metrics["f1"],
                "roc_auc":   metrics["roc_auc"],
                "precision": metrics["precision"],
                "recall":    metrics["recall"],
                "cv_f1":     metrics["cv_f1"],
                "threshold": args.threshold,
            })

            # Log model
            mlflow.sklearn.log_model(model, artifact_path=name)

            print(f"✅ {name} logged to MLflow")

    print(f"\n🎯 Run: mlflow ui → http://localhost:5000")