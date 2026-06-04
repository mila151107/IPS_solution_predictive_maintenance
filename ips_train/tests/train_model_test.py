import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from unittest.mock import Mock

import ips_train.train_model as tm


def make_data():
    X, y = make_classification(n_samples=500, n_features=8, weights=[0.97, 0.03], random_state=42)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]), pd.Series(y)


@pytest.fixture
def trained_rf():
    X, y = make_data()
    spw = (y == 0).sum() / (y == 1).sum()
    model = tm.get_models(spw)["Random Forest"]
    model.fit(X[:400], y[:400])
    return model, X[400:], y[400:]


class DummyModel:
    def __init__(self, proba):
        self.proba = np.asarray(proba)
        self.fit_calls = []

    def fit(self, X, y):
        self.fit_calls.append((X.copy(), y.copy()))
        return self

    def predict_proba(self, X):
        if self.proba.ndim == 1:
            return np.tile(self.proba[np.newaxis, :], (len(X), 1))
        return self.proba


class DummyScaler:
    def fit_transform(self, X):
        return X.to_numpy()

    def transform(self, X):
        return X.to_numpy()


def test_get_models_returns_expected_model_names():
    models = tm.get_models(spw=10.0)

    assert set(models.keys()) == {"XGBoost", "Random Forest", "Logistic Regression"}
    assert hasattr(models["XGBoost"], "predict_proba")
    assert hasattr(models["Random Forest"], "fit")
    assert hasattr(models["Logistic Regression"], "fit")


def test_scale_pos_weight():
    models = tm.get_models(spw=15.0)
    assert hasattr(models["XGBoost"], "predict_proba")


def test_evaluate_keys(trained_rf):
    model, X_test, y_test = trained_rf
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    result = tm.evaluate(model, X_test, X_test, y_test, y_test, "RF", 0.70, cv)
    assert set(result.keys()) == {"f1", "roc_auc", "precision", "recall", "cv_f1"}


def test_metrics_valid_range(trained_rf):
    model, X_test, y_test = trained_rf
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    result = tm.evaluate(model, X_test, X_test, y_test, y_test, "RF", 0.70, cv)
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_evaluate_returns_correct_metrics(monkeypatch):
    model = DummyModel([[0.1, 0.9], [0.9, 0.1]])
    X_tr = pd.DataFrame({"feature": [1, 2]})
    X_te = pd.DataFrame({"feature": [1, 2]})
    y_tr = pd.Series([0, 1])
    y_te = pd.Series([1, 0])

    monkeypatch.setattr(tm, "cross_val_score", lambda model, X, y, cv, scoring: np.array([0.8]))

    result = tm.evaluate(model, X_tr, X_te, y_tr, y_te, "test", threshold=0.5, cv=Mock())

    assert result["f1"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["cv_f1"] == pytest.approx(0.8)


def test_train_model_saves_models_and_scaler(monkeypatch):
    dummy_df = pd.DataFrame({"raw": [1, 2, 3, 4]})
    dummy_X = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
    dummy_y = pd.Series([0, 1, 0, 1])

    monkeypatch.setattr(tm.pd, "read_csv", lambda path: dummy_df)
    monkeypatch.setattr(tm, "preprocess", lambda df, fit=True: (dummy_X, dummy_y, None))
    monkeypatch.setattr(
        tm,
        "train_test_split",
        lambda X, y, test_size, random_state, stratify: (
            X.iloc[:2],
            X.iloc[2:],
            y.iloc[:2],
            y.iloc[2:],
        ),
    )
    monkeypatch.setattr(tm, "cross_val_score", lambda model, X, y, cv, scoring: np.array([0.9]))
    monkeypatch.setattr(tm, "MinMaxScaler", lambda: DummyScaler())
    monkeypatch.setattr(
        tm,
        "get_models",
        lambda spw: {
            "XGBoost": DummyModel([0.1, 0.9]),
            "Logistic Regression": DummyModel([0.1, 0.9]),
        },
    )
    monkeypatch.setattr(tm.os, "makedirs", lambda path, exist_ok=True: None)
    monkeypatch.setattr(
        tm,
        "PATHS",
        {
            "XGBoost": "xgb.joblib",
            "Random Forest": "rf.joblib",
            "Logistic Regression": "lr.joblib",
            "mm_scaler": "mm.joblib",
        },
    )

    dumped = []

    def fake_dump(obj, path):
        dumped.append((obj, path))

    monkeypatch.setattr(tm.joblib, "dump", fake_dump)

    tm.train_model("unused.csv", threshold=0.5)

    assert len(dumped) == 3
    assert dumped[0][1] == "xgb.joblib"
    assert dumped[1][1] == "lr.joblib"
    assert dumped[2][1] == "mm.joblib"
    assert isinstance(dumped[2][0], DummyScaler)
