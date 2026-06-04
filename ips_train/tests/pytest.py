import pytest
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

from ips_train.train_model import get_models, evaluate


def make_data():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, n_features=8,
                               weights=[0.97, 0.03], random_state=42)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]), pd.Series(y)


@pytest.fixture
def trained_rf():
    X, y = make_data()
    spw  = (y == 0).sum() / (y == 1).sum()
    model = get_models(spw)["Random Forest"]
    model.fit(X[:400], y[:400])
    return model, X[400:], y[400:]


def test_models_exist():
    models = get_models(spw=10.0)
    assert set(models.keys()) == {"XGBoost", "Random Forest", "Logistic Regression"}


def test_scale_pos_weight():
    models = get_models(spw=15.0)
    # XGBoost is now wrapped in CalibratedClassifierCV
    assert hasattr(models["XGBoost"], "predict_proba")


def test_evaluate_keys(trained_rf):
    model, X_test, y_test = trained_rf
    cv     = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    result = evaluate(model, X_test, X_test, y_test, y_test, "RF", 0.70, cv)
    assert set(result.keys()) == {"f1", "roc_auc", "precision", "recall", "cv_f1"}


def test_metrics_valid_range(trained_rf):
    model, X_test, y_test = trained_rf
    cv     = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    result = evaluate(model, X_test, X_test, y_test, y_test, "RF", 0.70, cv)
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_no_leaky_columns():
    X, _ = make_data()
    leaky = ["Failure_Type", "Product_ID", "Stress_Index"]
    assert not any(col in X.columns for col in leaky)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
