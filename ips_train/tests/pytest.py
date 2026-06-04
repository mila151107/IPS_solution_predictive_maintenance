import pytest
import pandas as pd
from sklearn.datasets import make_classification


def make_data():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=500, n_features=8, weights=[0.97, 0.03], random_state=42)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]), pd.Series(y)


def test_no_leaky_columns():
    X, _ = make_data()
    leaky = ["Failure_Type", "Product_ID", "Stress_Index"]
    assert not any(col in X.columns for col in leaky)
