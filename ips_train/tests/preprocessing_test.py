import os

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ips_train.data import (
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
    ROTATIONAL_SPEED_COL,
    STRESS_INDEX_COL,
    TARGET_COL,
    TOOL_WEAR_COL,
    TORQUE_COL,
    TORQUE_RPM_RATIO_COL,
    TYPE_COL,
    UDI_COL,
)
from ips_train.preprocessing import (
    ColumnDropper,
    CategoricalEncoder,
    NumericalScaler,
    build_pipeline,
    compute_features,
    load_preprocessor,
    preprocess,
    save_preprocessor,
)


def make_raw_row():
    return {
        UDI_COL: 1,
        PRODUCT_ID_COL: "P1",
        TYPE_COL: "L",
        FAILURE_TYPE_COL: "No Failure",
        ROTATIONAL_SPEED_COL: 1000.0,
        TORQUE_COL: 50.0,
        TOOL_WEAR_COL: 5.0,
        PROCESS_TEMPERATURE_COL: 800.0,
        AIR_TEMPERATURE_COL: 300.0,
        TARGET_COL: 0,
    }


def test_compute_features_creates_expected_columns():
    df = pd.DataFrame([make_raw_row()])
    result = compute_features(df)

    assert ANGULAR_SPEED_COL in result.columns
    assert POWER_COL in result.columns
    assert DELTA_TEMPERATURE_COL in result.columns
    assert TORQUE_RPM_RATIO_COL in result.columns
    assert STRESS_INDEX_COL in result.columns

    expected_angular_speed = 1000.0 * (2 * np.pi / 60)
    expected_power = (50.0 * expected_angular_speed) / 1000
    expected_delta_temp = 800.0 - 300.0
    expected_stress = expected_power * expected_delta_temp

    assert result.loc[0, ANGULAR_SPEED_COL] == pytest.approx(expected_angular_speed)
    assert result.loc[0, POWER_COL] == pytest.approx(expected_power)
    assert result.loc[0, DELTA_TEMPERATURE_COL] == pytest.approx(expected_delta_temp)
    assert result.loc[0, TORQUE_RPM_RATIO_COL] == pytest.approx(50.0 / 1000.0)
    assert result.loc[0, STRESS_INDEX_COL] == pytest.approx(expected_stress)


def test_column_dropper_drops_specified_columns():
    df = pd.DataFrame(
        [
            {
                UDI_COL: 1,
                PROCESS_TEMPERATURE_COL: 800.0,
                AIR_TEMPERATURE_COL: 300.0,
                TORQUE_RPM_RATIO_COL: 0.05,
                "Keep me": 42,
            }
        ]
    )
    transformer = ColumnDropper(columns=COLUMNS_TO_DROP)
    transformed = transformer.transform(df)

    assert set(transformed.columns) == {"Keep me"}
    assert transformed.loc[0, "Keep me"] == 42


def test_categorical_encoder_transforms_categorical_columns():
    df = pd.DataFrame(
        [
            {PRODUCT_ID_COL: "P1", TYPE_COL: "L", FAILURE_TYPE_COL: "No Failure"},
            {PRODUCT_ID_COL: "P2", TYPE_COL: "M", FAILURE_TYPE_COL: "Tool Wear"},
        ]
    )
    encoder = CategoricalEncoder(columns=CATEGORICAL_COLS)
    encoder.fit(df)
    transformed = encoder.transform(df)

    for col in CATEGORICAL_COLS:
        assert transformed[col].dtype == np.int32 or transformed[col].dtype == np.int64
        assert set(transformed[col]) == {0, 1}

    assert transformed.loc[0, PRODUCT_ID_COL] != transformed.loc[1, PRODUCT_ID_COL]


def test_numerical_scaler_standardizes_selected_columns():
    df = pd.DataFrame(
        {
            ROTATIONAL_SPEED_COL: [1000.0, 2000.0, 3000.0],
            TORQUE_COL: [10.0, 20.0, 30.0],
            TOOL_WEAR_COL: [1.0, 2.0, 3.0],
        }
    )
    scaler = NumericalScaler(columns=NUMERICAL_COLS)
    scaler.fit(df)
    transformed = scaler.transform(df)

    assert list(transformed.columns) == [
        ROTATIONAL_SPEED_COL,
        TORQUE_COL,
        TOOL_WEAR_COL,
    ]
    assert transformed[ROTATIONAL_SPEED_COL].mean() == pytest.approx(0.0, abs=1e-7)
    assert transformed[TORQUE_COL].std(ddof=0) == pytest.approx(1.0, abs=1e-7)
    assert transformed[TOOL_WEAR_COL].std(ddof=0) == pytest.approx(1.0, abs=1e-7)


def test_build_pipeline_returns_pipeline_with_expected_steps():
    pipeline = build_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == [
        "drop_columns",
        "encode_cats",
        "scale_numerics",
    ]


def test_preprocess_fit_returns_processed_features_and_target():
    df = pd.DataFrame([make_raw_row()])
    X_out, y, pipeline = preprocess(df, fit=True)

    assert TARGET_COL not in X_out.columns
    assert set(COLUMNS_TO_DROP).isdisjoint(X_out.columns)
    assert y.iloc[0] == 0
    assert isinstance(pipeline, Pipeline)
    assert X_out.shape[0] == 1


def test_save_and_load_preprocessor_roundtrip(tmp_path):
    pipeline = build_pipeline()
    path = tmp_path / "preprocessor.joblib"

    save_preprocessor(pipeline, str(path))
    loaded = load_preprocessor(str(path))

    assert isinstance(loaded, Pipeline)
    assert [name for name, _ in loaded.steps] == [name for name, _ in pipeline.steps]


def test_preprocess_inference_uses_saved_pipeline(tmp_path, monkeypatch):
    df = pd.DataFrame([make_raw_row()])
    _, _, pipeline = preprocess(df, fit=True)

    path = tmp_path / "preprocessor.joblib"
    save_preprocessor(pipeline, str(path))

    original_load_preprocessor = load_preprocessor

    def fake_load_preprocessor(load_path=str(path)):
        return original_load_preprocessor(load_path)

    monkeypatch.setattr("ips_train.preprocessing.load_preprocessor", fake_load_preprocessor)

    X_out, y, loaded_pipeline = preprocess(df, fit=False)

    assert isinstance(loaded_pipeline, Pipeline)
    assert X_out.shape[0] == 1
    assert y.iloc[0] == 0
    assert [name for name, _ in loaded_pipeline.steps] == [name for name, _ in pipeline.steps]
