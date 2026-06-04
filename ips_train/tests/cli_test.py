import pandas as pd
from click.testing import CliRunner
from unittest.mock import Mock

import ips_train.cli as cli_module
from ips_train.cli import preprocess as cli_preprocess


def test_cli_help_includes_input_and_output_options():
    runner = CliRunner()
    result = runner.invoke(cli_preprocess, ["--help"])

    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--output" in result.output


def test_cli_main_saves_preprocessor_with_mocked_dependencies(monkeypatch):
    dummy_df = pd.DataFrame({"col": [1]})
    dummy_X = pd.DataFrame({"feature": [1]})
    dummy_y = pd.Series([0])
    dummy_pipeline = Mock()

    saved = {}

    def fake_read_csv(path):
        saved["input_path"] = path
        return dummy_df

    def fake_preprocess(df, fit=True):
        assert df is dummy_df
        assert fit is True
        return dummy_X, dummy_y, dummy_pipeline

    def fake_save_preprocessor(pipeline, path):
        saved["pipeline"] = pipeline
        saved["path"] = path

    monkeypatch.setattr(cli_module.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(cli_module, "preprocess_data", fake_preprocess)
    monkeypatch.setattr(cli_module, "save_preprocessor", fake_save_preprocessor)

    runner = CliRunner()
    result = runner.invoke(cli_preprocess, ["--input", "input.csv", "--output", "output.joblib"])

    assert result.exit_code == 0
    assert saved["input_path"] == "input.csv"
    assert saved["pipeline"] is dummy_pipeline
    assert saved["path"] == "output.joblib"
    assert "Raw shape" in result.output
    assert "Processed shape" in result.output
