# ips_train

`ips_train` is the training package for the IPS predictive maintenance workflow. It includes data preprocessing and model training utilities built around a small command-line interface using `click`.

## Features

- Preprocess the raw predictive maintenance dataset
- Train classification models with configurable thresholding
- Save preprocessing pipeline and trained model artifacts

## Requirements

- Python 3.12+
- `pandas`
- `scikit-learn`
- `xgboost`
- `joblib`
- `click`

This package uses `xgboost` for the XGBoost classifier in the training workflow.

Install the package dependencies in the `ips_train` folder, for example with Poetry:

```bash
cd ips_train
poetry install
```

## Testing and formatting

From the `ips_train` package root, use Poetry to run tests and format code:

```bash
pytest
poetry run black .
```

## Package layout

- `ips_train/ips_train/cli.py` - CLI command definitions
- `ips_train/ips_train/preprocessing.py` - preprocessing pipeline
- `ips_train/ips_train/train_model.py` - model training workflow
- `ips_train/ips_train/data.py` - shared paths and dataset constants
- `ips_train/artifacts/` - output folder for saved artifacts

## CLI usage

This package installs two CLI scripts when configured via `pyproject.toml`:

- `ips_train_preprocess`
- `ips_train_train`

### Run preprocessing

The preprocessing command reads a CSV dataset, builds the feature pipeline, and saves the resulting preprocessor to disk.

Default behavior:

- Input dataset: `predictive_maintenance.csv`
- Output preprocessor: `artifacts/preprocessor.joblib`

Example:

```bash
ips_train_preprocess \
  --input predictive_maintenance.csv \
  --output artifacts/preprocessor.joblib
```

Options:

- `--input <path>`
  - Path to the raw CSV dataset
  - Default: `predictive_maintenance.csv`
- `--output <path>`
  - Path where the fitted preprocessor is saved
  - Default: `artifacts/preprocessor.joblib`

### Run training

The training command loads the dataset, runs preprocessing, fits the models, evaluates them, and saves trained artifacts.

Default behavior:

- Input dataset: `predictive_maintenance.csv`
- Decision threshold: `0.70`

Example:

```bash
ips_train_train \
  --input predictive_maintenance.csv \
  --threshold 0.70
```

Options:

- `--input <path>`
  - Path to the raw CSV dataset
  - Default: `predictive_maintenance.csv`
- `--threshold <float>`
  - Decision threshold used for binary classification predictions
  - Default: `0.70`

## Artifact outputs

Training saves the following artifacts under `artifacts/`:

- `xgb_model.joblib`
- `rf_model.joblib`
- `lr_model.joblib`
- `mm_scaler.joblib`
- `preprocessor.joblib`

## Notes

- Run commands from the `ips_train` package root so relative paths resolve correctly.
- If you add a console script entry point later, you can replace the `python -c` invocation with the installed command.
