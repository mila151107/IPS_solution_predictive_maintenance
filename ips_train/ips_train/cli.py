import click
import pandas as pd

from .data import INPUT_DATASET, PREPROCESSOR_PATH
from .preprocessing import preprocess as preprocess_data, save_preprocessor
from .train_model import train_model


@click.command()
@click.option(
    "--input",
    "input_path",
    type=str,
    default=INPUT_DATASET,
    help="Path to raw CSV",
)
@click.option(
    "--output",
    "output_path",
    type=str,
    default=PREPROCESSOR_PATH,
    help="Output path",
)
def preprocess(input_path: str, output_path: str) -> None:
    print(f"📂 Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Raw shape: {df.shape}")

    X, y, pipeline = preprocess_data(df, fit=True)

    print(f"\n✅ Processed shape: {X.shape}")
    print(f"   Features: {X.columns.tolist()}")
    print(f"\n   Target distribution:\n{y.value_counts().to_string()}")

    save_preprocessor(pipeline, output_path)


@click.command()
@click.option(
    "--input",
    "input_path",
    type=str,
    default=INPUT_DATASET,
    help="Path to raw CSV",
)
@click.option(
    "--threshold",
    "threshold",
    type=float,
    default=0.70,
    help="Decision threshold for binary classification",
)
def train(input_path: str, threshold: float) -> None:
    train_model(input_path, threshold)
