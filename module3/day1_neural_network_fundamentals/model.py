"""Model training helpers for the sensor status dataset."""

from pathlib import Path

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from inspect_data import clean_sensor_data

FEATURE_COLUMNS = ["temperature", "vibration", "pressure", "rpm", "current", "noise"]


def load_sensor_data(csv_path: str | Path = "sensor_dataset.csv", clean: bool = True) -> pd.DataFrame:
    """Read sensor data from CSV and optionally clean it."""
    data = pd.read_csv(csv_path)
    return clean_sensor_data(data) if clean else data


def prepare_data(data: pd.DataFrame, test_size: float = 0.3, validation_fraction: float = 0.5,
                 random_state: int = 42) -> dict:
    """Encode labels and split data into train, validation, and test sets."""
    features = data[FEATURE_COLUMNS]
    encoder = LabelEncoder()
    labels = encoder.fit_transform(data["status"])
    x_train, x_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=validation_fraction, random_state=random_state
    )
    return {
        "x_train": x_train, "x_val": x_val, "x_test": x_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "label_encoder": encoder,
    }


def train_baseline(x_train, y_train, max_iter: int = 1000) -> LogisticRegression:
    """Fit and return a balanced logistic regression baseline."""
    model = LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=42)
    return model.fit(x_train, y_train)


def evaluate_model(model, x, y, target_names=None) -> str:
    """Return a classification report for the supplied data."""
    predictions = model.predict(x)
    return classification_report(y, predictions, target_names=target_names)


def to_tensors(splits: dict) -> dict:
    """Convert split DataFrames/labels into PyTorch tensors."""
    tensors = {}
    for name in ("train", "val", "test"):
        tensors[f"x_{name}"] = torch.tensor(
            splits[f"x_{name}"].to_numpy(), dtype=torch.float32
        )
        tensors[f"y_{name}"] = torch.tensor(
            splits[f"y_{name}"], dtype=torch.long
        )
    return tensors


def run_pipeline(csv_path: str | Path = "sensor_dataset.csv") -> dict:
    """Load, split, train, evaluate, and tensorize the sensor dataset."""
    data = load_sensor_data(csv_path)
    splits = prepare_data(data)
    baseline = train_baseline(splits["x_train"], splits["y_train"])
    report = evaluate_model(
        baseline, splits["x_val"], splits["y_val"],
        target_names=splits["label_encoder"].classes_,
    )
    return {"data": data, "splits": splits, "model": baseline,
            "report": report, "tensors": to_tensors(splits)}


def main() -> None:
    results = run_pipeline()
    for name in ("train", "val", "test"):
        print(f"{name} shapes:", results["tensors"][f"x_{name}"].shape,
              results["tensors"][f"y_{name}"].shape)
    print(results["report"])


if __name__ == "__main__":
    main()
