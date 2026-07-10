import pandas as pd
import json

from pathlib import Path
from joblib import dump

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


FEATURES = [
    "Temperature",
    "Vibration",
    "Pressure",
    "RPM",
    "Current",
    "OperatingHours"
]


def load_data(filename):
    # Resolve path relative to this module so callers can run from any CWD
    file_path = Path(filename)
    if not file_path.is_absolute():
        file_path = Path(__file__).resolve().parent / filename

    df = pd.read_csv(file_path)
    return df


def validate_dataset(df):

    required_columns = set(FEATURES + ["Status"])

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.isna().any().any():
        raise ValueError("Dataset contains missing values")


def split_data(df):

    X = df[FEATURES]
    y = df["Status"]

    return train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )


def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )

    model.fit(X_train, y_train)

    return model


def save_model(model):

    Path("models").mkdir(exist_ok=True)

    dump(model, "models/random_forest.joblib")

    metadata = {
        "model_type": "RandomForestClassifier",
        "features": FEATURES,
        "classes": list(model.classes_),
        "random_state": 42
    }

    with open("models/metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)