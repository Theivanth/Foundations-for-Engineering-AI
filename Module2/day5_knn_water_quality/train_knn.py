import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


FEATURES = [
    "pH",
    "Turbidity",
    "DissolvedOxygen",
    "Conductivity",
    "Temperature",
    "Nitrate"
]


def load_data(filename):
    csv_path = Path(filename)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parent / filename

    df = pd.read_csv(csv_path)
    return df


def split_data(df):

    X = df[FEATURES]

    y = df["Quality"]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


def train_model(X_train, y_train):

    model = Pipeline(steps=[
        ("scale", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="minkowski",
            p=2
        ))
    ])

    model.fit(X_train, y_train)

    return model