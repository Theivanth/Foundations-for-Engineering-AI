from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/industrial_sensor_data.csv")
OUTPUT_DIR = Path("processed_data")

SENSOR_COLUMNS = [
    "temperature",
    "vibration",
    "pressure",
    "motor_current"
]

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20


def load_dataset(data_path=DATA_PATH, sensor_columns=SENSOR_COLUMNS):
    """Load, sort, and validate the raw sensor dataset."""
    df = pd.read_csv(
        data_path,
        parse_dates=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    required_columns = sensor_columns + ["timestamp", "anomaly"]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    if df[required_columns].isnull().any().any():
        raise ValueError("Required columns contain missing values")

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps must be monotonically increasing")

    return df


def split_dataset(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """Split rows chronologically into training, validation, and test sets."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1 or train_ratio + val_ratio > 1:
        raise ValueError(
            "val_ratio must be non-negative and ratios must sum to at most 1"
        )

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy()
    )


def scale_splits(train_df, val_df, test_df, sensor_columns=SENSOR_COLUMNS):
    """Fit scaling on normal training rows and transform all splits."""
    train_normal = train_df[
        train_df["anomaly"] == 0
    ].copy()

    if train_normal.empty:
        raise ValueError("Training split contains no normal samples")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        train_normal[sensor_columns]
    )
    X_val = scaler.transform(
        val_df[sensor_columns]
    )
    X_test = scaler.transform(
        test_df[sensor_columns]
    )

    y_train = train_normal["anomaly"].to_numpy()
    y_val = val_df["anomaly"].to_numpy()
    y_test = test_df["anomaly"].to_numpy()

    return scaler, X_train, X_val, X_test, y_train, y_val, y_test


def make_tensors(X, y):
    """Convert feature arrays and labels into PyTorch tensors."""
    return {
        "X": torch.tensor(
            np.asarray(X),
            dtype=torch.float32
        ),
        "y": torch.tensor(
            y,
            dtype=torch.long
        )
    }


def save_processed_data(
    train_tensors,
    val_tensors,
    test_tensors,
    scaler,
    sensor_columns=SENSOR_COLUMNS,
    output_dir=OUTPUT_DIR
):
    """Save the processed splits, fitted scaler, and feature order."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(train_tensors, output_dir / "train.pt")
    torch.save(val_tensors, output_dir / "val.pt")
    torch.save(test_tensors, output_dir / "test.pt")
    joblib.dump(scaler, output_dir / "scaler.pkl")
    joblib.dump(sensor_columns, output_dir / "sensor_columns.pkl")


def print_summary(
    df,
    train_df,
    val_df,
    test_df,
    train_tensors,
    val_tensors,
    test_tensors,
    y_train,
    y_val,
    y_test,
    scaler,
    sensor_columns=SENSOR_COLUMNS,
    output_dir=OUTPUT_DIR
):
    """Print dataset, tensor, label, and scaling summaries."""
    print("Total samples:", len(df))
    print("\nChronological splits:")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))
    print("\nNormal training samples:", len(train_tensors["X"]))

    print("\nTensor shapes:")
    print("Train:", train_tensors["X"].shape)
    print("Validation:", val_tensors["X"].shape)
    print("Test:", test_tensors["X"].shape)

    print("\nAnomaly counts:")
    print("Train:", int(sum(y_train)))
    print("Validation:", int(sum(y_val)))
    print("Test:", int(sum(y_test)))

    print("\nScaling statistics:")
    print(
        pd.DataFrame({
            "sensor": sensor_columns,
            "mean": scaler.mean_,
            "std": scaler.scale_
        }).to_string(index=False)
    )

    print("\nData preprocessing completed!")
    print("Files saved to:", Path(output_dir).resolve())


def prepare_data(
    data_path=DATA_PATH,
    output_dir=OUTPUT_DIR,
    sensor_columns=SENSOR_COLUMNS,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO
):
    """Prepare, save, and return the sensor dataset splits and scaler."""
    df = load_dataset(data_path, sensor_columns)
    train_df, val_df, test_df = split_dataset(
        df,
        train_ratio,
        val_ratio
    )
    (
        scaler,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = scale_splits(
        train_df,
        val_df,
        test_df,
        sensor_columns
    )

    train_tensors = make_tensors(X_train, y_train)
    val_tensors = make_tensors(X_val, y_val)
    test_tensors = make_tensors(X_test, y_test)

    save_processed_data(
        train_tensors,
        val_tensors,
        test_tensors,
        scaler,
        sensor_columns,
        output_dir
    )
    print_summary(
        df,
        train_df,
        val_df,
        test_df,
        train_tensors,
        val_tensors,
        test_tensors,
        y_train,
        y_val,
        y_test,
        scaler,
        sensor_columns,
        output_dir
    )

    return {
        "train": train_tensors,
        "val": val_tensors,
        "test": test_tensors,
        "scaler": scaler,
        "sensor_columns": sensor_columns
    }


def main():
    """Run preprocessing with the default input and output paths."""
    prepare_data()


if __name__ == "__main__":
    main()
