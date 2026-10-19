"""Callable data splitting helpers for model training."""

from pathlib import Path

from model import load_sensor_data, prepare_data


def split_sensor_data(data, test_size: float = 0.3, validation_fraction: float = 0.5,
                      random_state: int = 42) -> dict:
    """Split a sensor DataFrame into encoded train, validation, and test sets."""
    return prepare_data(data, test_size, validation_fraction, random_state)


def load_and_split_data(csv_path: str | Path = "sensor_dataset.csv", clean: bool = False,
                        test_size: float = 0.3, validation_fraction: float = 0.5,
                        random_state: int = 42) -> dict:
    """Load a CSV and return its train, validation, and test splits."""
    data = load_sensor_data(csv_path, clean=clean)
    return split_sensor_data(data, test_size, validation_fraction, random_state)


def main() -> None:
    splits = load_and_split_data()
    for name in ("train", "val", "test"):
        print(f"{name} shapes:", splits[f"x_{name}"].shape,
              splits[f"y_{name}"].shape)


if __name__ == "__main__":
    main()
