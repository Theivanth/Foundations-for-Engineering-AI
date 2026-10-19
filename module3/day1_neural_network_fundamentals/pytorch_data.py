"""Callable helpers for preparing sensor data for PyTorch."""

from pathlib import Path

from torch.utils.data import DataLoader, TensorDataset

from model import load_sensor_data, prepare_data, to_tensors


def make_datasets(csv_path: str | Path = "sensor_dataset.csv") -> dict:
    """Return train, validation, and test TensorDataset objects."""
    splits = prepare_data(load_sensor_data(csv_path))
    tensors = to_tensors(splits)
    return {
        name: TensorDataset(tensors[f"x_{name}"], tensors[f"y_{name}"])
        for name in ("train", "val", "test")
    }


def make_dataloaders(csv_path: str | Path = "sensor_dataset.csv", batch_size: int = 32,
                     shuffle_train: bool = True) -> dict:
    """Return DataLoaders for each split; only training shuffles by default."""
    datasets = make_datasets(csv_path)
    return {
        name: DataLoader(dataset, batch_size=batch_size,
                         shuffle=shuffle_train if name == "train" else False)
        for name, dataset in datasets.items()
    }


def main() -> None:
    loaders = make_dataloaders()
    for name, loader in loaders.items():
        features, labels = next(iter(loader))
        print(f"{name}: {len(loader.dataset)} rows; batch shapes {features.shape}, {labels.shape}")


if __name__ == "__main__":
    main()
