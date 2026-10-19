import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEED = 42
BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)


def configure_tk_environment() -> None:
    tcl_dir = Path(sys.base_prefix) / "tcl"
    tcl_library = tcl_dir / "tcl8.6"
    tk_library = tcl_dir / "tk8.6"

    if tcl_library.is_dir() and tk_library.is_dir():
        os.environ.setdefault("TCL_LIBRARY", str(tcl_library))
        os.environ.setdefault("TK_LIBRARY", str(tk_library))


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_dataset_root(search_path: Path) -> Path:
    for train_dir in search_path.rglob("train"):
        if (
            (train_dir.parent / "test").is_dir()
            and (train_dir / "def_front").is_dir()
            and (train_dir / "ok_front").is_dir()
        ):
            return train_dir.parent

    raise FileNotFoundError(
        f"Could not find a dataset with train/test folders under {search_path}"
    )


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])


def create_data_loaders(
    root: Path, seed: int = SEED
) -> tuple[DataLoader, DataLoader, DataLoader, datasets.ImageFolder]:
    train_dir = root / "train"
    test_dir = root / "test"

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    eval_transform = get_eval_transform()

    base_train = datasets.ImageFolder(train_dir, transform=train_transform)
    base_val = datasets.ImageFolder(train_dir, transform=eval_transform)
    test_data = datasets.ImageFolder(test_dir, transform=eval_transform)
    print("Class mapping:", base_train.class_to_idx)

    indices = np.arange(len(base_train))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=base_train.targets,
    )

    train_data = Subset(base_train, train_idx)
    val_data = Subset(base_val, val_idx)

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader, base_train


def inspect_training_batch(train_loader: DataLoader) -> None:
    images, _ = next(iter(train_loader))
    print("Tensor shape:", images.shape)
    print("Pixel range:", images.min().item(), images.max().item())


def display_sample(
    train_loader: DataLoader, class_names: list[str]
) -> None:
    train_data = train_loader.dataset
    img, label = train_data[0]

    print("Shape:", img.shape)
    print("Pixel values:")
    print(img[0, :5, :5])

    import matplotlib.pyplot as plt

    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"Class: {class_names[label]}")
    plt.axis("off")
    plt.show()


def main() -> None:
    configure_tk_environment()
    set_seed()

    dataset_path = Path(__file__).resolve().parent
    root = find_dataset_root(dataset_path)
    train_loader, _, _, base_train = create_data_loaders(root)

    inspect_training_batch(train_loader)
    display_sample(train_loader, base_train.classes)


if __name__ == "__main__":
    main()
