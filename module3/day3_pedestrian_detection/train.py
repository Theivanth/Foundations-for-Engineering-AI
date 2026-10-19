from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from transformer import (
    BATCH_SIZE,
    CHECKPOINT,
    DATA_ROOT,
    EPOCHS,
    LEARNING_RATE,
    MODEL_ARCHITECTURE,
    SEED,
    PennFudanDetectionDataset,
    build_detector,
    collate_fn,
    find_dataset_root,
    get_device,
    seed_everything,
)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    # Torchvision detection models return losses only in training mode.
    model.train()
    total_loss = 0.0
    image_count = 0

    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

        total_loss += loss.item() * len(images)
        image_count += len(images)

    if not training:
        model.eval()

    if image_count == 0:
        raise ValueError("Cannot calculate loss for an empty data loader.")

    return total_loss / image_count


def train_model(epochs: int = EPOCHS) -> None:
    seed_everything(SEED)
    device = get_device()
    dataset_root = find_dataset_root(Path(DATA_ROOT))

    dataset = PennFudanDetectionDataset(
        dataset_root,
        transform=transforms.ToTensor(),
    )
    indices = np.arange(len(dataset))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
    )

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = build_detector().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=5e-4,
    )

    print(f"Device: {device}")
    print(f"Dataset: {dataset_root}")
    print(
        f"Training images: {len(train_indices)} | "
        f"Validation images: {len(val_indices)}"
    )

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dataset_root": str(dataset_root),
                    "class_to_idx": {"background": 0, "pedestrian": 1},
                    "num_classes": 2,
                    "architecture": MODEL_ARCHITECTURE,
                    "seed": SEED,
                },
                CHECKPOINT,
            )

    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {CHECKPOINT}")


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()