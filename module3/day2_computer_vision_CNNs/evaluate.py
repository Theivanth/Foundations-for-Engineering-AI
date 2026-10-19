from pathlib import Path

import torch
import torch.nn as nn

from train import DefectCNN
from transformer import create_data_loaders, find_dataset_root


def evaluate_model(checkpoint_path: Path) -> None:
    dataset_root = find_dataset_root(Path(__file__).resolve().parent)
    _, _, test_loader, base_train = create_data_loaders(dataset_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )

    if checkpoint["class_to_idx"] != base_train.class_to_idx:
        raise ValueError(
            "Checkpoint class mapping does not match the current dataset."
        )

    model = DefectCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("The test dataset contains no images.")

    print(f"Test loss: {total_loss / total:.4f}")
    print(f"Test accuracy: {100 * correct / total:.2f}%")


def main() -> None:
    checkpoint_path = Path(__file__).resolve().parent / "best_defect_cnn.pth"
    evaluate_model(checkpoint_path)


if __name__ == "__main__":
    main()
