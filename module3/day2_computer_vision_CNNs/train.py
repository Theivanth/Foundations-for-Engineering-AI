from pathlib import Path

import torch
import torch.nn as nn

from transformer import SEED, create_data_loaders, find_dataset_root


class DefectCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def train_model(epochs: int = 20) -> None:
    dataset_root = find_dataset_root(Path(__file__).resolve().parent)
    train_loader, val_loader, _, base_train = create_data_loaders(
        dataset_root
    )
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = DefectCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(model)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_train_loss = running_loss / train_size
        avg_val_loss = val_loss / val_size
        val_accuracy = 100 * correct / total

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val loss: {avg_val_loss:.4f} | "
            f"Val accuracy: {val_accuracy:.2f}%"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": base_train.class_to_idx,
                    "image_size": [128, 128],
                    "seed": SEED,
                },
                Path(__file__).resolve().parent / "best_defect_cnn.pth",
            )


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
