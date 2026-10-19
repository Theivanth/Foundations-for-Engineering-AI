import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn


SEED = 42
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 5e-3
DATA_ROOT = Path(os.environ.get("DATASET_ROOT", Path(__file__).resolve().parent / "data"))
CHECKPOINT = Path(__file__).resolve().parent / "best_pedestrian_detector.pth"
MODEL_ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_320_fpn"


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def find_dataset_root(search_root: Path) -> Path:
    """Locate the Penn-Fudan dataset or a compatible ImageFolder dataset."""
    search_root = Path(search_root)

    for png_dir in search_root.rglob("PNGImages"):
        if (png_dir.parent / "PedMasks").is_dir():
            return png_dir.parent

    for train_dir in search_root.rglob("train"):
        root = train_dir.parent
        if (root / "test").is_dir() and (train_dir / "def_front").is_dir() and (train_dir / "ok_front").is_dir():
            return root

    raise FileNotFoundError(
        f"Could not find a compatible dataset under {search_root}. "
        "Expected a Penn-Fudan-style root with PNGImages/PedMasks or an "
        "ImageFolder-style train/test split. Set DATASET_ROOT to the dataset root."
    )


def is_pennfudan_dataset(root: Path) -> bool:
    root = Path(root)
    return (root / "PNGImages").is_dir() and (root / "PedMasks").is_dir()


def build_detector() -> torch.nn.Module:
    return fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=2,
    )


class PennFudanDetectionDataset(Dataset):
    """Dataset that turns each Penn-Fudan mask into a pedestrian detection target."""

    def __init__(self, root: Path, transform=None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.image_dir = self.root / "PNGImages"
        self.mask_dir = self.root / "PedMasks"
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        if not self.image_paths:
            raise ValueError(f"No PNG images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        mask_path = self.mask_dir / f"{image_path.stem}_mask.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")

        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        boxes = []
        for object_id in np.unique(mask):
            if object_id == 0:
                continue
            ys, xs = np.where(mask == object_id)
            if xs.size == 0:
                continue
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            boxes.append([float(x_min), float(y_min), float(x_max + 1), float(y_max + 1)])

        if boxes:
            boxes = np.asarray(boxes, dtype=np.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            area = torch.as_tensor(area, dtype=torch.float32)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": labels,
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.uint8),
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    # Detection models return loss dictionaries only in training mode.
    model.train()
    total_loss = 0.0
    count = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            total_loss += losses.item() * len(images)
        else:
            with torch.no_grad():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item() * len(images)
        count += len(images)

    if not training:
        model.eval()

    return total_loss / count if count else 0.0


def main() -> None:
    seed_everything()
    device = get_device()
    root = find_dataset_root(DATA_ROOT)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = PennFudanDetectionDataset(root, transform=transform)
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
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

    print(f"Dataset: {root}")
    print(f"Device: {device}")
    print(f"Training images: {len(train_idx)} | Validation images: {len(val_idx)}")

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "dataset_root": str(root),
                "architecture": MODEL_ARCHITECTURE,
            }, CHECKPOINT)

    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_val_loss = run_epoch(model, val_loader, device)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Model saved to: {CHECKPOINT}")


if __name__ == "__main__":
    main()