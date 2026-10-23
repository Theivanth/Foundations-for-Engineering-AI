from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


INPUT_DATASETS = {
    "office_supplies": Path(
        "/kaggle/input/python-object-detection-dataset-office-supplies"
    ),
}
OUTPUT = Path("/kaggle/working")
MERGED_DATASET = OUTPUT / "office_supplies_dataset"
RUNS = OUTPUT / "training_runs"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MAX_TRAIN_IMAGES = 50
RANDOM_SEED = 42
SOURCE_CLASS_NAMES = {
    15: "tape dispenser",
    16: "highlighter",
    17: "glue stick",
    18: "pen",
    19: "scissors",
    20: "protractor",
    21: "correction tape",
}


def read_class_names(dataset_yaml: Path, config: dict) -> list[str]:
    names = config.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        indexed = {int(index): str(name) for index, name in names.items()}
        expected = list(range(len(indexed)))
        if sorted(indexed) != expected:
            raise ValueError(f"Class IDs in {dataset_yaml} must start at zero.")
        return [indexed[index] for index in expected]
    raise ValueError(f"No supported class-name list found in {dataset_yaml}.")


def read_dataset_config(dataset_yaml: Path) -> tuple[dict, list[str], Path]:
    config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid dataset YAML: {dataset_yaml}")

    configured_root = config.get("path")
    dataset_root = Path(configured_root) if configured_root else dataset_yaml.parent
    if not dataset_root.is_absolute():
        dataset_root = dataset_yaml.parent / dataset_root
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    return config, read_class_names(dataset_yaml, config), dataset_root.resolve()


def resolve_split(config: dict, dataset_root: Path, split: str) -> Path:
    value = config.get(split)
    if value is None and split == "val":
        value = config.get("valid")
    if value is None:
        raise ValueError(f"Dataset YAML is missing the {split!r} split.")
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("Each dataset split must resolve to one image directory.")
        value = value[0]

    split_path = Path(str(value))
    if not split_path.is_absolute():
        split_path = dataset_root / split_path
    if not split_path.exists():
        raise FileNotFoundError(f"Dataset split path does not exist: {split_path}")
    if not split_path.is_dir():
        raise ValueError(f"Expected an image directory for dataset split: {split_path}")
    return split_path.resolve()


def labels_directory(images_directory: Path) -> Path:
    parts = list(images_directory.parts)
    image_indices = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not image_indices:
        raise ValueError(
            f"Expected YOLO images/<split> directory: {images_directory}"
        )
    parts[image_indices[-1]] = "labels"
    return Path(*parts)


def infer_dataset_config(
    input_root: Path,
) -> tuple[dict, list[str], Path, dict[int, int]]:
    dataset_root = input_root / "data"
    if not dataset_root.is_dir():
        dataset_root = input_root

    split_names = {
        "train": ("train",),
        "val": ("val", "valid", "validation"),
        "test": ("test",),
    }
    config = {"path": str(dataset_root)}
    split_directories = {}
    for split, candidates in split_names.items():
        for candidate in candidates:
            images_directory = dataset_root / candidate / "images"
            if images_directory.is_dir():
                config[split] = images_directory.relative_to(dataset_root).as_posix()
                split_directories[split] = images_directory
                break

    if "train" not in split_directories or "val" not in split_directories:
        raise FileNotFoundError(
            f"No data.yaml found in {input_root}, and expected train/images plus "
            "val, valid, or validation/images directories."
        )

    class_ids = set()
    for images_directory in split_directories.values():
        source_labels = labels_directory(images_directory)
        if not source_labels.is_dir():
            continue
        for label_file in source_labels.rglob("*.txt"):
            for line_number, line in enumerate(
                label_file.read_text(encoding="utf-8").splitlines(), 1
            ):
                fields = line.split()
                if not fields:
                    continue
                try:
                    class_id = int(fields[0])
                except ValueError as error:
                    raise ValueError(
                        f"Invalid YOLO class ID in {label_file}:{line_number}"
                    ) from error
                if class_id < 0:
                    raise ValueError(
                        f"Negative YOLO class ID in {label_file}:{line_number}"
                    )
                class_ids.add(class_id)

    if not class_ids:
        raise ValueError(f"No labeled classes found under {dataset_root}")

    source_class_ids = {
        source_id: target_id for target_id, source_id in enumerate(sorted(class_ids))
    }
    unknown_class_ids = class_ids - SOURCE_CLASS_NAMES.keys()
    if unknown_class_ids:
        raise ValueError(
            f"No object names are configured for source class IDs: "
            f"{sorted(unknown_class_ids)}"
        )
    class_names = [
        SOURCE_CLASS_NAMES[source_id] for source_id in source_class_ids
    ]
    config["names"] = class_names
    return config, class_names, dataset_root.resolve(), source_class_ids


def remap_label_file(
    source: Path | None,
    destination: Path,
    class_ids: dict[int, int],
) -> int:
    if source is None or not source.is_file():
        destination.write_text("", encoding="utf-8")
        return 0

    remapped_lines = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"Invalid YOLO label in {source}:{line_number}")

        try:
            source_class = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"Invalid YOLO label in {source}:{line_number}") from error

        if source_class not in class_ids:
            raise ValueError(
                f"Class ID {source_class} is not defined by the dataset YAML: {source}"
            )
        if len(coordinates) == 4:
            box = coordinates
        elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
            x_values = coordinates[0::2]
            y_values = coordinates[1::2]
            left, right = min(x_values), max(x_values)
            top, bottom = min(y_values), max(y_values)
            box = [
                (left + right) / 2,
                (top + bottom) / 2,
                right - left,
                bottom - top,
            ]
        else:
            raise ValueError(f"Unsupported YOLO annotation in {source}:{line_number}")

        if any(value < 0 or value > 1 for value in box):
            raise ValueError(f"Out-of-range normalized box in {source}:{line_number}")
        remapped_lines.append(
            f"{class_ids[source_class]} " + " ".join(f"{value:.6f}" for value in box)
        )

    destination.write_text(
        "\n".join(remapped_lines) + ("\n" if remapped_lines else ""),
        encoding="utf-8",
    )
    return len(remapped_lines)


def prepare_dataset() -> tuple[Path, list[str], dict[str, int]]:
    MERGED_DATASET.mkdir(parents=True, exist_ok=True)
    class_names: list[str] = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_box_counts = {"train": 0, "val": 0, "test": 0}

    for dataset_key, input_root in INPUT_DATASETS.items():
        if not input_root.is_dir():
            raise FileNotFoundError(f"Kaggle dataset is not attached: {input_root}")
        yaml_files = sorted(input_root.rglob("data.yaml"))
        if yaml_files:
            dataset_yaml = yaml_files[0]
            config, source_names, dataset_root = read_dataset_config(dataset_yaml)
            source_class_ids = {
                source_id: source_id for source_id in range(len(source_names))
            }
        else:
            config, source_names, dataset_root, source_class_ids = infer_dataset_config(
                input_root
            )
        class_offset = len(class_names)
        class_ids = {
            source_id: class_offset + target_id
            for source_id, target_id in source_class_ids.items()
        }
        for source_name in source_names:
            normalized_name = re.sub(r"\s+", "_", source_name.strip())
            class_names.append(f"{dataset_key}_{normalized_name}")

        for split in ("train", "val", "test"):
            if split == "test" and config.get("test") is None:
                continue
            images_directory = resolve_split(config, dataset_root, split)
            source_labels = labels_directory(images_directory)
            split_images = sorted(
                path
                for path in images_directory.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if split == "train" and len(split_images) > MAX_TRAIN_IMAGES:
                split_images = sorted(
                    random.Random(RANDOM_SEED).sample(
                        split_images,
                        MAX_TRAIN_IMAGES,
                    )
                )
            if not split_images:
                raise ValueError(f"No images found in {images_directory}")

            output_images = MERGED_DATASET / "images" / split
            output_labels = MERGED_DATASET / "labels" / split
            output_images.mkdir(parents=True, exist_ok=True)
            output_labels.mkdir(parents=True, exist_ok=True)

            for image in split_images:
                relative = image.relative_to(images_directory)
                unique_name = hashlib.sha1(
                    f"{dataset_key}/{relative.as_posix()}".encode("utf-8")
                ).hexdigest()[:10]
                output_name = f"{dataset_key}_{unique_name}_{image.name}"
                image_target = output_images / output_name
                label_target = output_labels / f"{Path(output_name).stem}.txt"
                image_target.symlink_to(image.resolve())

                label_source = source_labels / relative.with_suffix(".txt")
                split_box_counts[split] += remap_label_file(
                    label_source if label_source.is_file() else None,
                    label_target,
                    class_ids,
                )
                split_counts[split] += 1

    if (
        not class_names
        or not split_counts["train"]
        or not split_counts["val"]
        or not split_box_counts["train"]
        or not split_box_counts["val"]
    ):
        raise ValueError(
            "The office-supplies dataset must provide labeled training and validation data."
        )

    merged_config = {
        "path": str(MERGED_DATASET),
        "train": "images/train",
        "val": "images/val",
        "names": class_names,
    }
    if split_counts["test"]:
        merged_config["test"] = "images/test"

    data_yaml = MERGED_DATASET / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(merged_config, sort_keys=False),
        encoding="utf-8",
    )
    counts = {
        "class_count": len(class_names),
        "image_counts": split_counts,
        "box_counts": split_box_counts,
    }
    print("Merged dataset:", json.dumps(counts, indent=2))
    return data_yaml, class_names, counts


def main() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "ultralytics>=8.3,<9",
        ]
    )

    import torch
    from ultralytics import YOLO

    print("PyTorch:", torch.__version__)
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator before training.")
    print("GPU:", torch.cuda.get_device_name(0))

    dataset_yaml, class_names, counts = prepare_dataset()
    RUNS.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(dataset_yaml),
        epochs=20,
        imgsz=640,
        batch=4,
        device=0,
        project=str(RUNS),
        name="office_supplies_50",
        exist_ok=True,
        seed=42,
        workers=2,
        plots=True,
        cache=False,
    )

    best = RUNS / "office_supplies_50" / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training completed without producing {best}")

    exported = OUTPUT / "best.pt"
    shutil.copy2(best, exported)
    metrics = YOLO(str(exported)).val(
        data=str(dataset_yaml),
        imgsz=640,
        batch=8,
        device=0,
        split="val",
        project=str(RUNS),
        name="validation",
        exist_ok=True,
    )
    summary = {
        "model": "YOLO11 nano fine-tuned for office-supplies object detection",
        "dataset": {
            "ref": "immanuel0405/python-object-detection-dataset-office-supplies",
            "license": "CC BY-NC-SA 4.0 (non-commercial use)",
        },
        "classes": class_names,
        "training_data": counts,
        "max_train_images": MAX_TRAIN_IMAGES,
        "epochs": 20,
        "gpu": torch.cuda.get_device_name(0),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    (OUTPUT / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("Training complete. Download best.pt and training_summary.json.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
