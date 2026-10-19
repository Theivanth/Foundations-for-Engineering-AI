from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from transformer import (
    BATCH_SIZE,
    CHECKPOINT,
    DATA_ROOT,
    MODEL_ARCHITECTURE,
    SEED,
    PennFudanDetectionDataset,
    build_detector,
    collate_fn,
    find_dataset_root,
    get_device,
    seed_everything,
)


IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5


def box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)

    box_area = (box[2:] - box[:2]).clamp(min=0).prod()
    boxes_area = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=1)
    union = box_area + boxes_area - intersection
    return intersection / union.clamp(min=torch.finfo(union.dtype).eps)


def calculate_detection_metrics(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
) -> tuple[float, float, float]:
    total_ground_truths = sum(len(target["boxes"]) for target in targets)
    if total_ground_truths == 0:
        raise ValueError("Cannot calculate detection metrics without ground-truth boxes.")

    ranked_results: list[tuple[float, bool]] = []
    true_positives_at_threshold = 0
    false_positives_at_threshold = 0

    for prediction, target in zip(predictions, targets):
        predicted_boxes = prediction["boxes"]
        scores = prediction["scores"]
        ground_truth_boxes = target["boxes"]
        matched = torch.zeros(len(ground_truth_boxes), dtype=torch.bool)

        order = scores.argsort(descending=True)
        for prediction_index in order.tolist():
            score = float(scores[prediction_index])
            is_true_positive = False
            if len(ground_truth_boxes):
                overlaps = box_iou(
                    predicted_boxes[prediction_index],
                    ground_truth_boxes,
                )
                best_iou, best_match = overlaps.max(dim=0)
                if best_iou >= IOU_THRESHOLD and not matched[best_match]:
                    matched[best_match] = True
                    is_true_positive = True

            ranked_results.append((score, is_true_positive))
            if score >= SCORE_THRESHOLD:
                if is_true_positive:
                    true_positives_at_threshold += 1
                else:
                    false_positives_at_threshold += 1

    ranked_results.sort(key=lambda result: result[0], reverse=True)
    cumulative_true_positives = np.cumsum(
        [is_true_positive for _, is_true_positive in ranked_results],
        dtype=np.float64,
    )
    cumulative_false_positives = np.cumsum(
        [not is_true_positive for _, is_true_positive in ranked_results],
        dtype=np.float64,
    )

    if ranked_results:
        recall = cumulative_true_positives / total_ground_truths
        precision = cumulative_true_positives / np.maximum(
            cumulative_true_positives + cumulative_false_positives,
            np.finfo(np.float64).eps,
        )
        recall = np.concatenate(([0.0], recall, [1.0]))
        precision = np.concatenate(([0.0], precision, [0.0]))
        for index in range(len(precision) - 2, -1, -1):
            precision[index] = max(precision[index], precision[index + 1])
        change_points = np.flatnonzero(recall[1:] != recall[:-1])
        average_precision = float(
            np.sum(
                (recall[change_points + 1] - recall[change_points])
                * precision[change_points + 1]
            )
        )
    else:
        average_precision = 0.0

    false_negatives_at_threshold = total_ground_truths - true_positives_at_threshold
    precision_at_threshold = true_positives_at_threshold / max(
        true_positives_at_threshold + false_positives_at_threshold,
        1,
    )
    recall_at_threshold = true_positives_at_threshold / max(
        true_positives_at_threshold + false_negatives_at_threshold,
        1,
    )
    return average_precision, precision_at_threshold, recall_at_threshold


def calculate_validation_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

    total_loss = 0.0
    image_count = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [image.to(device) for image in images]
            device_targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]
            loss_dict = model(images, device_targets)
            loss = sum(loss_dict.values())
            total_loss += loss.item() * len(images)
            image_count += len(images)

    model.eval()
    if image_count == 0:
        raise ValueError("Cannot calculate validation loss for an empty data loader.")
    return total_loss / image_count


def evaluate_model(checkpoint_path: Path = CHECKPOINT) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}. Train the model first."
        )

    seed_everything(SEED)
    device = get_device()
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    checkpoint_architecture = checkpoint.get("architecture")
    if (
        checkpoint_architecture is not None
        and checkpoint_architecture != MODEL_ARCHITECTURE
    ):
        raise ValueError(
            f"Checkpoint architecture '{checkpoint_architecture}' does not match "
            f"the evaluator architecture '{MODEL_ARCHITECTURE}'."
        )

    dataset_root = find_dataset_root(Path(DATA_ROOT))
    dataset = PennFudanDetectionDataset(
        dataset_root,
        transform=transforms.ToTensor(),
    )
    indices = np.arange(len(dataset))
    _, validation_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = build_detector().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_loss = calculate_validation_loss(model, validation_loader, device)

    model.eval()
    predictions = []
    targets_for_metrics = []
    with torch.no_grad():
        for images, targets in validation_loader:
            outputs = model([image.to(device) for image in images])
            predictions.extend(
                {
                    key: value.detach().cpu()
                    for key, value in output.items()
                }
                for output in outputs
            )
            targets_for_metrics.extend(
                {
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu(),
                }
                for target in targets
            )

    average_precision, precision, recall = calculate_detection_metrics(
        predictions,
        targets_for_metrics,
    )
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Validation images: {len(validation_indices)}")
    print(f"Validation loss: {validation_loss:.4f}")
    print(f"Pedestrian AP@0.50 IoU: {average_precision:.4f}")
    print(
        f"Precision@{SCORE_THRESHOLD:.2f} score / {IOU_THRESHOLD:.2f} IoU: "
        f"{precision:.4f}"
    )
    print(
        f"Recall@{SCORE_THRESHOLD:.2f} score / {IOU_THRESHOLD:.2f} IoU: "
        f"{recall:.4f}"
    )


def main() -> None:
    evaluate_model()


if __name__ == "__main__":
    main()