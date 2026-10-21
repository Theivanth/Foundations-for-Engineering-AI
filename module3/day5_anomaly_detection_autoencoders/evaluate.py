from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from autoencoder import SensorAutoencoder


SEED = 42
THRESHOLD_PERCENTILE = 99
DATA_DIR = Path("processed_data")
MODEL_PATH = Path("saved_models/best_autoencoder.pt")
OUTPUT_DIR = Path("saved_models/evaluation")


def load_evaluation_data(data_dir=DATA_DIR):
    """Load and validate validation/test tensors and their binary labels."""
    data_dir = Path(data_dir)
    val_data = torch.load(
        data_dir / "val.pt",
        map_location="cpu",
        weights_only=True
    )
    test_data = torch.load(
        data_dir / "test.pt",
        map_location="cpu",
        weights_only=True
    )

    X_val = val_data["X"].float()
    y_val = val_data["y"].long().numpy()
    X_test = test_data["X"].float()
    y_test = test_data["y"].long().numpy()

    if X_val.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Validation and test features must be 2D")
    if not set(np.unique(y_val)).issubset({0, 1}):
        raise ValueError("Validation labels must be binary")
    if not set(np.unique(y_test)).issubset({0, 1}):
        raise ValueError("Test labels must be binary")

    return X_val, y_val, X_test, y_test


def load_model(model_path=MODEL_PATH, device=None):
    """Load the saved autoencoder checkpoint into evaluation mode."""
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )
    model = SensorAutoencoder(
        input_dim=checkpoint["input_dim"],
        latent_dim=checkpoint["latent_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint, device


def reconstruction_errors(model, X, device, batch_size=512):
    """Calculate mean squared reconstruction error for each sample."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    errors = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size].to(device)
            reconstructed = model(batch)
            sample_errors = torch.mean(
                (batch - reconstructed) ** 2,
                dim=1
            )
            errors.append(sample_errors.cpu().numpy())

    if not errors:
        return np.array([], dtype=float)
    return np.concatenate(errors)


def choose_threshold(val_errors, y_val, percentile=THRESHOLD_PERCENTILE):
    """Calibrate the anomaly threshold using normal validation errors only."""
    normal_val_errors = val_errors[y_val == 0]
    if len(normal_val_errors) == 0:
        raise ValueError("Validation data contains no normal samples")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")

    threshold = float(np.percentile(normal_val_errors, percentile))
    if not np.isfinite(threshold):
        raise ValueError("Calculated threshold is not finite")
    return threshold


def save_threshold(threshold, output_dir=OUTPUT_DIR, percentile=THRESHOLD_PERCENTILE):
    """Save the calibrated threshold configuration and return its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_data = {
        "threshold": threshold,
        "percentile": percentile,
        "method": "normal_validation_percentile",
        "comparison": "greater_than"
    }
    threshold_path = output_dir / "threshold.json"
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=4)
    return threshold_path


def calculate_metrics(y_test, y_pred, threshold):
    """Calculate test classification metrics and error breakdown."""
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        "precision": float(precision_score(
            y_test, y_pred, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, y_pred, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_test, y_pred, zero_division=0
        )),
        "false_alarm_rate": float(false_alarm_rate),
        "miss_rate": float(miss_rate),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "threshold": float(threshold)
    }
    return metrics, cm


def print_evaluation_summary(y_test, y_pred, metrics, cm):
    """Print classification and error breakdown summaries."""
    print("\nTEST EVALUATION RESULTS")
    print("-" * 35)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    ))
    print("\nError breakdown:")
    print("True negatives:", metrics["true_negative"])
    print("False positives:", metrics["false_positive"])
    print("False negatives:", metrics["false_negative"])
    print("True positives:", metrics["true_positive"])
    print(f"\nFalse alarm rate: {metrics['false_alarm_rate']:.2%}")
    print(f"Miss rate:        {metrics['miss_rate']:.2%}")


def save_predictions(y_test, y_pred, test_errors, output_dir=OUTPUT_DIR):
    """Save per-sample test predictions to CSV and return the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame({
        "sample_index": np.arange(len(y_test)),
        "actual_anomaly": y_test,
        "predicted_anomaly": y_pred,
        "reconstruction_error": test_errors
    })
    predictions_path = output_dir / "test_predictions.csv"
    results_df.to_csv(predictions_path, index=False)
    return predictions_path


def save_metrics(metrics, output_dir=OUTPUT_DIR):
    """Save calculated evaluation metrics as JSON and return the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    return metrics_path


def plot_reconstruction_errors(
    test_errors,
    y_pred,
    threshold,
    output_dir=OUTPUT_DIR,
    show=True
):
    """Save a plot of test reconstruction errors and detected anomalies."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13, 5))
    plt.plot(
        test_errors,
        linewidth=0.8,
        label="Reconstruction error"
    )
    plt.axhline(
        threshold,
        color="red",
        linestyle="--",
        label="Anomaly threshold"
    )
    detected = np.where(y_pred == 1)[0]
    plt.scatter(
        detected,
        test_errors[detected],
        color="orange",
        s=10,
        label="Predicted anomalies"
    )
    plt.xlabel("Test sample index")
    plt.ylabel("Reconstruction MSE")
    plt.title("Autoencoder Anomaly Detection - Test Dataset")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / "reconstruction_errors.png"
    plt.savefig(plot_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return plot_path


def plot_confusion_matrix(cm, output_dir=OUTPUT_DIR, show=True):
    """Save a plot of the test confusion matrix."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Anomaly"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Autoencoder Confusion Matrix")
    plt.tight_layout()

    plot_path = output_dir / "confusion_matrix.png"
    plt.savefig(plot_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(disp.figure_)
    return plot_path


def plot_actual_vs_predicted(
    y_test,
    y_pred,
    output_dir=OUTPUT_DIR,
    show=True
):
    """Save a plot comparing actual and predicted anomaly positions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13, 4))
    actual_indices = np.where(y_test == 1)[0]
    pred_indices = np.where(y_pred == 1)[0]
    plt.scatter(
        actual_indices,
        np.ones(len(actual_indices)),
        s=12,
        label="Actual anomalies"
    )
    plt.scatter(
        pred_indices,
        np.zeros(len(pred_indices)),
        s=12,
        label="Predicted anomalies"
    )
    plt.yticks([0, 1], ["Predicted", "Actual"])
    plt.xlabel("Test sample index")
    plt.title("Actual vs Predicted Anomalies")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / "actual_vs_predicted.png"
    plt.savefig(plot_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return plot_path


def evaluate_model(
    data_dir=DATA_DIR,
    model_path=MODEL_PATH,
    output_dir=OUTPUT_DIR,
    percentile=THRESHOLD_PERCENTILE,
    batch_size=512,
    show_plots=True
):
    """Calibrate the threshold, evaluate test data, and save outputs."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    X_val, y_val, X_test, y_test = load_evaluation_data(data_dir)
    print("\nValidation samples:", len(X_val))
    print("Test samples:", len(X_test))

    model, checkpoint, device = load_model(model_path, device)
    print("\nModel loaded successfully!")
    print("Best training epoch:", checkpoint["epoch"])
    print("Best validation loss:", checkpoint["best_val_loss"])

    val_errors = reconstruction_errors(model, X_val, device, batch_size)
    threshold = choose_threshold(val_errors, y_val, percentile)
    save_threshold(threshold, output_dir, percentile)
    print("\nThreshold percentile:", percentile)
    print("Anomaly threshold:", f"{threshold:.6f}")

    test_errors = reconstruction_errors(model, X_test, device, batch_size)
    y_pred = (test_errors > threshold).astype(int)
    metrics, cm = calculate_metrics(y_test, y_pred, threshold)
    print_evaluation_summary(y_test, y_pred, metrics, cm)

    save_predictions(y_test, y_pred, test_errors, output_dir)
    save_metrics(metrics, output_dir)
    plot_reconstruction_errors(
        test_errors,
        y_pred,
        threshold,
        output_dir,
        show_plots
    )
    plot_confusion_matrix(cm, output_dir, show_plots)
    plot_actual_vs_predicted(
        y_test,
        y_pred,
        output_dir,
        show_plots
    )

    return {
        "threshold": threshold,
        "metrics": metrics,
        "confusion_matrix": cm,
        "test_errors": test_errors,
        "predictions": y_pred
    }


def main():
    """Evaluate the default saved model and datasets."""
    evaluate_model()


if __name__ == "__main__":
    main()
