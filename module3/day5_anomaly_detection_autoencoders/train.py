from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import SensorAutoencoder


SEED = 42
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
DATA_DIR = Path("processed_data")
MODEL_DIR = Path("saved_models")
MODEL_PATH = MODEL_DIR / "best_autoencoder.pt"
PLOT_PATH = MODEL_DIR / "training_loss.png"


def set_seed(seed=SEED):
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_data(data_dir=DATA_DIR):
    """Load and validate the preprocessed train and validation tensors."""
    data_dir = Path(data_dir)
    train_data = torch.load(
        data_dir / "train.pt",
        map_location="cpu",
        weights_only=True
    )
    val_data = torch.load(
        data_dir / "val.pt",
        map_location="cpu",
        weights_only=True
    )

    X_train = train_data["X"].float()
    X_val = val_data["X"].float()
    y_train = train_data["y"]
    y_val = val_data["y"]

    if X_train.ndim != 2 or X_train.shape[1] != 4:
        raise ValueError("Training features must have shape (samples, 4)")
    if X_val.ndim != 2 or X_val.shape[1] != 4:
        raise ValueError("Validation features must have shape (samples, 4)")
    if not (y_train == 0).all():
        raise ValueError("Training data must contain normal samples only")

    X_val_normal = X_val[y_val == 0]
    if len(X_val_normal) == 0:
        raise ValueError("Validation data contains no normal samples")

    return X_train, X_val_normal


def create_data_loaders(X_train, X_val, batch_size=BATCH_SIZE):
    """Create shuffled training and deterministic validation data loaders."""
    train_loader = DataLoader(
        TensorDataset(X_train),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val),
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch and return average sample loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for (X_batch,) in train_loader:
        X_batch = X_batch.to(device)
        optimizer.zero_grad()
        reconstructed = model(X_batch)
        loss = criterion(reconstructed, X_batch)
        loss.backward()
        optimizer.step()

        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training loader contains no samples")
    return total_loss / total_samples


def validate(model, val_loader, criterion, device):
    """Calculate average validation loss without updating model weights."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for (X_batch,) in val_loader:
            X_batch = X_batch.to(device)
            reconstructed = model(X_batch)
            loss = criterion(reconstructed, X_batch)
            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Validation loader contains no samples")
    return total_loss / total_samples


def plot_training_history(train_losses, val_losses, plot_path=PLOT_PATH, show=True):
    """Save the training and validation loss plot."""
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label="Training Loss"
    )
    plt.plot(
        range(1, len(val_losses) + 1),
        val_losses,
        label="Validation Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Autoencoder Training History")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return plot_path


def verify_checkpoint(model_path, device):
    """Load the saved best checkpoint into an evaluation-mode model."""
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )
    loaded_model = SensorAutoencoder(
        input_dim=checkpoint["input_dim"],
        latent_dim=checkpoint["latent_dim"]
    ).to(device)
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.eval()
    return loaded_model, checkpoint


def train_model(
    data_dir=DATA_DIR,
    model_dir=MODEL_DIR,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    seed=SEED,
    show_plot=True
):
    """Train the autoencoder, save its best checkpoint, and return results."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    set_seed(seed)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_PATH.name
    plot_path = model_dir / PLOT_PATH.name
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    X_train, X_val_normal = load_training_data(data_dir)
    train_loader, val_loader = create_data_loaders(
        X_train,
        X_val_normal,
        batch_size
    )
    print("\nTraining samples:", len(X_train))
    print("Normal validation samples:", len(X_val_normal))

    model = SensorAutoencoder(
        input_dim=4,
        latent_dim=2
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )
    print("\nModel:")
    print(model)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0

    print("\nStarting training...\n")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )
        val_loss = validate(
            model,
            val_loader,
            criterion,
            device
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "input_dim": 4,
                "latent_dim": 2,
                "epoch": epoch,
                "best_val_loss": best_val_loss
            }
            torch.save(checkpoint, model_path)
            print("  Best model saved!")

    plot_training_history(
        train_losses,
        val_losses,
        plot_path,
        show_plot
    )
    print("\nTRAINING COMPLETED")
    print("Final train loss:", f"{train_losses[-1]:.6f}")
    print("Best validation loss:", f"{best_val_loss:.6f}")
    print("Model checkpoint:", model_path.resolve())
    print("Training plot:", plot_path.resolve())

    loaded_model, saved_checkpoint = verify_checkpoint(model_path, device)
    print("\nBest checkpoint epoch:", saved_checkpoint["epoch"])

    return {
        "model": loaded_model,
        "checkpoint": saved_checkpoint,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "model_path": model_path,
        "plot_path": plot_path,
        "device": device
    }


def main():
    """Train the model with the default data and configuration."""
    train_model()


if __name__ == "__main__":
    main()
