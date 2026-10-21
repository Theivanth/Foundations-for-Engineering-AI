import torch
from torch import nn


# ---------------------------------------
# 1. AUTOENCODER MODEL
# ---------------------------------------

class SensorAutoencoder(nn.Module):

    def __init__(
        self,
        input_dim=4,
        latent_dim=2
    ):
        super().__init__()

        # Encoder: 4 -> 8 -> 6 -> 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),

            nn.Linear(8, 6),
            nn.ReLU(),

            nn.Linear(6, latent_dim)
        )

        # Decoder: 2 -> 6 -> 8 -> 4
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 6),
            nn.ReLU(),

            nn.Linear(6, 8),
            nn.ReLU(),

            nn.Linear(8, input_dim)
        )

    def forward(self, x):

        # Compress the sensor readings
        latent = self.encoder(x)

        # Reconstruct original readings
        reconstructed = self.decoder(latent)

        return reconstructed


# ---------------------------------------
# 2. TEST THE MODEL
# ---------------------------------------

def demo_model():
    """Print the model architecture and run a small reconstruction example."""
    torch.manual_seed(42)

    model = SensorAutoencoder(
        input_dim=4,
        latent_dim=2
    )

    print("\nAUTOENCODER ARCHITECTURE")
    print(model)

    # Example: 5 sensor readings
    # Each sample has 4 sensor features.
    sample_input = torch.randn(5, 4)

    with torch.no_grad():
        reconstructed = model(sample_input)
        latent = model.encoder(sample_input)

    print("\nInput shape:")
    print(sample_input.shape)

    print("\nLatent representation shape:")
    print(latent.shape)

    print("\nReconstructed output shape:")
    print(reconstructed.shape)

    # Reconstruction loss
    criterion = nn.MSELoss()

    loss = criterion(
        reconstructed,
        sample_input
    )

    print("\nInitial reconstruction loss:")
    print(f"{loss.item():.6f}")

    # Number of trainable parameters
    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nTrainable parameters:")
    print(total_parameters)

    assert reconstructed.shape == sample_input.shape

    print("\nModel architecture test passed!")


def main():
    """Run the autoencoder demonstration."""
    demo_model()


if __name__ == "__main__":
    main()