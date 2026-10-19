import torch
import torch.nn as nn

class SimpleNeuralNetwork(nn.Module):
    """A small feed-forward classifier for tabular input features."""

    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
