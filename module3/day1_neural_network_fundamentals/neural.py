import torch
import torch.nn as nn

from create_model import SimpleNeuralNetwork

from pytorch_data import make_datasets, make_dataloaders

from failure_predictor import FailurePredictor

dt = make_datasets("sensor_dataset.csv")

x_train, y_train = dt["train"].tensors
x_val, y_val = dt["val"].tensors
x_test, y_test = dt["test"].tensors

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("First features:", x_train[:5])
print("First labels:", y_train[:5])

model_SNN = SimpleNeuralNetwork(x_train.shape[1], len(torch.unique(y_train)))
print(model_SNN)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model_SNN.parameters(),
    lr=0.001
)

epochs = 200

for epoch in range(epochs):

    # Training mode
    model_SNN.train()

    # Remove old gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model_SNN(x_train)

    # Calculate loss
    loss = criterion(outputs, y_train)

    # Calculate gradients
    loss.backward()

    # Update weights
    optimizer.step()

    # Validation
    model_SNN.eval()

    with torch.no_grad():
        val_outputs = model_SNN(x_val)

        val_loss = criterion(
            val_outputs,
            y_val
        )

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Train Loss: {loss.item():.4f} "
            f"Validation Loss: {val_loss.item():.4f}"
        )

# Test
model_SNN.eval()

with torch.no_grad():

    test_outputs = model_SNN(x_test)

    predictions = torch.argmax(
        test_outputs,
        dim=1
    )

print("Predictions:")
print(predictions[:20])

print("Actual:")
print(y_test[:20])

correct = (
    predictions == y_test
).sum().item()

total = y_test.size(0)

accuracy = correct / total

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

class_names = [
    "healthy",
    "warning",
    "failure"
]

predictor = FailurePredictor(
    model=model_SNN,
    class_names=class_names
)

predictor.run()