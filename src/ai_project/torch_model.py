"""PyTorch neural network for sports/F1 binary classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TorchNeuralNetwork(nn.Module):
    """2-layer fully-connected network for binary classification."""

    def __init__(self, input_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_model(
    x_data: list[list[float]],
    y_data: list[int],
    hidden_size: int = 16,
    epochs: int = 100,
    learning_rate: float = 0.01,
    batch_size: int = 16,
) -> TorchNeuralNetwork:
    """Train a TorchNeuralNetwork and return the trained model."""
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TorchNeuralNetwork(input_size=x_tensor.shape[1], hidden_size=hidden_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                all_preds = model(x_tensor)
                epoch_loss = criterion(all_preds, y_tensor).item()
            print(f"Epoch {epoch + 1}/{epochs}  loss={epoch_loss:.4f}")

    return model


def predict(model: TorchNeuralNetwork, x: list[float]) -> float:
    """Return a probability in [0, 1] for a single sample."""
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor([x], dtype=torch.float32)
        return model(tensor).item()
