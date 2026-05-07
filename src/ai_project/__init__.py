"""Base package for the AIProject neural-network starter."""

from .neural_network import SimpleNeuralNetwork
from .torch_model import TorchNeuralNetwork, predict, train_model

__all__ = ["SimpleNeuralNetwork", "TorchNeuralNetwork", "train_model", "predict"]
