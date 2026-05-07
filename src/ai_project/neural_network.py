"""Minimal neural network starter implementation (pure Python)."""

from __future__ import annotations

import math
import random
from typing import Iterable


class SimpleNeuralNetwork:
    """A tiny 2-layer network for binary classification with sigmoid activation."""

    def __init__(self, input_size: int, hidden_size: int = 4, learning_rate: float = 0.1) -> None:
        if input_size <= 0:
            raise ValueError("input_size must be greater than 0")
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w1 = [
            [random.uniform(-1.0, 1.0) for _ in range(hidden_size)] for _ in range(input_size)
        ]
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.w2 = [random.uniform(-1.0, 1.0) for _ in range(hidden_size)]
        self.b2 = 0.0

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _sigmoid_derivative(output: float) -> float:
        return output * (1.0 - output)

    def _forward(self, x: list[float]) -> tuple[list[float], float]:
        hidden = []
        for j in range(self.hidden_size):
            z = self.b1[j]
            for i in range(self.input_size):
                z += x[i] * self.w1[i][j]
            hidden.append(self._sigmoid(z))

        out = self.b2
        for j in range(self.hidden_size):
            out += hidden[j] * self.w2[j]
        output = self._sigmoid(out)
        return hidden, output

    def predict(self, x: Iterable[float]) -> float:
        values = list(x)
        if len(values) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {len(values)}")
        _, output = self._forward(values)
        return output

    def train(self, x_data: list[list[float]], y_data: list[int], epochs: int = 500) -> None:
        if len(x_data) != len(y_data):
            raise ValueError("x_data and y_data must have the same length")

        for _ in range(epochs):
            for x, target in zip(x_data, y_data):
                if len(x) != self.input_size:
                    raise ValueError(f"Expected {self.input_size} features, got {len(x)}")

                hidden, output = self._forward(x)
                output_error = target - output
                output_delta = output_error * self._sigmoid_derivative(output)

                hidden_deltas = []
                for j in range(self.hidden_size):
                    hidden_error = output_delta * self.w2[j]
                    hidden_deltas.append(hidden_error * self._sigmoid_derivative(hidden[j]))

                for j in range(self.hidden_size):
                    self.w2[j] += self.learning_rate * output_delta * hidden[j]
                self.b2 += self.learning_rate * output_delta

                for i in range(self.input_size):
                    for j in range(self.hidden_size):
                        self.w1[i][j] += self.learning_rate * hidden_deltas[j] * x[i]
                for j in range(self.hidden_size):
                    self.b1[j] += self.learning_rate * hidden_deltas[j]
