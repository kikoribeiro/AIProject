"""Starter runner for experimenting with a neural network on a simple sports-like dataset."""

from __future__ import annotations

from ai_project import SimpleNeuralNetwork


def get_sample_dataset() -> tuple[list[list[float]], list[int]]:
    """Simple 2-feature dataset: [pace, consistency] -> podium(1) or not(0)."""
    x_data = [
        [0.90, 0.90],
        [0.80, 0.70],
        [0.20, 0.30],
        [0.30, 0.20],
        [0.95, 0.85],
        [0.40, 0.35],
    ]
    y_data = [1, 1, 0, 0, 1, 0]
    return x_data, y_data


def main() -> None:
    x_data, y_data = get_sample_dataset()

    model = SimpleNeuralNetwork(input_size=2, hidden_size=4, learning_rate=0.4)
    model.train(x_data, y_data, epochs=1500)

    candidate_driver = [0.88, 0.82]
    prediction = model.predict(candidate_driver)

    print("Candidate features [pace, consistency]:", candidate_driver)
    print("Podium probability:", round(prediction, 3))


if __name__ == "__main__":
    main()
