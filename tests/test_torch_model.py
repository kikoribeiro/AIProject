import unittest

from ai_project.torch_model import predict, train_model


class TorchNeuralNetworkTests(unittest.TestCase):
    def test_prediction_is_probability(self) -> None:
        model = train_model(
            [[0.0, 0.0], [1.0, 1.0]],
            [0, 1],
            epochs=50,
            batch_size=2,
        )

        prediction = predict(model, [0.8, 0.7])

        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)
        self.assertGreater(predict(model, [0.9, 0.9]), 0.5)
        self.assertLess(predict(model, [0.1, 0.1]), 0.5)


if __name__ == "__main__":
    unittest.main()
