import unittest

from ai_project import SimpleNeuralNetwork


class SimpleNeuralNetworkTests(unittest.TestCase):
    def test_prediction_is_probability(self) -> None:
        model = SimpleNeuralNetwork(input_size=2)
        model.train([[0.0, 0.0], [1.0, 1.0]], [0, 1], epochs=300)

        prediction = model.predict([0.8, 0.7])

        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)


if __name__ == "__main__":
    unittest.main()
