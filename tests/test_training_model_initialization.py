from __future__ import annotations

import unittest

import atp_mlp_keras


class TrainingModelInitializationTest(unittest.TestCase):
    def test_relu_hidden_layers_use_he_initialization(self) -> None:
        model = atp_mlp_keras.build_model(input_dim=5)
        dense_layers = [
            layer
            for layer in model.layers
            if layer.__class__.__name__ == "Dense"
        ]

        hidden_initializers = [
            layer.kernel_initializer.__class__.__name__
            for layer in dense_layers[:2]
        ]

        self.assertEqual(hidden_initializers, ["HeNormal", "HeNormal"])


if __name__ == "__main__":
    unittest.main()
