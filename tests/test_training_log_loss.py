from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from sklearn.metrics import log_loss

import atp_mlp_keras


class IdentityScaler:
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x


class FixedSplitter:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def split(self, x: np.ndarray, y: np.ndarray):
        yield np.asarray([0, 1]), np.asarray([2, 3])


class FixedProbabilityModel:
    probabilities = np.asarray([0.2, 0.7], dtype=float)

    def fit(self, *args, **kwargs) -> None:
        return None

    def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
        return self.probabilities.reshape(-1, 1)

    @property
    def layers(self):
        weights = np.ones((5, 2), dtype=float)
        return [SimpleNamespace(get_weights=lambda: [weights])]


class TrainingLogLossTest(unittest.TestCase):
    def test_kfold_training_prints_log_loss_from_predicted_probabilities(self) -> None:
        x = np.zeros((4, 5), dtype=float)
        y = np.asarray([0, 1, 0, 1], dtype=int)
        buffer = io.StringIO()

        with (
            patch.object(atp_mlp_keras, "StratifiedKFold", FixedSplitter),
            patch.object(atp_mlp_keras, "StandardScaler", IdentityScaler),
            patch.object(atp_mlp_keras, "build_model", lambda input_dim: FixedProbabilityModel()),
            redirect_stdout(buffer),
        ):
            atp_mlp_keras.train_with_stratified_kfold(
                x,
                y,
                ["Rank_Diff", "Surface_Encoded", "Favorite_Form", "Underdog_Form", "Pts_Diff"],
                n_splits=1,
                epochs=1,
                batch_size=2,
            )

        expected = log_loss([0, 1], FixedProbabilityModel.probabilities, labels=atp_mlp_keras.LABELS)
        self.assertIn(f"Log loss (all folds): {expected:.4f}", buffer.getvalue())

    def test_kfold_training_uses_default_decision_threshold(self) -> None:
        class BorderlineProbabilityModel(FixedProbabilityModel):
            probabilities = np.asarray([0.55, 0.65], dtype=float)

        x = np.zeros((4, 5), dtype=float)
        y = np.asarray([0, 1, 0, 1], dtype=int)
        buffer = io.StringIO()

        with (
            patch.object(atp_mlp_keras, "StratifiedKFold", FixedSplitter),
            patch.object(atp_mlp_keras, "StandardScaler", IdentityScaler),
            patch.object(
                atp_mlp_keras,
                "build_model",
                lambda input_dim: BorderlineProbabilityModel(),
            ),
            redirect_stdout(buffer),
        ):
            atp_mlp_keras.train_with_stratified_kfold(
                x,
                y,
                ["Rank_Diff", "Surface_Encoded", "Favorite_Form", "Underdog_Form", "Pts_Diff"],
                n_splits=1,
                epochs=1,
                batch_size=2,
            )

        self.assertEqual(atp_mlp_keras.DEFAULT_DECISION_THRESHOLD, 0.60)
        self.assertIn("Decision threshold: 0.60", buffer.getvalue())
        self.assertIn("[[1 0]\n [0 1]]", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
