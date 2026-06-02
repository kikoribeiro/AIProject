from __future__ import annotations

import io
import os
import tempfile
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


class FixedHistoryModel(FixedProbabilityModel):
    probabilities = np.asarray([0.9, 0.1], dtype=float)

    def __init__(self) -> None:
        self.validation_data = None

    def fit(self, *args, **kwargs):
        self.validation_data = kwargs.get("validation_data")
        return SimpleNamespace(
            history={
                "loss": [0.60, 0.30],
                "val_loss": [0.62, 0.85],
                "accuracy": [0.65, 0.95],
                "val_accuracy": [0.64, 0.55],
            }
        )


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

    def test_overfitting_check_uses_validation_split_and_saves_graph(self) -> None:
        x = np.zeros((6, 5), dtype=float)
        y = np.asarray([0, 1, 0, 1, 0, 1], dtype=int)
        train_idx = np.asarray([0, 1])
        val_idx = np.asarray([2, 3])
        test_idx = np.asarray([4, 5])
        model = FixedHistoryModel()
        buffer = io.StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = os.path.join(tmpdir, "overfitting_check.png")
            with (
                patch.object(atp_mlp_keras, "StandardScaler", IdentityScaler),
                patch.object(atp_mlp_keras, "build_model", lambda input_dim: model),
                redirect_stdout(buffer),
            ):
                report = atp_mlp_keras.run_overfitting_check(
                    x,
                    y,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    epochs=2,
                    batch_size=2,
                    graph_path=graph_path,
                )

            self.assertTrue(os.path.exists(graph_path))

        _, validation_labels = model.validation_data
        np.testing.assert_array_equal(validation_labels, y[val_idx])
        self.assertEqual(report["graph_path"], graph_path)
        self.assertIn("Overfitting check:", buffer.getvalue())
        self.assertIn("Saved overfitting graph to:", buffer.getvalue())

    def test_overfitting_check_warns_when_validation_gap_is_large(self) -> None:
        x = np.zeros((6, 5), dtype=float)
        y = np.asarray([0, 1, 0, 1, 0, 1], dtype=int)
        buffer = io.StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(atp_mlp_keras, "StandardScaler", IdentityScaler),
                patch.object(atp_mlp_keras, "build_model", lambda input_dim: FixedHistoryModel()),
                redirect_stdout(buffer),
            ):
                atp_mlp_keras.run_overfitting_check(
                    x,
                    y,
                    train_idx=np.asarray([0, 1]),
                    val_idx=np.asarray([2, 3]),
                    test_idx=np.asarray([4, 5]),
                    epochs=2,
                    batch_size=2,
                    graph_path=os.path.join(tmpdir, "overfitting_check.png"),
                )

        self.assertIn("Possible overfitting", buffer.getvalue())

    def test_training_history_plot_uses_markers_for_single_epoch_visibility(self) -> None:
        class RecordingAxis:
            def __init__(self) -> None:
                self.plot_kwargs = []

            def plot(self, *args, **kwargs) -> None:
                self.plot_kwargs.append(kwargs)

            def set_title(self, *args, **kwargs) -> None:
                pass

            def set_xlabel(self, *args, **kwargs) -> None:
                pass

            def set_ylabel(self, *args, **kwargs) -> None:
                pass

            def legend(self, *args, **kwargs) -> None:
                pass

        class RecordingFigure:
            def tight_layout(self) -> None:
                pass

            def savefig(self, *args, **kwargs) -> None:
                pass

        loss_axis = RecordingAxis()
        accuracy_axis = RecordingAxis()
        history = {
            "loss": [0.75],
            "val_loss": [0.68],
            "accuracy": [0.52],
            "val_accuracy": [0.60],
        }

        with (
            patch.object(
                atp_mlp_keras.plt,
                "subplots",
                lambda *args, **kwargs: (RecordingFigure(), (loss_axis, accuracy_axis)),
            ),
            patch.object(atp_mlp_keras.plt, "close", lambda fig: None),
        ):
            atp_mlp_keras._plot_training_history(history, "unused.png")

        plot_kwargs = loss_axis.plot_kwargs + accuracy_axis.plot_kwargs
        self.assertEqual(len(plot_kwargs), 4)
        for kwargs in plot_kwargs:
            self.assertEqual(kwargs.get("marker"), "o")


if __name__ == "__main__":
    unittest.main()
