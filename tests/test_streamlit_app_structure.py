from __future__ import annotations

import ast
import unittest

import numpy as np
from sklearn.metrics import log_loss

from apps import confusion_matrix_app


class IdentityScaler:
    def transform(self, x: np.ndarray) -> np.ndarray:
        return x


class FixedProbabilityModel:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
        return self.probabilities.reshape(-1, 1)


class StreamlitAppStructureTest(unittest.TestCase):
    def test_streamlit_app_does_not_expose_training_helpers(self) -> None:
        training_helpers = [
            "set_seeds",
            "split_train_val_test",
            "build_model",
            "train_final_model",
            "run_kfold_training",
        ]

        for helper_name in training_helpers:
            with self.subTest(helper_name=helper_name):
                self.assertFalse(hasattr(confusion_matrix_app, helper_name))

    def test_streamlit_app_uses_shared_feature_helpers(self) -> None:
        duplicated_feature_helpers = [
            "validate_and_clean",
            "sort_matches",
            "build_surface_tools",
            "encode_surface",
            "build_player_stats",
            "build_features",
        ]

        for helper_name in duplicated_feature_helpers:
            with self.subTest(helper_name=helper_name):
                self.assertFalse(hasattr(confusion_matrix_app, helper_name))

    def test_streamlit_app_displays_favorite_baseline_comparison(self) -> None:
        self.assertTrue(hasattr(confusion_matrix_app, "favorite_wins_baseline"))

        source_path = confusion_matrix_app.__file__
        with open(source_path, encoding="utf-8") as fh:
            source = fh.read()

        self.assertIn("Baseline comparison", source)

    def test_streamlit_app_uses_the_whole_csv_by_default(self) -> None:
        source_path = confusion_matrix_app.__file__
        with open(source_path, encoding="utf-8") as fh:
            source = fh.read()

        self.assertIn("Rows after cleaning", source)

    def test_streamlit_prediction_form_uses_points_feature_without_odds_inputs(self) -> None:
        source_path = confusion_matrix_app.__file__
        with open(source_path, encoding="utf-8") as fh:
            source = fh.read()

        self.assertIn("Pts_Diff", source)
        self.assertNotIn("Player A decimal odds", source)
        self.assertNotIn("Player B decimal odds", source)
        self.assertNotIn("Odd_Diff", source)

    def test_streamlit_app_defaults_to_upset_friendly_threshold(self) -> None:
        self.assertEqual(confusion_matrix_app.DEFAULT_DECISION_THRESHOLD, 0.60)

    def test_cm_normalization_defaults_to_true_mode(self) -> None:
        source_path = confusion_matrix_app.__file__
        with open(source_path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "selectbox":
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            if node.args[0].value != "CM normalization":
                continue

            index_keywords = [keyword for keyword in node.keywords if keyword.arg == "index"]
            self.assertEqual(len(index_keywords), 1)
            self.assertIsInstance(index_keywords[0].value, ast.Constant)
            self.assertEqual(index_keywords[0].value.value, 1)
            return

        self.fail("CM normalization selectbox not found")

    def test_saved_model_evaluation_returns_log_loss_from_probabilities(self) -> None:
        x = np.zeros((4, 5), dtype=float)
        y = np.asarray([0, 1, 1, 0], dtype=int)
        probabilities = [0.1, 0.8, 0.4, 0.9]

        cm, metrics, importance, log_loss_value = confusion_matrix_app.evaluate_loaded_model(
            FixedProbabilityModel(probabilities),
            IdentityScaler(),
            x,
            y,
            threshold=0.5,
            feature_count=5,
        )

        self.assertEqual(cm, [[1, 1], [1, 1]])
        self.assertEqual(metrics["total"], 4)
        np.testing.assert_allclose(importance, np.zeros(5))
        self.assertAlmostEqual(
            log_loss_value,
            log_loss(y, probabilities, labels=confusion_matrix_app.LABELS),
        )


if __name__ == "__main__":
    unittest.main()
