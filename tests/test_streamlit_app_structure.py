from __future__ import annotations

import unittest

from apps import confusion_matrix_app


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
            "filter_years",
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


if __name__ == "__main__":
    unittest.main()
