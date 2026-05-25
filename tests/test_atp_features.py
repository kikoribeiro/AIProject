from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import ai_project.atp_features as atp_features
from ai_project.atp_features import (
    FEATURE_COLUMNS,
    build_features,
    build_player_stats,
    build_surface_tools,
    encode_surface,
    validate_and_clean,
)


class AtpFeaturesTest(unittest.TestCase):
    def test_build_features_uses_chronological_form_and_expected_order(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "Date": "2020-01-01",
                    "Rank_1": 1,
                    "Rank_2": 10,
                    "Pts_1": 1000,
                    "Pts_2": 100,
                    "Player_1": "A",
                    "Player_2": "B",
                    "Winner": "A",
                    "Surface": "Hard",
                },
                {
                    "Date": "2020-01-02",
                    "Rank_1": 10,
                    "Rank_2": 1,
                    "Pts_1": 100,
                    "Pts_2": 1000,
                    "Player_1": "B",
                    "Player_2": "A",
                    "Winner": "B",
                    "Surface": "Hard",
                },
            ]
        )

        encoder, mapping = build_surface_tools(df)
        x, y, feature_cols = build_features(
            df,
            form_window=10,
            surface_encoder=encoder,
            surface_mapping=mapping,
        )

        expected_cols = [
            "Rank_Diff",
            "Surface_Encoded",
            "Favorite_Form",
            "Underdog_Form",
            "Pts_Diff",
        ]
        self.assertEqual(feature_cols, expected_cols)
        self.assertEqual(FEATURE_COLUMNS, expected_cols)
        self.assertEqual(y.tolist(), [1, 0])
        np.testing.assert_allclose(x[:, 0], [-9.0, -9.0])
        np.testing.assert_allclose(x[:, 2], [0.5, 1.0])
        np.testing.assert_allclose(x[:, 3], [0.5, 0.0])
        np.testing.assert_allclose(x[:, 4], [900.0, 900.0])

    def test_helpers_support_streamlit_prediction_inputs(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "Date": "2021-01-01",
                    "Rank_1": 2,
                    "Rank_2": 8,
                    "Pts_1": 900,
                    "Pts_2": 300,
                    "Player_1": "A",
                    "Player_2": "B",
                    "Winner": "A",
                    "Surface": "Clay",
                }
            ]
        )

        cleaned, summary = validate_and_clean(df)
        encoder, mapping = build_surface_tools(cleaned)
        players, latest_rank, latest_points, player_form = build_player_stats(cleaned, form_window=10)

        self.assertEqual(summary["missing_rows"], 0)
        self.assertEqual(players, ["A", "B"])
        self.assertEqual(latest_rank["A"], 2.0)
        self.assertEqual(latest_points["A"], 900.0)
        self.assertEqual(player_form["A"], 1.0)
        self.assertEqual(encode_surface("Clay", surface_encoder=encoder, surface_mapping=mapping), 0.0)

    def test_feature_api_does_not_include_year_range_loader(self) -> None:
        helper_name = "_".join(["filter", "years"])

        self.assertFalse(hasattr(atp_features, helper_name))


if __name__ == "__main__":
    unittest.main()
