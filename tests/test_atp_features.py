from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ai_project.atp_features import (
    FEATURE_COLUMNS,
    build_features,
    build_player_stats,
    build_surface_tools,
    encode_surface,
    filter_years,
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
                    "Player_1": "A",
                    "Player_2": "B",
                    "Winner": "A",
                    "Surface": "Hard",
                },
                {
                    "Date": "2020-01-02",
                    "Rank_1": 10,
                    "Rank_2": 1,
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

        self.assertEqual(feature_cols, FEATURE_COLUMNS)
        self.assertEqual(y.tolist(), [1, 0])
        np.testing.assert_allclose(x[:, 0], [-9.0, -9.0])
        np.testing.assert_allclose(x[:, 2], [0.5, 1.0])
        np.testing.assert_allclose(x[:, 3], [0.5, 0.0])

    def test_helpers_support_streamlit_prediction_inputs(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "Date": "2021-01-01",
                    "Rank_1": 2,
                    "Rank_2": 8,
                    "Player_1": "A",
                    "Player_2": "B",
                    "Winner": "A",
                    "Surface": "Clay",
                }
            ]
        )

        filtered = filter_years(df, 2021, 2021)
        cleaned, summary = validate_and_clean(filtered)
        encoder, mapping = build_surface_tools(cleaned)
        players, latest_rank, player_form = build_player_stats(cleaned, form_window=10)

        self.assertEqual(summary["missing_rows"], 0)
        self.assertEqual(players, ["A", "B"])
        self.assertEqual(latest_rank["A"], 2.0)
        self.assertEqual(player_form["A"], 1.0)
        self.assertEqual(encode_surface("Clay", surface_encoder=encoder, surface_mapping=mapping), 0.0)


if __name__ == "__main__":
    unittest.main()
