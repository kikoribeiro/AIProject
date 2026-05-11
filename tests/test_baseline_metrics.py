from __future__ import annotations

import unittest

from ai_project.metrics.baselines import favorite_wins_baseline


class BaselineMetricsTest(unittest.TestCase):
    def test_favorite_wins_baseline_predicts_positive_class_for_every_match(self) -> None:
        baseline = favorite_wins_baseline([0, 1, 0, 1, 1], labels=[0, 1])

        self.assertEqual(baseline["name"], "Favorite always wins")
        self.assertEqual(baseline["predictions"], [1, 1, 1, 1, 1])
        self.assertEqual(baseline["confusion_matrix"], [[0, 2], [0, 3]])
        self.assertAlmostEqual(baseline["metrics"]["accuracy"], 0.6)

        favorite_metrics = next(
            metric for metric in baseline["metrics"]["per_class"] if metric.label == 1
        )
        self.assertAlmostEqual(favorite_metrics.precision, 0.6)
        self.assertAlmostEqual(favorite_metrics.recall, 1.0)
        self.assertAlmostEqual(favorite_metrics.f1, 0.75)


if __name__ == "__main__":
    unittest.main()
