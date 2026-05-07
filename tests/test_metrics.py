import unittest

from ai_project.metrics.classification_metrics import classification_metrics


class ClassificationMetricsTests(unittest.TestCase):
    def test_metrics_simple(self) -> None:
        cm = [
            [5, 0, 0],
            [0, 3, 1],
            [0, 2, 4],
        ]
        labels = [0, 1, 2]
        metrics = classification_metrics(cm, labels)

        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

        self.assertEqual(len(metrics["per_class"]), 3)
        self.assertIn("precision", metrics["macro_avg"])


if __name__ == "__main__":
    unittest.main()
