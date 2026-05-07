import unittest

from ai_project.metrics.confusion_matrix import confusion_matrix, normalize_confusion_matrix


class ConfusionMatrixTests(unittest.TestCase):
    def test_confusion_matrix_basic(self) -> None:
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        self.assertEqual(cm, [[1, 1], [1, 1]])

    def test_normalize_true_rows_sum_to_one(self) -> None:
        cm = [[1, 1], [0, 2]]
        norm = normalize_confusion_matrix(cm, mode="true")
        self.assertAlmostEqual(sum(norm[0]), 1.0)
        self.assertAlmostEqual(sum(norm[1]), 1.0)


if __name__ == "__main__":
    unittest.main()
