from __future__ import annotations

import unittest

import atp_mlp_keras
from ai_project.metrics.plots import plot_confusion_matrix


class ConfusionMatrixPlotColorsTest(unittest.TestCase):
    def test_shared_confusion_matrix_plot_uses_high_contrast_colormap(self) -> None:
        fig = plot_confusion_matrix(
            [[1, 2], [3, 4]],
            labels=["Upset", "Favorite Wins"],
        )

        try:
            self.assertEqual(fig.axes[0].images[0].cmap.name, "professional_confusion")
        finally:
            atp_mlp_keras.plt.close(fig)

    def test_shared_confusion_matrix_values_use_dark_text(self) -> None:
        fig = plot_confusion_matrix(
            [[1, 2], [3, 4]],
            labels=["Upset", "Favorite Wins"],
        )

        try:
            text_colors = [text.get_color() for text in fig.axes[0].texts]
            self.assertEqual(text_colors, ["#0f172a"] * 4)
        finally:
            atp_mlp_keras.plt.close(fig)

    def test_training_confusion_matrix_pdf_plot_uses_high_contrast_colormap(self) -> None:
        fig = atp_mlp_keras._plot_confusion_matrix(
            [[1, 2], [3, 4]],
            ["Upset", "Favorite Wins"],
            title="Confusion Matrix",
        )

        try:
            self.assertEqual(fig.axes[0].images[0].cmap.name, "professional_confusion")
        finally:
            atp_mlp_keras.plt.close(fig)

    def test_training_confusion_matrix_values_use_dark_text(self) -> None:
        fig = atp_mlp_keras._plot_confusion_matrix(
            [[1, 2], [3, 4]],
            ["Upset", "Favorite Wins"],
            title="Confusion Matrix",
        )

        try:
            text_colors = [text.get_color() for text in fig.axes[0].texts]
            self.assertEqual(text_colors, ["#0f172a"] * 4)
        finally:
            atp_mlp_keras.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
