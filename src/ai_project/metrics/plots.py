"""Plot helpers for confusion matrices."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt


def plot_confusion_matrix(
    cm: Sequence[Sequence[float]],
    *,
    labels: Sequence[str],
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Create a matplotlib figure of a confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i][j]:.2f}" if isinstance(cm[i][j], float) else str(cm[i][j]),
                    ha="center", va="center",
                    color="white" if cm[i][j] > (max(map(max, cm)) * 0.5) else "black")

    fig.tight_layout()
    return fig
