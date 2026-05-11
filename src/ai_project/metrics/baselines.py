"""Simple baseline models for comparing classifier performance."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ai_project.metrics.classification_metrics import classification_metrics
from ai_project.metrics.confusion_matrix import confusion_matrix

FAVORITE_WINS_BASELINE_NAME = "Favorite always wins"


def constant_prediction_baseline(
    y_true: Sequence[int],
    *,
    prediction_label: int,
    labels: list[int],
    name: str,
) -> dict[str, Any]:
    """Evaluate a baseline that predicts the same class for every row."""
    y_true_list = [int(label) for label in y_true]
    y_pred = [int(prediction_label)] * len(y_true_list)
    cm = confusion_matrix(y_true_list, y_pred, labels=labels)

    return {
        "name": name,
        "predictions": y_pred,
        "confusion_matrix": cm,
        "metrics": classification_metrics(cm, labels),
    }


def favorite_wins_baseline(
    y_true: Sequence[int],
    *,
    labels: list[int] | None = None,
) -> dict[str, Any]:
    """Evaluate the naive baseline that predicts the favorite always wins."""
    return constant_prediction_baseline(
        y_true,
        prediction_label=1,
        labels=labels or [0, 1],
        name=FAVORITE_WINS_BASELINE_NAME,
    )
