"""Classification metrics derived from a confusion matrix."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerClassMetrics:
    label: int
    support: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def _safe_div(n: float, d: float) -> float:
    return 0.0 if d == 0 else (n / d)


def classification_metrics(cm: list[list[int]], labels: list[int]) -> dict:
    """Compute accuracy and per-class precision/recall/F1 from a confusion matrix.

    Returns a dict shaped for easy display in the Streamlit app.
    """
    k = len(labels)
    if len(cm) != k or any(len(row) != k for row in cm):
        raise ValueError("cm size must match labels")

    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(k))
    accuracy = _safe_div(correct, total)

    per_class: list[PerClassMetrics] = []
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(k) if r != i)
        fn = sum(cm[i][c] for c in range(k) if c != i)
        support = sum(cm[i][c] for c in range(k))

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        per_class.append(
            PerClassMetrics(
                label=label,
                support=support,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    macro_precision = _safe_div(sum(m.precision for m in per_class), k)
    macro_recall = _safe_div(sum(m.recall for m in per_class), k)
    macro_f1 = _safe_div(sum(m.f1 for m in per_class), k)

    weighted_precision = _safe_div(sum(m.precision * m.support for m in per_class), total)
    weighted_recall = _safe_div(sum(m.recall * m.support for m in per_class), total)
    weighted_f1 = _safe_div(sum(m.f1 * m.support for m in per_class), total)

    return {
        "accuracy": accuracy,
        "total": total,
        "per_class": per_class,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "weighted_avg": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1,
        },
    }
