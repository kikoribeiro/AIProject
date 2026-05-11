"""Confusion matrix utilities.

We implement a confusion matrix from scratch to keep the project self-contained and
make the computation explicit.

Definitions
-----------
For multi-class classification with labels L0..LK-1:

- rows correspond to true labels
- columns correspond to predicted labels

So cm[i][j] is the number of samples whose true label is i and predicted label is j.
"""
from __future__ import annotations

from typing import Iterable, Sequence


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int] | None = None,
) -> list[list[int]]:
    """Compute a confusion matrix.

    Parameters
    ----------
    y_true:
        True class labels.
    y_pred:
        Predicted class labels.
    labels:
        Explicit label ordering. If None, labels are inferred from sorted unique
        values in y_true and y_pred.

    Returns
    -------
    list[list[int]]
        Square matrix (KxK) of integer counts.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if labels is None:
        labels = sorted(set(y_true).union(set(y_pred)))

    label_to_index = {label: idx for idx, label in enumerate(labels)}
    k = len(labels)

    cm = [[0 for _ in range(k)] for _ in range(k)]

    for t, p in zip(y_true, y_pred):
        if t not in label_to_index:
            raise ValueError(f"Unknown true label {t}. Provide labels explicitly.")
        if p not in label_to_index:
            raise ValueError(f"Unknown predicted label {p}. Provide labels explicitly.")

        i = label_to_index[t]
        j = label_to_index[p]
        cm[i][j] += 1

    return cm


def normalize_confusion_matrix(
    cm: Sequence[Sequence[int]], *, mode: str = "none"
) -> list[list[float]]:
    """Normalize a confusion matrix.

    mode:
      - "none": returns raw counts as floats
      - "true": row-normalized (each row sums to 1) -> recall-like view
      - "pred": column-normalized (each col sums to 1) -> precision-like view
    """
    if mode not in {"none", "true", "pred"}:
        raise ValueError("mode must be one of: none, true, pred")

    k = len(cm)
    if any(len(row) != k for row in cm):
        raise ValueError("cm must be square")

    cm_f = [[float(v) for v in row] for row in cm]

    if mode == "none":
        return cm_f

    if mode == "true":
        out: list[list[float]] = []
        for row in cm_f:
            s = sum(row)
            out.append([0.0 if s == 0 else (v / s) for v in row])
        return out

    # mode == "pred"
    col_sums = [0.0 for _ in range(k)]
    for i in range(k):
        for j in range(k):
            col_sums[j] += cm_f[i][j]

    out = [[0.0 for _ in range(k)] for _ in range(k)]
    for i in range(k):
        for j in range(k):
            s = col_sums[j]
            out[i][j] = 0.0 if s == 0 else (cm_f[i][j] / s)
    return out


def iter_cm_pairs(cm: Sequence[Sequence[int]]) -> Iterable[tuple[int, int, int]]:
    """Yield (true_index, pred_index, count) for each cell."""
    k = len(cm)
    for i in range(k):
        for j in range(k):
            yield i, j, int(cm[i][j])
