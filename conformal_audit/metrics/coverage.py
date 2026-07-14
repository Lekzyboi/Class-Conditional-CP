"""Coverage and prediction-set size metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def coverage(prediction_sets: Sequence[Sequence[int]], labels: np.ndarray) -> float:
    if len(prediction_sets) != len(labels):
        raise ValueError("prediction_sets and labels must have the same length")
    if len(labels) == 0:
        raise ValueError("labels must not be empty")
    covered = sum(int(int(label) in pred_set) for pred_set, label in zip(prediction_sets, labels))
    return covered / len(labels)


def average_set_size(prediction_sets: Sequence[Sequence[int]]) -> float:
    if len(prediction_sets) == 0:
        raise ValueError("prediction_sets must not be empty")
    return float(np.mean([len(pred_set) for pred_set in prediction_sets]))


def per_class_coverage(
    prediction_sets: Sequence[Sequence[int]],
    labels: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    if len(prediction_sets) != len(labels):
        raise ValueError("prediction_sets and labels must have the same length")

    out: dict[str, dict[str, float | int]] = {}
    for class_idx, class_name in enumerate(class_names):
        indices = np.where(labels == class_idx)[0]
        if len(indices) == 0:
            out[class_name] = {"count": 0, "covered": 0, "coverage_rate": 0.0, "avg_set_size": 0.0}
            continue
        covered = sum(int(int(labels[i]) in prediction_sets[i]) for i in indices)
        out[class_name] = {
            "count": int(len(indices)),
            "covered": int(covered),
            "coverage_rate": covered / len(indices),
            "avg_set_size": float(np.mean([len(prediction_sets[i]) for i in indices])),
        }
    return out

