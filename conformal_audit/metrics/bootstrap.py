"""Bootstrap confidence intervals for audit metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Return percentile bootstrap CI for a one-sample statistic."""

    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")
    if len(values) == 0:
        raise ValueError("values must not be empty")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    statistic = statistic or (lambda sample: float(np.mean(sample)))
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sample_indices = rng.integers(0, len(values), size=len(values))
        estimates[idx] = statistic(values[sample_indices])

    alpha = 1.0 - confidence
    return {
        "estimate": float(statistic(values)),
        "ci_low": float(np.quantile(estimates, alpha / 2.0)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
        "confidence": confidence,
        "n_bootstrap": int(n_bootstrap),
    }


def bootstrap_prediction_set_metrics(
    prediction_sets: list[list[int]],
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Bootstrap marginal coverage and average set size."""

    labels = np.asarray(labels)
    if len(prediction_sets) != len(labels):
        raise ValueError("prediction_sets and labels must have the same length")
    indicators = np.asarray([int(int(label) in pred_set) for pred_set, label in zip(prediction_sets, labels)])
    set_sizes = np.asarray([len(pred_set) for pred_set in prediction_sets], dtype=float)
    return {
        "coverage": bootstrap_ci(indicators, n_bootstrap=n_bootstrap, confidence=confidence, seed=seed),
        "average_set_size": bootstrap_ci(set_sizes, n_bootstrap=n_bootstrap, confidence=confidence, seed=seed + 1),
    }


def bootstrap_equity_gap(
    prediction_sets: list[list[int]],
    labels: np.ndarray,
    class_names: list[str] | tuple[str, ...],
    high_prevalence_classes: list[str] | tuple[str, ...],
    low_prevalence_classes: list[str] | tuple[str, ...],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap macro high-minus-low prevalence coverage gap."""

    labels = np.asarray(labels)
    if len(prediction_sets) != len(labels):
        raise ValueError("prediction_sets and labels must have the same length")
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    high_indices = [class_to_idx[name] for name in high_prevalence_classes if name in class_to_idx]
    low_indices = [class_to_idx[name] for name in low_prevalence_classes if name in class_to_idx]
    if not high_indices or not low_indices:
        return {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "confidence": confidence, "n_bootstrap": n_bootstrap}

    covered = np.asarray([int(int(label) in pred_set) for pred_set, label in zip(prediction_sets, labels)])
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=float)
    indices = np.arange(len(labels))

    def estimate_gap(sample_indices: np.ndarray) -> float:
        class_coverages: dict[int, float] = {}
        for class_idx in high_indices + low_indices:
            mask = labels[sample_indices] == class_idx
            if np.any(mask):
                class_coverages[class_idx] = float(np.mean(covered[sample_indices][mask]))
        high = [class_coverages[idx] for idx in high_indices if idx in class_coverages]
        low = [class_coverages[idx] for idx in low_indices if idx in class_coverages]
        if not high or not low:
            return 0.0
        return float(np.mean(high) - np.mean(low))

    for idx in range(n_bootstrap):
        sample = rng.choice(indices, size=len(indices), replace=True)
        estimates[idx] = estimate_gap(sample)

    estimate = estimate_gap(indices)
    alpha = 1.0 - confidence
    return {
        "estimate": float(estimate),
        "ci_low": float(np.quantile(estimates, alpha / 2.0)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
        "confidence": confidence,
        "n_bootstrap": int(n_bootstrap),
    }


def summarize_bootstrap_for_result(
    prediction_sets: list[list[int]],
    labels: np.ndarray,
    class_names: list[str] | tuple[str, ...],
    high_prevalence_classes: list[str] | tuple[str, ...] = (),
    low_prevalence_classes: list[str] | tuple[str, ...] = (),
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    summary: dict[str, Any] = bootstrap_prediction_set_metrics(
        prediction_sets,
        labels,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    summary["coverage_equity_gap"] = bootstrap_equity_gap(
        prediction_sets,
        labels,
        class_names,
        high_prevalence_classes,
        low_prevalence_classes,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed + 2,
    )
    return summary
