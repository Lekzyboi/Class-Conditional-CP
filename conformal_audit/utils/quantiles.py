"""Conformal quantile helpers."""

from __future__ import annotations

import numpy as np


def conformal_quantile_rank(n: int, alpha: float) -> int:
    """Return the one-indexed finite-sample conformal rank."""

    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    return int(np.ceil((n + 1) * (1.0 - alpha)))


def conformal_quantile_index(n: int, alpha: float) -> int:
    """Return the zero-indexed conformal order statistic, or -1 for infinity."""

    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    rank = conformal_quantile_rank(n, alpha)
    if rank > n:
        return -1
    return rank - 1


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the finite-sample conformal order statistic."""

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D array")
    if len(scores) == 0:
        raise ValueError("scores must not be empty")
    sorted_scores = np.sort(scores)
    index = conformal_quantile_index(len(sorted_scores), alpha)
    if index < 0:
        return float("inf")
    return float(sorted_scores[index])
