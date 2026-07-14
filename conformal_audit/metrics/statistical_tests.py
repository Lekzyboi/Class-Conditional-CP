"""Multiplicity-adjusted statistical tests for coverage audits."""

from __future__ import annotations

from math import exp, isinf, lgamma, log
from typing import Any

import numpy as np


def binomial_less_p_value(successes: int, trials: int, target_probability: float) -> float:
    """Exact one-sided binomial p-value P[X <= successes]."""

    if trials < 0:
        raise ValueError("trials must be non-negative")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")
    if not 0.0 <= target_probability <= 1.0:
        raise ValueError("target_probability must be in [0, 1]")
    if trials == 0:
        return 1.0
    if target_probability == 0.0:
        return 1.0
    if target_probability == 1.0:
        return 1.0 if successes >= trials else 0.0

    log_p = log(target_probability)
    log_q = log(1.0 - target_probability)
    log_terms = [
        _log_binomial_pmf(k, trials, log_p, log_q)
        for k in range(successes + 1)
    ]
    max_log = max(log_terms)
    if isinf(max_log):
        return 0.0
    total = sum(exp(term - max_log) for term in log_terms)
    return float(min(max(exp(max_log) * total, 0.0), 1.0))


def adjust_p_values(p_values: list[float] | np.ndarray, method: str = "holm") -> np.ndarray:
    """Adjust p-values using Holm or Benjamini-Hochberg correction."""

    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1:
        raise ValueError("p_values must be a 1D array")
    if len(p_values) == 0:
        return p_values
    if np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError("p_values must be between 0 and 1")

    method = method.lower()
    if method in {"holm", "holm-bonferroni"}:
        return _holm_adjust(p_values)
    if method in {"bh", "fdr_bh", "benjamini-hochberg"}:
        return _benjamini_hochberg_adjust(p_values)
    raise ValueError(f"Unknown p-value adjustment method: {method}")


def per_class_below_target_tests(
    per_class: dict[str, dict[str, float | int]],
    target_coverage: float,
    adjustment: str = "holm",
) -> list[dict[str, Any]]:
    """Exact one-sided tests for class coverage below target."""

    rows: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for class_name, stats in per_class.items():
        count = int(stats["count"])
        covered = int(stats.get("covered", round(float(stats["coverage_rate"]) * count)))
        p_value = binomial_less_p_value(covered, count, target_coverage)
        raw_p_values.append(p_value)
        rows.append(
            {
                "class_name": class_name,
                "count": count,
                "covered": covered,
                "coverage_rate": float(stats["coverage_rate"]),
                "target_coverage": target_coverage,
                "p_value": p_value,
            }
        )

    adjusted = adjust_p_values(raw_p_values, adjustment)
    for row, adjusted_p in zip(rows, adjusted):
        row["adjustment"] = adjustment
        row["adjusted_p_value"] = float(adjusted_p)
        row["below_target"] = bool(row["coverage_rate"] < target_coverage)
        row["significant_below_target"] = bool(row["coverage_rate"] < target_coverage and adjusted_p < 0.05)
    return rows


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted_sorted = np.empty(n, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = (n - rank) * p_values[idx]
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = min(running_max, 1.0)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def _log_binomial_pmf(k: int, n: int, log_p: float, log_q: float) -> float:
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log_p + (n - k) * log_q


def _benjamini_hochberg_adjust(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted_sorted = np.empty(n, dtype=float)
    running_min = 1.0
    for reverse_rank in range(n - 1, -1, -1):
        rank = reverse_rank + 1
        adjusted = sorted_p[reverse_rank] * n / rank
        running_min = min(running_min, adjusted)
        adjusted_sorted[reverse_rank] = min(running_min, 1.0)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted
