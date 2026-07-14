"""Stratified coverage metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from conformal_audit.utils.bins import create_2d_bins


def coverage_indicator(prediction_sets: Sequence[Sequence[int]], labels: np.ndarray) -> np.ndarray:
    if len(prediction_sets) != len(labels):
        raise ValueError("prediction_sets and labels must have the same length")
    return np.asarray([int(int(label) in pred_set) for pred_set, label in zip(prediction_sets, labels)])


def true_label_ranks(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array")
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have the same length")

    ranks = np.zeros(len(labels), dtype=int)
    for i, (row, label) in enumerate(zip(probs, labels)):
        sorted_indices = np.argsort(row)[::-1]
        ranks[i] = int(np.where(sorted_indices == int(label))[0][0]) + 1
    return ranks


def size_stratified_coverage(
    prediction_sets: Sequence[Sequence[int]],
    labels: np.ndarray,
    target_coverage: float,
    max_size: int,
) -> tuple[float, dict[str, dict[str, float | int]]]:
    set_sizes = np.asarray([len(pred_set) for pred_set in prediction_sets])
    indicators = coverage_indicator(prediction_sets, labels)
    stats: dict[str, dict[str, float | int]] = {}
    deviations: list[float] = []

    for size in range(1, max_size + 1):
        if size == max_size:
            mask = set_sizes >= size
            label = f"{size}+"
        else:
            mask = set_sizes == size
            label = str(size)
        if np.any(mask):
            cov = float(np.mean(indicators[mask]))
            deviation = abs(cov - target_coverage)
            deviations.append(deviation)
            stats[label] = {
                "count": int(np.sum(mask)),
                "coverage": cov,
                "deviation": float(deviation),
            }

    return (max(deviations) if deviations else 0.0), stats


def _covgap_2d(
    values1: np.ndarray,
    values2: np.ndarray,
    indicators: np.ndarray,
    target_coverage: float,
    num_bins1: int,
    num_bins2: int,
) -> tuple[float, np.ndarray, dict[str, np.ndarray]]:
    assignments, edges1, edges2 = create_2d_bins(values1, values2, num_bins1, num_bins2, "quantile")
    n_bins = num_bins1 * num_bins2
    coverages = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for bin_idx in range(n_bins):
        mask = assignments == bin_idx
        if np.any(mask):
            coverages[bin_idx] = np.mean(indicators[mask])
            counts[bin_idx] = np.sum(mask)

    non_empty = counts > 0
    covgap = float(np.mean(np.abs(coverages[non_empty] - target_coverage))) if np.any(non_empty) else 0.0
    info = {"edges1": edges1, "edges2": edges2, "bin_counts": counts, "bin_coverages": coverages}
    return covgap, coverages, info


def conf_rank_covgap(
    confidences: np.ndarray,
    ranks: np.ndarray,
    indicators: np.ndarray,
    target_coverage: float,
    num_conf_bins: int = 4,
    num_rank_bins: int = 4,
) -> tuple[float, np.ndarray, dict[str, np.ndarray]]:
    return _covgap_2d(confidences, ranks, indicators, target_coverage, num_conf_bins, num_rank_bins)


def conf_trust_covgap(
    confidences: np.ndarray,
    trust_scores: np.ndarray,
    indicators: np.ndarray,
    target_coverage: float,
    num_conf_bins: int = 4,
    num_trust_bins: int = 4,
) -> tuple[float, np.ndarray, dict[str, np.ndarray]]:
    return _covgap_2d(confidences, trust_scores, indicators, target_coverage, num_conf_bins, num_trust_bins)
