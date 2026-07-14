"""Binning utilities."""

from __future__ import annotations

import numpy as np


def create_bins(
    values: np.ndarray,
    num_bins: int,
    method: str = "equal_width",
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")
    if len(values) == 0:
        raise ValueError("values must not be empty")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    if method == "equal_width":
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        bin_edges = np.linspace(min_val, max_val + 1e-10, num_bins + 1)
    elif method == "quantile":
        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(values, percentiles)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.array([float(np.min(values)), float(np.max(values)) + 1e-10])
        bin_edges[-1] += 1e-10
    else:
        raise ValueError(f"Unknown binning method: {method}")

    assignments = np.digitize(values, bin_edges[1:-1])
    return assignments.astype(int), bin_edges


def create_2d_bins(
    values1: np.ndarray,
    values2: np.ndarray,
    num_bins1: int,
    num_bins2: int,
    method: str = "equal_width",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values1 = np.asarray(values1)
    values2 = np.asarray(values2)
    if len(values1) != len(values2):
        raise ValueError("values1 and values2 must have the same length")

    bins1, edges1 = create_bins(values1, num_bins1, method)
    bins2, edges2 = create_bins(values2, num_bins2, method)
    combined = bins1 * num_bins2 + bins2
    return combined.astype(int), edges1, edges2
