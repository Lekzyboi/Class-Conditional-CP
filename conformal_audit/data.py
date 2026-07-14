"""Data containers and validation helpers for probability-based audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ProbabilityDataset:
    """Probability matrix, true labels, and class names for conformal evaluation."""

    probs: np.ndarray
    labels: np.ndarray
    class_names: Sequence[str]

    def validate(self) -> None:
        if self.probs.ndim != 2:
            raise ValueError("probs must be a 2D array with shape (n_samples, n_classes)")
        if self.labels.ndim != 1:
            raise ValueError("labels must be a 1D array")
        if len(self.probs) != len(self.labels):
            raise ValueError("probs and labels must contain the same number of samples")
        if self.probs.shape[1] != len(self.class_names):
            raise ValueError("number of probability columns must match class_names")

