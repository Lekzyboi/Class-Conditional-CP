"""Shared method interface for conformal predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ConformalMethod(ABC):
    """Base interface for all conformal prediction methods."""

    name: str = "base"

    @abstractmethod
    def fit(self, probs_cal: np.ndarray, labels_cal: np.ndarray, **kwargs: Any) -> "ConformalMethod":
        """Fit/calibrate the method on calibration probabilities and labels."""

    @abstractmethod
    def predict_sets(self, probs_test: np.ndarray, **kwargs: Any) -> list[list[int]]:
        """Return prediction sets for test probabilities."""

    def diagnostics(self) -> dict[str, Any]:
        """Return method-specific diagnostics."""
        return {}
