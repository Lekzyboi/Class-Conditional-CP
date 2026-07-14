"""Adaptive prediction sets."""

from __future__ import annotations

import numpy as np

from conformal_audit.methods.base import ConformalMethod
from conformal_audit.utils.quantiles import conformal_quantile


class APS(ConformalMethod):
    name = "aps"

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.threshold_: float | None = None

    def fit(self, probs_cal: np.ndarray, labels_cal: np.ndarray, **kwargs) -> "APS":
        scores = []
        for row, label in zip(probs_cal, labels_cal):
            sorted_probs = np.sort(row)[::-1]
            cumsum = np.cumsum(sorted_probs)
            true_prob = row[int(label)]
            true_rank = int(np.sum(row >= true_prob))
            scores.append(cumsum[true_rank - 1] if true_rank > 0 else 0.0)
        self.threshold_ = conformal_quantile(np.asarray(scores), self.alpha)
        return self

    def predict_sets(self, probs_test: np.ndarray) -> list[list[int]]:
        if self.threshold_ is None:
            raise RuntimeError("APS must be fitted before prediction")

        prediction_sets: list[list[int]] = []
        for row in probs_test:
            sorted_indices = np.argsort(row)[::-1]
            sorted_probs = row[sorted_indices]
            cumsum = np.cumsum(sorted_probs)
            pred_set = sorted_indices[cumsum <= self.threshold_].astype(int).tolist()
            if not pred_set:
                pred_set = [int(sorted_indices[0])]
            prediction_sets.append(pred_set)
        return prediction_sets

    def diagnostics(self) -> dict[str, float | str]:
        return {"method": self.name, "alpha": self.alpha, "threshold": self.threshold_}
