"""Mondrian/class-conditional conformal prediction."""

from __future__ import annotations

import numpy as np

from conformal_audit.methods.base import ConformalMethod
from conformal_audit.utils.quantiles import conformal_quantile


class MondrianCP(ConformalMethod):
    name = "mondrian_cp"

    def __init__(self, alpha: float = 0.10, n_classes: int | None = None):
        self.alpha = alpha
        self.n_classes = n_classes
        self.thresholds_: list[float] | None = None

    def fit(self, probs_cal: np.ndarray, labels_cal: np.ndarray, **kwargs) -> "MondrianCP":
        n_classes = self.n_classes or probs_cal.shape[1]
        global_scores = 1.0 - probs_cal[np.arange(len(labels_cal)), labels_cal]
        global_threshold = conformal_quantile(global_scores, self.alpha)

        thresholds: list[float] = []
        for class_idx in range(n_classes):
            mask = labels_cal == class_idx
            if np.any(mask):
                class_scores = 1.0 - probs_cal[mask, class_idx]
                thresholds.append(conformal_quantile(class_scores, self.alpha))
            else:
                thresholds.append(global_threshold)
        self.thresholds_ = thresholds
        self.n_classes = n_classes
        return self

    def predict_sets(self, probs_test: np.ndarray) -> list[list[int]]:
        if self.thresholds_ is None:
            raise RuntimeError("MondrianCP must be fitted before prediction")

        prediction_sets: list[list[int]] = []
        for row in probs_test:
            pred_set = [
                class_idx
                for class_idx, threshold in enumerate(self.thresholds_)
                if row[class_idx] >= (1.0 - threshold)
            ]
            if not pred_set:
                pred_set = [int(np.argmax(row))]
            prediction_sets.append(pred_set)
        return prediction_sets

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": self.name,
            "alpha": self.alpha,
            "n_classes": self.n_classes,
            "thresholds": self.thresholds_,
        }
