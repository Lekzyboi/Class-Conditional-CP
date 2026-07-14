"""Regularized adaptive prediction sets."""

from __future__ import annotations

import numpy as np

from conformal_audit.methods.base import ConformalMethod
from conformal_audit.utils.quantiles import conformal_quantile


class RAPS(ConformalMethod):
    name = "raps"

    def __init__(self, alpha: float = 0.10, lambda_reg: float = 0.01, k_reg: int = 3):
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.k_reg = k_reg
        self.threshold_: float | None = None

    def fit(self, probs_cal: np.ndarray, labels_cal: np.ndarray, **kwargs) -> "RAPS":
        probs_cal = np.asarray(probs_cal)
        labels_cal = np.asarray(labels_cal)
        scores = np.asarray([self._label_score(row, int(label)) for row, label in zip(probs_cal, labels_cal)])
        self.threshold_ = conformal_quantile(scores, self.alpha)
        return self

    def predict_sets(self, probs_test: np.ndarray) -> list[list[int]]:
        if self.threshold_ is None:
            raise RuntimeError("RAPS must be fitted before prediction")

        prediction_sets: list[list[int]] = []
        for row in np.asarray(probs_test):
            sorted_indices = np.argsort(row)[::-1]
            sorted_probs = row[sorted_indices]
            cumulative = np.cumsum(sorted_probs)
            pred_set: list[int] = []
            for rank_zero, class_idx in enumerate(sorted_indices):
                rank = rank_zero + 1
                score = cumulative[rank_zero] + self.lambda_reg * max(rank - self.k_reg, 0)
                if score <= self.threshold_:
                    pred_set.append(int(class_idx))
            if not pred_set:
                pred_set = [int(sorted_indices[0])]
            prediction_sets.append(pred_set)
        return prediction_sets

    def _label_score(self, row: np.ndarray, label: int) -> float:
        sorted_indices = np.argsort(row)[::-1]
        sorted_probs = row[sorted_indices]
        rank_zero = int(np.where(sorted_indices == label)[0][0])
        rank = rank_zero + 1
        cumulative_score = float(np.cumsum(sorted_probs)[rank_zero])
        regularizer = self.lambda_reg * max(rank - self.k_reg, 0)
        return cumulative_score + regularizer

    def diagnostics(self) -> dict[str, float | int | str]:
        return {
            "method": self.name,
            "alpha": self.alpha,
            "lambda_reg": self.lambda_reg,
            "k_reg": self.k_reg,
            "threshold": self.threshold_,
        }
