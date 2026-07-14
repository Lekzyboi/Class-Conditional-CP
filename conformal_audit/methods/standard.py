"""Standard split conformal prediction."""

from __future__ import annotations

import numpy as np

from conformal_audit.methods.base import ConformalMethod
from conformal_audit.utils.quantiles import conformal_quantile


class StandardCP(ConformalMethod):
    name = "standard_cp"

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.threshold_: float | None = None

    def fit(self, probs_cal: np.ndarray, labels_cal: np.ndarray, **kwargs) -> "StandardCP":
        scores = 1.0 - probs_cal[np.arange(len(labels_cal)), labels_cal]
        self.threshold_ = conformal_quantile(scores, self.alpha)
        return self

    def predict_sets(self, probs_test: np.ndarray) -> list[list[int]]:
        if self.threshold_ is None:
            raise RuntimeError("StandardCP must be fitted before prediction")
        cutoff = 1.0 - self.threshold_
        sets = []
        for row in probs_test:
            pred_set = np.where(row >= cutoff)[0].tolist()
            if not pred_set:
                pred_set = [int(np.argmax(row))]
            sets.append(pred_set)
        return sets

    def diagnostics(self) -> dict[str, float | str]:
        return {"method": self.name, "alpha": self.alpha, "threshold": self.threshold_}

