"""Temperature scaling probability calibration comparator."""

from __future__ import annotations

import numpy as np

class TemperatureScaler:
    """Non-conformal probability calibration comparator."""

    def __init__(self, temperatures: np.ndarray | None = None):
        self.temperatures = temperatures
        self.temperature_: float | None = None
        self.nll_: float | None = None

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels).astype(int)
        if logits.ndim != 2:
            raise ValueError("logits must be a 2D array")
        if len(logits) != len(labels):
            raise ValueError("logits and labels must have the same length")

        candidates = self.temperatures
        if candidates is None:
            candidates = np.logspace(-2, 2, 401)

        losses = np.asarray([negative_log_likelihood(self.transform(logits, temp), labels) for temp in candidates])
        best_idx = int(np.argmin(losses))
        self.temperature_ = float(candidates[best_idx])
        self.nll_ = float(losses[best_idx])
        return self

    def transform(self, logits: np.ndarray, temperature: float | None = None) -> np.ndarray:
        temp = self.temperature_ if temperature is None else temperature
        if temp is None:
            raise RuntimeError("TemperatureScaler must be fitted before transform")
        if temp <= 0:
            raise ValueError("temperature must be positive")
        return softmax(np.asarray(logits, dtype=float) / temp)

    def diagnostics(self) -> dict[str, float | str | None]:
        return {"method": "temperature_scaling", "temperature": self.temperature_, "nll": self.nll_}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels).astype(int)
    clipped = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels).astype(int)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(correct[mask])) - float(np.mean(confidences[mask])))
    return ece
