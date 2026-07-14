"""Confidence-trust conformal prediction methods."""

from __future__ import annotations

import numpy as np

from conformal_audit.methods.base import ConformalMethod
from conformal_audit.utils.bins import create_2d_bins
from conformal_audit.utils.quantiles import conformal_quantile


class ConfTrustNaiveCP(ConformalMethod):
    """2D-binned conformal thresholds over model confidence and trust score."""

    name = "conf_trust_naive"
    requires_trust_scores = True

    def __init__(
        self,
        alpha: float = 0.10,
        num_conf_bins: int = 4,
        num_trust_bins: int = 4,
        binning_method: str = "quantile",
        min_bin_size: int = 5,
    ):
        self.alpha = alpha
        self.num_conf_bins = num_conf_bins
        self.num_trust_bins = num_trust_bins
        self.binning_method = binning_method
        self.min_bin_size = min_bin_size
        self.conf_edges_: np.ndarray | None = None
        self.trust_edges_: np.ndarray | None = None
        self.bin_thresholds_: dict[int, float] | None = None
        self.global_threshold_: float | None = None

    def fit(self, probs_cal: np.ndarray, labels_cal: np.ndarray, **kwargs) -> "ConfTrustNaiveCP":
        trust_cal = _require_vector(kwargs.get("trust_cal"), len(probs_cal), "trust_cal")
        probs_cal = np.asarray(probs_cal)
        labels_cal = np.asarray(labels_cal).astype(int)
        confidences = np.max(probs_cal, axis=1)
        scores = 1.0 - probs_cal[np.arange(len(labels_cal)), labels_cal]
        self.global_threshold_ = conformal_quantile(scores, self.alpha)

        assignments, self.conf_edges_, self.trust_edges_ = create_2d_bins(
            confidences,
            trust_cal,
            self.num_conf_bins,
            self.num_trust_bins,
            self.binning_method,
        )
        n_bins = self.num_conf_bins * self.num_trust_bins
        self.bin_thresholds_ = {}
        for bin_idx in range(n_bins):
            bin_scores = scores[assignments == bin_idx]
            if len(bin_scores) >= self.min_bin_size:
                self.bin_thresholds_[bin_idx] = conformal_quantile(bin_scores, self.alpha)
            else:
                self.bin_thresholds_[bin_idx] = self.global_threshold_
        return self

    def predict_sets(self, probs_test: np.ndarray, **kwargs) -> list[list[int]]:
        if self.bin_thresholds_ is None or self.conf_edges_ is None or self.trust_edges_ is None:
            raise RuntimeError("ConfTrustNaiveCP must be fitted before prediction")

        probs_test = np.asarray(probs_test)
        trust_test = _require_vector(kwargs.get("trust_test"), len(probs_test), "trust_test")
        confidences = np.max(probs_test, axis=1)

        prediction_sets: list[list[int]] = []
        for row, conf, trust in zip(probs_test, confidences, trust_test):
            threshold = self._threshold_for_sample(float(conf), float(trust))
            pred_set = np.where(row >= (1.0 - threshold))[0].astype(int).tolist()
            if not pred_set:
                pred_set = [int(np.argmax(row))]
            prediction_sets.append(pred_set)
        return prediction_sets

    def _threshold_for_sample(self, confidence: float, trust: float) -> float:
        conf_bin = int(np.digitize([confidence], self.conf_edges_[1:-1])[0])
        trust_bin = int(np.digitize([trust], self.trust_edges_[1:-1])[0])
        conf_bin = min(max(conf_bin, 0), self.num_conf_bins - 1)
        trust_bin = min(max(trust_bin, 0), self.num_trust_bins - 1)
        bin_idx = conf_bin * self.num_trust_bins + trust_bin
        return float(self.bin_thresholds_.get(bin_idx, self.global_threshold_))

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": self.name,
            "alpha": self.alpha,
            "num_conf_bins": self.num_conf_bins,
            "num_trust_bins": self.num_trust_bins,
            "binning_method": self.binning_method,
            "min_bin_size": self.min_bin_size,
            "global_threshold": self.global_threshold_,
            "conf_edges": self.conf_edges_.tolist() if self.conf_edges_ is not None else None,
            "trust_edges": self.trust_edges_.tolist() if self.trust_edges_ is not None else None,
            "bin_thresholds": self.bin_thresholds_,
        }


class ConfTrustConditionalCP(ConformalMethod):
    """Polynomial quantile model over confidence and trust score."""

    name = "conf_trust_conditional"
    requires_trust_scores = True

    def __init__(
        self,
        alpha: float = 0.10,
        degree: int = 3,
        learning_rate: float = 0.05,
        n_epochs: int = 1000,
        l2_reg: float = 1e-3,
    ):
        self.alpha = alpha
        self.degree = degree
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.l2_reg = l2_reg
        self.theta_: np.ndarray | None = None
        self.conf_mean_: float | None = None
        self.conf_std_: float | None = None
        self.trust_mean_: float | None = None
        self.trust_std_: float | None = None
        self.global_threshold_: float | None = None
        self.best_loss_: float | None = None

    def fit(self, probs_cal: np.ndarray, labels_cal: np.ndarray, **kwargs) -> "ConfTrustConditionalCP":
        trust_cal = _require_vector(kwargs.get("trust_cal"), len(probs_cal), "trust_cal")
        probs_cal = np.asarray(probs_cal)
        labels_cal = np.asarray(labels_cal).astype(int)
        confidences = np.max(probs_cal, axis=1)
        scores = 1.0 - probs_cal[np.arange(len(labels_cal)), labels_cal]
        self.global_threshold_ = conformal_quantile(scores, self.alpha)

        conf_norm, trust_norm = self._fit_normalization(confidences, trust_cal)
        phi = _polynomial_features(conf_norm, trust_norm, self.degree)
        tau = 1.0 - self.alpha
        theta = np.zeros(phi.shape[1], dtype=float)
        best_theta = theta.copy()
        best_loss = float("inf")

        for _ in range(self.n_epochs):
            predictions = phi @ theta
            residuals = scores - predictions
            weights = np.where(residuals >= 0.0, -tau, 1.0 - tau)
            gradient = np.mean(phi * weights[:, None], axis=0) + 2.0 * self.l2_reg * theta
            theta = theta - self.learning_rate * gradient
            loss = _pinball_loss(predictions, scores, tau) + self.l2_reg * float(np.sum(theta * theta))
            if loss < best_loss:
                best_loss = loss
                best_theta = theta.copy()

        self.theta_ = best_theta
        self.best_loss_ = best_loss
        return self

    def predict_sets(self, probs_test: np.ndarray, **kwargs) -> list[list[int]]:
        if self.theta_ is None:
            raise RuntimeError("ConfTrustConditionalCP must be fitted before prediction")

        probs_test = np.asarray(probs_test)
        trust_test = _require_vector(kwargs.get("trust_test"), len(probs_test), "trust_test")
        confidences = np.max(probs_test, axis=1)
        conf_norm, trust_norm = self._normalize(confidences, trust_test)
        phi = _polynomial_features(conf_norm, trust_norm, self.degree)
        thresholds = np.clip(phi @ self.theta_, 0.0, 1.0)

        prediction_sets: list[list[int]] = []
        for row, threshold in zip(probs_test, thresholds):
            pred_set = np.where(row >= (1.0 - threshold))[0].astype(int).tolist()
            if not pred_set:
                pred_set = [int(np.argmax(row))]
            prediction_sets.append(pred_set)
        return prediction_sets

    def _fit_normalization(self, confidence: np.ndarray, trust: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.conf_mean_ = float(np.mean(confidence))
        self.conf_std_ = float(np.std(confidence))
        self.trust_mean_ = float(np.mean(trust))
        self.trust_std_ = float(np.std(trust))
        return self._normalize(confidence, trust)

    def _normalize(self, confidence: np.ndarray, trust: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        conf = (confidence - self.conf_mean_) / ((self.conf_std_ or 0.0) + 1e-8)
        trust_norm = (trust - self.trust_mean_) / ((self.trust_std_ or 0.0) + 1e-8)
        return conf, trust_norm

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": self.name,
            "alpha": self.alpha,
            "degree": self.degree,
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "l2_reg": self.l2_reg,
            "global_threshold": self.global_threshold_,
            "best_loss": self.best_loss_,
            "theta": self.theta_.tolist() if self.theta_ is not None else None,
        }


class ConfTrustCP(ConfTrustNaiveCP):
    """Backward-compatible alias for the naive Conf-Trust baseline."""

    name = "conf_trust_cp"


def _require_vector(values: object, expected_len: int, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} is required for Conf-Trust methods")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if len(array) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}")
    return array


def _polynomial_features(conf: np.ndarray, trust: np.ndarray, degree: int) -> np.ndarray:
    features = []
    for total_degree in range(degree + 1):
        for conf_power in range(total_degree + 1):
            trust_power = total_degree - conf_power
            features.append((conf ** conf_power) * (trust ** trust_power))
    return np.column_stack(features)


def _pinball_loss(predictions: np.ndarray, targets: np.ndarray, tau: float) -> float:
    residuals = targets - predictions
    losses = np.where(residuals >= 0.0, tau * residuals, (tau - 1.0) * residuals)
    return float(np.mean(losses))
