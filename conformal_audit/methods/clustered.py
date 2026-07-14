"""Frequency-clustered class-conditional conformal prediction."""

from __future__ import annotations

import numpy as np

from conformal_audit.methods.base import ConformalMethod
from conformal_audit.utils.quantiles import conformal_quantile


class ClusteredCP(ConformalMethod):
    name = "clustered_cp"

    def __init__(
        self,
        alpha: float = 0.10,
        n_classes: int | None = None,
        n_clusters: int = 3,
        min_cluster_size: int = 1,
    ):
        self.alpha = alpha
        self.n_classes = n_classes
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.class_to_cluster_: dict[int, int] | None = None
        self.cluster_thresholds_: dict[int, float] | None = None
        self.class_counts_: list[int] | None = None
        self.global_threshold_: float | None = None

    def fit(self, probs_cal: np.ndarray, labels_cal: np.ndarray, **kwargs) -> "ClusteredCP":
        probs_cal = np.asarray(probs_cal)
        labels_cal = np.asarray(labels_cal).astype(int)
        n_classes = self.n_classes or probs_cal.shape[1]
        scores = 1.0 - probs_cal[np.arange(len(labels_cal)), labels_cal]
        self.global_threshold_ = conformal_quantile(scores, self.alpha)

        train_frequencies = kwargs.get("train_frequencies")
        if train_frequencies is None:
            class_counts = np.bincount(labels_cal, minlength=n_classes)
        else:
            class_counts = np.asarray(train_frequencies, dtype=int)
            if len(class_counts) != n_classes:
                raise ValueError("train_frequencies length must match n_classes")
        self.class_counts_ = class_counts.astype(int).tolist()

        clusters = self._build_frequency_clusters(class_counts)
        class_to_cluster: dict[int, int] = {}
        cluster_thresholds: dict[int, float] = {}

        for cluster_idx, class_indices in enumerate(clusters):
            for class_idx in class_indices:
                class_to_cluster[int(class_idx)] = cluster_idx
            mask = np.isin(labels_cal, class_indices)
            cluster_scores = scores[mask]
            if len(cluster_scores) >= self.min_cluster_size:
                cluster_thresholds[cluster_idx] = conformal_quantile(cluster_scores, self.alpha)
            else:
                cluster_thresholds[cluster_idx] = self.global_threshold_

        self.n_classes = n_classes
        self.class_to_cluster_ = class_to_cluster
        self.cluster_thresholds_ = cluster_thresholds
        return self

    def predict_sets(self, probs_test: np.ndarray) -> list[list[int]]:
        if self.class_to_cluster_ is None or self.cluster_thresholds_ is None:
            raise RuntimeError("ClusteredCP must be fitted before prediction")

        prediction_sets: list[list[int]] = []
        for row in np.asarray(probs_test):
            pred_set: list[int] = []
            for class_idx in range(len(row)):
                cluster_idx = self.class_to_cluster_.get(class_idx)
                threshold = self.cluster_thresholds_.get(cluster_idx, self.global_threshold_)
                if threshold is not None and row[class_idx] >= (1.0 - threshold):
                    pred_set.append(class_idx)
            if not pred_set:
                pred_set = [int(np.argmax(row))]
            prediction_sets.append(pred_set)
        return prediction_sets

    def _build_frequency_clusters(self, class_counts: np.ndarray) -> list[np.ndarray]:
        n_classes = len(class_counts)
        n_clusters = min(max(1, self.n_clusters), n_classes)
        sorted_classes = np.argsort(class_counts)
        return [cluster.astype(int) for cluster in np.array_split(sorted_classes, n_clusters) if len(cluster) > 0]

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": self.name,
            "alpha": self.alpha,
            "n_classes": self.n_classes,
            "n_clusters": self.n_clusters,
            "class_counts": self.class_counts_,
            "class_to_cluster": self.class_to_cluster_,
            "cluster_thresholds": self.cluster_thresholds_,
            "global_threshold": self.global_threshold_,
        }
