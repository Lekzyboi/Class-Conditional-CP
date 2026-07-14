"""Frequency-grouped class-conditional conformal prediction."""

from __future__ import annotations

import numpy as np

from conformal_audit.methods.base import ConformalMethod
from conformal_audit.utils.quantiles import conformal_quantile


class FrequencyGroupedCP(ConformalMethod):
    name = "frequency_grouped_cp"

    def __init__(
        self,
        alpha: float = 0.10,
        n_classes: int | None = None,
        n_groups: int = 3,
        min_group_size: int = 1,
    ):
        self.alpha = alpha
        self.n_classes = n_classes
        self.n_groups = n_groups
        self.min_group_size = min_group_size
        self.class_to_group_: dict[int, int] | None = None
        self.group_thresholds_: dict[int, float] | None = None
        self.class_counts_: list[int] | None = None
        self.global_threshold_: float | None = None

    def fit(self, probs_cal: np.ndarray, labels_cal: np.ndarray, **kwargs) -> "FrequencyGroupedCP":
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

        groups = self._build_frequency_groups(class_counts)
        class_to_group: dict[int, int] = {}
        group_thresholds: dict[int, float] = {}

        for group_idx, class_indices in enumerate(groups):
            for class_idx in class_indices:
                class_to_group[int(class_idx)] = group_idx
            mask = np.isin(labels_cal, class_indices)
            group_scores = scores[mask]
            if len(group_scores) >= self.min_group_size:
                group_thresholds[group_idx] = conformal_quantile(group_scores, self.alpha)
            else:
                group_thresholds[group_idx] = self.global_threshold_

        self.n_classes = n_classes
        self.class_to_group_ = class_to_group
        self.group_thresholds_ = group_thresholds
        return self

    def predict_sets(self, probs_test: np.ndarray) -> list[list[int]]:
        if self.class_to_group_ is None or self.group_thresholds_ is None:
            raise RuntimeError("FrequencyGroupedCP must be fitted before prediction")

        prediction_sets: list[list[int]] = []
        for row in np.asarray(probs_test):
            pred_set: list[int] = []
            for class_idx in range(len(row)):
                group_idx = self.class_to_group_.get(class_idx)
                threshold = self.group_thresholds_.get(group_idx, self.global_threshold_)
                if threshold is not None and row[class_idx] >= (1.0 - threshold):
                    pred_set.append(class_idx)
            if not pred_set:
                pred_set = [int(np.argmax(row))]
            prediction_sets.append(pred_set)
        return prediction_sets

    def _build_frequency_groups(self, class_counts: np.ndarray) -> list[np.ndarray]:
        n_classes = len(class_counts)
        n_groups = min(max(1, self.n_groups), n_classes)
        sorted_classes = np.argsort(class_counts)
        return [group.astype(int) for group in np.array_split(sorted_classes, n_groups) if len(group) > 0]

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": self.name,
            "alpha": self.alpha,
            "n_classes": self.n_classes,
            "n_groups": self.n_groups,
            "class_counts": self.class_counts_,
            "class_to_group": self.class_to_group_,
            "group_thresholds": self.group_thresholds_,
            "global_threshold": self.global_threshold_,
        }
