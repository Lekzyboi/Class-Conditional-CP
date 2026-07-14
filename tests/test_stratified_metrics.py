import numpy as np

from conformal_audit.metrics.stratified import coverage_indicator, size_stratified_coverage, true_label_ranks


def test_coverage_indicator():
    pred_sets = [[0], [0], [1, 2]]
    labels = np.array([0, 1, 2])
    assert coverage_indicator(pred_sets, labels).tolist() == [1, 0, 1]


def test_true_label_ranks():
    probs = np.array([[0.8, 0.2], [0.3, 0.7]])
    labels = np.array([0, 0])
    assert true_label_ranks(probs, labels).tolist() == [1, 2]


def test_size_stratified_coverage():
    pred_sets = [[0], [0], [1, 2]]
    labels = np.array([0, 1, 2])
    violation, stats = size_stratified_coverage(pred_sets, labels, target_coverage=0.9, max_size=2)
    assert "1" in stats
    assert "2+" in stats
    assert violation >= 0

