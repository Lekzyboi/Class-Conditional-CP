import numpy as np

from conformal_audit.metrics.coverage import average_set_size, coverage, per_class_coverage


def test_coverage_counts_labels_inside_sets():
    pred_sets = [[0], [0, 1], [2]]
    labels = np.array([0, 1, 1])
    assert coverage(pred_sets, labels) == 2 / 3


def test_average_set_size():
    assert average_set_size([[0], [0, 1], [0, 1, 2]]) == 2.0


def test_per_class_coverage():
    pred_sets = [[0], [0], [1, 2], [2]]
    labels = np.array([0, 1, 1, 2])
    out = per_class_coverage(pred_sets, labels, ["A", "B", "C"])
    assert out["A"]["coverage_rate"] == 1.0
    assert out["B"]["coverage_rate"] == 0.5
    assert out["C"]["coverage_rate"] == 1.0

