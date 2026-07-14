import numpy as np

from conformal_audit.methods.aps import APS
from conformal_audit.methods.clustered import ClusteredCP
from conformal_audit.methods.mondrian import MondrianCP
from conformal_audit.methods.raps import RAPS
from conformal_audit.methods.rc3p import RC3P
from conformal_audit.methods.standard import StandardCP


def test_standard_cp_returns_one_set_per_sample():
    cal_probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    cal_labels = np.array([0, 1, 0])
    test_probs = np.array([[0.7, 0.3], [0.4, 0.6]])
    sets = StandardCP(alpha=0.2).fit(cal_probs, cal_labels).predict_sets(test_probs)
    assert len(sets) == 2
    assert all(len(pred_set) >= 1 for pred_set in sets)


def test_aps_returns_one_set_per_sample():
    cal_probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    cal_labels = np.array([0, 1, 0])
    test_probs = np.array([[0.7, 0.3], [0.4, 0.6]])
    sets = APS(alpha=0.2).fit(cal_probs, cal_labels).predict_sets(test_probs)
    assert len(sets) == 2
    assert all(len(pred_set) >= 1 for pred_set in sets)


def test_mondrian_returns_one_set_per_sample():
    cal_probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    cal_labels = np.array([0, 1, 0, 1])
    test_probs = np.array([[0.7, 0.3], [0.4, 0.6]])
    sets = MondrianCP(alpha=0.2).fit(cal_probs, cal_labels).predict_sets(test_probs)
    assert len(sets) == 2
    assert all(len(pred_set) >= 1 for pred_set in sets)


def test_raps_returns_one_set_per_sample():
    cal_probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    cal_labels = np.array([0, 1, 0, 1])
    test_probs = np.array([[0.7, 0.3], [0.4, 0.6]])
    sets = RAPS(alpha=0.2, lambda_reg=0.01, k_reg=1).fit(cal_probs, cal_labels).predict_sets(test_probs)
    assert len(sets) == 2
    assert all(len(pred_set) >= 1 for pred_set in sets)


def test_clustered_cp_returns_one_set_per_sample():
    cal_probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    cal_labels = np.array([0, 1, 0, 1])
    test_probs = np.array([[0.7, 0.3], [0.4, 0.6]])
    sets = ClusteredCP(alpha=0.2, n_clusters=2).fit(cal_probs, cal_labels).predict_sets(test_probs)
    assert len(sets) == 2
    assert all(len(pred_set) >= 1 for pred_set in sets)


def test_rc3p_substitute_returns_one_set_per_sample():
    cal_probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    cal_labels = np.array([0, 1, 0, 1])
    test_probs = np.array([[0.7, 0.3], [0.4, 0.6]])
    method = RC3P(alpha=0.2, n_clusters=2).fit(cal_probs, cal_labels)
    sets = method.predict_sets(test_probs)
    assert len(sets) == 2
    assert "substitute" in method.diagnostics()["implementation_note"]
