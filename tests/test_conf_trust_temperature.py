import numpy as np

from conformal_audit.methods.conf_trust import ConfTrustConditionalCP, ConfTrustNaiveCP
from conformal_audit.methods.temperature_scaling import (
    TemperatureScaler,
    expected_calibration_error,
    negative_log_likelihood,
    softmax,
)


def test_conf_trust_naive_returns_prediction_sets():
    cal_probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.65, 0.35], [0.35, 0.65]])
    cal_labels = np.array([0, 1, 0, 1])
    cal_trust = np.array([0.8, 0.7, 0.6, 0.5])
    test_probs = np.array([[0.7, 0.3], [0.45, 0.55]])
    test_trust = np.array([0.75, 0.55])
    sets = (
        ConfTrustNaiveCP(alpha=0.2, num_conf_bins=2, num_trust_bins=2, min_bin_size=1)
        .fit(cal_probs, cal_labels, trust_cal=cal_trust)
        .predict_sets(test_probs, trust_test=test_trust)
    )
    assert len(sets) == 2
    assert all(len(pred_set) >= 1 for pred_set in sets)


def test_conf_trust_conditional_returns_prediction_sets():
    cal_probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.65, 0.35], [0.35, 0.65]])
    cal_labels = np.array([0, 1, 0, 1])
    cal_trust = np.array([0.8, 0.7, 0.6, 0.5])
    test_probs = np.array([[0.7, 0.3], [0.45, 0.55]])
    test_trust = np.array([0.75, 0.55])
    sets = (
        ConfTrustConditionalCP(alpha=0.2, degree=1, n_epochs=20)
        .fit(cal_probs, cal_labels, trust_cal=cal_trust)
        .predict_sets(test_probs, trust_test=test_trust)
    )
    assert len(sets) == 2
    assert all(len(pred_set) >= 1 for pred_set in sets)


def test_temperature_scaling_outputs_probabilities():
    logits = np.array([[3.0, 1.0], [0.5, 2.0], [1.5, 0.2]])
    labels = np.array([0, 1, 0])
    scaler = TemperatureScaler(temperatures=np.array([0.5, 1.0, 2.0])).fit(logits, labels)
    probs = scaler.transform(logits)
    assert probs.shape == logits.shape
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert negative_log_likelihood(probs, labels) >= 0.0
    assert expected_calibration_error(softmax(logits), labels, n_bins=2) >= 0.0
