import numpy as np

from conformal_audit.metrics.bootstrap import bootstrap_ci, summarize_bootstrap_for_result
from conformal_audit.metrics.statistical_tests import (
    adjust_p_values,
    binomial_less_p_value,
    per_class_below_target_tests,
)


def test_bootstrap_ci_returns_ordered_interval():
    out = bootstrap_ci(np.array([0, 1, 1, 1]), n_bootstrap=50, seed=1)
    assert out["ci_low"] <= out["estimate"] <= out["ci_high"]


def test_bootstrap_prediction_summary_contains_equity_gap():
    pred_sets = [[0], [1], [0], [1]]
    labels = np.array([0, 1, 1, 1])
    out = summarize_bootstrap_for_result(
        pred_sets,
        labels,
        class_names=["A", "B"],
        high_prevalence_classes=["B"],
        low_prevalence_classes=["A"],
        n_bootstrap=20,
    )
    assert "coverage" in out
    assert "coverage_equity_gap" in out


def test_binomial_less_p_value_range():
    p_value = binomial_less_p_value(successes=7, trials=10, target_probability=0.9)
    assert 0.0 <= p_value <= 1.0


def test_binomial_less_p_value_handles_large_counts():
    p_value = binomial_less_p_value(successes=1200, trials=1272, target_probability=0.9)
    assert 0.0 <= p_value <= 1.0


def test_adjust_p_values_holm_monotone_bounds():
    adjusted = adjust_p_values([0.01, 0.04, 0.20], method="holm")
    assert np.all(adjusted >= np.array([0.01, 0.04, 0.20]))
    assert np.all(adjusted <= 1.0)


def test_per_class_below_target_tests_adds_adjusted_p_values():
    per_class = {
        "A": {"count": 10, "covered": 7, "coverage_rate": 0.7, "avg_set_size": 1.0},
        "B": {"count": 10, "covered": 10, "coverage_rate": 1.0, "avg_set_size": 1.0},
    }
    rows = per_class_below_target_tests(per_class, target_coverage=0.9)
    assert rows[0]["below_target"] is True
    assert "adjusted_p_value" in rows[0]
