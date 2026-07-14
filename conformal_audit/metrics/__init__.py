"""Metric functions for coverage-equity auditing."""

from .bootstrap import bootstrap_ci, summarize_bootstrap_for_result
from .coverage import coverage, average_set_size, per_class_coverage
from .equity import coverage_equity_gap
from .statistical_tests import adjust_p_values, binomial_less_p_value, per_class_below_target_tests

__all__ = [
    "adjust_p_values",
    "average_set_size",
    "binomial_less_p_value",
    "bootstrap_ci",
    "coverage",
    "coverage_equity_gap",
    "per_class_below_target_tests",
    "per_class_coverage",
    "summarize_bootstrap_for_result",
]
