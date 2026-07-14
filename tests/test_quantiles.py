import math

import numpy as np

from conformal_audit.utils.quantiles import (
    conformal_quantile,
    conformal_quantile_index,
    conformal_quantile_level,
    conformal_quantile_rank,
)


def test_conformal_quantile_level_is_capped_at_one():
    assert conformal_quantile_level(5, 0.01) == 1.0


def test_conformal_quantile_uses_order_statistic():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    alpha = 0.25
    assert conformal_quantile_rank(len(scores), alpha) == 4
    assert conformal_quantile_index(len(scores), alpha) == 3
    assert conformal_quantile(scores, alpha) == 0.4


def test_conformal_quantile_returns_infinity_when_rank_exceeds_sample_size():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    alpha = 0.01
    assert conformal_quantile_rank(len(scores), alpha) == 5
    assert conformal_quantile_index(len(scores), alpha) == -1
    assert math.isinf(conformal_quantile(scores, alpha))
