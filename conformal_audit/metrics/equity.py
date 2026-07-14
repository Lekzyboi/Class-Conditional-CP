"""Coverage-equity metrics."""

from __future__ import annotations

from collections.abc import Sequence


def coverage_equity_gap(
    per_class: dict[str, dict[str, float | int]],
    high_prevalence_classes: Sequence[str],
    low_prevalence_classes: Sequence[str],
) -> float:
    """Compute macro high-minus-low prevalence coverage gap."""

    high = [
        float(per_class[class_name]["coverage_rate"])
        for class_name in high_prevalence_classes
        if int(per_class[class_name]["count"]) > 0
    ]
    low = [
        float(per_class[class_name]["coverage_rate"])
        for class_name in low_prevalence_classes
        if int(per_class[class_name]["count"]) > 0
    ]
    if not high or not low:
        return 0.0
    return sum(high) / len(high) - sum(low) / len(low)

