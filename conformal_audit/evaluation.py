"""Config-driven conformal evaluation runners."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from conformal_audit.config import ExperimentConfig, require_path
from conformal_audit.methods import (
    APS,
    ClasswiseCP,
    ConfTrustConditionalCP,
    ConfTrustCP,
    ConfTrustNaiveCP,
    FrequencyGroupedCP,
    MondrianCP,
    RAPS,
    StandardCP,
)
from conformal_audit.metrics.coverage import average_set_size, coverage, per_class_coverage
from conformal_audit.metrics.equity import coverage_equity_gap
from conformal_audit.metrics.bootstrap import summarize_bootstrap_for_result
from conformal_audit.metrics.statistical_tests import per_class_below_target_tests
from conformal_audit.utils.io import read_json, write_json


METHODS = {
    "standard_cp": StandardCP,
    "standard": StandardCP,
    "aps": APS,
    "raps": RAPS,
    "mondrian_cp": MondrianCP,
    "mondrian": MondrianCP,
    "classwise_cp": ClasswiseCP,
    "classwise": ClasswiseCP,
    "frequency_grouped_cp": FrequencyGroupedCP,
    "frequency_grouped": FrequencyGroupedCP,
    "conf_trust_cp": ConfTrustCP,
    "conf_trust_naive": ConfTrustNaiveCP,
    "conf_trust_conditional": ConfTrustConditionalCP,
}


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Run an experiment from arrays, or summarize existing paper outputs."""

    output_dir = Path(config.paths.get("framework_output_dir", f"assets/output/framework_runs/{config.name}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if _has_probability_arrays(config):
        result = _run_from_probability_arrays(config, output_dir)
    else:
        result = _run_from_existing_summary(config, output_dir)

    write_json(output_dir / "results.json", result)
    _write_comparison_csv(output_dir / "comparison_table.csv", result.get("comparison_table", []))
    return result


def _has_probability_arrays(config: ExperimentConfig) -> bool:
    required = {"calibration_probs", "calibration_labels", "test_probs", "test_labels"}
    return required.issubset(config.paths)


def _run_from_probability_arrays(config: ExperimentConfig, output_dir: Path) -> dict[str, Any]:
    cal_probs = np.load(require_path(config.paths["calibration_probs"], "calibration probabilities"))
    cal_labels = np.load(require_path(config.paths["calibration_labels"], "calibration labels")).astype(int)
    test_probs = np.load(require_path(config.paths["test_probs"], "test probabilities"))
    test_labels = np.load(require_path(config.paths["test_labels"], "test labels")).astype(int)
    cal_trust = _load_optional_array(config, "calibration_trust")
    test_trust = _load_optional_array(config, "test_trust")

    class_names = tuple(config.metadata.get("class_names", [f"class_{i}" for i in range(test_probs.shape[1])]))
    high_classes = tuple(config.metadata.get("high_prevalence_classes", []))
    low_classes = tuple(config.metadata.get("low_prevalence_classes", []))
    fit_kwargs: dict[str, Any] = {}
    if "train_frequencies" in config.metadata:
        fit_kwargs["train_frequencies"] = config.metadata["train_frequencies"]
    n_bootstrap = int(config.metadata.get("n_bootstrap", 200))
    bootstrap_confidence = float(config.metadata.get("bootstrap_confidence", 0.95))
    p_value_adjustment = str(config.metadata.get("p_value_adjustment", "holm"))

    detailed_results: list[dict[str, Any]] = []
    comparison_table: list[dict[str, Any]] = []
    for alpha in config.alpha_values:
        target_coverage = 1.0 - alpha
        for method_name in config.methods:
            method_cls = METHODS.get(method_name)
            if method_cls is None:
                raise ValueError(f"Unknown conformal method in config: {method_name}")
            method = method_cls(alpha=alpha)
            method_fit_kwargs = dict(fit_kwargs)
            predict_kwargs: dict[str, Any] = {}
            if getattr(method, "requires_trust_scores", False):
                if cal_trust is None or test_trust is None:
                    raise ValueError(f"{method.name} requires paths.calibration_trust and paths.test_trust")
                method_fit_kwargs["trust_cal"] = cal_trust
                predict_kwargs["trust_test"] = test_trust
            prediction_sets = method.fit(cal_probs, cal_labels, **method_fit_kwargs).predict_sets(
                test_probs,
                **predict_kwargs,
            )
            per_class = per_class_coverage(prediction_sets, test_labels, class_names)
            equity_gap = coverage_equity_gap(per_class, high_classes, low_classes)
            marginal_coverage = coverage(prediction_sets, test_labels)
            avg_size = average_set_size(prediction_sets)
            failing_classes = [
                class_name
                for class_name, stats in per_class.items()
                if int(stats["count"]) > 0 and float(stats["coverage_rate"]) < target_coverage
            ]
            bootstrap = summarize_bootstrap_for_result(
                prediction_sets,
                test_labels,
                class_names,
                high_classes,
                low_classes,
                n_bootstrap=n_bootstrap,
                confidence=bootstrap_confidence,
                seed=int(config.metadata.get("bootstrap_seed", 42)),
            )
            class_tests = per_class_below_target_tests(per_class, target_coverage, p_value_adjustment)

            row = {
                "Method": method.name,
                "alpha": alpha,
                "Coverage": marginal_coverage,
                "Avg Size": avg_size,
                "Coverage Equity Gap": equity_gap,
                "# Below Target": len(failing_classes),
                "Coverage CI Low": bootstrap["coverage"]["ci_low"],
                "Coverage CI High": bootstrap["coverage"]["ci_high"],
            }
            comparison_table.append(row)
            detailed_results.append(
                {
                    **row,
                    "target_coverage": target_coverage,
                    "failing_classes": failing_classes,
                    "per_class": per_class,
                    "bootstrap": bootstrap,
                    "per_class_tests": class_tests,
                    "diagnostics": method.diagnostics(),
                }
            )

    return {
        "name": config.name,
        "mode": "probability_arrays",
        "output_dir": str(output_dir),
        "comparison_table": comparison_table,
        "detailed_results": detailed_results,
    }


def _load_optional_array(config: ExperimentConfig, key: str) -> np.ndarray | None:
    path = config.paths.get(key)
    if path is None:
        return None
    return np.load(require_path(path, key))


def _run_from_existing_summary(config: ExperimentConfig, output_dir: Path) -> dict[str, Any]:
    source_path = config.paths.get("source_results_json") or config.paths.get("existing_results_json")
    if source_path is None:
        raise ValueError(
            "Config must provide probability array paths or paths.source_results_json for summary reproduction."
        )

    source = read_json(require_path(source_path, "source results JSON"))
    detailed_results = source.get("detailed_results", [])
    comparison_table = source.get("comparison_table", [])

    return {
        "name": config.name,
        "mode": "existing_summary",
        "source_results_json": source_path,
        "output_dir": str(output_dir),
        "comparison_table": comparison_table,
        "detailed_results": detailed_results,
        "note": (
            "This run reproduces the saved paper-result summary. Add calibration/test probability "
            "arrays to the config to recompute conformal sets directly."
        ),
    }


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
