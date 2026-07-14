"""CSV and LaTeX table generation for framework results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from conformal_audit.utils.io import read_json, write_json


def generate_tables_from_results(results_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Generate publication-support tables from a framework results JSON."""

    results_path = Path(results_path)
    results = read_json(results_path)
    target_dir = Path(output_dir) if output_dir else results_path.parent / "tables"
    target_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = normalize_rows(results.get("comparison_table", []))
    per_class_rows = build_per_class_rows(results.get("detailed_results", []))
    class_test_rows = build_class_test_rows(results.get("detailed_results", []))
    bootstrap_rows = build_bootstrap_rows(results.get("detailed_results", []))

    outputs: dict[str, str] = {}
    if comparison_rows:
        outputs["comparison_csv"] = str(target_dir / "comparison_table.csv")
        outputs["comparison_tex"] = str(target_dir / "comparison_table.tex")
        write_csv(outputs["comparison_csv"], comparison_rows)
        write_latex_table(
            outputs["comparison_tex"],
            comparison_rows,
            caption="Method comparison summary.",
            label="tab:method_comparison",
        )
    if per_class_rows:
        outputs["per_class_csv"] = str(target_dir / "per_class_coverage.csv")
        outputs["per_class_tex"] = str(target_dir / "per_class_coverage.tex")
        write_csv(outputs["per_class_csv"], per_class_rows)
        write_latex_table(
            outputs["per_class_tex"],
            per_class_rows,
            caption="Per-class conformal coverage.",
            label="tab:per_class_coverage",
        )
    if class_test_rows:
        outputs["class_tests_csv"] = str(target_dir / "per_class_below_target_tests.csv")
        outputs["class_tests_tex"] = str(target_dir / "per_class_below_target_tests.tex")
        write_csv(outputs["class_tests_csv"], class_test_rows)
        write_latex_table(
            outputs["class_tests_tex"],
            class_test_rows,
            caption="Multiplicity-adjusted per-class below-target coverage tests.",
            label="tab:per_class_tests",
        )
    if bootstrap_rows:
        outputs["bootstrap_csv"] = str(target_dir / "bootstrap_intervals.csv")
        outputs["bootstrap_tex"] = str(target_dir / "bootstrap_intervals.tex")
        write_csv(outputs["bootstrap_csv"], bootstrap_rows)
        write_latex_table(
            outputs["bootstrap_tex"],
            bootstrap_rows,
            caption="Bootstrap uncertainty intervals.",
            label="tab:bootstrap_intervals",
        )

    manifest = {
        "source_results": str(results_path),
        "output_dir": str(target_dir),
        "generated_files": outputs,
        "row_counts": {
            "comparison": len(comparison_rows),
            "per_class": len(per_class_rows),
            "class_tests": len(class_test_rows),
            "bootstrap": len(bootstrap_rows),
        },
    }
    write_json(target_dir / "table_manifest.json", manifest)
    return manifest


def write_comparison_table(results: dict[str, Any], path: str | Path) -> None:
    """Backward-compatible helper for writing only the comparison table."""

    rows = normalize_rows(results.get("comparison_table", []))
    write_csv(path, rows)


def build_per_class_rows(detailed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in detailed_results:
        method = result.get("Method") or result.get("method")
        alpha = result.get("alpha") or result.get("Alpha")
        target = result.get("target_coverage")
        for class_name, stats in result.get("per_class", {}).items():
            rows.append(
                {
                    "Method": method,
                    "alpha": alpha,
                    "target_coverage": target,
                    "class_name": class_name,
                    "count": stats.get("count"),
                    "covered": stats.get("covered"),
                    "coverage_rate": stats.get("coverage_rate", stats.get("coverage")),
                    "avg_set_size": stats.get("avg_set_size"),
                }
            )
    return normalize_rows(rows)


def build_class_test_rows(detailed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in detailed_results:
        method = result.get("Method") or result.get("method")
        alpha = result.get("alpha") or result.get("Alpha")
        for row in result.get("per_class_tests", []):
            rows.append({"Method": method, "alpha": alpha, **row})
    return normalize_rows(rows)


def build_bootstrap_rows(detailed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in detailed_results:
        method = result.get("Method") or result.get("method")
        alpha = result.get("alpha") or result.get("Alpha")
        for metric, stats in result.get("bootstrap", {}).items():
            rows.append(
                {
                    "Method": method,
                    "alpha": alpha,
                    "metric": metric,
                    "estimate": stats.get("estimate"),
                    "ci_low": stats.get("ci_low"),
                    "ci_high": stats.get("ci_high"),
                    "confidence": stats.get("confidence"),
                    "n_bootstrap": stats.get("n_bootstrap"),
                }
            )
    return normalize_rows(rows)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(
    path: str | Path,
    rows: list[dict[str, Any]],
    caption: str,
    label: str,
    max_rows: int = 40,
) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    displayed_rows = rows[:max_rows]
    alignment = "l" * len(columns)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{escape_latex(caption)}}}",
        f"\\label{{{escape_latex(label)}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\hline",
        " & ".join(escape_latex(column) for column in columns) + " \\\\",
        "\\hline",
    ]
    for row in displayed_rows:
        lines.append(" & ".join(format_latex_value(row.get(column)) for column in columns) + " \\\\")
    if len(rows) > max_rows:
        lines.append(f"\\multicolumn{{{len(columns)}}}{{l}}{{Table truncated to first {max_rows} rows.}} \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: normalize_value(value) for key, value in row.items()} for row in rows]


def normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, bool):
        return str(value)
    if value is None:
        return ""
    return value


def format_latex_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return escape_latex(str("" if value is None else value))


def escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)
