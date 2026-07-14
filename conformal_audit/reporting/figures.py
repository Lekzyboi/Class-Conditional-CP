"""Publication-support figure generation for framework results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from conformal_audit.utils.io import read_json, write_json


PUBLICATION_METHOD_ORDER = [
    "Standard CP",
    "APS",
    "RAPS",
    "Mondrian/classwise CP",
    "Frequency-grouped CP",
]

PUBLICATION_METHOD_NAMES = {
    "standard_cp": "Standard CP",
    "aps": "APS",
    "raps": "RAPS",
    "mondrian_cp": "Mondrian/classwise CP",
    "classwise_cp": "Mondrian/classwise CP",
    "frequency_grouped_cp": "Frequency-grouped CP",
}

PUBLICATION_METHOD_PRIORITY = {
    "standard_cp": 0,
    "aps": 1,
    "raps": 2,
    "mondrian_cp": 3,
    "classwise_cp": 4,
    "frequency_grouped_cp": 5,
}


def generate_figures_from_results(results_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Generate core PNG figures from a framework results JSON."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_path = Path(results_path)
    results = read_json(results_path)
    target_dir = Path(output_dir) if output_dir else results_path.parent / "figures"
    target_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = publication_rows(results.get("comparison_table", []))
    detailed_results = publication_rows(results.get("detailed_results", []))
    outputs: dict[str, str] = {}

    if comparison_rows:
        outputs["coverage_by_alpha"] = str(target_dir / "coverage_by_alpha.png")
        outputs["average_set_size_by_alpha"] = str(target_dir / "average_set_size_by_alpha.png")
        outputs["coverage_equity_gap_by_alpha"] = str(target_dir / "coverage_equity_gap_by_alpha.png")
        plot_metric_by_alpha(
            comparison_rows,
            metric="Coverage",
            ylabel="Marginal coverage",
            path=outputs["coverage_by_alpha"],
            target_line=True,
            plt=plt,
        )
        plot_metric_by_alpha(
            comparison_rows,
            metric="Avg Size",
            ylabel="Average prediction-set size",
            path=outputs["average_set_size_by_alpha"],
            target_line=False,
            plt=plt,
        )
        plot_metric_by_alpha(
            comparison_rows,
            metric="Coverage Equity Gap",
            ylabel="Prevalence coverage gap",
            path=outputs["coverage_equity_gap_by_alpha"],
            target_line=False,
            plt=plt,
        )

        primary_alpha = closest_alpha(comparison_rows, 0.10)
        outputs["method_tradeoff_primary_alpha"] = str(target_dir / f"method_tradeoff_alpha_{format_alpha(primary_alpha)}.png")
        plot_method_tradeoff(comparison_rows, primary_alpha, outputs["method_tradeoff_primary_alpha"], plt)

    if detailed_results:
        primary_alpha = closest_alpha(detailed_results, 0.10)
        outputs["per_class_coverage_primary_alpha"] = str(
            target_dir / f"per_class_coverage_alpha_{format_alpha(primary_alpha)}.png"
        )
        plot_per_class_coverage(detailed_results, primary_alpha, outputs["per_class_coverage_primary_alpha"], plt)

    manifest = {
        "source_results": str(results_path),
        "output_dir": str(target_dir),
        "generated_files": outputs,
    }
    write_json(target_dir / "figure_manifest.json", manifest)
    return manifest


def plot_metric_by_alpha(
    rows: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    path: str | Path,
    target_line: bool,
    plt,
) -> None:
    by_method = group_rows_by_method(rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for method, method_rows in by_method.items():
        ordered = sorted(method_rows, key=lambda row: float(row["alpha"]))
        alphas = [float(row["alpha"]) for row in ordered]
        values = [float(row[metric]) for row in ordered]
        ax.plot(alphas, values, marker="o", linewidth=1.8, label=method)
    if target_line:
        alpha_values = sorted({float(row["alpha"]) for row in rows})
        ax.plot(alpha_values, [1.0 - alpha for alpha in alpha_values], linestyle="--", color="black", label="Target")
    ax.set_xlabel("Miscoverage level alpha")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_per_class_coverage(
    detailed_results: list[dict[str, Any]],
    alpha: float,
    path: str | Path,
    plt=None,
) -> None:
    if plt is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    rows = [row for row in publication_rows(detailed_results) if float(row.get("alpha", row.get("Alpha"))) == alpha]
    methods = [str(row.get("Method") or row.get("method")) for row in rows]
    class_names = sorted({class_name for row in rows for class_name in row.get("per_class", {})})
    matrix = np.full((len(methods), len(class_names)), np.nan, dtype=float)
    for i, row in enumerate(rows):
        per_class = row.get("per_class", {})
        for j, class_name in enumerate(class_names):
            stats = per_class.get(class_name, {})
            value = stats.get("coverage_rate", stats.get("coverage"))
            if value is not None:
                matrix[i, j] = float(value)

    fig_width = max(7.0, 0.65 * len(class_names) + 3.0)
    fig_height = max(4.5, 0.42 * len(methods) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    observed_values = matrix[~np.isnan(matrix)]
    color_min = 0.0
    color_max = 1.0
    if observed_values.size:
        color_min = max(0.0, np.floor(float(np.min(observed_values)) / 0.05) * 0.05)
    image = ax.imshow(matrix, aspect="auto", vmin=color_min, vmax=color_max, cmap="viridis")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title(f"Per-class coverage at alpha={alpha:g}")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                color = "white" if matrix[i, j] < color_min + 0.35 * (color_max - color_min) else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Coverage")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_method_tradeoff(rows: list[dict[str, Any]], alpha: float, path: str | Path, plt) -> None:
    subset = [row for row in publication_rows(rows) if float(row["alpha"]) == alpha]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    sizes = np.asarray([float(row["Avg Size"]) for row in subset])
    gaps = np.asarray([float(row["Coverage Equity Gap"]) for row in subset])
    coverages = np.asarray([float(row["Coverage"]) for row in subset])
    scatter = ax.scatter(sizes, gaps, c=coverages, cmap="viridis", s=70, edgecolor="black", linewidth=0.5)
    for row, size, gap in zip(subset, sizes, gaps):
        ax.annotate(str(row["Method"]), (size, gap), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Average prediction-set size")
    ax.set_ylabel("Prevalence coverage gap")
    ax.grid(True, alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Marginal coverage")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def group_rows_by_method(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["Method"]), []).append(row)
    return {method: grouped[method] for method in PUBLICATION_METHOD_ORDER if method in grouped}


def publication_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse method aliases used internally into publication-facing methods."""

    selected: dict[tuple[str, float], dict[str, Any]] = {}
    priorities: dict[tuple[str, float], int] = {}
    passthrough: list[dict[str, Any]] = []

    for row in rows:
        raw_method = str(row.get("Method") or row.get("method"))
        display_method = PUBLICATION_METHOD_NAMES.get(raw_method)
        if display_method is None:
            passthrough.append(dict(row))
            continue

        alpha_value = float(row.get("alpha", row.get("Alpha")))
        key = (display_method, alpha_value)
        priority = PUBLICATION_METHOD_PRIORITY.get(raw_method, 999)
        if key not in selected or priority < priorities[key]:
            normalized = dict(row)
            normalized["Method"] = display_method
            selected[key] = normalized
            priorities[key] = priority

    ordered_rows = sorted(
        selected.values(),
        key=lambda row: (
            float(row.get("alpha", row.get("Alpha"))),
            PUBLICATION_METHOD_ORDER.index(str(row["Method"]))
            if str(row["Method"]) in PUBLICATION_METHOD_ORDER
            else len(PUBLICATION_METHOD_ORDER),
        ),
    )
    return ordered_rows + passthrough


def closest_alpha(rows: list[dict[str, Any]], preferred: float) -> float:
    alpha_values = sorted({float(row.get("alpha", row.get("Alpha"))) for row in rows})
    return min(alpha_values, key=lambda value: abs(value - preferred))


def format_alpha(alpha: float) -> str:
    return str(alpha).replace(".", "_")
