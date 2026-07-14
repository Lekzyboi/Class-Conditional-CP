"""Reporting helpers for tables, figures, and manuscript checks."""

from .figures import generate_figures_from_results, plot_per_class_coverage
from .tables import generate_tables_from_results, write_comparison_table

__all__ = [
    "generate_figures_from_results",
    "generate_tables_from_results",
    "plot_per_class_coverage",
    "write_comparison_table",
]
