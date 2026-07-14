from pathlib import Path
from tempfile import TemporaryDirectory

from conformal_audit.reporting.figures import generate_figures_from_results
from conformal_audit.utils.io import write_json


def test_generate_figures_from_results_writes_pngs():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        results_path = root / "results.json"
        write_json(
            results_path,
            {
                "comparison_table": [
                    {
                        "Method": "standard_cp",
                        "alpha": 0.1,
                        "Coverage": 0.9,
                        "Avg Size": 2.0,
                        "Coverage Equity Gap": 0.1,
                    },
                    {
                        "Method": "aps",
                        "alpha": 0.1,
                        "Coverage": 0.95,
                        "Avg Size": 3.0,
                        "Coverage Equity Gap": 0.05,
                    },
                ],
                "detailed_results": [
                    {
                        "Method": "standard_cp",
                        "alpha": 0.1,
                        "per_class": {
                            "A": {"coverage_rate": 0.8},
                            "B": {"coverage_rate": 1.0},
                        },
                    },
                    {
                        "Method": "aps",
                        "alpha": 0.1,
                        "per_class": {
                            "A": {"coverage_rate": 0.9},
                            "B": {"coverage_rate": 1.0},
                        },
                    },
                ],
            },
        )
        manifest = generate_figures_from_results(results_path)
        assert Path(manifest["generated_files"]["coverage_by_alpha"]).exists()
        assert Path(manifest["generated_files"]["average_set_size_by_alpha"]).exists()
        assert Path(manifest["generated_files"]["per_class_coverage_primary_alpha"]).exists()
