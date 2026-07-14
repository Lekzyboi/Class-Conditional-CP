from pathlib import Path
from tempfile import TemporaryDirectory

from conformal_audit.reporting.tables import generate_tables_from_results
from conformal_audit.utils.io import write_json


def test_generate_tables_from_results_writes_csv_and_latex():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        results_path = root / "results.json"
        write_json(
            results_path,
            {
                "comparison_table": [
                    {"Method": "standard_cp", "alpha": 0.1, "Coverage": 0.9, "Avg Size": 2.0}
                ],
                "detailed_results": [
                    {
                        "Method": "standard_cp",
                        "alpha": 0.1,
                        "target_coverage": 0.9,
                        "per_class": {
                            "A": {"count": 2, "covered": 2, "coverage_rate": 1.0, "avg_set_size": 1.5}
                        },
                        "per_class_tests": [
                            {
                                "class_name": "A",
                                "count": 2,
                                "covered": 2,
                                "coverage_rate": 1.0,
                                "adjusted_p_value": 1.0,
                            }
                        ],
                        "bootstrap": {
                            "coverage": {
                                "estimate": 0.9,
                                "ci_low": 0.8,
                                "ci_high": 1.0,
                                "confidence": 0.95,
                                "n_bootstrap": 20,
                            }
                        },
                    }
                ],
            },
        )
        manifest = generate_tables_from_results(results_path)
        assert Path(manifest["generated_files"]["comparison_csv"]).exists()
        assert Path(manifest["generated_files"]["comparison_tex"]).exists()
        assert Path(manifest["generated_files"]["per_class_csv"]).exists()
        assert Path(manifest["generated_files"]["class_tests_csv"]).exists()
        assert Path(manifest["generated_files"]["bootstrap_csv"]).exists()
