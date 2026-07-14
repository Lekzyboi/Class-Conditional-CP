from conformal_audit.config import load_experiment_config
from conformal_audit.evaluation import run_experiment


def test_summary_reproduction_returns_existing_rows():
    config = load_experiment_config("configs/isic_resnet50_baseline.yaml")
    result = run_experiment(config)
    assert result["mode"] == "existing_summary"
    assert len(result["comparison_table"]) == 15
    assert result["comparison_table"][0]["Method"] == "standard_cp"
    assert result["comparison_table"][4]["Method"] == "frequency_grouped_cp"
