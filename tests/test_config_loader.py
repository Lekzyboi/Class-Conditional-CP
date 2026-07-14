from conformal_audit.config import load_experiment_config


def test_load_baseline_yaml_config():
    config = load_experiment_config("configs/isic_resnet50_baseline.yaml")
    assert config.name == "isic_resnet50_baseline"
    assert config.alpha_values == (0.05, 0.10, 0.20)
    assert "source_results_json" in config.paths
