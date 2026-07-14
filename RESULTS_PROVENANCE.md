# Results Provenance

## Purpose

This document maps the current generated result artifacts to the scripts that produced them and to the framework outputs that must replace them.

The new CMPB framework must reproduce the current baseline outputs before any legacy script is archived.

## Main Output Folders

| Output folder | Current role | Source script(s) | New-framework target |
|---|---|---|---|
| `assets/output/paper1_conformal_fairness/` | Main ISIC 2019 three-method conformal/fairness results | `paper1.py` | `conformal_audit run --config configs/isic_resnet50_baseline.yaml` |
| `assets/output/conf_trust_experiments/` | ISIC 2019 five-method Conf-Trust experiments | `conf_trust_cp.py` | `conformal_audit run --config configs/isic_conf_trust.yaml` |
| `assets/output/paper1_bcn20000_validation/` | BCN20000 three-method sensitivity results | `validation.py` | `conformal_audit run --config configs/bcn_sensitivity.yaml` |
| `assets/output/bcn_distribution_shift/` | BCN/ISIC confidence and nonconformity shift figures | `bcn_distribution_shift.py` | `conformal_audit audit distribution-shift --config configs/bcn_sensitivity.yaml` |
| `assets/output/ablation_studies/` | Ablations for degree, binning, k-NN, method comparisons | `run_ablations.py` | `conformal_audit run --config configs/ablations/*.yaml` |
| `assets/output/test_evaluation/` | Raw test prediction/evaluation artifacts | `scripts/regenerate_test_predictions.py`, `scripts/generate_confusion_matrix.py` | `conformal_audit predict` and `conformal_audit report confusion` |
| `assets/output/xai_visualizations/` | XAI sample images | Historical/unclear from current refactor | Not primary CMPB framework target |

## Current Result Artifacts

### Legacy Single-Method Output

| Artifact | Role | Source |
|---|---|---|
| `assets/output/conformal_prediction_results.json` | Legacy conformal prediction summary | `measure.py` |
| `assets/output/xai_draps_results.json` | Historical XAI/DRAPS result artifact | Needs later classification |

### Main ISIC Three-Method Results

Folder: `assets/output/paper1_conformal_fairness/`

| Artifact | Role |
|---|---|
| `comparison_table.csv` | Main comparison table |
| `comparison_table.tex` | LaTeX comparison table |
| `complete_results.json` | Detailed result store |
| `mondrian_bootstrap_ci.csv` | Mondrian threshold bootstrap intervals |
| `results.json` | Additional result summary |
| `figures/method_comparison.png` | Method-comparison figure |
| `figures/per_class_coverage_analysis.png` | Per-class coverage figure |
| `figures/fairness_gap_trends.png` | Fairness gap trend figure |
| `figures/coverage_equity_gap_trends.png` | Coverage equity gap trend figure |
| `figures/confusion_matrix.png` | Confusion matrix |
| `figures/nonconformity_score_distributions.png` | Score distribution figure |
| `figures/bootstrap_ci_width_vs_sample_size.png` | Bootstrap stability figure |

### ISIC Conf-Trust Experiments

Folder: `assets/output/conf_trust_experiments/`

| Artifact | Role |
|---|---|
| `comparison_table.csv` | Five-method Conf-Trust comparison |
| `complete_results.json` | Detailed five-method results |
| `ablation_degree.csv` | Polynomial-degree ablation |
| `figures/coverage_size_comparison.png` | Coverage and set-size comparison |
| `figures/coverage_equity_gap_comparison.png` | Coverage equity gap figure |
| `figures/covgap_comparison.png` | CovGap comparison figure |
| `figures/fairness_gap_comparison.png` | Fairness-gap comparison figure |
| `figures/per_class_coverage.png` | Per-class coverage figure |

### BCN Sensitivity Results

Folder: `assets/output/paper1_bcn20000_validation/`

| Artifact | Role |
|---|---|
| `bcn20000_complete_results.json` | Detailed BCN results |
| `bcn20000_results.csv` | BCN method comparison |
| `bcn20000_table2_per_class.csv` | BCN per-class coverage table |
| `isic_vs_bcn_comparison.csv` | Cross-dataset comparison |
| `per_class_comparison.csv` | Per-class ISIC/BCN comparison |
| `figures/per_class_coverage_comparison.png` | Per-class coverage comparison |
| `figures/isic_vs_bcn_coverage_equity_gaps.png` | Coverage equity gap comparison |
| `figures/isic_vs_bcn_fairness_gaps.png` | Fairness gap comparison |
| `figures/coverage_equity_gap_robustness_010.png` | Alpha 0.10 coverage gap robustness |
| `figures/fairness_gap_robustness_010.png` | Alpha 0.10 fairness gap robustness |

### BCN Distribution Shift Results

Folder: `assets/output/bcn_distribution_shift/`

| Artifact | Role |
|---|---|
| `bcn_vs_isic_confidence_distribution.png` | Softmax confidence distribution comparison |
| `bcn_vs_isic_nonconformity_scores.png` | Nonconformity score distribution comparison |

### Ablation Results

Folder: `assets/output/ablation_studies/`

| Artifact | Role |
|---|---|
| `ablation_binning_strategy.csv` | Conf-Trust binning strategy ablation |
| `ablation_knn.csv` | Trust-score k-NN ablation |
| `ablation_polynomial_degree.csv` | Polynomial-degree ablation |
| `figures/ablation_degree.png` | Degree ablation figure |
| `figures/mondrian_vs_conf_trust.png` | Mondrian versus Conf-Trust figure |

### Test Evaluation Results

Folder: `assets/output/test_evaluation/`

| Artifact | Role |
|---|---|
| `test_set_results.json` | Test-set evaluation summary |
| `correlation_plot.png` | Correlation plot |
| `per_class_coverage_analysis.png` | Test per-class coverage figure |
| `stratified_coverage.png` | Stratified coverage figure |
| `confusion_analysis/confusion_matrix.png` | Confusion matrix figure |
| `confusion_analysis/confusion_matrix_normalized.png` | Normalized confusion matrix figure |

## Manuscript Figure Dependencies

The current manuscript references bare figure filenames. These must eventually be mapped to actual output paths or copied into an Overleaf-ready figure folder.

| Manuscript include | Current likely source |
|---|---|
| `method_comparison.png` | `assets/output/paper1_conformal_fairness/figures/method_comparison.png` |
| `confusion_matrix.png` | `assets/output/paper1_conformal_fairness/figures/confusion_matrix.png` or `assets/output/test_evaluation/confusion_analysis/confusion_matrix.png` |
| `per_class_coverage_analysis.png` | `assets/output/paper1_conformal_fairness/figures/per_class_coverage_analysis.png` |
| `fairness_gap_trends.png` | `assets/output/paper1_conformal_fairness/figures/fairness_gap_trends.png` |
| `isic_vs_bcn_fairness_gaps.png` | `assets/output/paper1_bcn20000_validation/figures/isic_vs_bcn_fairness_gaps.png` |

## Baseline Reproduction Targets

The new framework must reproduce, at minimum:

1. `paper1_conformal_fairness/comparison_table.csv`
2. `paper1_conformal_fairness/complete_results.json`
3. `paper1_conformal_fairness/mondrian_bootstrap_ci.csv`
4. `conf_trust_experiments/comparison_table.csv`
5. `conf_trust_experiments/complete_results.json`
6. `paper1_bcn20000_validation/bcn20000_results.csv`
7. `paper1_bcn20000_validation/isic_vs_bcn_comparison.csv`
8. `test_evaluation/test_set_results.json`

## Required New Provenance Features

The framework should write a machine-readable provenance block for every run:

```json
{
  "run_id": "isic_resnet50_seed_unknown_alpha010",
  "config_file": "configs/isic_resnet50_baseline.yaml",
  "model_checkpoint": "assets/output/best_model.pth",
  "probabilities_source": "generated",
  "calibration_indices": "splits/isic_cal_seed42.json",
  "test_indices": "splits/isic_test_seed42.json",
  "alpha_values": [0.05, 0.10, 0.20],
  "methods": ["standard_cp", "aps", "mondrian_cp"],
  "created_at": "ISO-8601 timestamp",
  "code_version": "git commit or local hash",
  "data_manifest": "DATA_MANIFEST.md",
  "checkpoint_manifest": "CHECKPOINT_MANIFEST.md"
}
```

## Known Provenance Issues

1. The current scripts are not fully config-driven.
2. Some outputs are generated by monolithic scripts with embedded constants.
3. The root-level `metadata.csv` expected by BCN scripts is missing.
4. Manuscript figure paths are not directly aligned with output paths.
5. The current result values should be automatically checked against manuscript tables before submission.

