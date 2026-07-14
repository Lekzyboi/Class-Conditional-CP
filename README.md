# Conformal Biomedical CP Framework

This repository contains the code and lightweight reproducibility artifacts for the manuscript:

**A Reproducible Evaluation Framework for Class-Conditional Conformal Prediction in Imbalanced Biomedical Image Classification**

The framework evaluates conformal prediction methods from exported probability arrays and labels. It reports marginal, per-class, prevalence-stratified, and size-stratified coverage, bootstrap confidence intervals, per-class below-target tests, and dataset provenance checks.

## What Is Included

- `conformal_audit/` contains the reusable framework code.
- `configs/` contains configuration files used for the reported framework runs and audits.
- `scripts/` contains export and utility scripts for producing framework-ready arrays from trained checkpoints.
- `tests/` contains unit tests for methods, metrics, reporting, config parsing, and audit utilities.
- `reproducibility_artifacts/` contains lightweight generated tables, figures, JSON summaries, and audit manifests used to support the manuscript.
- `assets/` contains metadata CSV files only. Raw images and model checkpoints are not included.

## What Is Not Included

The public ISIC 2019 and BCN20000 images are not redistributed here. Local reruns use the original dataset releases under the paths recorded in the configuration files. Model checkpoints and exported NumPy arrays are tracked through the Zenodo archive manifest rather than stored in GitHub.

## Environment

Conda environment:

```bash
conda env create -f environment.yml
conda activate conformal-biomedical-cp
```

For lightweight framework tests, a modern Python environment with `numpy`, `scipy`, `pandas`, `matplotlib`, `pillow`, and `pytest` is sufficient. The export scripts require PyTorch and torchvision.

## Quick Checks

Test suite:

```bash
pytest
```

Regenerate tables from an existing result file:

```bash
python -m conformal_audit.cli report tables --results reproducibility_artifacts/isic_resnet50_framework_inputs/results.json
```

Regenerate figures from an existing result file:

```bash
python -m conformal_audit.cli report figures --results reproducibility_artifacts/isic_resnet50_framework_inputs/results.json
```

## Framework Evaluation Command

Framework evaluations use configuration files such as `configs/isic_resnet50_framework_inputs.yaml`:

```bash
python -m conformal_audit.cli run --config configs/isic_resnet50_framework_inputs.yaml
```

The config file records the input arrays, class names, target miscoverage levels, bootstrap settings, method list, and output directory.

## Main Output Tables

The primary ISIC run artifacts are in:

```text
reproducibility_artifacts/isic_resnet50_framework_inputs/
```

The per-class raw and Holm-adjusted p-value table is:

```text
reproducibility_artifacts/isic_resnet50_framework_inputs/tables/per_class_below_target_tests.csv
```

The partial identifier-overlap BCN20000 sensitivity artifacts are in:

```text
reproducibility_artifacts/bcn_kept_overlap_aware_sensitivity/
```

The binary array and checkpoint files associated with the Zenodo release are listed in:

```text
docs/ZENODO_ARCHIVE_MANIFEST.md
```

## Code Availability Statement

The archived release DOI is `10.5281/zenodo.21362039`.

## Citation

Citation metadata are provided in `CITATION.cff`. The archived release is available at https://doi.org/10.5281/zenodo.21362039.
