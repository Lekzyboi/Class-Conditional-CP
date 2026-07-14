# Checkpoint Manifest

## Purpose

This manifest records the model checkpoints currently present in the repository. It exists so the new CMPB framework can reproduce current results before any legacy scripts are retired.

No checkpoint should be deleted, renamed, or overwritten without explicit approval.

## Checkpoint Inventory

| Path | Approx. size | Last modified | Known role |
|---|---:|---|---|
| `assets/output/best_model.pth` | 179.8 MB | 2025-11-27 19:33:40 | Main ResNet-50 checkpoint used by current manuscript analyses |
| `assets/output/final_model.pth` | 90.0 MB | 2025-11-27 19:33:40 | Final model save from older training flow |
| `assets/output/checkpoint_epoch_4.pth` | 179.8 MB | 2025-11-27 15:24:08 | Training checkpoint |
| `assets/output/checkpoint_epoch_9.pth` | 179.8 MB | 2025-11-27 16:11:54 | Training checkpoint |
| `assets/output/checkpoint_epoch_14.pth` | 179.8 MB | 2025-11-27 17:00:44 | Training checkpoint |
| `assets/output/checkpoint_epoch_19.pth` | 179.8 MB | 2025-11-27 17:53:12 | Training checkpoint |
| `assets/output/checkpoint_epoch_24.pth` | 179.8 MB | 2025-11-27 18:46:57 | Training checkpoint |
| `assets/output/efficientnet_b3_8ep.pth` | 41.4 MB | 2026-03-30 23:15:17 | EfficientNet-B3 sensitivity/check architecture checkpoint |

## Current Checkpoint Usage

| Checkpoint | Used by |
|---|---|
| `assets/output/best_model.pth` | `measure.py`, `paper1.py`, `conf_trust_cp.py`, `validation.py`, `conf_trust_bcn_validation.py`, `bcn_distribution_shift.py`, `scripts/regenerate_test_predictions.py`, `run_ablations.py` |
| `assets/output/efficientnet_b3_8ep.pth` | `train_efficientnet_cp.py` output and architecture-sensitivity comparison |
| `assets/output/checkpoint_epoch_*.pth` | Training history/reference |
| `assets/output/final_model.pth` | Older final model artifact |

## Required Framework Checkpoint Metadata

The framework should eventually generate a structured checkpoint metadata file for each model:

```yaml
model_id: resnet50_skincon_seed_unknown
architecture: resnet50
num_classes: 8
training_data: assets/ISIC_2019/train
validation_data: assets/ISIC_2019/val
soft_label_file: assets/SkinCON.csv
training_seed: unknown
checkpoint_path: assets/output/best_model.pth
validation_accuracy: 0.8725
notes: Main checkpoint used by current manuscript.
```

## Known Issues

1. The main ResNet-50 training seed is not documented.
2. The exact code state that produced `best_model.pth` is not fully tracked.
3. EfficientNet-B3 comparison is not training-matched to ResNet-50.
4. Future CMPB experiments should use multiple independently seeded checkpoints.
5. Checkpoints should be archived separately from the code repository for public release.

## Minimum CMPB Requirement

For submission, the reproducibility package should include:

- Checkpoint download links or DOI archive.
- Model architecture and class-index mapping.
- Training configuration.
- Training seed.
- Validation metrics.
- Dataset version and split indices.
- Hashes for released checkpoint files.

