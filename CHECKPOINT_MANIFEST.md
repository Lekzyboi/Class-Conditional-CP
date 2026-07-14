# Checkpoint Archive Manifest

## Purpose

This manifest describes the model checkpoint used to export the probability arrays for the manuscript analyses. The checkpoint is not included in the GitHub repository because it is a large binary artifact. The archived release is identified by DOI `10.5281/zenodo.21362039`.

## Main Checkpoint

| File for archive | Local source path | Size | SHA-256 | Role |
|---|---|---:|---|---|
| `best_model.pth` | `assets/output/best_model.pth` | 188,563,359 bytes | `B4D483C63A13469067ECAF8EE8C2D46F6CBB90ABD74613955989E8C03AA84512` | ResNet-50 checkpoint used to export the ISIC and BCN probability arrays |

## Model Metadata

| Field | Value |
|---|---|
| Architecture | ResNet-50 |
| Number of classes | 8 |
| Initialization | ImageNet-pretrained weights |
| Training data | Organized ISIC 2019 training and validation folders |
| Soft-label source | `assets/SkinCON.csv` |
| Selected checkpoint | Best validation-accuracy checkpoint from the recorded run |
| Selected epoch | 28 |
| Training seed | Not fully recoverable from the legacy training run |
| Conformal test top-1 accuracy | 71.2 percent |

## Important Limitation

The conformal evaluation is deterministic given the exported arrays. The original training run did not fully record all random seeds, so independently retraining the same architecture may not reproduce the exact checkpoint. This limitation is stated in the manuscript.

## Archive Record

The archive record for this release is `10.5281/zenodo.21362039`. The binary reproducibility set tracked for this manuscript consists of the checkpoint, exported probability arrays, labels, split indices, class mapping, final configs, generated tables, and the Git commit identifier for the public code release.
