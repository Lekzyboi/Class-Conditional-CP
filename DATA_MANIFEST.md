# Data Manifest

## Purpose

This manifest documents the datasets and metadata currently present in the working repository. It is part of the CMPB framework refactor traceability layer and should be updated whenever data are added, restored, regenerated, or intentionally relocated.

No dataset should be deleted, moved, renamed, or overwritten without explicit approval.

## Dataset Inventory

| Path | Role | Current status | Notes |
|---|---|---:|---|
| `assets/ISIC_2019/` | ISIC 2019 train/validation image folders used by training scripts | Present | Contains `train/` and `val/` class folders |
| `assets/ISIC_2019_Test/` | Raw ISIC 2019 test images | Present | Used as source data/reference |
| `assets/ISIC_2019_Test_Organized/` | ISIC 2019 test images organized by class | Present | Used by `paper1.py`, `conf_trust_cp.py`, and test-evaluation scripts |
| `assets/bcn_20000_data/` | BCN20000 image data | Present | Used for overlap-aware/sensitivity analyses |
| `assets/SkinCON.csv` | SkinCON soft-label annotations | Present | Needed by `datasets.py`/`train.py` soft-label training |
| `assets/ISIC_2019_Training_GroundTruth.csv` | ISIC training-label CSV | Present | Needed for overlap and dataset provenance checks |
| `assets/ISIC_2019_Test_GroundTruth.csv` | ISIC test-label CSV | Present | Needed for test-set provenance and organization checks |
| `metadata.csv` | BCN metadata CSV expected by BCN scripts | Missing at repo root | Several BCN scripts expect this exact file path |
| `assets/test/` | Unclassified local asset folder | Present | Needs later inspection before any use in framework |

## ISIC 2019 Train/Validation Counts

Path: `assets/ISIC_2019/`

The effective SkinCON-supervised training cohort contains 25,329 images. Two ISIC 2019 training images, one NV and one DF, lacked matching SkinCON soft-label records and were excluded from the soft-label training population.

| Split | Files |
|---|---:|
| `train` | 22,796 |
| `val` | 2,533 |

### ISIC 2019 Training Class Counts

Path: `assets/ISIC_2019/train/`

| Class | Files |
|---|---:|
| AK | 714 |
| BCC | 2,731 |
| BKL | 2,372 |
| DF | 211 |
| MEL | 3,971 |
| NV | 12,031 |
| SCC | 545 |
| VASC | 221 |

### ISIC 2019 Validation Class Counts

Path: `assets/ISIC_2019/val/`

| Class | Files |
|---|---:|
| AK | 153 |
| BCC | 592 |
| BKL | 252 |
| DF | 27 |
| MEL | 551 |
| NV | 843 |
| SCC | 83 |
| VASC | 32 |

## ISIC 2019 Organized Test Counts

Path: `assets/ISIC_2019_Test_Organized/`

| Class folder | Files |
|---|---:|
| actinic keratosis | 374 |
| basal cell carcinoma | 975 |
| benign keratosis | 660 |
| dermatofibroma | 91 |
| melanoma | 1,327 |
| nevus | 2,495 |
| squamous cell carcinoma | 165 |
| vascular lesion | 104 |
| **Total** | **6,191** |

## BCN20000 Data

Path: `assets/bcn_20000_data/`

Current direct image-file count: **18,947**

Important caveat:

- The BCN scripts expect a root-level `metadata.csv`.
- At the time this manifest was created, `metadata.csv` was not present at the repository root.
- The framework should not silently proceed with BCN analyses unless the metadata file is present and validated.

## Metadata Files

| Path | Size | Role |
|---|---:|---|
| `assets/SkinCON.csv` | 1,913,565 bytes | Soft labels and class-distribution annotations |
| `assets/ISIC_2019_Training_GroundTruth.csv` | 1,291,479 bytes | ISIC training labels |
| `assets/ISIC_2019_Test_GroundTruth.csv` | 464,810 bytes | ISIC test labels |

## Known Data Issues To Address

1. `datasets.py` currently hardcodes `/home/liyunqi/SkinCON.csv`; it should use `assets/SkinCON.csv` or a config value.
2. `metadata.csv` is required by BCN scripts but is missing from the root.
3. BCN20000 should not be described as clean external validation unless overlap-free metadata and duplicate audits support that.
4. Primary ISIC calibration/test splits should be checked for lesion/patient overlap if identifiers are available.
5. Perceptual duplicate detection should be added for any image-level split where lesion/patient IDs are missing.

## Required Framework Data Checks

The new framework should validate:

- Required files exist.
- Expected class folders exist.
- Class counts match expected values or are explicitly updated.
- Metadata columns required by each analysis exist.
- Calibration and test splits are disjoint.
- Optional lesion/patient groups do not cross split boundaries.
- BCN/ISIC overlap status is recorded before analysis.
