# Data Manifest

This manifest documents the public datasets, derived metadata, and lightweight artifacts associated with the released framework.

## Public Source Datasets

The study uses publicly available ISIC 2019 and BCN20000 dermatoscopic image data. Raw images are not redistributed in this repository.

| Dataset | Role in this release |
|---|---|
| ISIC 2019 Challenge | Main eight-class dermatoscopic classification case study |
| BCN20000 | Partial identifier-overlap sensitivity analysis |
| SkinCON annotations | Soft-label supervision source for the fixed ResNet-50 checkpoint |

## Included Metadata

| Path | Role |
|---|---|
| `assets/ISIC_2019_Training_GroundTruth.csv` | ISIC training labels used for provenance and frequency summaries |
| `assets/ISIC_2019_Test_GroundTruth.csv` | ISIC test labels used for split and overlap checks |
| `assets/SkinCON.csv` | SkinCON soft-label annotations |

## Effective Training Cohort

The effective SkinCON-supervised training cohort contains 25,329 images. Two ISIC 2019 training images, one NV and one DF, lacked matching SkinCON soft-label records and were excluded from the soft-label training population.

| Split | Files |
|---|---:|
| Train | 22,796 |
| Validation | 2,533 |

## Training Class Counts

| Class | Files |
|---|---:|
| AK | 867 |
| BCC | 3,323 |
| BKL | 2,624 |
| DF | 238 |
| MEL | 4,522 |
| NV | 12,874 |
| SCC | 628 |
| VASC | 253 |

## ISIC 2019 Organized Test Counts

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
| Total | 6,191 |

## Included Derived Artifacts

| Path | Contents |
|---|---|
| `reproducibility_artifacts/isic_resnet50_framework_inputs/` | Main ISIC result tables, figures, JSON summaries, and statistical-test outputs |
| `reproducibility_artifacts/bcn_kept_overlap_aware_sensitivity/` | BCN sensitivity result tables, figures, and summaries |
| `reproducibility_artifacts/audits/` | Split, overlap, duplicate-screening, and kept-image audit outputs |

## Excluded Large Artifacts

Raw images, exported NumPy probability arrays, and model checkpoints are tracked through the Zenodo archive manifest rather than stored in GitHub. Their hashes are listed in `ZENODO_ARCHIVE_MANIFEST.md`.
