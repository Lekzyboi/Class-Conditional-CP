# Data And Artifact Notes

This repository does not redistribute raw ISIC 2019 or BCN20000 images. The released framework stores metadata, generated result artifacts, and audit manifests.

Included metadata and artifacts:

- `assets/ISIC_2019_Training_GroundTruth.csv`
- `assets/ISIC_2019_Test_GroundTruth.csv`
- `assets/SkinCON.csv`
- `reproducibility_artifacts/isic_resnet50_framework_inputs/`
- `reproducibility_artifacts/bcn_kept_overlap_aware_sensitivity/`
- `reproducibility_artifacts/audits/`

The generated artifacts are included so reviewers can inspect the reported tables, figures, bootstrap intervals, below-target tests, and overlap manifests without downloading image files.

Large local artifacts intentionally excluded:

- raw ISIC 2019 images
- raw BCN20000 images
- model checkpoints
- exported NumPy probability arrays
- local Python caches

Full reruns use local dataset and checkpoint paths recorded in the configuration files.
