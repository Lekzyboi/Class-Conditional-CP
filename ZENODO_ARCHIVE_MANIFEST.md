# Zenodo Archive Manifest

This manifest lists binary and provenance files associated with the public code release. They are not committed to GitHub because they are generated or binary artifacts.

## Code Snapshot

| Field | Value |
|---|---|
| GitHub repository | `https://github.com/Lekzyboi/Class-Conditional-CP` |
| Current local commit | `9eaec6380dfc57bca8d79d53a7cad1ff4d2bcff8` |
| DOI | `10.5281/zenodo.21361646` |

## Binary Inputs

| File | Local source path | SHA-256 |
|---|---|---|
| `isic_calibration_probs.npy` | `assets/output/framework_inputs/isic_calibration_probs.npy` | `F6152B77B7C6053DA4F5DF2FCC81C8AD11CBADACFA9126F98B8C5A7BC6492D46` |
| `isic_calibration_labels.npy` | `assets/output/framework_inputs/isic_calibration_labels.npy` | `079BCEA9C0B853457CB9D909A9ABB04BCCAC551932395F4B6858FD6077D3B277` |
| `isic_test_probs.npy` | `assets/output/framework_inputs/isic_test_probs.npy` | `A0952853CAE906BEBFE0D788ECAB32889E9A9E09FA951DEC1973716D192B8B99` |
| `isic_test_labels.npy` | `assets/output/framework_inputs/isic_test_labels.npy` | `279D288E2812ECD9CBB2D8AE0C9A0EBD19044057FF810E8613CA63DAB232371D` |
| `bcn_kept_probs.npy` | `assets/output/framework_inputs/bcn_kept_probs.npy` | `10B6D6C27B04751840C564ED4D20B6C81F995D9DAA0EFBEEB0914A4FE048E4A7` |
| `bcn_kept_labels.npy` | `assets/output/framework_inputs/bcn_kept_labels.npy` | `7003B40BA47EE3140C0E99F1A31B4C07B0C46B92AD0B48F589E78600E560FED2` |
| `best_model.pth` | `assets/output/best_model.pth` | `B4D483C63A13469067ECAF8EE8C2D46F6CBB90ABD74613955989E8C03AA84512` |

## Provenance Files

| File | Local source path | SHA-256 |
|---|---|---|
| `class_index_mapping.json` | `assets/output/framework_inputs/class_index_mapping.json` | `114E16F3CCA708CB40893AE22A7736760BAC5CB072E6D06BBE8A624AC15D875D` |
| `split_indices.json` | `assets/output/framework_inputs/split_indices.json` | `C4BADE00CDF77189AC8AD48DF21FBF3A07514B2EF2CF6216124018BCFD9F3B93` |

## Generated Outputs

The lightweight GitHub release contains generated tables, figures, and audit outputs derived from:

- `assets/output/framework_runs/isic_resnet50_framework_inputs/`
- `assets/output/framework_runs/bcn_kept_overlap_aware_sensitivity/`
- `assets/output/audits/`

The GitHub repository includes lightweight copies of these generated tables, figures, and audit reports under `reproducibility_artifacts/`.
