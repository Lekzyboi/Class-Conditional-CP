# Ablation Configs

This folder holds config files for expanded method grids, polynomial degree, binning strategy, and trust-score k-NN ablations.

The expanded conformal baseline grid includes:

- `standard_cp`
- `aps`
- `raps`
- `mondrian_cp`
- `frequency_grouped_cp`
- `conf_trust_naive`
- `conf_trust_conditional`
- `temperature_scaling`

`frequency_grouped_cp` partitions classes by training-set frequency. It is not a score-distribution clustering method.

The Conf-Trust methods require extra arrays:

- `paths.calibration_trust`
- `paths.test_trust`

Temperature scaling is available as a non-conformal calibration utility, not as a coverage-guaranteeing conformal method.
