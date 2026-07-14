# Ablation Configs

This folder holds config files for expanded method grids, polynomial degree, binning strategy, and trust-score k-NN ablations.

The full conformal baseline grid should include:

- `standard_cp`
- `aps`
- `raps`
- `mondrian_cp`
- `classwise_cp`
- `clustered_cp`
- `rc3p`
- `conf_trust_naive`
- `conf_trust_conditional`
- `temperature_scaling`

`rc3p` is currently registered as a documented frequency-clustered substitute until a faithful RC3P implementation is added.

The Conf-Trust methods require extra arrays:

- `paths.calibration_trust`
- `paths.test_trust`

Temperature scaling is available as a non-conformal calibration utility, not as a coverage-guaranteeing conformal method.
