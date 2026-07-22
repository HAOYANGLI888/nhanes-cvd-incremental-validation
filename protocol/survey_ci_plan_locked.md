# Survey-aware confidence interval plan locked

This post-Stage-3A module estimates design-aware uncertainty for the prespecified survey-weighted probability metrics, calibration metrics, and paired incremental differences. It does not refit or retune either model family.

- Data scope: NHANES 2005-2018 outer-validation predictions only.
- Forbidden scope: NHANES 2021-2023 remains locked and must not be read by this module.
- Design key: cycle plus masked variance stratum; PSU is nested within this key.
- Replicate method: Rao-Wu rescaled bootstrap.
- Replicate construction: within each stratum with n_h PSUs, sample n_h-1 PSUs with replacement and multiply each selected PSU weight by n_h/(n_h-1) times its selection multiplicity.
- Replicates: 2,000, seed 20260715.
- Interval: two-sided 95% percentile interval.
- Pairing: core and extended models use the same participant rows, the extended-model analysis weight, and identical replicate factors.
- Pooled performance: equal-cycle mean of the cycle-specific performance estimator, matching the locked Stage 3A pooled summary.
- Pooled incremental value: participant-pooled paired estimator across the seven development cycles, matching the locked Stage 3A definition.
- Main CI scope: survey-weighted AUROC, survey-weighted PR-AUC, weighted Brier score, weighted log loss, calibration quantities, and paired incremental changes.
- Threshold metrics remain secondary point summaries and are not used for the survey CI release gate.
- Ordinary participant-level bootstrap and SMOTE are prohibited.
- A run passes only if all replicate batches complete, point estimates reconcile to Stage 3A within 1e-7, at least 95% of replicates are valid for every reported metric, and no forbidden-cycle rows are present.
