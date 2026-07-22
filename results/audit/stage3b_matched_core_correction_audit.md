# Stage 3B matched-core temporal validation correction

- Correction version: `stage3b_matched_fasting_core_v1_1`
- Status: **PASS**
- External cycle: NHANES 2021-2023.
- Original Stage 3B v1 outputs were archived with SHA-256 hashes before replacement.
- Frozen pipelines verified: 14/14 (10 original models plus 4 fasting-core aliases).
- Fasting metabolic and combined comparisons use cores trained in the identical fasting domain.
- The two fasting-core aliases share one byte-identical fitted pipeline per algorithm because their development samples and specifications are identical.
- No temporal-result-dependent hyperparameter tuning, feature selection, recalibration, or model updating was performed.
- Prediction groups: 14/14.
- Rao-Wu rescaled bootstrap replicates: 2000.

## Interpretation

This version is the canonical Stage 3B result for manuscript reporting. The archived v1 result is retained only as an audit trail of the implementation issue.
