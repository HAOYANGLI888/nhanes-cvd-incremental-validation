# Stage 3B matched fasting-core implementation amendment

## Version

- Amendment: `stage3b_matched_fasting_core_v1_1`
- Date: 2026-07-20
- Scope: implementation correction; no change to the scientific estimand, predictor sets, outcome, algorithms, hyperparameter grids, training cycles, temporal cycle, metrics, or survey-bootstrap method.

## Audit finding

The original Stage 3B code evaluated the metabolic and combined extensions on the correct fasting participants and weights, but generated their core probabilities from the MEC-trained Model 0 artifact. This did not satisfy the locked requirement that incremental comparisons use a core trained in the same analysis domain as the extension.

## Correction

For each algorithm, a fasting-domain Model 0 pipeline is selected and fitted using only NHANES 2005-2018, the original complete locked grid, the original survey-weighted inner-cycle log-loss rule, and the original training-only preprocessing. The metabolic and combined comparisons use identical fasting eligibility, features, weights, cycles, and tuning rules; therefore, their two logical aliases are byte-identical copies of one fitted fasting-core pipeline per algorithm.

The corrected frozen pipelines are then applied to NHANES 2021-2023 without temporal-result-dependent tuning, feature selection, recalibration, threshold selection, or model updating. The original unmatched-core outputs are archived with SHA-256 hashes before canonical outputs are replaced.

## Interpretation constraint

Because the implementation issue was detected after the first temporal run, the correction cannot recreate prospective blinding. The correction rule is dictated by the paired-domain specification and development data, not by the direction or magnitude of temporal performance. Manuscript reporting must disclose the audit correction and treat the corrected version as canonical.
