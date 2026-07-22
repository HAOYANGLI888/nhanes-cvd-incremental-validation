# Matched fasting core freeze audit

- Correction version: `stage3b_matched_fasting_core_v1_1`
- Status: **PASS**
- Development data only: NHANES 2005-2018.
- Selection metric and preprocessing: unchanged from the locked Stage 3A protocol.
- Candidate limits: absent; complete locked grids were evaluated.
- The two fasting-core aliases are byte-identical within algorithm because their sample, features, weights, and tuning rule are identical.
- NHANES 2021-2023 performance was not used for tuning, selection, recalibration, or model updating.

## Frozen artifacts

- model0_paired_combined_elastic_net: candidates=45; n=15956; events=1813; sha256=953352a0052dd3b69e4013c948de0506eccee0fddd72495d088d5875ce4f28f0
- model0_paired_metabolic_elastic_net: candidates=45; n=15956; events=1813; sha256=953352a0052dd3b69e4013c948de0506eccee0fddd72495d088d5875ce4f28f0
- model0_paired_combined_xgboost: candidates=256; n=15956; events=1813; sha256=62e15e7b6251504035ff1d8efa970b75c09d61f1d8bafe4678d42a603ef12ae2
- model0_paired_metabolic_xgboost: candidates=256; n=15956; events=1813; sha256=62e15e7b6251504035ff1d8efa970b75c09d61f1d8bafe4678d42a603ef12ae2
