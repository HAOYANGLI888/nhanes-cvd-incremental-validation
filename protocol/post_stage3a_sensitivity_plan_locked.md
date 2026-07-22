# Post-Stage-3A sensitivity plan locked

This plan was frozen before running the remaining sensitivity results. It cannot alter the main model selection or unlock NHANES 2021-2023.

## UACR missingness

- Model 1 and Model 4 are evaluated with the paired Model 0 comparator on identical samples and weights.
- Main fold-specific median results are retained as the reference.
- Multiple imputation uses 20 posterior draws from a Bayesian-ridge iterative imputer fitted only in each outer training fold. The outcome is excluded because it would be unavailable when imputing held-out participants.
- Main-analysis hyperparameters selected without the outer holdout are fixed; the sensitivity does not use its results to retune or select model complexity.
- Complete-case refits use only observed UACR in both training and validation samples.
- The IPW availability model is fitted only in the outer training fold using model-specific predictors excluding UACR. Stabilized weights use training-fold prevalence, propensity clipping at 0.05 and 0.995, and a training-derived 99th-percentile upper cap.

## Class imbalance

- Main hyperparameters remain fixed.
- Elastic Net uses class weights n/(2 n_class), calculated from each outer training fold.
- XGBoost uses training negatives divided by training positives as scale_pos_weight.
- No SMOTE, ADASYN, oversampling, or undersampling is allowed.

## Cycle meta-analysis

- Main paired incremental estimates and survey-bootstrap standard errors are pooled across the seven disjoint cycles.
- Between-cycle variance is estimated by REML.
- The primary interval uses modified Hartung-Knapp with a t distribution and k-1 degrees of freedom.

## Decision-curve analysis

- Net benefit is evaluated at thresholds 0.01 through 0.30 in increments of 0.01.
- Core and extended models use identical participants, survey weights, and survey replicate factors.
- Treat-all and treat-none are reported with extended-minus-core incremental net benefit.
- Percentile confidence intervals use the same 2,000 Rao-Wu survey replicates as the survey CI module.

## Final freeze

- The five Elastic Net and five XGBoost development artifacts already frozen after Stage 3A are not refitted or replaced based on sensitivity results.
- Final release requires PASS gates for Stage 3A, survey CI, UACR sensitivity, class imbalance, meta-analysis, and DCA, plus exact model-file hash verification.
- NHANES 2021-2023 remains locked after this stage.
