# Missing Data Analysis Plan

- Exclude missing CVD outcome.
- Exclude missing required survey weight/strata/PSU for the target analysis.
- Fit imputation inside training folds only.
- Continuous predictors: median imputation fitted inside training folds.
- Categorical predictors: most-frequent imputation or explicit missing category, locked before Stage 3 implementation.
- UACR sensitivity: complete-case, fold-specific median imputation, and IPW for UACR availability when feasible.
- Nested multiple imputation is allowed only if implemented without leakage inside outer/inner folds.
