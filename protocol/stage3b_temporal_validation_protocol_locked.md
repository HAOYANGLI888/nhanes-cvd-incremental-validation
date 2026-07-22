# Stage 3B Temporal Validation Protocol — Locked Before Performance Evaluation

Lock date: 2026-07-20  
Authorization: explicitly provided by the study owner on 2026-07-20.

## Objective

Evaluate the one-time temporal transportability of the ten already frozen NHANES 2005–2018 development pipelines in NHANES 2021–2023. The outcome remains prevalent self-reported CVD; this is not incident-event prediction.

## Immutable analysis rules

- Apply the five prespecified predictor sets for Elastic Net and XGBoost exactly as frozen.
- Do not refit preprocessing, imputation, scaling, encoding, coefficients, trees, hyperparameters, or probability calibration.
- Do not select thresholds, variables, samples, transformations, or missing-data strategies from 2021–2023 performance.
- Use adults aged at least 20 years who are not pregnant, have an observed CVD outcome, valid masked stratum and PSU identifiers, and a positive domain-specific weight.
- Use the MEC weight for the core, renal, and inflammatory comparisons and the fasting-subsample weight for metabolic and combined comparisons.
- Calculate paired increments on identical participants using the extended-model domain weight. Positive values denote improvement: extended minus core for AUROC and PR-AUC; core minus extended for Brier score and log loss.
- Estimate 95% confidence intervals with 2,000 deterministic Rao–Wu rescaled bootstrap replicates within 2021–2023 masked strata and PSUs. Each paired comparison shares the same replicate factors.
- Elastic Net remains primary; XGBoost remains a secondary nonlinear sensitivity analysis.
- Report the result regardless of direction. No result-dependent model repair is permitted.

## Primary outputs

Survey-weighted AUROC, PR-AUC, Brier score, log loss, calibration intercept, calibration slope, observed/expected ratio, weighted observed prevalence, weighted mean predicted probability, and calibration-in-the-large. Paired incremental versions are reported for renal, metabolic, inflammatory, and combined extensions.

## Release requirements

The Stage 3B gate passes only if the final Stage 3A/postprocessing gate was already PASS, all ten frozen model hashes match the freeze manifest before and after prediction, all expected model and comparison rows are present, paired samples and weights are identical, 2,000 bootstrap replicates complete with adequate valid-replicate fractions, and no forbidden fitting operation occurs.
