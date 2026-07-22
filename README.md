# NHANES CVD incremental validation reproducibility archive

This GitHub-ready archive contains analytical scripts, locked configurations and protocols, aggregate results, figure source data, audit gates, ten original frozen development-model objects, and the audited fasting-matched core correction objects for the NHANES CVD manuscript.

Individual-level harmonised data and prediction exports are intentionally excluded. Recreate the participant-level analytic file from public-use NHANES files using the included scripts and metadata. NHANES data and documentation are available from the US National Center for Health Statistics.

Development period: NHANES 2005-2018. One-time temporal evaluation: NHANES 2021-2023. Elastic Net is primary; XGBoost is secondary. A post-run implementation audit replaced unmatched MEC-trained core predictions in fasting comparisons with a core selected and fitted solely in 2005-2018 under the unchanged locked grid; original outputs were archived with hashes. No GBD or other cohort data, no SMOTE, and no temporal-result-dependent tuning, refitting, or recalibration were used.

## Permanent archive and citation

- Frozen GitHub release: [v1.0.0](https://github.com/HAOYANGLI888/nhanes-cvd-incremental-validation/releases/tag/v1.0.0)
- Version-specific Zenodo DOI: [10.5281/zenodo.21487069](https://doi.org/10.5281/zenodo.21487069)
- All-versions Zenodo DOI: [10.5281/zenodo.21487068](https://doi.org/10.5281/zenodo.21487068)

Use the version-specific DOI when citing the exact release used for the manuscript. The all-versions DOI resolves to the latest archived version.

## Licensing

- Analytical code, locked configurations, and frozen model objects are released under the MIT License; see `LICENSE-CODE-MIT.txt`.
- Documentation, protocols, aggregate result tables, audit records, and figure source data are released under the Creative Commons Attribution 4.0 International License; see `LICENSE-DATA-CC-BY-4.0.txt`.
- Public-use NHANES source data remain subject to the terms and documentation of the US National Center for Health Statistics and are not redistributed here.

## Funding

This work was supported by the National Outstanding Young Physician Program (国家优秀青年医师; document no. 2024 (41)). The funder had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.
