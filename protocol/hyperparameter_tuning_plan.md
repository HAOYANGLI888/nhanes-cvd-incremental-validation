# Hyperparameter Tuning Plan

Elastic Net and XGBoost use finite, prespecified search grids in `config/elastic_net_hyperparameter_grid.yml` and `config/xgboost_hyperparameter_grid.yml`. Hyperparameters are selected using inner-cycle validation and weighted log loss. NHANES 2021-2023 is not used for tuning.
