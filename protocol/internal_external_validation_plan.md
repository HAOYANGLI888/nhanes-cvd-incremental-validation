# Internal-External Validation Plan

Outer validation is leave-one-NHANES-cycle-out across the seven development cycles from 2005-2006 through 2017-2018. Inner tuning is leave-one-training-cycle-out within each outer training set. The primary tuning metric is weighted log loss. Weighted Brier score is the key secondary metric. The held-out outer cycle is never used to fit imputation, scaling, transformations, hyperparameters, early stopping, or thresholds.
