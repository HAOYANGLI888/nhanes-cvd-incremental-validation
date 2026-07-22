# Class Imbalance Analysis Plan

Main analyses preserve the observed class distribution. Elastic Net uses class_weight=None and XGBoost uses scale_pos_weight=1. Sensitivity analyses evaluate class weighting calculated only from training folds. SMOTE, ADASYN, random oversampling, and random undersampling are not used in the main analysis.
