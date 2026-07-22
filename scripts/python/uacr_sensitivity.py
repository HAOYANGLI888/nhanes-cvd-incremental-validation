from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import stage3a_core as core
from post_stage3a_common import (
    DEVELOPMENT_CYCLES,
    atomic_csv,
    atomic_text,
    canonical_json,
    checkpoint_dir,
    fit_predict,
    load_config,
    load_development_data,
    load_selected_parameters,
    load_task_frames,
    log,
    module_signature,
    now,
    paired_increment_row,
    performance_row,
    save_task_frames,
    selected_params,
    sha256_text,
    validate_upstream,
    write_gate,
)


def dense_one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def task_seed(global_seed: int, comparison: str, algorithm: str, cycle: str) -> int:
    digest = hashlib.sha256(f"{global_seed}|{comparison}|{algorithm}|{cycle}".encode()).hexdigest()
    return int(digest[:8], 16)


def build_uacr_imputer(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
) -> tuple[IterativeImputer, np.ndarray, np.ndarray]:
    continuous = list(config["auxiliary_continuous"])
    categorical = list(config["auxiliary_categorical"])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", dense_one_hot()),
        ]
    )
    categorical_pipe.fit(train[categorical])
    train_cat = categorical_pipe.transform(train[categorical])
    validation_cat = categorical_pipe.transform(validation[categorical])
    train_matrix = np.column_stack([train[continuous].astype(float).to_numpy(), train_cat])
    validation_matrix = np.column_stack([validation[continuous].astype(float).to_numpy(), validation_cat])
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        sample_posterior=True,
        max_iter=int(config["imputer_max_iter"]),
        initial_strategy="median",
        skip_complete=True,
        keep_empty_features=True,
        random_state=seed,
    )
    imputer.fit(train_matrix)
    return imputer, train_matrix, validation_matrix


def propensity_weights(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    model: str,
    features: list[str],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    predictors = [feature for feature in features if feature != "log2_uacr"]
    preprocessor = core.build_preprocessor(predictors, "elastic_net", train)
    classifier = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=float(config["propensity_C"]),
        max_iter=3000,
        random_state=core.RANDOM_SEED,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", classifier)])
    observed_train = train["log2_uacr"].notna().astype(int).to_numpy()
    base_train = train[core.weight_column(model)].astype(float).to_numpy()
    pipeline.fit(train[predictors], observed_train, model__sample_weight=base_train)
    raw_train = pipeline.predict_proba(train[predictors])[:, 1]
    raw_validation = pipeline.predict_proba(validation[predictors])[:, 1]
    lower = float(config["propensity_clip_lower"])
    upper = float(config["propensity_clip_upper"])
    clipped_train = np.clip(raw_train, lower, upper)
    clipped_validation = np.clip(raw_validation, lower, upper)
    prevalence = float(np.average(observed_train, weights=base_train))
    factor_train = prevalence / clipped_train
    factor_validation = prevalence / clipped_validation
    cap = float(np.quantile(factor_train[observed_train == 1], float(config["stabilized_weight_upper_quantile"])))
    factor_train = np.minimum(factor_train, cap)
    factor_validation = np.minimum(factor_validation, cap)
    observed_validation = validation["log2_uacr"].notna().to_numpy()
    base_validation = validation[core.weight_column(model)].astype(float).to_numpy()
    weighted_train = base_train[observed_train == 1] * factor_train[observed_train == 1]
    weighted_validation = base_validation[observed_validation] * factor_validation[observed_validation]
    diagnostics = {
        "availability_weighted_prevalence_train": prevalence,
        "raw_propensity_min_train": float(raw_train.min()),
        "raw_propensity_max_train": float(raw_train.max()),
        "raw_propensity_min_validation": float(raw_validation.min()),
        "raw_propensity_max_validation": float(raw_validation.max()),
        "stabilized_factor_cap_training_p99": cap,
        "stabilized_factor_max_train_observed": float(factor_train[observed_train == 1].max()),
        "stabilized_factor_max_validation_observed": float(factor_validation[observed_validation].max()),
        "ipw_ess_train": float(weighted_train.sum() ** 2 / np.sum(weighted_train**2)),
        "ipw_ess_validation": float(weighted_validation.sum() ** 2 / np.sum(weighted_validation**2)),
    }
    return factor_train, factor_validation, diagnostics


def prediction_frame(
    frame: pd.DataFrame,
    probability: np.ndarray,
    weight: np.ndarray,
    comparison: str,
    algorithm: str,
    model: str,
    method: str,
    prediction_sd: np.ndarray | float = 0.0,
) -> pd.DataFrame:
    if np.isscalar(prediction_sd):
        prediction_sd = np.full(len(frame), float(prediction_sd))
    return pd.DataFrame(
        {
            "seqn": frame["seqn"].to_numpy(),
            "cycle": frame["cycle"].to_numpy(),
            "comparison": comparison,
            "algorithm": algorithm,
            "model": model,
            "uacr_method": method,
            "cvd": frame["cvd"].astype(int).to_numpy(),
            "predicted_probability": probability,
            "between_imputation_prediction_sd": prediction_sd,
            "analysis_weight": weight,
            "strata": frame["strata"].to_numpy(),
            "psu": frame["psu"].to_numpy(),
        }
    )


def run_task(
    project_root: Path,
    data: pd.DataFrame,
    features_by_model: dict[str, list[str]],
    selected: pd.DataFrame,
    stage3_predictions: pd.DataFrame,
    config: dict[str, Any],
    signature_hash: str,
    comparison: str,
    algorithm: str,
    fold: dict[str, Any],
    resume: bool,
) -> dict[str, pd.DataFrame]:
    cycle = str(fold["holdout_cycle"])
    task_id = f"{comparison}__{algorithm}__{cycle.replace('-', '_')}"
    task_signature = sha256_text(canonical_json({"base": signature_hash, "task": task_id}))
    path = checkpoint_dir(project_root, "uacr", task_id)
    if resume:
        loaded = load_task_frames(path, task_signature)
        if loaded is not None:
            return loaded
    item = config["comparisons"][comparison]
    core_model = item["core_model"]
    extended_model = item["extended_model"]
    mask = core.sample_mask(data, extended_model)
    train = data[mask & data["cycle"].isin(fold["training_cycles"])].sort_values(["cycle", "seqn"], kind="mergesort").copy()
    validation = data[mask & data["cycle"].eq(cycle)].sort_values("seqn", kind="mergesort").copy()
    core_params = selected_params(selected, cycle, core_model, algorithm)
    extended_params = selected_params(selected, cycle, extended_model, algorithm)
    base_train_weight = train[core.weight_column(extended_model)].astype(float).to_numpy()
    base_validation_weight = validation[core.weight_column(extended_model)].astype(float).to_numpy()
    performance: list[dict[str, Any]] = []
    incremental: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []

    main_core = stage3_predictions[
        stage3_predictions["algorithm"].eq(algorithm)
        & stage3_predictions["cycle"].eq(cycle)
        & stage3_predictions["model"].eq(core_model)
    ][["seqn", "predicted_probability"]].rename(columns={"predicted_probability": "p_core"})
    main_extended = stage3_predictions[
        stage3_predictions["algorithm"].eq(algorithm)
        & stage3_predictions["cycle"].eq(cycle)
        & stage3_predictions["model"].eq(extended_model)
    ][["seqn", "predicted_probability"]].rename(columns={"predicted_probability": "p_extended"})
    main = validation[["seqn"]].merge(main_core, on="seqn", how="left", validate="one_to_one").merge(
        main_extended, on="seqn", how="left", validate="one_to_one"
    )
    if main[["p_core", "p_extended"]].isna().any().any():
        raise RuntimeError(f"Main prediction reconciliation failed for {task_id}.")
    for model, probability in [(core_model, main["p_core"].to_numpy()), (extended_model, main["p_extended"].to_numpy())]:
        performance.append(performance_row(validation, probability, base_validation_weight, model, algorithm, cycle, "uacr_method", "main_fold_median"))
        predictions.append(prediction_frame(validation, probability, base_validation_weight, comparison, algorithm, model, "main_fold_median"))
    incremental.append(paired_increment_row(validation, main["p_core"].to_numpy(), main["p_extended"].to_numpy(), base_validation_weight, comparison, algorithm, cycle, core_model, extended_model, "uacr_method", "main_fold_median"))

    _, p_core_mi = fit_predict(train, validation, features_by_model[core_model], core_model, algorithm, core_params, base_train_weight)
    imputer, train_matrix, validation_matrix = build_uacr_imputer(
        train,
        validation,
        config,
        task_seed(int(config.get("random_seed", 20260715)), comparison, algorithm, cycle),
    )
    m = int(config["multiple_imputations"])
    sum_probability = np.zeros(len(validation))
    sum_square_probability = np.zeros(len(validation))
    for _ in range(m):
        train_imputed = train.copy()
        validation_imputed = validation.copy()
        train_imputed["log2_uacr"] = imputer.transform(train_matrix)[:, 0]
        validation_imputed["log2_uacr"] = imputer.transform(validation_matrix)[:, 0]
        _, probability = fit_predict(
            train_imputed,
            validation_imputed,
            features_by_model[extended_model],
            extended_model,
            algorithm,
            extended_params,
            base_train_weight,
        )
        sum_probability += probability
        sum_square_probability += probability**2
    p_extended_mi = sum_probability / m
    p_extended_mi_sd = np.sqrt(np.maximum(0.0, (sum_square_probability - m * p_extended_mi**2) / max(1, m - 1)))
    for model, probability, sd in [
        (core_model, p_core_mi, 0.0),
        (extended_model, p_extended_mi, p_extended_mi_sd),
    ]:
        performance.append(performance_row(validation, probability, base_validation_weight, model, algorithm, cycle, "uacr_method", "nested_multiple_imputation"))
        predictions.append(prediction_frame(validation, probability, base_validation_weight, comparison, algorithm, model, "nested_multiple_imputation", sd))
    incremental.append(paired_increment_row(validation, p_core_mi, p_extended_mi, base_validation_weight, comparison, algorithm, cycle, core_model, extended_model, "uacr_method", "nested_multiple_imputation"))

    train_observed = train[train["log2_uacr"].notna()].copy()
    validation_observed = validation[validation["log2_uacr"].notna()].copy()
    cc_train_weight = train_observed[core.weight_column(extended_model)].astype(float).to_numpy()
    cc_validation_weight = validation_observed[core.weight_column(extended_model)].astype(float).to_numpy()
    _, p_core_cc = fit_predict(train_observed, validation_observed, features_by_model[core_model], core_model, algorithm, core_params, cc_train_weight)
    _, p_extended_cc = fit_predict(train_observed, validation_observed, features_by_model[extended_model], extended_model, algorithm, extended_params, cc_train_weight)
    for model, probability in [(core_model, p_core_cc), (extended_model, p_extended_cc)]:
        performance.append(performance_row(validation_observed, probability, cc_validation_weight, model, algorithm, cycle, "uacr_method", "complete_case_refit"))
        predictions.append(prediction_frame(validation_observed, probability, cc_validation_weight, comparison, algorithm, model, "complete_case_refit"))
    incremental.append(paired_increment_row(validation_observed, p_core_cc, p_extended_cc, cc_validation_weight, comparison, algorithm, cycle, core_model, extended_model, "uacr_method", "complete_case_refit"))

    factor_train, factor_validation, propensity_diagnostics = propensity_weights(
        train, validation, extended_model, features_by_model[extended_model], config
    )
    observed_train_mask = train["log2_uacr"].notna().to_numpy()
    observed_validation_mask = validation["log2_uacr"].notna().to_numpy()
    ipw_train_weight = base_train_weight[observed_train_mask] * factor_train[observed_train_mask]
    ipw_validation_weight = base_validation_weight[observed_validation_mask] * factor_validation[observed_validation_mask]
    _, p_core_ipw = fit_predict(train_observed, validation_observed, features_by_model[core_model], core_model, algorithm, core_params, ipw_train_weight)
    _, p_extended_ipw = fit_predict(train_observed, validation_observed, features_by_model[extended_model], extended_model, algorithm, extended_params, ipw_train_weight)
    for model, probability in [(core_model, p_core_ipw), (extended_model, p_extended_ipw)]:
        row = performance_row(validation_observed, probability, ipw_validation_weight, model, algorithm, cycle, "uacr_method", "stabilized_ipw_complete_case")
        row["weight_variable"] = f"{core.weight_column(extended_model)}_x_stabilized_uacr_ipw"
        performance.append(row)
        predictions.append(prediction_frame(validation_observed, probability, ipw_validation_weight, comparison, algorithm, model, "stabilized_ipw_complete_case"))
    inc_row = paired_increment_row(validation_observed, p_core_ipw, p_extended_ipw, ipw_validation_weight, comparison, algorithm, cycle, core_model, extended_model, "uacr_method", "stabilized_ipw_complete_case")
    inc_row["weight_variable"] = f"{core.weight_column(extended_model)}_x_stabilized_uacr_ipw"
    incremental.append(inc_row)

    diagnostics = pd.DataFrame(
        [
            {
                "comparison": comparison,
                "algorithm": algorithm,
                "cycle": cycle,
                "n_train": len(train),
                "n_validation": len(validation),
                "uacr_observed_train": len(train_observed),
                "uacr_observed_validation": len(validation_observed),
                "uacr_missing_fraction_train": float(train["log2_uacr"].isna().mean()),
                "uacr_missing_fraction_validation": float(validation["log2_uacr"].isna().mean()),
                "multiple_imputations": m,
                **propensity_diagnostics,
            }
        ]
    )
    frames = {
        "performance": pd.DataFrame(performance),
        "incremental": pd.DataFrame(incremental),
        "predictions": pd.concat(predictions, ignore_index=True),
        "diagnostics": diagnostics,
    }
    save_task_frames(path, task_signature, frames)
    return frames


def run_uacr(project_root: Path, resume: bool, workers: int, log_path: Path) -> str:
    validate_upstream(project_root)
    config_all = load_config(project_root)
    config = {**config_all["uacr"], "random_seed": config_all["random_seed"]}
    signature = module_signature(project_root, "uacr", ["scripts/python/uacr_sensitivity.py"])
    data = load_development_data(project_root)
    features = core.read_frozen_features(project_root)
    selected = load_selected_parameters(project_root)
    stage3_predictions = pd.read_csv(project_root / "results" / "predictions" / "stage3a_outer_predictions.csv")
    if set(stage3_predictions["cycle"]) != set(DEVELOPMENT_CYCLES):
        raise RuntimeError("UACR module received unexpected cycles.")
    tasks = [
        (comparison, algorithm, fold)
        for comparison in config["comparisons"]
        for algorithm in config["algorithms"]
        for fold in core.read_cycle_folds(project_root)
    ]
    results: list[dict[str, pd.DataFrame]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 4))) as executor:
        futures = {
            executor.submit(
                run_task,
                project_root,
                data,
                features,
                selected,
                stage3_predictions,
                config,
                signature["signature_hash"],
                comparison,
                algorithm,
                fold,
                resume,
            ): (comparison, algorithm, fold["holdout_cycle"])
            for comparison, algorithm, fold in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            log(log_path, f"UACR task complete {futures[future]}; tasks={len(results)}/{len(tasks)}")
    combined = {name: pd.concat([result[name] for result in results], ignore_index=True) for name in results[0]}
    combined["performance"].sort_values(["uacr_method", "comparison_sample", "algorithm", "cycle", "model"], inplace=True)
    combined["incremental"].sort_values(["uacr_method", "comparison", "algorithm", "cycle"], inplace=True)
    combined["predictions"].sort_values(["uacr_method", "comparison", "algorithm", "cycle", "model", "seqn"], inplace=True)
    combined["diagnostics"].sort_values(["comparison", "algorithm", "cycle"], inplace=True)
    atomic_csv(project_root / "results" / "tables" / "uacr_method_sensitivity.csv", combined["performance"])
    atomic_csv(project_root / "results" / "tables" / "uacr_incremental_sensitivity.csv", combined["incremental"])
    atomic_csv(project_root / "results" / "predictions" / "uacr_sensitivity_predictions.csv.gz", combined["predictions"], compression="gzip")
    atomic_csv(project_root / "results" / "audit" / "uacr_sensitivity_diagnostics.csv", combined["diagnostics"])

    expected_performance = len(tasks) * 8
    expected_incremental = len(tasks) * 4
    methods = set(config["methods"])
    observed_methods = set(combined["incremental"]["uacr_method"])
    main_rows = combined["incremental"][combined["incremental"]["uacr_method"].eq("main_fold_median")]
    reference = pd.read_csv(project_root / "results" / "tables" / "cycle_specific_incremental_value.csv")
    reference = reference[reference["comparison"].isin(config["comparisons"])]
    merged = main_rows.merge(reference, on=["comparison", "algorithm", "cycle"], suffixes=("_new", "_reference"))
    metrics = ["delta_AUROC", "delta_PR_AUC", "delta_Brier_improvement", "delta_log_loss_improvement"]
    max_difference = max(float((merged[f"{metric}_new"] - merged[f"{metric}_reference"]).abs().max()) for metric in metrics)
    predictions_unique = not combined["predictions"].duplicated(["seqn", "cycle", "comparison", "algorithm", "model", "uacr_method"]).any()
    finite = np.isfinite(combined["incremental"][metrics].to_numpy()).all()
    status = "PASS" if all(
        [
            len(results) == len(tasks),
            len(combined["performance"]) == expected_performance,
            len(combined["incremental"]) == expected_incremental,
            observed_methods == methods,
            predictions_unique,
            finite,
            max_difference <= 1e-10,
            set(combined["incremental"]["cycle"]) == set(DEVELOPMENT_CYCLES),
        ]
    ) else "FAIL"
    write_gate(
        project_root,
        "uacr_sensitivity",
        {
            "module": "uacr_nested_mi_ipw",
            "status": status,
            "tasks_completed": len(results),
            "tasks_expected": len(tasks),
            "performance_rows": len(combined["performance"]),
            "incremental_rows": len(combined["incremental"]),
            "methods": "|".join(sorted(observed_methods)),
            "multiple_imputations": config["multiple_imputations"],
            "minimum_raw_propensity": combined["diagnostics"][["raw_propensity_min_train", "raw_propensity_min_validation"]].min().min(),
            "maximum_stabilized_factor": combined["diagnostics"][["stabilized_factor_max_train_observed", "stabilized_factor_max_validation_observed"]].max().max(),
            "minimum_ipw_ess": combined["diagnostics"][["ipw_ess_train", "ipw_ess_validation"]].min().min(),
            "main_point_reconciliation_max_difference": max_difference,
            "paired_prediction_uniqueness": predictions_unique,
            "forbidden_cycle_absent": True,
            "signature_hash": signature["signature_hash"],
        },
    )
    audit = [
        "# UACR nested MI and IPW sensitivity audit",
        "",
        f"- Status: {status}",
        f"- Tasks: {len(results)}/{len(tasks)}",
        f"- Multiple imputations: {config['multiple_imputations']} per outer fold.",
        "- Imputation and UACR-availability propensity models were fitted only on outer training cycles.",
        "- Main training-selected hyperparameters were fixed and sensitivity results did not retune the main models.",
        "- Complete-case and stabilized IPW core/extended models used identical participants and weights.",
        "- Outcome was excluded from held-out-cycle imputation.",
        "- NHANES 2021-2023 was not accessed.",
    ]
    atomic_text(project_root / "results" / "audit" / "uacr_stage3a_sensitivity_audit.md", "\n".join(audit) + "\n")
    return status
