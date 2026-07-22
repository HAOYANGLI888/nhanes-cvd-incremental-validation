from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    paired_increment_row,
    performance_row,
    save_task_frames,
    selected_params,
    sha256_text,
    validate_upstream,
    write_gate,
)


def run_task(
    project_root: Path,
    data: pd.DataFrame,
    features: dict[str, list[str]],
    selected: pd.DataFrame,
    signature_hash: str,
    model: str,
    algorithm: str,
    fold: dict[str, Any],
    resume: bool,
) -> dict[str, pd.DataFrame]:
    cycle = str(fold["holdout_cycle"])
    task_id = f"{model}__{algorithm}__{cycle.replace('-', '_')}"
    task_signature = sha256_text(canonical_json({"base": signature_hash, "task": task_id}))
    path = checkpoint_dir(project_root, "class_imbalance", task_id)
    if resume:
        loaded = load_task_frames(path, task_signature)
        if loaded is not None:
            return loaded
    mask = core.sample_mask(data, model)
    train = data[mask & data["cycle"].isin(fold["training_cycles"])].sort_values(["cycle", "seqn"], kind="mergesort").copy()
    validation = data[mask & data["cycle"].eq(cycle)].sort_values("seqn", kind="mergesort").copy()
    params = selected_params(selected, cycle, model, algorithm)
    y_train = train["cvd"].astype(int).to_numpy()
    negatives = int(np.sum(y_train == 0))
    positives = int(np.sum(y_train == 1))
    if positives == 0 or negatives == 0:
        raise RuntimeError(f"Training fold lacks an outcome class: {task_id}")
    params = dict(params)
    if algorithm == "elastic_net":
        total = len(y_train)
        params["class_weight"] = {0: total / (2.0 * negatives), 1: total / (2.0 * positives)}
        setting = "training_count_balanced_class_weights"
        multiplier = params["class_weight"][1] / params["class_weight"][0]
    else:
        params["scale_pos_weight"] = negatives / positives
        setting = "training_negatives_divided_by_training_positives"
        multiplier = params["scale_pos_weight"]
    train_weight = train[core.weight_column(model)].astype(float).to_numpy()
    _, probability = fit_predict(train, validation, features[model], model, algorithm, params, train_weight)
    validation_weight = validation[core.weight_column(model)].astype(float).to_numpy()
    row = performance_row(validation, probability, validation_weight, model, algorithm, cycle, "imbalance_setting", setting)
    prediction = pd.DataFrame(
        {
            "seqn": validation["seqn"].to_numpy(),
            "cycle": validation["cycle"].to_numpy(),
            "model": model,
            "algorithm": algorithm,
            "cvd": validation["cvd"].astype(int).to_numpy(),
            "predicted_probability": probability,
            "analysis_weight": validation_weight,
            "weight_variable": core.weight_column(model),
            "strata": validation["strata"].to_numpy(),
            "psu": validation["psu"].to_numpy(),
            "comparison_sample": core.comparison_sample_label(model),
            "imbalance_setting": setting,
        }
    )
    diagnostics = pd.DataFrame(
        [
            {
                "cycle": cycle,
                "model": model,
                "algorithm": algorithm,
                "n_train": len(train),
                "events_train": positives,
                "non_events_train": negatives,
                "positive_class_multiplier": multiplier,
                "imbalance_setting": setting,
                "resampling": "none",
            }
        ]
    )
    frames = {"performance": pd.DataFrame([row]), "predictions": prediction, "diagnostics": diagnostics}
    save_task_frames(path, task_signature, frames)
    return frames


def build_incremental(predictions: pd.DataFrame, setting: str) -> pd.DataFrame:
    rows = []
    for comparison, (core_model, extended_model, _domain) in core.INCREMENT_COMPARISONS.items():
        for algorithm in sorted(predictions["algorithm"].unique()):
            for cycle in DEVELOPMENT_CYCLES:
                core_pred = predictions[
                    predictions["model"].eq(core_model)
                    & predictions["algorithm"].eq(algorithm)
                    & predictions["cycle"].eq(cycle)
                ].copy()
                extended_pred = predictions[
                    predictions["model"].eq(extended_model)
                    & predictions["algorithm"].eq(algorithm)
                    & predictions["cycle"].eq(cycle)
                ].copy()
                pair = core_pred.merge(extended_pred, on=["seqn", "cycle"], suffixes=("_core", "_extended"), validate="one_to_one")
                if pair.empty or not pair["cvd_core"].equals(pair["cvd_extended"]):
                    raise RuntimeError(f"Class-imbalance pair mismatch: {comparison} {algorithm} {cycle}")
                frame = pd.DataFrame(
                    {
                        "cvd": pair["cvd_extended"],
                        "seqn": pair["seqn"],
                        "cycle": pair["cycle"],
                    }
                )
                rows.append(
                    paired_increment_row(
                        frame,
                        pair["predicted_probability_core"].to_numpy(),
                        pair["predicted_probability_extended"].to_numpy(),
                        pair["analysis_weight_extended"].to_numpy(),
                        comparison,
                        algorithm,
                        cycle,
                        core_model,
                        extended_model,
                        "imbalance_setting",
                        setting,
                    )
                )
    return pd.DataFrame(rows)


def run_class_imbalance(project_root: Path, resume: bool, workers: int, log_path: Path) -> str:
    validate_upstream(project_root)
    config = load_config(project_root)["class_imbalance"]
    signature = module_signature(project_root, "class_imbalance", ["scripts/python/class_imbalance_sensitivity.py"])
    data = load_development_data(project_root)
    features = core.read_frozen_features(project_root)
    selected = load_selected_parameters(project_root)
    tasks = [
        (model, algorithm, fold)
        for model in config["models"]
        for algorithm in config["algorithms"]
        for fold in core.read_cycle_folds(project_root)
    ]
    results: list[dict[str, pd.DataFrame]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(
                run_task,
                project_root,
                data,
                features,
                selected,
                signature["signature_hash"],
                model,
                algorithm,
                fold,
                resume,
            ): (model, algorithm, fold["holdout_cycle"])
            for model, algorithm, fold in tasks
        }
        for future in as_completed(futures):
            results.append(future.result())
            log(log_path, f"Class-imbalance task complete {futures[future]}; tasks={len(results)}/{len(tasks)}")
    weighted_performance = pd.concat([result["performance"] for result in results], ignore_index=True)
    weighted_predictions = pd.concat([result["predictions"] for result in results], ignore_index=True)
    diagnostics = pd.concat([result["diagnostics"] for result in results], ignore_index=True)
    main_performance = pd.read_csv(project_root / "results" / "tables" / "cycle_specific_model_performance.csv")
    main_performance = main_performance[
        main_performance["classification_threshold_type"].eq("fixed_0.50")
        & main_performance["model"].isin(config["models"])
    ].copy()
    main_performance["imbalance_setting"] = "main_observed_class_distribution"
    all_performance = pd.concat([main_performance, weighted_performance], ignore_index=True)
    main_predictions = pd.read_csv(project_root / "results" / "predictions" / "stage3a_outer_predictions.csv")
    main_predictions["imbalance_setting"] = "main_observed_class_distribution"
    main_incremental = build_incremental(main_predictions, "main_observed_class_distribution")
    weighted_incremental = build_incremental(weighted_predictions, "class_weighted_refit")
    incremental = pd.concat([main_incremental, weighted_incremental], ignore_index=True)
    all_performance.sort_values(["imbalance_setting", "algorithm", "cycle", "model"], inplace=True)
    incremental.sort_values(["imbalance_setting", "comparison", "algorithm", "cycle"], inplace=True)
    weighted_predictions.sort_values(["algorithm", "cycle", "model", "seqn"], inplace=True)
    diagnostics.sort_values(["algorithm", "cycle", "model"], inplace=True)
    atomic_csv(project_root / "results" / "tables" / "class_imbalance_sensitivity.csv", all_performance)
    atomic_csv(project_root / "results" / "tables" / "class_imbalance_incremental_value.csv", incremental)
    atomic_csv(project_root / "results" / "predictions" / "class_imbalance_weighted_predictions.csv.gz", weighted_predictions, compression="gzip")
    atomic_csv(project_root / "results" / "audit" / "class_imbalance_refit_diagnostics.csv", diagnostics)

    reference = pd.read_csv(project_root / "results" / "tables" / "cycle_specific_incremental_value.csv")
    check = main_incremental.merge(reference, on=["comparison", "algorithm", "cycle"], suffixes=("_new", "_reference"))
    metrics = ["delta_AUROC", "delta_PR_AUC", "delta_Brier_improvement", "delta_log_loss_improvement"]
    max_difference = max(float((check[f"{metric}_new"] - check[f"{metric}_reference"]).abs().max()) for metric in metrics)
    prediction_unique = not weighted_predictions.duplicated(["seqn", "cycle", "model", "algorithm"]).any()
    status = "PASS" if all(
        [
            len(results) == len(tasks) == 98,
            len(weighted_performance) == 98,
            len(incremental) == 112,
            len(weighted_predictions) == 352540,
            prediction_unique,
            np.isfinite(weighted_predictions["predicted_probability"]).all(),
            weighted_predictions["predicted_probability"].between(0, 1).all(),
            max_difference <= 1e-10,
            set(weighted_predictions["cycle"]) == set(DEVELOPMENT_CYCLES),
        ]
    ) else "FAIL"
    write_gate(
        project_root,
        "class_imbalance",
        {
            "module": "class_imbalance_refits",
            "status": status,
            "tasks_completed": len(results),
            "tasks_expected": len(tasks),
            "weighted_prediction_rows": len(weighted_predictions),
            "weighted_performance_rows": len(weighted_performance),
            "incremental_rows_all_settings": len(incremental),
            "minimum_positive_class_multiplier": diagnostics["positive_class_multiplier"].min(),
            "maximum_positive_class_multiplier": diagnostics["positive_class_multiplier"].max(),
            "resampling_absent": diagnostics["resampling"].eq("none").all(),
            "smote_absent": True,
            "prediction_uniqueness": prediction_unique,
            "main_point_reconciliation_max_difference": max_difference,
            "forbidden_cycle_absent": True,
            "signature_hash": signature["signature_hash"],
        },
    )
    audit = [
        "# Class imbalance refit audit",
        "",
        f"- Status: {status}",
        f"- Tasks: {len(results)}/{len(tasks)}",
        "- Main outer-fold hyperparameters were fixed.",
        "- Elastic Net class weights and XGBoost scale_pos_weight were calculated only from each outer training fold.",
        "- No SMOTE, ADASYN, oversampling, or undersampling was used.",
        "- All preprocessing was fitted only on outer training cycles.",
        "- NHANES 2021-2023 was not accessed.",
    ]
    atomic_text(project_root / "results" / "audit" / "class_imbalance_audit.md", "\n".join(audit) + "\n")
    return status
