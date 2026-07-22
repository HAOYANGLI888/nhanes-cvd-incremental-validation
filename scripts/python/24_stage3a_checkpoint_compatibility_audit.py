"""Audit and attest legacy Elastic Net Stage 3A checkpoints.

This script never evaluates XGBoost and never reads 2021-2023 performance.
It independently recomputes one complete outer task for each legacy code-hash
group, then migrates compatible checkpoint metadata to the locked analysis
semantics signature without changing any stored result file.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.python import stage3a_core as core


ANALYTIC_FUNCTIONS = [
    "adult_nonpreg_outcome",
    "development_df",
    "sample_mask",
    "weight_column",
    "build_preprocessor",
    "elastic_net_candidates",
    "xgboost_candidates",
    "xgb_complexity_key",
    "en_complexity_key",
    "make_estimator",
    "make_pipeline",
    "fit_pipeline",
    "predict_proba",
    "weighted_mean",
    "weighted_log_loss",
    "weighted_brier",
    "safe_auc",
    "safe_ap",
    "calibration_stats",
    "youden_threshold",
    "class_metrics",
    "all_metrics",
    "metric_row",
    "get_X_y_w",
    "select_hyperparameters",
    "evaluate_candidate_inner",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def analytic_ast_hash(path: Path) -> tuple[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = sorted(set(ANALYTIC_FUNCTIONS) - set(functions))
    payload = []
    for name in ANALYTIC_FUNCTIONS:
        if name in functions:
            node = functions[name]
            for item in ast.walk(node):
                for attr in ["lineno", "col_offset", "end_lineno", "end_col_offset"]:
                    if hasattr(item, attr):
                        setattr(item, attr, None)
            payload.append((name, ast.dump(node, annotate_fields=True, include_attributes=False)))
    return core.sha256_text(core.canonical_json(payload)), missing


def read_raw_checkpoint(checkpoint_dir: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    meta_path = checkpoint_dir / "checkpoint.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frames = {}
    for key, record in meta["files"].items():
        path = checkpoint_dir / record["file"]
        if not path.exists() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Checkpoint file hash failure: {path}")
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            if int(record.get("rows", len(frame))) != len(frame):
                raise RuntimeError(f"Checkpoint row-count failure: {path}")
            frames[key] = frame
    return meta, frames


def recompute_outer_task(
    ctx: core.Stage3AContext,
    df: pd.DataFrame,
    features: dict[str, list[str]],
    fold: dict[str, Any],
    model_name: str,
) -> dict[str, pd.DataFrame]:
    algorithm = "elastic_net"
    holdout = str(fold["holdout_cycle"])
    train_cycles = list(fold["training_cycles"])
    candidates = core.elastic_net_candidates(ctx.project_root)
    core.log(ctx, f"Compatibility recomputation started: {holdout} {model_name}; candidates {len(candidates)}")
    selected, inner_rows = core.tune_one(
        ctx,
        df,
        features[model_name],
        model_name,
        algorithm,
        train_cycles,
        candidates,
        checkpoint_dir=None,
        checkpoint_signature=None,
        resume=False,
    )
    inner = pd.DataFrame([{**row, "outer_holdout_cycle": holdout, "fold": fold["fold"]} for row in inner_rows])
    params = {key: value for key, value in selected.items() if not key.startswith("_")}
    mask = core.sample_mask(df, model_name)
    train = df[mask & df["cycle"].isin(train_cycles)].copy()
    validation = df[mask & df["cycle"].eq(holdout)].copy()
    x_train, y_train, w_train = core.get_X_y_w(train, features[model_name], model_name)
    x_validation, y_validation, w_validation = core.get_X_y_w(validation, features[model_name], model_name)
    pipeline = core.make_pipeline(features[model_name], algorithm, train, params)
    core.fit_pipeline(pipeline, x_train, y_train, w_train)
    p_train = core.predict_proba(pipeline, x_train)
    p_validation = core.predict_proba(pipeline, x_validation)
    youden = core.youden_threshold(y_train, p_train, w_train)
    weight_var = core.weight_column(model_name)
    comparison_sample = core.comparison_sample_label(model_name)
    performance_rows = []
    for threshold, threshold_type in [(0.5, "fixed_0.50"), (youden, "training_weighted_youden")]:
        performance_rows.append(
            core.metric_row(
                model_name,
                algorithm,
                holdout,
                y_validation,
                p_validation,
                w_validation,
                threshold,
                threshold_type,
                len(validation),
                int(y_validation.sum()),
                weight_var,
                comparison_sample,
            )
        )
    selected_frame = pd.DataFrame(
        [
            {
                "fold": fold["fold"],
                "outer_holdout_cycle": holdout,
                "model": model_name,
                "algorithm": algorithm,
                "selected_candidate_id": selected["_selected_candidate_id"],
                "mean_inner_weighted_log_loss": selected["_mean_weighted_log_loss"],
                "sd_inner_weighted_log_loss": selected["_sd_weighted_log_loss"],
                "training_cycles": ";".join(train_cycles),
                "hyperparameters_json": json.dumps(params, sort_keys=True),
                "n_outer_train": len(train),
                "n_outer_validation": len(validation),
                "events_outer_validation": int(y_validation.sum()),
                "training_youden_threshold": youden,
                "config_hash": json.dumps(core.config_hashes_dict(ctx.project_root), sort_keys=True),
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "seqn": seqn,
                "cycle": cycle,
                "model": model_name,
                "algorithm": algorithm,
                "cvd": int(outcome),
                "predicted_probability": float(probability),
                "analysis_weight": float(weight),
                "weight_variable": weight_var,
                "strata": strata,
                "psu": psu,
                "outer_holdout_cycle": holdout,
                "comparison_sample": comparison_sample,
            }
            for seqn, cycle, outcome, probability, weight, strata, psu in zip(
                validation["seqn"],
                validation["cycle"],
                y_validation,
                p_validation,
                w_validation,
                validation["strata"],
                validation["psu"],
            )
        ]
    )
    core.log(ctx, f"Compatibility recomputation finished: {holdout} {model_name}")
    return {
        "performance": pd.DataFrame(performance_rows),
        "inner": inner,
        "selected": selected_frame,
        "predictions": predictions,
    }


def compare_frames(
    stored: pd.DataFrame,
    recomputed: pd.DataFrame,
    keys: list[str],
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "stored_rows": len(stored),
        "recomputed_rows": len(recomputed),
        "rtol": rtol,
        "atol": atol,
        "max_absolute_difference": 0.0,
        "mismatched_columns": [],
    }
    if len(stored) != len(recomputed) or set(stored.columns) != set(recomputed.columns):
        result["status"] = "FAIL"
        result["mismatched_columns"] = ["__shape_or_columns__"]
        return result
    left = stored.sort_values(keys).reset_index(drop=True)
    right = recomputed[left.columns].sort_values(keys).reset_index(drop=True)
    max_difference = 0.0
    mismatches = []
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
            a = left[column].to_numpy(dtype=float)
            b = right[column].to_numpy(dtype=float)
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.any():
                max_difference = max(max_difference, float(np.max(np.abs(a[finite] - b[finite]))))
            if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                mismatches.append(column)
        else:
            a = left[column].fillna("<NA>").astype(str).to_numpy()
            b = right[column].fillna("<NA>").astype(str).to_numpy()
            if not np.array_equal(a, b):
                mismatches.append(column)
    result["max_absolute_difference"] = max_difference
    result["mismatched_columns"] = mismatches
    result["status"] = "PASS" if not mismatches else "FAIL"
    return result


def select_representative(entries: list[dict[str, Any]]) -> dict[str, Any]:
    priority = {"model4_combined": 0, "model1_renal": 1, "model0_core": 2}
    return sorted(entries, key=lambda item: (priority.get(item["signature"]["model"], 9), item["signature"]["holdout_cycle"]))[0]


def migrate_checkpoint_metadata(
    meta_path: Path,
    base_signature: dict[str, Any],
    attestation_id: str,
    attestation_hash: str,
) -> None:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    signature = meta["signature"]
    signature["analysis_semantics_version"] = base_signature["analysis_semantics_version"]
    signature["analysis_semantics_hash"] = base_signature["analysis_semantics_hash"]
    signature["analysis_software"] = base_signature["analysis_software"]
    signature["compatibility_attestation_id"] = attestation_id
    signature["compatibility_attestation_sha256"] = attestation_hash
    meta["compatibility_migration"] = {
        "migrated_at": core.now(),
        "attestation_id": attestation_id,
        "attestation_sha256": attestation_hash,
        "original_code_hash_preserved": signature["code_hash"],
    }
    core.atomic_write_text(meta_path, json.dumps(meta, indent=2, sort_keys=True))


def run(project_root: Path) -> int:
    project_root = project_root.resolve()
    core.ensure_dirs(project_root)
    core.check_temporal_lock(project_root)
    log_path = project_root / "results/logs/stage3a_checkpoint_compatibility_audit.log"
    core.atomic_write_text(log_path, "")
    ctx = core.Stage3AContext(
        project_root=project_root,
        data_path=project_root / "data/interim/stage2_harmonized_audit_dataset.csv",
        run_time=core.now(),
        log_path=log_path,
    )
    base_signature = core.checkpoint_base_signature(project_root)
    current_code_hash = base_signature["code_hash"]
    archive_source = project_root / "archive/stage3a_checkpoint_compatibility/pre_fix_20260715/stage3a_core_pre_fix.py"
    if not archive_source.exists():
        raise FileNotFoundError(archive_source)
    archived_code_hash = sha256_file(archive_source)
    archived_ast_hash, archived_missing = analytic_ast_hash(archive_source)
    current_ast_hash, current_missing = analytic_ast_hash(Path(core.__file__).resolve())
    analytic_source_pass = archived_ast_hash == current_ast_hash and not archived_missing and not current_missing

    outer_entries: list[dict[str, Any]] = []
    for meta_path in sorted((project_root / "results/checkpoints/stage3a/outer").glob("*/checkpoint.json")):
        meta, frames = read_raw_checkpoint(meta_path.parent)
        signature = meta["signature"]
        if signature.get("algorithm") != "elastic_net":
            continue
        if signature.get("candidate_count") != 45 or signature.get("candidate_limit") is not None:
            raise RuntimeError(f"Non-full-grid Elastic Net checkpoint: {meta_path}")
        outer_entries.append({"meta_path": meta_path, "meta": meta, "signature": signature, "frames": frames})
    if len(outer_entries) != 49:
        raise RuntimeError(f"Expected 49 Elastic Net outer checkpoints, found {len(outer_entries)}")

    by_hash: dict[str, list[dict[str, Any]]] = {}
    for entry in outer_entries:
        by_hash.setdefault(entry["signature"]["code_hash"], []).append(entry)
    legacy_hashes = sorted(code_hash for code_hash in by_hash if code_hash != archived_code_hash)
    if len(legacy_hashes) != 2:
        raise RuntimeError(f"Expected two legacy code-hash groups besides archived baseline; found {legacy_hashes}")

    dataframe = core.development_df(core.read_harmonized_data(project_root))
    features = core.read_frozen_features(project_root)
    folds = {str(fold["holdout_cycle"]): fold for fold in core.read_cycle_folds(project_root)}
    equivalence_results = []
    for code_hash in legacy_hashes:
        representative = select_representative(by_hash[code_hash])
        signature = representative["signature"]
        holdout = str(signature["holdout_cycle"])
        model_name = str(signature["model"])
        recomputed = recompute_outer_task(ctx, dataframe, features, folds[holdout], model_name)
        comparisons = {
            "inner": compare_frames(
                representative["frames"]["inner"],
                recomputed["inner"],
                ["candidate_id", "inner_validation_cycle"],
            ),
            "selected": compare_frames(representative["frames"]["selected"], recomputed["selected"], ["model"]),
            "predictions": compare_frames(
                representative["frames"]["predictions"],
                recomputed["predictions"],
                ["seqn", "cycle"],
            ),
            "performance": compare_frames(
                representative["frames"]["performance"],
                recomputed["performance"],
                ["classification_threshold_type"],
            ),
        }
        status = "PASS" if all(item["status"] == "PASS" for item in comparisons.values()) else "FAIL"
        equivalence_results.append(
            {
                "legacy_code_hash": code_hash,
                "representative_holdout": holdout,
                "representative_model": model_name,
                "status": status,
                "comparisons": comparisons,
            }
        )

    equivalence_pass = all(result["status"] == "PASS" for result in equivalence_results)
    overall_pass = bool(analytic_source_pass and equivalence_pass)
    attestation_id = f"stage3a-compat-{core.now().replace(':', '').replace('-', '')}"
    compatible_hashes = sorted(set(by_hash) | {archived_code_hash, current_code_hash}) if overall_pass else []
    attestation = {
        "attestation_id": attestation_id,
        "created": core.now(),
        "status": "PASS" if overall_pass else "FAIL",
        "scope": "NHANES 2005-2018 Elastic Net Stage 3A checkpoints only",
        "xgboost_evaluated": False,
        "external_cycle_performance_accessed": False,
        "analysis_semantics_version": core.ANALYSIS_SEMANTICS_VERSION,
        "analysis_semantics_hash": core.analysis_semantics_hash(),
        "analysis_software": core.analysis_software_versions(),
        "input_hash": base_signature["input_hash"],
        "config_hashes": base_signature["config_hashes"],
        "current_code_hash": current_code_hash,
        "archived_pre_fix_code_hash": archived_code_hash,
        "compatible_code_hashes": compatible_hashes,
        "analytic_ast_comparison": {
            "status": "PASS" if analytic_source_pass else "FAIL",
            "archived_ast_hash": archived_ast_hash,
            "current_ast_hash": current_ast_hash,
            "archived_missing_functions": archived_missing,
            "current_missing_functions": current_missing,
            "functions": ANALYTIC_FUNCTIONS,
        },
        "legacy_full_task_equivalence": equivalence_results,
        "policy": "Unknown code hashes fail. Code-hash exceptions require this immutable PASS attestation and identical analytic signatures.",
    }
    attestation_path = project_root / core.CHECKPOINT_ATTESTATION_RELATIVE_PATH
    core.atomic_write_text(attestation_path, json.dumps(attestation, indent=2, sort_keys=True))
    attestation_hash = sha256_file(attestation_path)

    audit_rows = []
    if overall_pass:
        metadata_paths = [entry["meta_path"] for entry in outer_entries]
        metadata_paths.extend(sorted((project_root / "results/checkpoints/stage3a/frozen").glob("*/checkpoint.json")))
        for meta_path in metadata_paths:
            migrate_checkpoint_metadata(meta_path, base_signature, attestation_id, attestation_hash)
        for meta_path in metadata_paths:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            signature = meta["signature"]
            status = core.checkpoint_compatibility_status(project_root, signature)
            audit_rows.append(
                {
                    "checkpoint": str(meta_path.relative_to(project_root)),
                    "checkpoint_type": signature["checkpoint_type"],
                    "algorithm": signature["algorithm"],
                    "model": signature["model"],
                    "holdout_cycle": signature.get("holdout_cycle", "all"),
                    "original_code_hash": signature["code_hash"],
                    "analysis_semantics_hash": signature["analysis_semantics_hash"],
                    "attestation_id": signature["compatibility_attestation_id"],
                    "compatibility_status": status,
                }
            )
        if not all(row["compatibility_status"] == "PASS" for row in audit_rows):
            raise RuntimeError("Post-migration compatibility verification failed")

    audit_frame = pd.DataFrame(audit_rows)
    core.atomic_write_csv(audit_frame, project_root / "results/audit/stage3a_checkpoint_compatibility_audit.csv")
    lines = [
        "# Stage 3A Checkpoint Compatibility Audit",
        "",
        f"- Status: {'PASS' if overall_pass else 'FAIL'}.",
        f"- Analysis semantics: {core.ANALYSIS_SEMANTICS_VERSION} ({core.analysis_semantics_hash()}).",
        f"- Archived pre-fix analytic AST comparison: {'PASS' if analytic_source_pass else 'FAIL'}.",
        f"- Legacy code-hash groups fully recomputed: {len(equivalence_results)}.",
        f"- Migrated compatible outer/frozen checkpoints: {len(audit_rows)}.",
        "- XGBoost was not evaluated.",
        "- NHANES 2021-2023 performance was not accessed.",
    ]
    for result in equivalence_results:
        lines.append(
            f"- {result['legacy_code_hash'][:12]}: {result['status']} using "
            f"{result['representative_holdout']} {result['representative_model']}."
        )
    core.atomic_write_text(project_root / "results/audit/stage3a_checkpoint_compatibility_audit.md", "\n".join(lines) + "\n")
    return 0 if overall_pass else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage 3A Elastic Net checkpoint compatibility.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    if args.n_jobs is not None:
        os.environ["STAGE3A_N_JOBS"] = str(max(1, args.n_jobs))
    return run(Path(args.project_root))


if __name__ == "__main__":
    raise SystemExit(main())
