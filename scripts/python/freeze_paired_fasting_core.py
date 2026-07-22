from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

import stage3a_core as s3a


CORRECTION_VERSION = "stage3b_matched_fasting_core_v1_1"
SOURCE_MODEL = "model0_paired_metabolic"
ALIASES = ("model0_paired_metabolic", "model0_paired_combined")


def publish_alias(root: Path, source_dir: Path, alias: str, algorithm: str, source_meta: dict) -> dict:
    destination = root / "results/models/frozen_paired_core_correction" / f"{alias}_{algorithm}"
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("model_pipeline.joblib", "feature_list.json", "hyperparameters.json"):
        s3a.atomic_copy(source_dir / filename, destination / filename)
    metadata = {
        **source_meta,
        "model": alias,
        "correction_version": CORRECTION_VERSION,
        "shared_fit_source": SOURCE_MODEL,
        "shared_fit_justification": (
            "The metabolic and combined comparisons use the identical fasting-subsample "
            "eligibility mask, Model 0 feature set, weights, training cycles, and locked grid."
        ),
    }
    metadata_path = destination / "training_metadata.json"
    s3a.atomic_write_text(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    artifact = destination / "model_pipeline.joblib"
    return {
        "model_key": f"{alias}_{algorithm}",
        "model": alias,
        "algorithm": algorithm,
        "status": "PASS",
        "model_path": str(artifact.relative_to(root)).replace("\\", "/"),
        "model_sha256": s3a.sha256_file(artifact),
        "metadata_sha256": s3a.sha256_file(metadata_path),
        "features_sha256": s3a.sha256_file(destination / "feature_list.json"),
        "hyperparameters_sha256": s3a.sha256_file(destination / "hyperparameters.json"),
        "candidate_count": int(source_meta["candidate_count"]),
        "candidate_limit": source_meta.get("candidate_limit"),
        "candidate_grid_hash": source_meta["candidate_grid_hash"],
        "n_training": int(source_meta["n_training"]),
        "events_training": int(source_meta["events_training"]),
        "weight_variable": source_meta["weight_variable"],
        "selected_candidate_id": source_meta["selected_candidate_id"],
        "mean_inner_weighted_log_loss": source_meta["mean_inner_weighted_log_loss"],
        "training_period": "NHANES 2005-2018",
        "correction_version": CORRECTION_VERSION,
    }


def fit_one(root: Path, ctx: s3a.Stage3AContext, algorithm: str, resume: bool) -> tuple[Path, dict]:
    data = s3a.development_df(s3a.read_harmonized_data(root))
    features = s3a.read_frozen_features(root)[SOURCE_MODEL]
    candidates, limit = s3a.candidates_for_algorithm(root, algorithm, None, None)
    base = s3a.checkpoint_base_signature(root)
    signature = s3a.frozen_task_signature(base, SOURCE_MODEL, algorithm, features, candidates, limit)
    signature["correction_version"] = CORRECTION_VERSION
    signature["correction_code_hash"] = s3a.sha256_file(Path(__file__).resolve())
    loaded = s3a.load_frozen_task(root, SOURCE_MODEL, algorithm, signature) if resume else None
    checkpoint_dir = s3a.frozen_checkpoint_dir(root, SOURCE_MODEL, algorithm)
    if loaded is not None:
        checkpoint, metadata = loaded
        s3a.log(ctx, f"Resume hit corrected fasting core {algorithm}")
        return checkpoint_dir, metadata

    s3a.log(ctx, f"Tuning corrected fasting core {algorithm}; full locked grid candidates={len(candidates)}")
    selected, _ = s3a.tune_one(
        ctx,
        data,
        features,
        SOURCE_MODEL,
        algorithm,
        s3a.DEVELOPMENT_CYCLES,
        candidates,
        checkpoint_dir=checkpoint_dir,
        checkpoint_signature=signature,
        resume=resume,
    )
    params = {k: v for k, v in selected.items() if not k.startswith("_")}
    train = data[s3a.sample_mask(data, SOURCE_MODEL)].copy()
    X, y, w = s3a.get_X_y_w(train, features, SOURCE_MODEL)
    pipeline = s3a.make_pipeline(features, algorithm, train, params)
    s3a.fit_pipeline(pipeline, X, y, w)
    metadata = {
        "model": SOURCE_MODEL,
        "algorithm": algorithm,
        "training_period": "NHANES 2005-2018",
        "n_training": len(train),
        "events_training": int(y.sum()),
        "weight_variable": s3a.weight_column(SOURCE_MODEL),
        "calibration_status": "none",
        "temporal_validation_status": "audit_correction; no temporal-result-dependent tuning",
        "software": s3a.software_versions(),
        "config_hash": s3a.config_hashes_dict(root),
        "creation_timestamp": s3a.now(),
        "selected_candidate_id": selected["_selected_candidate_id"],
        "mean_inner_weighted_log_loss": selected["_mean_weighted_log_loss"],
        "sd_inner_weighted_log_loss": selected["_sd_weighted_log_loss"],
        "candidate_count": len(candidates),
        "candidate_limit": limit,
        "candidate_grid_hash": signature["candidate_grid_hash"],
        "correction_version": CORRECTION_VERSION,
    }
    _, metadata = s3a.write_frozen_task(checkpoint_dir, signature, pipeline, features, params, metadata)
    s3a.log(ctx, f"Saved corrected fasting core checkpoint {algorithm}")
    return checkpoint_dir, metadata


def run(root: Path, algorithms: list[str], resume: bool) -> int:
    root = root.resolve()
    s3a.ensure_dirs(root)
    log_path = root / "results/logs/stage3b_matched_core_correction.log"
    ctx = s3a.Stage3AContext(root, root / "data/interim/stage2_harmonized_audit_dataset.csv", s3a.now(), log_path)
    if not resume:
        log_path.write_text("", encoding="utf-8")
    rows = []
    existing_path = root / "results/audit/paired_fasting_core_freeze_manifest.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        existing = existing[~existing["algorithm"].isin(algorithms)].copy()
        for item in existing.to_dict(orient="records"):
            artifact = root / str(item["model_path"])
            if item.get("status") == "PASS" and artifact.exists() and s3a.sha256_file(artifact) == str(item.get("model_sha256")):
                rows.append(item)
    for algorithm in algorithms:
        source_dir, meta = fit_one(root, ctx, algorithm, resume)
        for alias in ALIASES:
            rows.append(publish_alias(root, source_dir, alias, algorithm, meta))
    manifest = pd.DataFrame(rows).sort_values(["algorithm", "model"]).reset_index(drop=True)
    expected_algorithms = sorted(set(manifest["algorithm"]))
    expected = len(expected_algorithms) * len(ALIASES)
    full_counts = {"elastic_net": len(s3a.elastic_net_candidates(root)), "xgboost": len(s3a.xgboost_candidates(root))}
    passed = (
        len(manifest) == expected
        and manifest["status"].eq("PASS").all()
        and manifest["candidate_limit"].isna().all()
        and all(int(row.candidate_count) == full_counts[row.algorithm] for row in manifest.itertuples())
    )
    s3a.atomic_write_csv(manifest, root / "results/audit/paired_fasting_core_freeze_manifest.csv")
    audit = [
        "# Matched fasting core freeze audit",
        "",
        f"- Correction version: `{CORRECTION_VERSION}`",
        f"- Status: **{'PASS' if passed else 'FAIL'}**",
        "- Development data only: NHANES 2005-2018.",
        "- Selection metric and preprocessing: unchanged from the locked Stage 3A protocol.",
        "- Candidate limits: absent; complete locked grids were evaluated.",
        "- The two fasting-core aliases are byte-identical within algorithm because their sample, features, weights, and tuning rule are identical.",
        "- NHANES 2021-2023 performance was not used for tuning, selection, recalibration, or model updating.",
        "",
        "## Frozen artifacts",
        "",
    ]
    for row in manifest.itertuples():
        audit.append(
            f"- {row.model_key}: candidates={row.candidate_count}; n={row.n_training}; events={row.events_training}; sha256={row.model_sha256}"
        )
    s3a.atomic_write_text(root / "results/audit/paired_fasting_core_freeze_audit.md", "\n".join(audit) + "\n")
    s3a.log(ctx, f"Matched fasting core freeze gate={'PASS' if passed else 'FAIL'}; artifacts={len(manifest)}/{expected}")
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze fasting-matched Model 0 pipelines for the Stage 3B audit correction.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--algorithms", default="elastic_net,xgboost")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    algorithms = [x.strip() for x in args.algorithms.split(",") if x.strip()]
    unsupported = set(algorithms) - set(s3a.SUPPORTED_ALGORITHMS)
    if unsupported:
        raise ValueError(f"Unsupported algorithms: {sorted(unsupported)}")
    return run(Path(args.project_root), algorithms, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
