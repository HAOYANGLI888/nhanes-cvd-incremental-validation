from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import stage3b_core as s3b
from stage3a_core import MODEL_DISPLAY, comparison_sample_label
from survey_ci_core import atomic_write_csv, atomic_write_text, canonical_json, sha256_file, sha256_text


CORRECTION_VERSION = "stage3b_matched_fasting_core_v1_1"
PAIRED_MODELS = ("model0_paired_metabolic", "model0_paired_combined")


def validate_upstream(root: Path) -> pd.DataFrame:
    release = pd.read_csv(root / "results/audit/final_analysis_release_gate.csv")
    if len(release) != 1 or release.iloc[0]["status"] != "PASS":
        raise RuntimeError("Final post-Stage-3A release gate is not PASS.")
    main = pd.read_csv(root / "results/audit/final_model_freeze_manifest.csv")
    paired = pd.read_csv(root / "results/audit/paired_fasting_core_freeze_manifest.csv")
    if len(main) != 10 or not main["status"].eq("PASS").all():
        raise RuntimeError("The original freeze manifest does not contain ten PASS models.")
    if len(paired) != 4 or not paired["status"].eq("PASS").all():
        raise RuntimeError("The matched fasting-core manifest does not contain four PASS aliases.")
    if not paired["candidate_limit"].isna().all():
        raise RuntimeError("A candidate limit was used for a matched fasting core.")
    for frame in (main, paired):
        for row in frame.itertuples(index=False):
            path = root / str(row.model_path)
            if sha256_file(path) != str(row.model_sha256):
                raise RuntimeError(f"Frozen model hash mismatch: {row.model_key}")
    combined = pd.concat([main, paired], ignore_index=True, sort=False)
    if combined["model_key"].duplicated().any() or len(combined) != 14:
        raise RuntimeError("Combined freeze manifest must contain fourteen unique model keys.")
    return combined.sort_values("model_key").reset_index(drop=True)


def signature(root: Path, config: dict[str, Any], manifest: pd.DataFrame) -> dict[str, Any]:
    inputs = [
        "config/stage3b_temporal_validation.yml",
        "config/temporal_validation_locked.yml",
        "protocol/stage3b_temporal_validation_protocol_locked.md",
        "protocol/stage3b_matched_core_correction_amendment.md",
        "data/interim/stage2_harmonized_audit_dataset.csv",
        "results/audit/final_analysis_release_gate.csv",
        "results/audit/final_model_freeze_manifest.csv",
        "results/audit/paired_fasting_core_freeze_manifest.csv",
        "results/audit/stage3b_v1_unmatched_fasting_core_archive/archive_manifest.csv",
    ]
    hashes = {rel: sha256_file(root / rel) for rel in inputs}
    hashes.update({str(row.model_path): str(row.model_sha256) for row in manifest.itertuples()})
    value = {
        "analysis_version": CORRECTION_VERSION,
        "protocol_basis": config["analysis_version"],
        "correction_reason": "Fasting comparisons require a fasting-trained matched Model 0 core.",
        "external_cycle": s3b.EXTERNAL_CYCLE,
        "authorization_date": config["authorization_date"],
        "input_hashes": hashes,
        "code_hash": sha256_file(Path(__file__).resolve()),
        "base_stage3b_code_hash": sha256_file(Path(s3b.__file__).resolve()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    value["signature_hash"] = sha256_text(canonical_json(value))
    return value


def build_predictions(root: Path, config: dict[str, Any], manifest: pd.DataFrame) -> pd.DataFrame:
    data = pd.read_csv(root / "data/interim/stage2_harmonized_audit_dataset.csv", low_memory=False)
    if set(data.loc[data["cycle"].eq(s3b.EXTERNAL_CYCLE), "cycle"].astype(str)) != {s3b.EXTERNAL_CYCLE}:
        raise RuntimeError("The external cycle is unavailable.")
    manifest_by_key = manifest.set_index("model_key")
    rows: list[pd.DataFrame] = []
    for algorithm in s3b.ALGORITHMS:
        loaded: dict[str, Any] = {}
        features: dict[str, list[str]] = {}
        for model in s3b.PREDICTION_MODELS:
            key = f"{model}_{algorithm}"
            item = manifest_by_key.loc[key]
            model_path = root / str(item["model_path"])
            loaded[model] = joblib.load(model_path)
            features[model] = json.loads((model_path.parent / "feature_list.json").read_text(encoding="utf-8"))
        for model in s3b.PREDICTION_MODELS:
            domain = config["sample_domains"][model]
            sample, weight_column = s3b.external_sample(data, domain, config)
            probability = np.asarray(loaded[model].predict_proba(sample[features[model]])[:, 1], dtype=float)
            probability = np.clip(probability, *map(float, config["probability_clip"]))
            if not np.isfinite(probability).all():
                raise RuntimeError(f"Non-finite probabilities: {model} {algorithm}")
            out = sample[["seqn", "cycle", "cvd", "strata", "psu"]].copy()
            out["model"] = model
            out["model_label"] = MODEL_DISPLAY[model]
            out["algorithm"] = algorithm
            out["predicted_probability"] = probability
            out["analysis_weight"] = sample[weight_column].to_numpy(dtype=float)
            out["weight_variable"] = weight_column
            out["target_population"] = "fasting subsample" if domain == "fasting" else "MEC examination sample"
            out["comparison_sample"] = comparison_sample_label(model)
            rows.append(out)
    predictions = pd.concat(rows, ignore_index=True)
    if predictions.groupby(["algorithm", "model"]).ngroups != 14:
        raise RuntimeError("Temporal prediction task count is incomplete.")
    return predictions


def checkpoint_paths(root: Path, ids: np.ndarray) -> tuple[Path, Path]:
    base = root / "results/checkpoints/stage3b_matched_core_v1_1"
    stem = f"batch_{int(ids[0]):04d}_{int(ids[-1]):04d}"
    return base / f"{stem}.npz", base / f"{stem}.json"


def relabel_outputs(root: Path) -> None:
    gate_path = root / "results/audit/stage3b_completion_gate.csv"
    gate = pd.read_csv(gate_path)
    gate["module"] = CORRECTION_VERSION
    old = "ten_frozen_model_hashes_match_before_and_after"
    if old in gate.columns:
        gate = gate.rename(columns={old: "fourteen_frozen_model_hashes_match_before_and_after"})
    atomic_write_csv(gate_path, gate)
    atomic_write_csv(root / "results/audit/stage3b_matched_core_correction_gate.csv", gate)
    status = str(gate.iloc[0]["status"])
    prediction = pd.read_csv(root / "results/predictions/stage3b_temporal_predictions.csv")
    audit = [
        "# Stage 3B matched-core temporal validation correction",
        "",
        f"- Correction version: `{CORRECTION_VERSION}`",
        f"- Status: **{status}**",
        "- External cycle: NHANES 2021-2023.",
        "- Original Stage 3B v1 outputs were archived with SHA-256 hashes before replacement.",
        "- Frozen pipelines verified: 14/14 (10 original models plus 4 fasting-core aliases).",
        "- Fasting metabolic and combined comparisons use cores trained in the identical fasting domain.",
        "- The two fasting-core aliases share one byte-identical fitted pipeline per algorithm because their development samples and specifications are identical.",
        "- No temporal-result-dependent hyperparameter tuning, feature selection, recalibration, or model updating was performed.",
        f"- Prediction groups: {prediction.groupby(['algorithm', 'model']).ngroups}/14.",
        "- Rao-Wu rescaled bootstrap replicates: 2000.",
        "",
        "## Interpretation",
        "",
        "This version is the canonical Stage 3B result for manuscript reporting. The archived v1 result is retained only as an audit trail of the implementation issue.",
    ]
    atomic_write_text(root / "results/audit/stage3b_temporal_validation_audit.md", "\n".join(audit) + "\n")
    atomic_write_text(root / "results/audit/stage3b_matched_core_correction_audit.md", "\n".join(audit) + "\n")


def run(root: Path, resume: bool, workers: int) -> int:
    s3b.validate_upstream = validate_upstream
    s3b.signature = signature
    s3b.build_predictions = build_predictions
    s3b.checkpoint_paths = checkpoint_paths
    code = s3b.run(root, resume, workers)
    relabel_outputs(root.resolve())
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the audited matched-fasting-core correction to Stage 3B.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    return run(Path(args.project_root), args.resume, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
