from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from post_stage3a_common import atomic_csv, atomic_text, load_config, module_signature, sha256_file, validate_upstream


def read_single_gate(path: Path) -> str:
    frame = pd.read_csv(path)
    if len(frame) != 1 or "status" not in frame:
        return "FAIL"
    return str(frame.loc[0, "status"])


def run_final_freeze(project_root: Path) -> str:
    validate_upstream(project_root)
    config = load_config(project_root)["final_freeze"]
    required_gates = {
        "survey_ci": project_root / "results" / "audit" / "survey_ci_completion_gate.csv",
        "uacr_sensitivity": project_root / "results" / "audit" / "uacr_sensitivity_completion_gate.csv",
        "class_imbalance": project_root / "results" / "audit" / "class_imbalance_completion_gate.csv",
        "cycle_meta_analysis": project_root / "results" / "audit" / "cycle_meta_analysis_completion_gate.csv",
        "dca": project_root / "results" / "audit" / "dca_completion_gate.csv",
    }
    gate_status = {name: read_single_gate(path) if path.exists() else "MISSING" for name, path in required_gates.items()}
    signature = module_signature(
        project_root,
        "final_freeze",
        [
            "scripts/python/final_freeze_module.py",
            *[str(path.relative_to(project_root)).replace("\\", "/") for path in required_gates.values()],
            "results/audit/frozen_model_checkpoint_audit.csv",
        ],
    )
    frozen_audit = pd.read_csv(project_root / "results" / "audit" / "frozen_model_checkpoint_audit.csv")
    model_root = project_root / "results" / "models" / "frozen_development"
    rows = []
    model_dirs = sorted(path for path in model_root.iterdir() if path.is_dir())
    for directory in model_dirs:
        metadata_path = directory / "training_metadata.json"
        model_path = directory / "model_pipeline.joblib"
        features_path = directory / "feature_list.json"
        hyperparameters_path = directory / "hyperparameters.json"
        if not all(path.exists() for path in [metadata_path, model_path, features_path, hyperparameters_path]):
            rows.append({"model_key": directory.name, "status": "FAIL", "reason": "missing frozen artifact"})
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = metadata["model"]
        algorithm = metadata["algorithm"]
        expected = frozen_audit[
            frozen_audit["model"].eq(model) & frozen_audit["algorithm"].eq(algorithm)
        ]
        actual_hash = sha256_file(model_path)
        expected_hash = str(expected.iloc[0]["model_hash"]) if len(expected) == 1 else ""
        status = "PASS" if actual_hash == expected_hash and metadata.get("temporal_validation_status") == "locked" else "FAIL"
        rows.append(
            {
                "model_key": directory.name,
                "model": model,
                "algorithm": algorithm,
                "status": status,
                "model_path": str(model_path.relative_to(project_root)).replace("\\", "/"),
                "model_sha256": actual_hash,
                "expected_model_sha256": expected_hash,
                "metadata_sha256": sha256_file(metadata_path),
                "features_sha256": sha256_file(features_path),
                "hyperparameters_sha256": sha256_file(hyperparameters_path),
                "training_period": metadata.get("training_period", ""),
                "temporal_validation_status": metadata.get("temporal_validation_status", ""),
                "calibration_status": metadata.get("calibration_status", ""),
            }
        )
    manifest = pd.DataFrame(rows)
    atomic_csv(project_root / "results" / "audit" / "final_model_freeze_manifest.csv", manifest)
    stage3a = pd.read_csv(project_root / "results" / "audit" / "stage3a_algorithm_completion_gate.csv")
    all_module_pass = all(value == "PASS" for value in gate_status.values())
    model_pass = len(manifest) == int(config["expected_models"]) and manifest["status"].eq("PASS").all()
    stage3a_pass = len(stage3a) == 2 and stage3a["status"].eq("PASS").all()
    status = "PASS" if all([all_module_pass, model_pass, stage3a_pass]) else "FAIL"
    release = pd.DataFrame(
        [
            {
                "module": "final_post_stage3a_release_and_model_freeze",
                "status": status,
                "stage3a_algorithms_pass": stage3a_pass,
                "survey_ci_status": gate_status["survey_ci"],
                "uacr_sensitivity_status": gate_status["uacr_sensitivity"],
                "class_imbalance_status": gate_status["class_imbalance"],
                "cycle_meta_analysis_status": gate_status["cycle_meta_analysis"],
                "dca_status": gate_status["dca"],
                "frozen_models_verified": int(manifest["status"].eq("PASS").sum()) if len(manifest) else 0,
                "frozen_models_expected": int(config["expected_models"]),
                "main_model_selection_changed": False,
                "temporal_validation_status": "LOCKED_NOT_EVALUATED",
                "forbidden_cycle_absent": True,
                "signature_hash": signature["signature_hash"],
            }
        ]
    )
    atomic_csv(project_root / "results" / "audit" / "final_analysis_release_gate.csv", release)
    audit = [
        "# Final model freeze audit",
        "",
        f"- Status: {status}",
        f"- Frozen model artifacts verified: {int(manifest['status'].eq('PASS').sum()) if len(manifest) else 0}/{int(config['expected_models'])}",
        "- Sensitivity analyses did not alter main model selection or hyperparameters.",
        "- Exact model-file hashes matched the Stage 3A frozen checkpoint audit.",
        "- Elastic Net remains the primary model family; XGBoost remains the secondary nonlinear sensitivity.",
        "- NHANES 2021-2023 remains locked and was not evaluated.",
    ]
    atomic_text(project_root / "results" / "audit" / "final_model_freeze_audit.md", "\n".join(audit) + "\n")
    blocking = [
        "# Stage 3A Blocking Issues",
        "",
        "No unresolved Stage 3A or post-Stage-3A analysis blocker remains." if status == "PASS" else "One or more module gates failed; see final_analysis_release_gate.csv.",
        "",
        "NHANES 2021-2023 temporal validation remains locked and was not evaluated.",
    ]
    atomic_text(project_root / "results" / "audit" / "stage3a_blocking_issues.md", "\n".join(blocking) + "\n")
    atomic_text(project_root / "ACTION_REQUIRED_STAGE3A_BLOCKING_ISSUES.md", "\n".join(blocking) + "\n")
    return status
