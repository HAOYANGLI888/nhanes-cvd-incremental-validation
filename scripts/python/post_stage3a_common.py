from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import stage3a_core as core


DEVELOPMENT_CYCLES = list(core.DEVELOPMENT_CYCLES)
SUPPORTED_ALGORITHMS = ["elastic_net", "xgboost"]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    frame.to_csv(tmp, index=False, compression=compression)
    os.replace(tmp, path)


def log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{now()} | {message}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def load_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "config" / "post_stage3a_sensitivity_locked.yml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("analysis_version") != "post_stage3a_locked_v1":
        raise RuntimeError("Unexpected post-Stage-3A analysis version.")
    if list(config.get("development_cycles", [])) != DEVELOPMENT_CYCLES:
        raise RuntimeError("Development-cycle lock mismatch.")
    return config


def validate_upstream(project_root: Path, require_survey_ci: bool = True) -> None:
    stage3a = pd.read_csv(project_root / "results" / "audit" / "stage3a_algorithm_completion_gate.csv")
    if set(stage3a["algorithm"]) != set(SUPPORTED_ALGORITHMS) or not stage3a["status"].eq("PASS").all():
        raise RuntimeError("Both Stage 3A algorithms must pass.")
    if require_survey_ci:
        survey = pd.read_csv(project_root / "results" / "audit" / "survey_ci_completion_gate.csv")
        if len(survey) != 1 or survey.loc[0, "status"] != "PASS":
            raise RuntimeError("Survey CI must pass.")


def module_signature(project_root: Path, module: str, extra_paths: list[str] | None = None) -> dict[str, Any]:
    paths = [
        "config/post_stage3a_sensitivity_locked.yml",
        "data/interim/stage2_harmonized_audit_dataset.csv",
        "results/models/outer_fold_hyperparameters.csv",
        "results/predictions/stage3a_outer_predictions.csv",
        "results/audit/stage3a_algorithm_completion_gate.csv",
        "scripts/python/stage3a_core.py",
    ]
    paths.extend(extra_paths or [])
    hashes = {}
    for relative in paths:
        path = project_root / relative
        if not path.exists():
            raise FileNotFoundError(path)
        hashes[relative] = sha256_file(path)
    value = {
        "analysis_version": "post_stage3a_locked_v1",
        "module": module,
        "input_hashes": hashes,
        "python": sys.version.split()[0],
    }
    value["signature_hash"] = sha256_text(canonical_json(value))
    return value


def load_development_data(project_root: Path) -> pd.DataFrame:
    data = core.development_df(core.read_harmonized_data(project_root))
    observed = set(data["cycle"].astype(str))
    if observed != set(DEVELOPMENT_CYCLES):
        raise RuntimeError(f"Unexpected development cycles: {sorted(observed)}")
    if data.get("is_external_2021_2023", pd.Series(False, index=data.index)).fillna(False).astype(bool).any():
        raise RuntimeError("External temporal-validation rows entered post-Stage-3A analysis.")
    return data


def load_selected_parameters(project_root: Path) -> pd.DataFrame:
    selected = pd.read_csv(project_root / "results" / "models" / "outer_fold_hyperparameters.csv")
    if len(selected) != 98:
        raise RuntimeError(f"Expected 98 selected outer task rows, found {len(selected)}.")
    if selected.duplicated(["outer_holdout_cycle", "model", "algorithm"]).any():
        raise RuntimeError("Duplicate selected outer hyperparameters.")
    return selected


def selected_params(selected: pd.DataFrame, cycle: str, model: str, algorithm: str) -> dict[str, Any]:
    row = selected[
        selected["outer_holdout_cycle"].eq(cycle)
        & selected["model"].eq(model)
        & selected["algorithm"].eq(algorithm)
    ]
    if len(row) != 1:
        raise RuntimeError(f"Missing selected parameters for {cycle} {model} {algorithm}.")
    return json.loads(row.iloc[0]["hyperparameters_json"])


def fit_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    model: str,
    algorithm: str,
    params: dict[str, Any],
    train_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    Xtr = train[features].copy()
    Xva = validation[features].copy()
    ytr = train["cvd"].astype(int).to_numpy()
    if train_weight is None:
        train_weight = train[core.weight_column(model)].astype(float).to_numpy()
    pipe = core.make_pipeline(features, algorithm, train, params)
    core.fit_pipeline(pipe, Xtr, ytr, np.asarray(train_weight, dtype=float))
    return core.predict_proba(pipe, Xtr), core.predict_proba(pipe, Xva)


def performance_row(
    frame: pd.DataFrame,
    probability: np.ndarray,
    weight: np.ndarray,
    model: str,
    algorithm: str,
    cycle: str,
    method_column: str,
    method: str,
) -> dict[str, Any]:
    y = frame["cvd"].astype(int).to_numpy()
    row = core.metric_row(
        model,
        algorithm,
        cycle,
        y,
        probability,
        np.asarray(weight, dtype=float),
        0.5,
        "fixed_0.50",
        len(frame),
        int(y.sum()),
        core.weight_column(model),
        core.comparison_sample_label(model),
    )
    row[method_column] = method
    return row


def paired_increment_row(
    frame: pd.DataFrame,
    core_probability: np.ndarray,
    extended_probability: np.ndarray,
    weight: np.ndarray,
    comparison: str,
    algorithm: str,
    cycle: str,
    core_model: str,
    extended_model: str,
    method_column: str,
    method: str,
) -> dict[str, Any]:
    y = frame["cvd"].astype(int).to_numpy()
    weight = np.asarray(weight, dtype=float)
    mc = core.all_metrics(y, core_probability, weight, 0.5)
    me = core.all_metrics(y, extended_probability, weight, 0.5)
    row = {
        "comparison": comparison,
        "algorithm": algorithm,
        "cycle": cycle,
        "n": len(frame),
        "events": int(y.sum()),
        "weight_variable": core.weight_column(extended_model),
        "core_model": core_model,
        "extended_model": extended_model,
        "delta_AUROC": me["survey_weighted_AUROC"] - mc["survey_weighted_AUROC"],
        "delta_PR_AUC": me["survey_weighted_PR_AUC"] - mc["survey_weighted_PR_AUC"],
        "delta_Brier_improvement": mc["weighted_Brier"] - me["weighted_Brier"],
        "delta_log_loss_improvement": mc["weighted_log_loss"] - me["weighted_log_loss"],
        "core_calibration_intercept": mc["calibration_intercept"],
        "extended_calibration_intercept": me["calibration_intercept"],
        "core_calibration_slope": mc["calibration_slope"],
        "extended_calibration_slope": me["calibration_slope"],
        "absolute_intercept_deviation_change": abs(mc["calibration_intercept"]) - abs(me["calibration_intercept"]),
        "absolute_slope_deviation_change": abs(mc["calibration_slope"] - 1) - abs(me["calibration_slope"] - 1),
        method_column: method,
    }
    return row


def checkpoint_dir(project_root: Path, module: str, task_id: str) -> Path:
    safe = task_id.replace("/", "_").replace("\\", "_")
    return project_root / "results" / "checkpoints" / "post_stage3a" / module / safe


def save_task_frames(path: Path, signature_hash: str, frames: dict[str, pd.DataFrame]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"status": "complete", "signature_hash": signature_hash, "created": now(), "files": {}}
    for name, frame in frames.items():
        target = path / f"{name}.csv.gz"
        atomic_csv(target, frame, compression="gzip")
        manifest["files"][name] = {"name": target.name, "sha256": sha256_file(target), "rows": len(frame)}
    atomic_text(path / "checkpoint.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def load_task_frames(path: Path, signature_hash: str) -> dict[str, pd.DataFrame] | None:
    metadata_path = path / "checkpoint.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete" or metadata.get("signature_hash") != signature_hash:
            return None
        frames = {}
        for name, info in metadata["files"].items():
            target = path / info["name"]
            if not target.exists() or sha256_file(target) != info["sha256"]:
                return None
            frame = pd.read_csv(target)
            if len(frame) != int(info["rows"]):
                return None
            frames[name] = frame
        return frames
    except Exception:
        return None


def write_gate(project_root: Path, name: str, row: dict[str, Any]) -> None:
    atomic_csv(project_root / "results" / "audit" / f"{name}_completion_gate.csv", pd.DataFrame([row]))
