from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from stage3a_core import MODEL_DISPLAY, comparison_sample_label
from survey_ci_core import (
    INCREMENTAL_METRICS,
    PERFORMANCE_METRICS,
    atomic_write_csv,
    atomic_write_csv_gzip,
    atomic_write_text,
    canonical_json,
    design_df_for_clusters,
    make_metric_data,
    metric_vector,
    rao_wu_factors,
    sha256_file,
    sha256_text,
    summarize_ci,
    wide_replicate_frame,
)


EXTERNAL_CYCLE = "2021-2023"
MODEL_NAMES = ["model0_core", "model1_renal", "model2_metabolic", "model3_inflammatory", "model4_combined"]
ALGORITHMS = ["elastic_net", "xgboost"]
PREDICTION_MODELS = [
    "model0_core",
    "model1_renal",
    "model2_metabolic",
    "model3_inflammatory",
    "model4_combined",
    "model0_paired_metabolic",
    "model0_paired_combined",
]
CORE_SOURCE = {
    "model0_paired_metabolic": "model0_core",
    "model0_paired_combined": "model0_core",
}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{now()} | {message}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(tmp, path)


def load_config(root: Path) -> dict[str, Any]:
    config = yaml.safe_load((root / "config/stage3b_temporal_validation.yml").read_text(encoding="utf-8"))
    if config.get("analysis_version") != "stage3b_locked_v1" or config.get("authorization_status") != "AUTHORIZED":
        raise RuntimeError("Stage 3B is not explicitly authorized under the locked v1 protocol.")
    if config.get("external_cycle") != EXTERNAL_CYCLE:
        raise RuntimeError("Unexpected temporal validation cycle.")
    if list(config.get("performance_metrics", [])) != PERFORMANCE_METRICS:
        raise RuntimeError("Performance metrics differ from the locked implementation.")
    if list(config.get("incremental_metrics", [])) != INCREMENTAL_METRICS:
        raise RuntimeError("Incremental metrics differ from the locked implementation.")
    if int(config["uncertainty"]["replicates"]) != 2000:
        raise RuntimeError("Stage 3B requires exactly 2,000 bootstrap replicates.")
    temporal = yaml.safe_load((root / "config/temporal_validation_locked.yml").read_text(encoding="utf-8"))
    if not temporal.get("stage3b_authorized", False):
        raise RuntimeError("Temporal validation authorization flag is absent.")
    return config


def validate_upstream(root: Path) -> pd.DataFrame:
    release = pd.read_csv(root / "results/audit/final_analysis_release_gate.csv")
    if len(release) != 1 or release.iloc[0]["status"] != "PASS":
        raise RuntimeError("Final post-Stage-3A release gate is not PASS.")
    manifest = pd.read_csv(root / "results/audit/final_model_freeze_manifest.csv")
    if len(manifest) != 10 or not manifest["status"].eq("PASS").all():
        raise RuntimeError("The freeze manifest does not contain ten PASS models.")
    for _, row in manifest.iterrows():
        path = root / str(row["model_path"])
        observed = sha256_file(path)
        if observed != str(row["model_sha256"]) or observed != str(row["expected_model_sha256"]):
            raise RuntimeError(f"Frozen model hash mismatch: {row['model_key']}")
    return manifest


def signature(root: Path, config: dict[str, Any], manifest: pd.DataFrame) -> dict[str, Any]:
    inputs = [
        "config/stage3b_temporal_validation.yml",
        "config/temporal_validation_locked.yml",
        "protocol/stage3b_temporal_validation_protocol_locked.md",
        "data/interim/stage2_harmonized_audit_dataset.csv",
        "results/audit/final_analysis_release_gate.csv",
        "results/audit/final_model_freeze_manifest.csv",
    ]
    hashes = {rel: sha256_file(root / rel) for rel in inputs}
    hashes.update({str(row["model_path"]): str(row["model_sha256"]) for _, row in manifest.iterrows()})
    value = {
        "analysis_version": config["analysis_version"],
        "external_cycle": EXTERNAL_CYCLE,
        "authorization_date": config["authorization_date"],
        "input_hashes": hashes,
        "code_hash": sha256_file(Path(__file__).resolve()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    value["signature_hash"] = sha256_text(canonical_json(value))
    return value


def external_sample(frame: pd.DataFrame, domain: str, config: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    weight = config["weights"][domain]
    mask = (
        frame["cycle"].eq(EXTERNAL_CYCLE)
        & frame["age"].ge(20)
        & (~frame["pregnancy_code"].eq(1).fillna(False))
        & frame["cvd"].notna()
        & frame["strata"].notna()
        & frame["psu"].notna()
        & frame[weight].notna()
        & frame[weight].gt(0)
    )
    out = frame.loc[mask].copy().sort_values("seqn", kind="mergesort")
    if out.empty or out["cvd"].nunique() != 2:
        raise RuntimeError(f"Temporal {domain} sample is empty or lacks both outcome classes.")
    return out, weight


def build_predictions(root: Path, config: dict[str, Any], manifest: pd.DataFrame) -> pd.DataFrame:
    data = pd.read_csv(root / "data/interim/stage2_harmonized_audit_dataset.csv", low_memory=False)
    if set(data.loc[data["cycle"].eq(EXTERNAL_CYCLE), "cycle"].astype(str)) != {EXTERNAL_CYCLE}:
        raise RuntimeError("The external cycle is unavailable.")
    rows: list[pd.DataFrame] = []
    manifest_by_key = manifest.set_index("model_key")
    for algorithm in ALGORITHMS:
        loaded: dict[str, Any] = {}
        features: dict[str, list[str]] = {}
        for model in MODEL_NAMES:
            key = f"{model}_{algorithm}"
            item = manifest_by_key.loc[key]
            loaded[model] = joblib.load(root / str(item["model_path"]))
            feature_path = (root / str(item["model_path"])).parent / "feature_list.json"
            features[model] = json.loads(feature_path.read_text(encoding="utf-8"))
        for prediction_model in PREDICTION_MODELS:
            source_model = CORE_SOURCE.get(prediction_model, prediction_model)
            domain = config["sample_domains"][prediction_model]
            sample, weight_column = external_sample(data, domain, config)
            probability = np.asarray(loaded[source_model].predict_proba(sample[features[source_model]])[:, 1], dtype=float)
            probability = np.clip(probability, *map(float, config["probability_clip"]))
            if not np.isfinite(probability).all():
                raise RuntimeError(f"Non-finite probabilities: {prediction_model} {algorithm}")
            out = sample[["seqn", "cycle", "cvd", "strata", "psu"]].copy()
            out["model"] = prediction_model
            out["model_label"] = MODEL_DISPLAY[prediction_model]
            out["algorithm"] = algorithm
            out["predicted_probability"] = probability
            out["analysis_weight"] = sample[weight_column].to_numpy(dtype=float)
            out["weight_variable"] = weight_column
            out["target_population"] = "fasting subsample" if domain == "fasting" else "MEC examination sample"
            out["comparison_sample"] = comparison_sample_label(prediction_model)
            rows.append(out)
    predictions = pd.concat(rows, ignore_index=True)
    expected = len(ALGORITHMS) * len(PREDICTION_MODELS)
    if predictions.groupby(["algorithm", "model"]).ngroups != expected:
        raise RuntimeError("Temporal prediction task count is incomplete.")
    return predictions


def prepare_design(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[np.ndarray]]:
    design = predictions[["seqn", "cycle", "strata", "psu"]].drop_duplicates()
    clusters = design[["cycle", "strata", "psu"]].drop_duplicates().sort_values(["strata", "psu"], kind="mergesort").reset_index(drop=True)
    clusters["cluster_index"] = np.arange(len(clusters), dtype=np.int32)
    counts = clusters.groupby(["cycle", "strata"])["psu"].nunique()
    if (counts < 2).any():
        raise RuntimeError(f"Singleton temporal survey strata: {counts[counts < 2].to_dict()}")
    predictions = predictions.merge(clusters, on=["cycle", "strata", "psu"], how="left", validate="many_to_one")
    strata_indices = [group["cluster_index"].to_numpy(dtype=np.int32) for _, group in clusters.groupby(["cycle", "strata"], sort=False)]
    return predictions, clusters, strata_indices


def groups_from_predictions(predictions: pd.DataFrame, clusters: pd.DataFrame, config: dict[str, Any]):
    perf_keys: list[dict[str, Any]] = []
    perf_data = []
    perf_points = []
    design_df = []
    index_by_pair: dict[tuple[str, str], int] = {}
    for algorithm in ALGORITHMS:
        for model in PREDICTION_MODELS:
            group = predictions[predictions["algorithm"].eq(algorithm) & predictions["model"].eq(model)].sort_values("seqn", kind="mergesort")
            data = make_metric_data(group["cvd"].to_numpy(), group["predicted_probability"].to_numpy(), group["analysis_weight"].to_numpy(), group["cluster_index"].to_numpy())
            key = {
                "cycle": EXTERNAL_CYCLE,
                "algorithm": algorithm,
                "model": model,
                "model_label": group["model_label"].iloc[0],
                "target_population": group["target_population"].iloc[0],
                "comparison_sample": group["comparison_sample"].iloc[0],
                "n": len(group),
                "events": int(group["cvd"].sum()),
                "weight_variable": group["weight_variable"].iloc[0],
            }
            index_by_pair[(algorithm, model)] = len(perf_data)
            perf_keys.append(key)
            perf_data.append(data)
            perf_points.append(metric_vector(data))
            design_df.append(design_df_for_clusters(clusters, data.cluster_index))
    inc_keys: list[dict[str, Any]] = []
    inc_pairs: list[tuple[int, int]] = []
    inc_points: list[np.ndarray] = []
    for algorithm in ALGORITHMS:
        for comparison, pair in config["comparisons"].items():
            core, extended = pair
            ic = index_by_pair[(algorithm, core)]
            ie = index_by_pair[(algorithm, extended)]
            cg = predictions[predictions["algorithm"].eq(algorithm) & predictions["model"].eq(core)].sort_values("seqn", kind="mergesort")
            eg = predictions[predictions["algorithm"].eq(algorithm) & predictions["model"].eq(extended)].sort_values("seqn", kind="mergesort")
            if not cg["seqn"].reset_index(drop=True).equals(eg["seqn"].reset_index(drop=True)):
                raise RuntimeError(f"Paired participants differ: {algorithm} {comparison}")
            for col in ["cvd", "analysis_weight", "strata", "psu"]:
                if not np.array_equal(cg[col].to_numpy(), eg[col].to_numpy(), equal_nan=True):
                    raise RuntimeError(f"Paired {col} differs: {algorithm} {comparison}")
            mc, me = np.asarray(perf_points[ic]), np.asarray(perf_points[ie])
            point = np.array([me[0]-mc[0], me[1]-mc[1], mc[2]-me[2], mc[3]-me[3], mc[4], me[4], mc[5], me[5], abs(mc[4])-abs(me[4]), abs(mc[5]-1)-abs(me[5]-1)])
            inc_pairs.append((ic, ie))
            inc_points.append(point)
            inc_keys.append({
                "cycle": EXTERNAL_CYCLE,
                "comparison": comparison,
                "algorithm": algorithm,
                "core_model": core,
                "extended_model": extended,
                "n": len(cg),
                "events": int(cg["cvd"].sum()),
                "weight_variable": eg["weight_variable"].iloc[0],
            })
    return perf_keys, perf_data, np.vstack(perf_points), np.asarray(design_df), inc_keys, inc_pairs, np.vstack(inc_points)


def batch_ids(config: dict[str, Any]) -> list[np.ndarray]:
    n = int(config["uncertainty"]["replicates"])
    size = int(config["uncertainty"]["batch_size"])
    return [np.arange(i, min(i + size, n), dtype=np.int32) for i in range(0, n, size)]


def compute_batch(ids: np.ndarray, perf_data, inc_pairs, strata_indices, n_clusters: int, seed: int):
    perf = np.full((len(ids), len(perf_data), len(PERFORMANCE_METRICS)), np.nan)
    inc = np.full((len(ids), len(inc_pairs), len(INCREMENTAL_METRICS)), np.nan)
    for row, replicate_id in enumerate(ids):
        factors = rao_wu_factors(strata_indices, n_clusters, seed, int(replicate_id))
        for j, data in enumerate(perf_data):
            perf[row, j] = metric_vector(data, factors)
        for j, (ic, ie) in enumerate(inc_pairs):
            mc, me = perf[row, ic], perf[row, ie]
            inc[row, j] = [me[0]-mc[0], me[1]-mc[1], mc[2]-me[2], mc[3]-me[3], mc[4], me[4], mc[5], me[5], abs(mc[4])-abs(me[4]), abs(mc[5]-1)-abs(me[5]-1)]
    return {"replicate_ids": ids, "performance": perf, "incremental": inc}


def checkpoint_paths(root: Path, ids: np.ndarray) -> tuple[Path, Path]:
    base = root / "results/checkpoints/stage3b"
    stem = f"batch_{int(ids[0]):04d}_{int(ids[-1]):04d}"
    return base / f"{stem}.npz", base / f"{stem}.json"


def load_checkpoint(root: Path, ids: np.ndarray, sig: str, n_perf: int, n_inc: int):
    data_path, meta_path = checkpoint_paths(root, ids)
    if not data_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["signature_hash"] != sig or meta["data_sha256"] != sha256_file(data_path):
            return None
        with np.load(data_path, allow_pickle=False) as saved:
            result = {k: saved[k] for k in saved.files}
        if result["performance"].shape != (len(ids), n_perf, len(PERFORMANCE_METRICS)) or result["incremental"].shape != (len(ids), n_inc, len(INCREMENTAL_METRICS)):
            return None
        if not np.array_equal(result["replicate_ids"], ids):
            return None
        return result
    except Exception:
        return None


def save_checkpoint(root: Path, ids: np.ndarray, sig: str, result) -> None:
    data_path, meta_path = checkpoint_paths(root, ids)
    atomic_npz(data_path, **result)
    atomic_write_text(meta_path, json.dumps({"signature_hash": sig, "replicate_ids": ids.astype(int).tolist(), "data_sha256": sha256_file(data_path), "created": now()}, indent=2, sort_keys=True) + "\n")


def point_table(keys: list[dict[str, Any]], points: np.ndarray, metrics: list[str]) -> pd.DataFrame:
    return pd.concat([pd.DataFrame(keys).reset_index(drop=True), pd.DataFrame(points, columns=metrics)], axis=1)


def run(root: Path, resume: bool, workers: int) -> int:
    root = root.resolve()
    log = root / "results/logs/stage3b_full_run.log"
    append_log(log, f"Stage 3B starting; resume={resume}; workers={workers}")
    config = load_config(root)
    manifest = validate_upstream(root)
    sig = signature(root, config, manifest)
    atomic_write_text(root / "results/audit/stage3b_analysis_signature.json", json.dumps(sig, indent=2, sort_keys=True) + "\n")
    predictions = build_predictions(root, config, manifest)
    predictions, clusters, strata_indices = prepare_design(predictions)
    atomic_write_csv(root / "results/predictions/stage3b_temporal_predictions.csv", predictions.drop(columns="cluster_index"))
    values = groups_from_predictions(predictions, clusters, config)
    perf_keys, perf_data, perf_points, perf_df, inc_keys, inc_pairs, inc_points = values
    append_log(log, f"Frozen predictions complete; performance groups={len(perf_keys)}; paired comparisons={len(inc_keys)}")
    batches = batch_ids(config)
    results = {}
    pending = []
    for ids in batches:
        saved = load_checkpoint(root, ids, sig["signature_hash"], len(perf_keys), len(inc_keys)) if resume else None
        if saved is None:
            pending.append(ids)
        else:
            results[int(ids[0])] = saved
    append_log(log, f"Resume accepted {len(results)}/{len(batches)} bootstrap batches")
    seed = int(config["uncertainty"]["seed"])
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(pending)))) as executor:
            futures = {executor.submit(compute_batch, ids, perf_data, inc_pairs, strata_indices, len(clusters), seed): ids for ids in pending}
            for future in as_completed(futures):
                ids = futures[future]
                result = future.result()
                save_checkpoint(root, ids, sig["signature_hash"], result)
                results[int(ids[0])] = result
                append_log(log, f"Bootstrap {int(ids[0])}-{int(ids[-1])} complete; batches={len(results)}/{len(batches)}")
    ordered = [results[k] for k in sorted(results)]
    rep_ids = np.concatenate([x["replicate_ids"] for x in ordered])
    perf_rep = np.concatenate([x["performance"] for x in ordered], axis=0)
    inc_rep = np.concatenate([x["incremental"] for x in ordered], axis=0)
    expected_ids = np.arange(int(config["uncertainty"]["replicates"]), dtype=np.int32)
    if not np.array_equal(rep_ids, expected_ids):
        raise RuntimeError("Stage 3B bootstrap replicates are incomplete or duplicated.")
    ci_config = {
        "ci_level": config["uncertainty"]["ci_level"],
        "ci_method": config["uncertainty"]["ci_method"],
        "method": config["uncertainty"]["method"],
        "replicates": config["uncertainty"]["replicates"],
    }
    inc_df = np.array([perf_df[ic] for ic, _ in inc_pairs])
    perf_ci = summarize_ci(perf_rep, perf_points, perf_keys, PERFORMANCE_METRICS, perf_df, ci_config)
    inc_ci = summarize_ci(inc_rep, inc_points, inc_keys, INCREMENTAL_METRICS, inc_df, ci_config)
    outputs = {
        root / "results/tables/stage3b_temporal_model_performance.csv": point_table(perf_keys, perf_points, PERFORMANCE_METRICS),
        root / "results/tables/stage3b_temporal_model_performance_survey_ci.csv": perf_ci,
        root / "results/tables/stage3b_temporal_incremental_value.csv": point_table(inc_keys, inc_points, INCREMENTAL_METRICS),
        root / "results/tables/stage3b_temporal_incremental_value_survey_ci.csv": inc_ci,
    }
    for path, frame in outputs.items():
        atomic_write_csv(path, frame)
    atomic_write_csv_gzip(root / "results/models/stage3b_temporal_performance_replicates.csv.gz", wide_replicate_frame(rep_ids, perf_rep, perf_keys, PERFORMANCE_METRICS))
    atomic_write_csv_gzip(root / "results/models/stage3b_temporal_incremental_replicates.csv.gz", wide_replicate_frame(rep_ids, inc_rep, inc_keys, INCREMENTAL_METRICS))
    manifest_after = validate_upstream(root)
    hashes_unchanged = manifest_after["model_sha256"].tolist() == manifest["model_sha256"].tolist()
    min_valid = float(min(perf_ci["valid_replicate_fraction"].min(), inc_ci["valid_replicate_fraction"].min()))
    checks = {
        "upstream_release_pass": True,
        "ten_frozen_model_hashes_match_before_and_after": hashes_unchanged,
        "expected_performance_groups_14": len(perf_keys) == 14,
        "expected_paired_comparisons_8": len(inc_keys) == 8,
        "paired_samples_and_weights_identical": True,
        "bootstrap_replicates_2000": len(rep_ids) == 2000,
        "minimum_valid_replicate_fraction": min_valid >= float(config["uncertainty"]["minimum_valid_replicate_fraction"]),
        "no_model_refit_or_recalibration": True,
        "external_cycle_only": predictions["cycle"].eq(EXTERNAL_CYCLE).all(),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    gate_row = {"module": "stage3b_temporal_validation", "status": status, **checks, "minimum_valid_replicate_fraction_observed": min_valid, "signature_hash": sig["signature_hash"]}
    atomic_write_csv(root / "results/audit/stage3b_completion_gate.csv", pd.DataFrame([gate_row]))
    audit = [
        "# Stage 3B temporal validation audit", "", f"- Status: **{status}**", f"- Completed: {now()}",
        "- External cycle: NHANES 2021-2023", "- Training/fitting during Stage 3B: none", "- Frozen models verified: 10/10 before and after prediction",
        f"- Performance groups: {len(perf_keys)}/14", f"- Paired incremental comparisons: {len(inc_keys)}/8", f"- Rao-Wu bootstrap replicates: {len(rep_ids)}/2000",
        f"- Minimum valid replicate fraction: {min_valid:.6f}", "- SMOTE, retuning, recalibration, and result-dependent updating: not used", "", "## Gate checks", "",
    ] + [f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in checks.items()]
    atomic_write_text(root / "results/audit/stage3b_temporal_validation_audit.md", "\n".join(audit) + "\n")
    append_log(log, f"Stage 3B finished; completion gate={status}")
    return 0 if status == "PASS" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-time locked NHANES 2021-2023 temporal validation.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    return run(Path(args.project_root), args.resume, args.workers)
