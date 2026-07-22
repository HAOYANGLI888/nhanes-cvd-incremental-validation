from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


EPS = 1e-12

PERFORMANCE_METRICS = [
    "survey_weighted_AUROC",
    "survey_weighted_PR_AUC",
    "weighted_Brier",
    "weighted_log_loss",
    "calibration_intercept",
    "calibration_slope",
    "observed_expected_ratio",
    "weighted_observed_prevalence",
    "weighted_mean_predicted_probability",
    "calibration_in_the_large",
]

INCREMENTAL_METRICS = [
    "delta_AUROC",
    "delta_PR_AUC",
    "delta_Brier_improvement",
    "delta_log_loss_improvement",
    "core_calibration_intercept",
    "extended_calibration_intercept",
    "core_calibration_slope",
    "extended_calibration_slope",
    "absolute_intercept_deviation_change",
    "absolute_slope_deviation_change",
]

INCREMENT_COMPARISONS = {
    "renal": ("model0_core", "model1_renal"),
    "metabolic": ("model0_paired_metabolic", "model2_metabolic"),
    "inflammatory": ("model0_core", "model3_inflammatory"),
    "combined": ("model0_paired_combined", "model4_combined"),
}

PERFORMANCE_KEY_COLUMNS = [
    "target_population",
    "comparison_sample",
    "model",
    "model_label",
    "algorithm",
    "cycle",
    "n",
    "events",
    "weight_variable",
]

POOLED_PERFORMANCE_KEY_COLUMNS = [
    "target_population",
    "comparison_sample",
    "model",
    "model_label",
    "algorithm",
    "weight_variable",
]


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


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    frame.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{now()} | {message}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


@dataclass
class MetricData:
    y: np.ndarray
    p: np.ndarray
    base_weight: np.ndarray
    cluster_index: np.ndarray
    sort_order: np.ndarray
    score_group_starts: np.ndarray
    logit_p: np.ndarray
    brier_loss: np.ndarray
    log_loss: np.ndarray


@dataclass
class PerformanceGroup:
    key: dict[str, Any]
    data: MetricData
    point: np.ndarray
    design_df: int


@dataclass
class IncrementalGroup:
    key: dict[str, Any]
    core: MetricData
    extended: MetricData
    point: np.ndarray
    design_df: int


@dataclass
class PreparedInputs:
    config: dict[str, Any]
    signature: dict[str, Any]
    clusters: pd.DataFrame
    strata_cluster_indices: list[np.ndarray]
    performance_groups: list[PerformanceGroup]
    pooled_performance_keys: list[dict[str, Any]]
    pooled_performance_indices: list[np.ndarray]
    pooled_performance_points: np.ndarray
    pooled_performance_design_df: np.ndarray
    cycle_incremental_groups: list[IncrementalGroup]
    pooled_incremental_groups: list[IncrementalGroup]
    reconciliation_rows: list[dict[str, Any]]


def load_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "config" / "survey_bootstrap_locked.yml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("analysis_version") != "survey_ci_locked_v1":
        raise RuntimeError("Unexpected survey CI analysis version.")
    if config.get("method") != "rao_wu_rescaled_bootstrap":
        raise RuntimeError("Only the locked Rao-Wu rescaled bootstrap is allowed.")
    if int(config.get("replicates", 0)) < 1000:
        raise RuntimeError("Locked survey CI requires at least 1000 replicates.")
    if PERFORMANCE_METRICS != list(config.get("performance_metrics", [])):
        raise RuntimeError("Performance metric configuration does not match the implementation.")
    if INCREMENTAL_METRICS != list(config.get("incremental_metrics", [])):
        raise RuntimeError("Incremental metric configuration does not match the implementation.")
    return config


def validate_stage3a_gate(project_root: Path) -> pd.DataFrame:
    path = project_root / "results" / "audit" / "stage3a_algorithm_completion_gate.csv"
    gate = pd.read_csv(path)
    expected = {"elastic_net", "xgboost"}
    if set(gate["algorithm"].astype(str)) != expected:
        raise RuntimeError("Stage 3A gate does not contain both locked algorithms.")
    if not gate["status"].eq("PASS").all() or not gate["algorithm_gate"].eq("PASS").all():
        raise RuntimeError("Both Stage 3A algorithms must pass before survey CI.")
    if not gate["all_candidate_limits_absent"].astype(str).str.lower().eq("true").all():
        raise RuntimeError("Candidate-limited Stage 3A results are not eligible for survey CI.")
    if not gate["all_frozen_checkpoint_compatibility_pass"].astype(str).str.lower().eq("true").all():
        raise RuntimeError("Stage 3A checkpoint compatibility did not pass.")
    return gate


def build_signature(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    inputs = [
        "config/survey_bootstrap_locked.yml",
        "results/audit/stage3a_algorithm_completion_gate.csv",
        "results/predictions/stage3a_outer_predictions.csv",
        "results/tables/cycle_specific_model_performance.csv",
        "results/tables/pooled_out_of_cycle_performance.csv",
        "results/tables/cycle_specific_incremental_value.csv",
        "results/tables/pooled_incremental_value.csv",
    ]
    hashes = {}
    for rel in inputs:
        path = project_root / rel
        if not path.exists():
            raise FileNotFoundError(path)
        hashes[rel] = sha256_file(path)
    code_path = Path(__file__).resolve()
    signature = {
        "analysis_version": config["analysis_version"],
        "method": config["method"],
        "replicates": int(config["replicates"]),
        "seed": int(config["seed"]),
        "ci_level": float(config["ci_level"]),
        "ci_method": config["ci_method"],
        "config_hash": hashes["config/survey_bootstrap_locked.yml"],
        "code_hash": sha256_file(code_path),
        "input_hashes": hashes,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    signature["signature_hash"] = sha256_text(canonical_json(signature))
    return signature


def make_metric_data(
    y: np.ndarray,
    p: np.ndarray,
    weight: np.ndarray,
    cluster_index: np.ndarray,
) -> MetricData:
    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    weight = np.asarray(weight, dtype=float)
    cluster_index = np.asarray(cluster_index, dtype=np.int32)
    if not (len(y) == len(p) == len(weight) == len(cluster_index)):
        raise ValueError("Metric arrays have inconsistent lengths.")
    if len(y) == 0 or len(np.unique(y)) < 2:
        raise ValueError("Metric group must contain events and non-events.")
    if (~np.isfinite(weight)).any() or (weight <= 0).any():
        raise ValueError("Analysis weights must be finite and positive.")
    order = np.argsort(-p, kind="mergesort")
    sorted_p = p[order]
    starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1].astype(np.int32)
    return MetricData(
        y=y,
        p=p,
        base_weight=weight,
        cluster_index=cluster_index,
        sort_order=order.astype(np.int32),
        score_group_starts=starts,
        logit_p=np.log(p / (1 - p)),
        brier_loss=(y - p) ** 2,
        log_loss=-(y * np.log(p) + (1 - y) * np.log1p(-p)),
    )


def fit_weighted_calibration(y: np.ndarray, x: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    total = float(weight.sum())
    if total <= 0:
        return float("nan"), float("nan")
    w = weight / total
    beta = np.zeros(2, dtype=float)
    converged = False
    for _ in range(100):
        eta = np.clip(beta[0] + beta[1] * x, -35.0, 35.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        v = w * mu * (1.0 - mu)
        h00 = float(v.sum())
        h01 = float(np.dot(v, x))
        h11 = float(np.dot(v, x * x))
        g0 = float(np.dot(w, y - mu))
        g1 = float(np.dot(w, (y - mu) * x))
        det = h00 * h11 - h01 * h01
        if not np.isfinite(det) or det <= EPS * max(1.0, h00 * h11):
            return float("nan"), float("nan")
        step0 = (h11 * g0 - h01 * g1) / det
        step1 = (-h01 * g0 + h00 * g1) / det
        if not np.isfinite(step0 + step1):
            return float("nan"), float("nan")
        beta += np.array([step0, step1])
        if max(abs(step0), abs(step1)) < 1e-10:
            converged = True
            break
        if np.max(np.abs(beta)) > 1e4:
            return float("nan"), float("nan")
    if not converged:
        return float("nan"), float("nan")
    return float(beta[0]), float(beta[1])


def metric_vector(data: MetricData, cluster_factors: np.ndarray | None = None) -> np.ndarray:
    if cluster_factors is None:
        weight = data.base_weight
    else:
        weight = data.base_weight * cluster_factors[data.cluster_index]
    total = float(weight.sum())
    positive = float(np.dot(weight, data.y))
    negative = total - positive
    if total <= 0 or positive <= 0 or negative <= 0:
        return np.full(len(PERFORMANCE_METRICS), np.nan)

    order = data.sort_order
    ordered_weight = weight[order]
    ordered_y = data.y[order]
    positive_by_score = np.add.reduceat(ordered_weight * ordered_y, data.score_group_starts)
    negative_by_score = np.add.reduceat(ordered_weight * (1 - ordered_y), data.score_group_starts)
    cumulative_negative = np.cumsum(negative_by_score)
    negative_below = negative - cumulative_negative
    auc = float(np.sum(positive_by_score * (negative_below + 0.5 * negative_by_score)) / (positive * negative))
    cumulative_positive = np.cumsum(positive_by_score)
    cumulative_total = cumulative_positive + cumulative_negative
    precision = np.divide(cumulative_positive, cumulative_total, out=np.zeros_like(cumulative_positive), where=cumulative_total > 0)
    ap = float(np.sum((positive_by_score / positive) * precision))

    obs = positive / total
    mean_p = float(np.dot(weight, data.p) / total)
    brier = float(np.dot(weight, data.brier_loss) / total)
    logloss = float(np.dot(weight, data.log_loss) / total)
    ratio = float(obs / mean_p) if mean_p > 0 else float("nan")
    if 0 < obs < 1 and 0 < mean_p < 1:
        citl = float(math.log(obs / (1 - obs)) - math.log(mean_p / (1 - mean_p)))
    else:
        citl = float("nan")
    intercept, slope = fit_weighted_calibration(data.y.astype(float), data.logit_p, weight)
    return np.array([auc, ap, brier, logloss, intercept, slope, ratio, obs, mean_p, citl], dtype=float)


def incremental_vector(core: MetricData, extended: MetricData, cluster_factors: np.ndarray | None = None) -> np.ndarray:
    mc = metric_vector(core, cluster_factors)
    me = metric_vector(extended, cluster_factors)
    return np.array(
        [
            me[0] - mc[0],
            me[1] - mc[1],
            mc[2] - me[2],
            mc[3] - me[3],
            mc[4],
            me[4],
            mc[5],
            me[5],
            abs(mc[4]) - abs(me[4]),
            abs(mc[5] - 1.0) - abs(me[5] - 1.0),
        ],
        dtype=float,
    )


def design_df_for_clusters(clusters: pd.DataFrame, cluster_indices: np.ndarray) -> int:
    used = clusters.loc[np.unique(cluster_indices), ["cycle", "strata", "psu"]]
    counts = used.groupby(["cycle", "strata"], sort=False)["psu"].nunique()
    return int((counts - 1).clip(lower=0).sum())


def record_reconciliation(
    rows: list[dict[str, Any]],
    scope: str,
    key: dict[str, Any],
    metric: str,
    expected: float,
    observed: float,
) -> None:
    if pd.isna(expected) and pd.isna(observed):
        difference = 0.0
    elif pd.isna(expected) or pd.isna(observed):
        difference = float("inf")
    else:
        difference = abs(float(expected) - float(observed))
    rows.append(
        {
            "scope": scope,
            "key_json": canonical_json(key),
            "metric": metric,
            "expected_point_estimate": expected,
            "recomputed_point_estimate": observed,
            "absolute_difference": difference,
        }
    )


def enrich_predictions(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[np.ndarray]]:
    required = {
        "seqn",
        "cycle",
        "model",
        "algorithm",
        "cvd",
        "predicted_probability",
        "analysis_weight",
        "weight_variable",
        "strata",
        "psu",
        "comparison_sample",
    }
    missing = sorted(required - set(pred.columns))
    if missing:
        raise RuntimeError(f"Prediction file is missing columns: {missing}")
    if pred[list(required)].isna().any().any():
        raise RuntimeError("Prediction file contains missing required survey CI values.")
    participant_design = pred[["seqn", "cycle", "strata", "psu"]].drop_duplicates()
    if participant_design.duplicated(["seqn", "cycle"]).any():
        raise RuntimeError("A participant maps to multiple survey design clusters.")
    clusters = participant_design[["cycle", "strata", "psu"]].drop_duplicates().sort_values(
        ["cycle", "strata", "psu"], kind="mergesort"
    ).reset_index(drop=True)
    clusters["cluster_index"] = np.arange(len(clusters), dtype=np.int32)
    counts = clusters.groupby(["cycle", "strata"], sort=False)["psu"].nunique()
    if (counts < 2).any():
        bad = counts[counts < 2]
        raise RuntimeError(f"Singleton survey strata are not allowed: {bad.to_dict()}")
    pred = pred.merge(clusters, on=["cycle", "strata", "psu"], how="left", validate="many_to_one")
    strata_cluster_indices = [
        group["cluster_index"].to_numpy(dtype=np.int32)
        for _, group in clusters.groupby(["cycle", "strata"], sort=False)
    ]
    return pred, clusters, strata_cluster_indices


def prepare_inputs(project_root: Path) -> PreparedInputs:
    config = load_config(project_root)
    validate_stage3a_gate(project_root)
    signature = build_signature(project_root, config)
    pred = pd.read_csv(project_root / "results" / "predictions" / "stage3a_outer_predictions.csv")
    allowed = set(config["development_cycles"])
    observed = set(pred["cycle"].astype(str))
    forbidden = set(config.get("forbidden_cycles", []))
    if observed & forbidden:
        raise RuntimeError("Forbidden temporal-validation predictions were accessed.")
    if observed != allowed:
        raise RuntimeError(f"Unexpected prediction cycles: {sorted(observed)}")
    if pred["predicted_probability"].isna().any() or not pred["predicted_probability"].between(0, 1).all():
        raise RuntimeError("Predicted probabilities are missing or outside [0,1].")
    pred, clusters, strata_cluster_indices = enrich_predictions(pred)

    reconciliation: list[dict[str, Any]] = []
    perf = pd.read_csv(project_root / "results" / "tables" / "cycle_specific_model_performance.csv")
    perf = perf[perf["classification_threshold_type"].eq("fixed_0.50")].copy()
    performance_groups: list[PerformanceGroup] = []
    for _, row in perf.iterrows():
        mask = (
            pred["algorithm"].eq(row["algorithm"])
            & pred["cycle"].eq(row["cycle"])
            & pred["model"].eq(row["model"])
            & pred["comparison_sample"].eq(row["comparison_sample"])
        )
        group = pred.loc[mask].sort_values("seqn", kind="mergesort")
        if len(group) != int(row["n"]) or int(group["cvd"].sum()) != int(row["events"]):
            raise RuntimeError(f"Performance sample mismatch for {row['algorithm']} {row['cycle']} {row['model']}")
        data = make_metric_data(
            group["cvd"].to_numpy(),
            group["predicted_probability"].to_numpy(),
            group["analysis_weight"].to_numpy(),
            group["cluster_index"].to_numpy(),
        )
        point = metric_vector(data)
        key = {column: row[column] for column in PERFORMANCE_KEY_COLUMNS}
        for index, metric in enumerate(PERFORMANCE_METRICS):
            record_reconciliation(reconciliation, "cycle_performance", key, metric, row[metric], point[index])
        performance_groups.append(
            PerformanceGroup(key=key, data=data, point=point, design_df=design_df_for_clusters(clusters, data.cluster_index))
        )

    pooled_table = pd.read_csv(project_root / "results" / "tables" / "pooled_out_of_cycle_performance.csv")
    pooled_keys: list[dict[str, Any]] = []
    pooled_indices: list[np.ndarray] = []
    pooled_points: list[np.ndarray] = []
    pooled_df: list[int] = []
    for _, row in pooled_table.iterrows():
        key = {column: row[column] for column in POOLED_PERFORMANCE_KEY_COLUMNS}
        indices = np.array(
            [
                index
                for index, group in enumerate(performance_groups)
                if all(group.key[column] == key[column] for column in POOLED_PERFORMANCE_KEY_COLUMNS)
            ],
            dtype=np.int32,
        )
        if len(indices) != len(config["development_cycles"]):
            raise RuntimeError(f"Pooled performance does not map to seven cycles: {key}")
        point = np.nanmean(np.vstack([performance_groups[i].point for i in indices]), axis=0)
        for index, metric in enumerate(PERFORMANCE_METRICS):
            column = f"{metric}_mean"
            if column in row.index:
                record_reconciliation(reconciliation, "pooled_performance", key, metric, row[column], point[index])
        pooled_keys.append(key)
        pooled_indices.append(indices)
        pooled_points.append(point)
        used_clusters = np.concatenate([performance_groups[i].data.cluster_index for i in indices])
        pooled_df.append(design_df_for_clusters(clusters, used_clusters))

    cycle_incremental_table = pd.read_csv(project_root / "results" / "tables" / "cycle_specific_incremental_value.csv")
    pooled_incremental_table = pd.read_csv(project_root / "results" / "tables" / "pooled_incremental_value.csv")
    cycle_incremental_groups: list[IncrementalGroup] = []
    pooled_incremental_groups: list[IncrementalGroup] = []

    def make_incremental_group(comparison: str, algorithm: str, cycle: str | None) -> IncrementalGroup:
        core_name, extended_name = INCREMENT_COMPARISONS[comparison]
        base_mask = pred["algorithm"].eq(algorithm)
        if cycle is not None:
            base_mask &= pred["cycle"].eq(cycle)
        core = pred.loc[base_mask & pred["model"].eq(core_name)].copy()
        extended = pred.loc[base_mask & pred["model"].eq(extended_name)].copy()
        pair = core.merge(extended, on=["seqn", "cycle"], suffixes=("_core", "_extended"), validate="one_to_one")
        if pair.empty:
            raise RuntimeError(f"Empty incremental pair for {comparison} {algorithm} {cycle}")
        for column in ["cvd", "strata", "psu", "cluster_index"]:
            if not pair[f"{column}_core"].equals(pair[f"{column}_extended"]):
                raise RuntimeError(f"Paired {column} mismatch for {comparison} {algorithm} {cycle}")
        y = pair["cvd_extended"].to_numpy()
        weight = pair["analysis_weight_extended"].to_numpy()
        cluster_index = pair["cluster_index_extended"].to_numpy()
        core_data = make_metric_data(y, pair["predicted_probability_core"].to_numpy(), weight, cluster_index)
        extended_data = make_metric_data(y, pair["predicted_probability_extended"].to_numpy(), weight, cluster_index)
        point = incremental_vector(core_data, extended_data)
        key = {
            "comparison": comparison,
            "algorithm": algorithm,
            "cycle": cycle if cycle is not None else "pooled_2005_2018",
            "n": len(pair),
            "events": int(np.sum(y)),
            "weight_variable": pair["weight_variable_extended"].iloc[0],
            "core_model": core_name,
            "extended_model": extended_name,
        }
        return IncrementalGroup(
            key=key,
            core=core_data,
            extended=extended_data,
            point=point,
            design_df=design_df_for_clusters(clusters, cluster_index),
        )

    for comparison in INCREMENT_COMPARISONS:
        for algorithm in sorted(pred["algorithm"].unique()):
            for cycle in config["development_cycles"]:
                group = make_incremental_group(comparison, algorithm, cycle)
                expected = cycle_incremental_table[
                    cycle_incremental_table["comparison"].eq(comparison)
                    & cycle_incremental_table["algorithm"].eq(algorithm)
                    & cycle_incremental_table["cycle"].eq(cycle)
                ]
                if len(expected) != 1:
                    raise RuntimeError(f"Missing cycle incremental point row: {comparison} {algorithm} {cycle}")
                expected_row = expected.iloc[0]
                for index, metric in enumerate(INCREMENTAL_METRICS):
                    if metric in expected_row.index:
                        record_reconciliation(reconciliation, "cycle_incremental", group.key, metric, expected_row[metric], group.point[index])
                cycle_incremental_groups.append(group)
            pooled_group = make_incremental_group(comparison, algorithm, None)
            expected = pooled_incremental_table[
                pooled_incremental_table["comparison"].eq(comparison)
                & pooled_incremental_table["algorithm"].eq(algorithm)
            ]
            if len(expected) != 1:
                raise RuntimeError(f"Missing pooled incremental point row: {comparison} {algorithm}")
            expected_row = expected.iloc[0]
            for index, metric in enumerate(INCREMENTAL_METRICS):
                if metric in expected_row.index:
                    record_reconciliation(reconciliation, "pooled_incremental", pooled_group.key, metric, expected_row[metric], pooled_group.point[index])
            pooled_incremental_groups.append(pooled_group)

    return PreparedInputs(
        config=config,
        signature=signature,
        clusters=clusters,
        strata_cluster_indices=strata_cluster_indices,
        performance_groups=performance_groups,
        pooled_performance_keys=pooled_keys,
        pooled_performance_indices=pooled_indices,
        pooled_performance_points=np.vstack(pooled_points),
        pooled_performance_design_df=np.asarray(pooled_df, dtype=int),
        cycle_incremental_groups=cycle_incremental_groups,
        pooled_incremental_groups=pooled_incremental_groups,
        reconciliation_rows=reconciliation,
    )


def rao_wu_factors(
    strata_cluster_indices: list[np.ndarray],
    n_clusters: int,
    seed: int,
    replicate_id: int,
) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(replicate_id)]))
    factors = np.zeros(n_clusters, dtype=float)
    for cluster_indices in strata_cluster_indices:
        n_h = len(cluster_indices)
        if n_h < 2:
            raise RuntimeError("Rao-Wu bootstrap requires at least two PSUs per stratum.")
        draws = rng.integers(0, n_h, size=n_h - 1)
        multiplicities = np.bincount(draws, minlength=n_h)
        factors[cluster_indices] = (n_h / (n_h - 1.0)) * multiplicities
    return factors


def compute_batch(prepared: PreparedInputs, replicate_ids: np.ndarray) -> dict[str, np.ndarray]:
    n_rep = len(replicate_ids)
    n_perf = len(prepared.performance_groups)
    n_pool_perf = len(prepared.pooled_performance_keys)
    n_cycle_inc = len(prepared.cycle_incremental_groups)
    n_pool_inc = len(prepared.pooled_incremental_groups)
    performance = np.full((n_rep, n_perf, len(PERFORMANCE_METRICS)), np.nan)
    pooled_performance = np.full((n_rep, n_pool_perf, len(PERFORMANCE_METRICS)), np.nan)
    cycle_incremental = np.full((n_rep, n_cycle_inc, len(INCREMENTAL_METRICS)), np.nan)
    pooled_incremental = np.full((n_rep, n_pool_inc, len(INCREMENTAL_METRICS)), np.nan)
    for row_index, replicate_id in enumerate(replicate_ids):
        factors = rao_wu_factors(
            prepared.strata_cluster_indices,
            len(prepared.clusters),
            int(prepared.config["seed"]),
            int(replicate_id),
        )
        for group_index, group in enumerate(prepared.performance_groups):
            performance[row_index, group_index, :] = metric_vector(group.data, factors)
        for group_index, indices in enumerate(prepared.pooled_performance_indices):
            selected = performance[row_index, indices, :]
            valid_count = np.isfinite(selected).sum(axis=0)
            pooled_performance[row_index, group_index, :] = np.divide(
                np.nansum(selected, axis=0),
                valid_count,
                out=np.full(selected.shape[1], np.nan),
                where=valid_count > 0,
            )
        for group_index, group in enumerate(prepared.cycle_incremental_groups):
            cycle_incremental[row_index, group_index, :] = incremental_vector(group.core, group.extended, factors)
        for group_index, group in enumerate(prepared.pooled_incremental_groups):
            pooled_incremental[row_index, group_index, :] = incremental_vector(group.core, group.extended, factors)
    return {
        "replicate_ids": np.asarray(replicate_ids, dtype=np.int32),
        "performance": performance,
        "pooled_performance": pooled_performance,
        "cycle_incremental": cycle_incremental,
        "pooled_incremental": pooled_incremental,
    }


def checkpoint_paths(project_root: Path, replicate_ids: np.ndarray) -> tuple[Path, Path]:
    start = int(replicate_ids[0])
    end = int(replicate_ids[-1])
    root = project_root / "results" / "checkpoints" / "survey_ci"
    return root / f"batch_{start:04d}_{end:04d}.npz", root / f"batch_{start:04d}_{end:04d}.json"


def save_batch_checkpoint(
    project_root: Path,
    prepared: PreparedInputs,
    result: dict[str, np.ndarray],
) -> None:
    data_path, metadata_path = checkpoint_paths(project_root, result["replicate_ids"])
    data_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = data_path.with_name(f"{data_path.name}.tmp-{uuid.uuid4().hex}")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **result)
    os.replace(tmp, data_path)
    metadata = {
        "status": "complete",
        "analysis_signature_hash": prepared.signature["signature_hash"],
        "replicate_ids": result["replicate_ids"].astype(int).tolist(),
        "data_file": data_path.name,
        "data_sha256": sha256_file(data_path),
        "shapes": {key: list(value.shape) for key, value in result.items()},
        "created": now(),
    }
    atomic_write_text(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def load_batch_checkpoint(
    project_root: Path,
    prepared: PreparedInputs,
    replicate_ids: np.ndarray,
) -> dict[str, np.ndarray] | None:
    data_path, metadata_path = checkpoint_paths(project_root, replicate_ids)
    if not data_path.exists() or not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            return None
        if metadata.get("analysis_signature_hash") != prepared.signature["signature_hash"]:
            return None
        if metadata.get("replicate_ids") != replicate_ids.astype(int).tolist():
            return None
        if metadata.get("data_sha256") != sha256_file(data_path):
            return None
        with np.load(data_path, allow_pickle=False) as saved:
            result = {key: saved[key] for key in saved.files}
        expected_shapes = {
            "replicate_ids": (len(replicate_ids),),
            "performance": (len(replicate_ids), len(prepared.performance_groups), len(PERFORMANCE_METRICS)),
            "pooled_performance": (
                len(replicate_ids),
                len(prepared.pooled_performance_keys),
                len(PERFORMANCE_METRICS),
            ),
            "cycle_incremental": (
                len(replicate_ids),
                len(prepared.cycle_incremental_groups),
                len(INCREMENTAL_METRICS),
            ),
            "pooled_incremental": (
                len(replicate_ids),
                len(prepared.pooled_incremental_groups),
                len(INCREMENTAL_METRICS),
            ),
        }
        if set(result) != set(expected_shapes):
            return None
        if any(tuple(result[key].shape) != shape for key, shape in expected_shapes.items()):
            return None
        if not np.array_equal(result["replicate_ids"], replicate_ids):
            return None
        return result
    except Exception:
        return None


def batch_definitions(config: dict[str, Any]) -> list[np.ndarray]:
    replicates = int(config["replicates"])
    size = int(config["batch_size"])
    return [np.arange(start, min(start + size, replicates), dtype=np.int32) for start in range(0, replicates, size)]


def combine_batches(batch_results: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    ordered = sorted(batch_results, key=lambda item: int(item["replicate_ids"][0]))
    combined = {key: np.concatenate([item[key] for item in ordered], axis=0) for key in ordered[0]}
    expected_ids = np.arange(len(combined["replicate_ids"]), dtype=np.int32)
    if not np.array_equal(combined["replicate_ids"], expected_ids):
        raise RuntimeError("Survey bootstrap replicate IDs are incomplete or duplicated.")
    return combined


def wide_replicate_frame(
    replicate_ids: np.ndarray,
    values: np.ndarray,
    keys: list[dict[str, Any]],
    metrics: list[str],
) -> pd.DataFrame:
    n_replicates, n_groups, _ = values.shape
    frame = pd.DataFrame(values.reshape(n_replicates * n_groups, len(metrics)), columns=metrics)
    frame.insert(0, "replicate_id", np.repeat(replicate_ids, n_groups))
    key_columns = list(keys[0])
    for position, column in enumerate(key_columns, start=1):
        frame.insert(position, column, np.tile([key[column] for key in keys], n_replicates))
    return frame


def summarize_ci(
    values: np.ndarray,
    points: np.ndarray,
    keys: list[dict[str, Any]],
    metrics: list[str],
    design_df: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    alpha = 1.0 - float(config["ci_level"])
    requested = int(config["replicates"])
    rows: list[dict[str, Any]] = []
    for group_index, key in enumerate(keys):
        for metric_index, metric in enumerate(metrics):
            sample = values[:, group_index, metric_index]
            valid = sample[np.isfinite(sample)]
            row = dict(key)
            row.update(
                {
                    "metric": metric,
                    "point_estimate": float(points[group_index, metric_index]),
                    "bootstrap_mean": float(np.mean(valid)) if len(valid) else float("nan"),
                    "bootstrap_bias": float(np.mean(valid) - points[group_index, metric_index]) if len(valid) else float("nan"),
                    "bootstrap_se": float(np.std(valid, ddof=1)) if len(valid) > 1 else float("nan"),
                    "ci_lower": float(np.quantile(valid, alpha / 2.0)) if len(valid) else float("nan"),
                    "ci_upper": float(np.quantile(valid, 1.0 - alpha / 2.0)) if len(valid) else float("nan"),
                    "ci_level": float(config["ci_level"]),
                    "ci_method": config["ci_method"],
                    "replicate_method": config["method"],
                    "n_replicates_requested": requested,
                    "n_replicates_valid": int(len(valid)),
                    "valid_replicate_fraction": float(len(valid) / requested),
                    "design_df": int(design_df[group_index]),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def output_keys(prepared: PreparedInputs) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    performance = [group.key for group in prepared.performance_groups]
    cycle_incremental = [group.key for group in prepared.cycle_incremental_groups]
    pooled_incremental = [group.key for group in prepared.pooled_incremental_groups]
    return performance, cycle_incremental, pooled_incremental


def write_outputs(project_root: Path, prepared: PreparedInputs, combined: dict[str, np.ndarray]) -> dict[str, Any]:
    performance_keys, cycle_incremental_keys, pooled_incremental_keys = output_keys(prepared)
    performance_points = np.vstack([group.point for group in prepared.performance_groups])
    performance_df = np.asarray([group.design_df for group in prepared.performance_groups], dtype=int)
    cycle_incremental_points = np.vstack([group.point for group in prepared.cycle_incremental_groups])
    cycle_incremental_df = np.asarray([group.design_df for group in prepared.cycle_incremental_groups], dtype=int)
    pooled_incremental_points = np.vstack([group.point for group in prepared.pooled_incremental_groups])
    pooled_incremental_df = np.asarray([group.design_df for group in prepared.pooled_incremental_groups], dtype=int)

    summaries = {
        "cycle_performance": summarize_ci(
            combined["performance"], performance_points, performance_keys, PERFORMANCE_METRICS, performance_df, prepared.config
        ),
        "pooled_performance": summarize_ci(
            combined["pooled_performance"],
            prepared.pooled_performance_points,
            prepared.pooled_performance_keys,
            PERFORMANCE_METRICS,
            prepared.pooled_performance_design_df,
            prepared.config,
        ),
        "cycle_incremental": summarize_ci(
            combined["cycle_incremental"],
            cycle_incremental_points,
            cycle_incremental_keys,
            INCREMENTAL_METRICS,
            cycle_incremental_df,
            prepared.config,
        ),
        "pooled_incremental": summarize_ci(
            combined["pooled_incremental"],
            pooled_incremental_points,
            pooled_incremental_keys,
            INCREMENTAL_METRICS,
            pooled_incremental_df,
            prepared.config,
        ),
    }
    summary_paths = {
        "cycle_performance": project_root / "results" / "tables" / "cycle_specific_model_performance_survey_ci.csv",
        "pooled_performance": project_root / "results" / "tables" / "pooled_out_of_cycle_performance_survey_ci.csv",
        "cycle_incremental": project_root / "results" / "tables" / "cycle_specific_incremental_value_survey_ci.csv",
        "pooled_incremental": project_root / "results" / "tables" / "pooled_incremental_value_survey_ci.csv",
    }
    for key, frame in summaries.items():
        atomic_write_csv(summary_paths[key], frame)

    replicate_frames = {
        "cycle_performance": wide_replicate_frame(
            combined["replicate_ids"], combined["performance"], performance_keys, PERFORMANCE_METRICS
        ),
        "pooled_performance": wide_replicate_frame(
            combined["replicate_ids"],
            combined["pooled_performance"],
            prepared.pooled_performance_keys,
            PERFORMANCE_METRICS,
        ),
        "cycle_incremental": wide_replicate_frame(
            combined["replicate_ids"],
            combined["cycle_incremental"],
            cycle_incremental_keys,
            INCREMENTAL_METRICS,
        ),
        "pooled_incremental": wide_replicate_frame(
            combined["replicate_ids"],
            combined["pooled_incremental"],
            pooled_incremental_keys,
            INCREMENTAL_METRICS,
        ),
    }
    replicate_paths = {
        key: project_root / "results" / "models" / f"survey_ci_{key}_replicates.csv.gz"
        for key in replicate_frames
    }
    for key, frame in replicate_frames.items():
        atomic_write_csv_gzip(replicate_paths[key], frame)

    reconciliation = pd.DataFrame(prepared.reconciliation_rows)
    reconciliation_path = project_root / "results" / "audit" / "survey_ci_point_reconciliation.csv"
    atomic_write_csv(reconciliation_path, reconciliation)
    finite_differences = reconciliation.loc[np.isfinite(reconciliation["absolute_difference"]), "absolute_difference"]
    max_difference = float(finite_differences.max()) if len(finite_differences) else float("inf")
    has_nonfinite_difference = bool((~np.isfinite(reconciliation["absolute_difference"])).any())
    all_summary = pd.concat(summaries.values(), ignore_index=True)
    min_valid_fraction = float(all_summary["valid_replicate_fraction"].min())
    return {
        "summary_paths": summary_paths,
        "replicate_paths": replicate_paths,
        "reconciliation_path": reconciliation_path,
        "max_point_difference": max_difference,
        "has_nonfinite_point_difference": has_nonfinite_difference,
        "min_valid_fraction": min_valid_fraction,
        "summary_row_counts": {key: len(frame) for key, frame in summaries.items()},
        "replicate_row_counts": {key: len(frame) for key, frame in replicate_frames.items()},
    }


def write_gate_and_audit(
    project_root: Path,
    prepared: PreparedInputs,
    output_info: dict[str, Any],
    completed_batches: int,
) -> str:
    config = prepared.config
    tolerance = float(config["point_reconciliation_tolerance"])
    minimum_valid = float(config["minimum_valid_replicate_fraction"])
    point_pass = (
        not output_info["has_nonfinite_point_difference"]
        and output_info["max_point_difference"] <= tolerance
    )
    valid_pass = output_info["min_valid_fraction"] >= minimum_valid
    expected_replicates = int(config["replicates"])
    expected_batches = len(batch_definitions(config))
    cycles = sorted(prepared.clusters["cycle"].astype(str).unique().tolist())
    forbidden_absent = not bool(set(cycles) & set(config.get("forbidden_cycles", [])))
    strata_counts = prepared.clusters.groupby(["cycle", "strata"])["psu"].nunique()
    design_pass = bool((strata_counts >= 2).all())
    output_hashes = {
        str(path.relative_to(project_root)): sha256_file(path)
        for path in list(output_info["summary_paths"].values())
        + list(output_info["replicate_paths"].values())
        + [output_info["reconciliation_path"]]
    }
    status = "PASS" if all(
        [point_pass, valid_pass, forbidden_absent, design_pass, completed_batches == expected_batches]
    ) else "FAIL"
    gate = pd.DataFrame(
        [
            {
                "module": "survey_aware_confidence_intervals",
                "status": status,
                "analysis_version": config["analysis_version"],
                "method": config["method"],
                "ci_method": config["ci_method"],
                "ci_level": config["ci_level"],
                "replicates_completed": expected_replicates,
                "replicates_expected": expected_replicates,
                "replicate_fraction": f"{expected_replicates}/{expected_replicates}",
                "batches_completed": completed_batches,
                "batches_expected": expected_batches,
                "seed": config["seed"],
                "design_strata": len(strata_counts),
                "design_psu_clusters": len(prepared.clusters),
                "minimum_psus_per_stratum": int(strata_counts.min()),
                "design_gate": "PASS" if design_pass else "FAIL",
                "point_reconciliation_gate": "PASS" if point_pass else "FAIL",
                "maximum_point_absolute_difference": output_info["max_point_difference"],
                "point_reconciliation_tolerance": tolerance,
                "valid_replicate_gate": "PASS" if valid_pass else "FAIL",
                "minimum_valid_replicate_fraction_observed": output_info["min_valid_fraction"],
                "minimum_valid_replicate_fraction_required": minimum_valid,
                "forbidden_cycle_absent": forbidden_absent,
                "development_cycles": "|".join(cycles),
                "algorithms": "elastic_net|xgboost",
                "analysis_signature_hash": prepared.signature["signature_hash"],
            }
        ]
    )
    gate_path = project_root / "results" / "audit" / "survey_ci_completion_gate.csv"
    atomic_write_csv(gate_path, gate)
    signature_path = project_root / "results" / "audit" / "survey_ci_analysis_signature.json"
    atomic_write_text(signature_path, json.dumps(prepared.signature, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    inventory = {
        "status": status,
        "created": now(),
        "summary_row_counts": output_info["summary_row_counts"],
        "replicate_row_counts": output_info["replicate_row_counts"],
        "output_hashes": output_hashes,
    }
    atomic_write_text(
        project_root / "results" / "audit" / "survey_ci_output_inventory.json",
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    audit_lines = [
        "# Survey-aware confidence interval implementation",
        "",
        f"- Status: {status}",
        f"- Analysis version: {config['analysis_version']}",
        "- Design: NHANES cycle-specific masked strata and PSU identifiers; cycle is included in the stratum key.",
        f"- Replicates: {expected_replicates} deterministic Rao-Wu rescaled bootstrap replicates.",
        "- Within each stratum containing n_h PSUs, n_h-1 PSUs were sampled with replacement and selected PSU weights were multiplied by n_h/(n_h-1) times their multiplicity.",
        "- A single replicate factor was shared by every prediction row from the same cycle-stratum-PSU cluster across models and algorithms.",
        "- Incremental comparisons used identical replicate factors, identical paired participants, and the extended-model analysis weight.",
        f"- Confidence intervals: {100 * float(config['ci_level']):.1f}% percentile intervals.",
        f"- Minimum valid replicate fraction observed: {output_info['min_valid_fraction']:.6f}.",
        f"- Maximum point-estimate reconciliation difference: {output_info['max_point_difference']:.12g}.",
        f"- Development cycles: {', '.join(cycles)}.",
        "- NHANES 2021-2023 was not read or evaluated.",
        "- Ordinary participant-level bootstrap and SMOTE were not used.",
        "",
        "Primary methodological sources:",
        "- NHANES Variance Estimation Tutorial: https://wwwn.cdc.gov/nchs/NHANES/tutorials/VarianceEstimation.aspx",
        "- Rao JNK, Wu CFJ. Resampling Inference with Complex Survey Data. JASA. 1988;83:231-241. doi:10.1080/01621459.1988.10478591",
    ]
    atomic_write_text(
        project_root / "results" / "audit" / "survey_bootstrap_implementation.md",
        "\n".join(audit_lines) + "\n",
    )
    report_lines = [
        "# Survey CI completion report",
        "",
        f"- Completion time: {now()}",
        f"- Gate: {status}",
        f"- Replicates: {expected_replicates}/{expected_replicates}",
        f"- Batches: {completed_batches}/{expected_batches}",
        f"- Cycle performance CI rows: {output_info['summary_row_counts']['cycle_performance']}",
        f"- Pooled performance CI rows: {output_info['summary_row_counts']['pooled_performance']}",
        f"- Cycle incremental CI rows: {output_info['summary_row_counts']['cycle_incremental']}",
        f"- Pooled incremental CI rows: {output_info['summary_row_counts']['pooled_incremental']}",
        "- Stage 3B remained locked.",
    ]
    atomic_write_text(
        project_root / "results" / "audit" / "survey_ci_completion_report.md",
        "\n".join(report_lines) + "\n",
    )
    return status


def update_blocking_files(project_root: Path, status: str) -> None:
    if status != "PASS":
        return
    remaining = [
        "Nested multiple imputation and IPW sensitivity analyses for UACR availability have not yet been completed.",
        "Class-imbalance sensitivity refits have not yet been completed.",
        "Random-effects cycle meta-analysis and decision-curve analysis have not yet been completed.",
    ]
    lines = [
        "# Stage 3A Blocking Issues",
        "",
        "Survey-aware confidence intervals passed the independent post-Stage-3A release gate.",
        "",
        "Blocking or compliance issues detected:",
    ] + [f"- {item}" for item in remaining]
    value = "\n".join(lines) + "\n"
    atomic_write_text(project_root / "results" / "audit" / "stage3a_blocking_issues.md", value)
    atomic_write_text(project_root / "ACTION_REQUIRED_STAGE3A_BLOCKING_ISSUES.md", value)


def run_survey_ci(project_root: Path, resume: bool = False, workers: int | None = None) -> int:
    project_root = Path(project_root).resolve()
    log_path = project_root / "results" / "logs" / "survey_ci_full_run.log"
    append_log(log_path, f"Survey CI starting; resume={resume}")
    try:
        prepared = prepare_inputs(project_root)
        config = prepared.config
        batches = batch_definitions(config)
        workers = int(workers if workers is not None else config["default_workers"])
        workers = max(1, workers)
        append_log(
            log_path,
            f"Prepared {len(prepared.performance_groups)} cycle performance groups, "
            f"{len(prepared.cycle_incremental_groups)} cycle incremental groups, "
            f"{len(prepared.pooled_incremental_groups)} pooled incremental groups; "
            f"replicates={config['replicates']}; workers={workers}",
        )
        results: dict[int, dict[str, np.ndarray]] = {}
        pending: list[np.ndarray] = []
        for replicate_ids in batches:
            checkpoint = load_batch_checkpoint(project_root, prepared, replicate_ids) if resume else None
            if checkpoint is None:
                pending.append(replicate_ids)
            else:
                results[int(replicate_ids[0])] = checkpoint
        append_log(log_path, f"Resume accepted {len(results)}/{len(batches)} completed bootstrap batches")
        if pending:
            with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as executor:
                futures = {executor.submit(compute_batch, prepared, ids): ids for ids in pending}
                for future in as_completed(futures):
                    replicate_ids = futures[future]
                    result = future.result()
                    save_batch_checkpoint(project_root, prepared, result)
                    results[int(replicate_ids[0])] = result
                    append_log(
                        log_path,
                        f"Bootstrap batch {int(replicate_ids[0])}-{int(replicate_ids[-1])} complete; "
                        f"batches={len(results)}/{len(batches)}",
                    )
        if len(results) != len(batches):
            raise RuntimeError("Not all survey bootstrap batches completed.")
        combined = combine_batches(list(results.values()))
        if len(combined["replicate_ids"]) != int(config["replicates"]):
            raise RuntimeError("Survey bootstrap replicate count is incomplete.")
        append_log(log_path, "All bootstrap batches complete; writing CI tables and replicate archives")
        output_info = write_outputs(project_root, prepared, combined)
        status = write_gate_and_audit(project_root, prepared, output_info, len(results))
        update_blocking_files(project_root, status)
        append_log(log_path, f"Survey CI finished; completion gate={status}")
        if status != "PASS":
            return 2
        return 0
    except Exception as exc:
        append_log(log_path, f"Survey CI failed: {type(exc).__name__}: {exc}")
        failure = pd.DataFrame(
            [
                {
                    "module": "survey_aware_confidence_intervals",
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "failed_at": now(),
                }
            ]
        )
        atomic_write_csv(project_root / "results" / "audit" / "survey_ci_completion_gate.csv", failure)
        raise
