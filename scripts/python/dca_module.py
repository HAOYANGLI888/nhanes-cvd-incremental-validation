from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from post_stage3a_common import atomic_csv, atomic_text, load_config, module_signature, validate_upstream, write_gate
from survey_ci_core import enrich_predictions, rao_wu_factors


def cluster_matrix(values: np.ndarray, cluster_index: np.ndarray, n_clusters: int) -> np.ndarray:
    if values.ndim == 1:
        values = values[:, None]
    output = np.zeros((n_clusters, values.shape[1]), dtype=float)
    np.add.at(output, cluster_index, values)
    return output


def net_benefit(
    probability: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    cluster_index: np.ndarray,
    thresholds: np.ndarray,
    factors: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    decision = probability[:, None] >= thresholds[None, :]
    tp_cluster = cluster_matrix(weight[:, None] * y[:, None] * decision, cluster_index, n_clusters)
    fp_cluster = cluster_matrix(weight[:, None] * (1 - y)[:, None] * decision, cluster_index, n_clusters)
    total_cluster = np.bincount(cluster_index, weights=weight, minlength=n_clusters)
    total = float(weight.sum())
    odds = thresholds / (1.0 - thresholds)
    point = tp_cluster.sum(axis=0) / total - fp_cluster.sum(axis=0) / total * odds
    replicate_total = factors @ total_cluster
    replicate = (factors @ tp_cluster) / replicate_total[:, None] - (factors @ fp_cluster) / replicate_total[:, None] * odds
    return point, replicate


def summarize_strategy(
    key: dict,
    strategy: str,
    thresholds: np.ndarray,
    point: np.ndarray,
    replicates: np.ndarray,
    ci_level: float,
) -> list[dict]:
    alpha = 1.0 - ci_level
    rows = []
    for index, threshold in enumerate(thresholds):
        values = replicates[:, index]
        rows.append(
            {
                **key,
                "strategy": strategy,
                "threshold_probability": threshold,
                "net_benefit": point[index],
                "bootstrap_se": float(np.std(values, ddof=1)),
                "ci_lower": float(np.quantile(values, alpha / 2.0)),
                "ci_upper": float(np.quantile(values, 1.0 - alpha / 2.0)),
                "valid_replicates": int(np.isfinite(values).sum()),
                "ci_level": ci_level,
                "ci_method": "survey_Rao_Wu_percentile",
            }
        )
    return rows


def run_dca(project_root: Path) -> str:
    validate_upstream(project_root)
    config = load_config(project_root)["dca"]
    signature = module_signature(
        project_root,
        "dca",
        ["scripts/python/dca_module.py", "config/survey_bootstrap_locked.yml", "scripts/python/survey_ci_core.py"],
    )
    predictions = pd.read_csv(project_root / "results" / "predictions" / "stage3a_outer_predictions.csv")
    predictions, clusters, strata_indices = enrich_predictions(predictions)
    n_replicates = int(config["survey_replicates"])
    factors = np.vstack(
        [
            rao_wu_factors(strata_indices, len(clusters), int(config["survey_seed"]), replicate_id)
            for replicate_id in range(n_replicates)
        ]
    )
    thresholds = np.round(
        np.arange(float(config["threshold_start"]), float(config["threshold_stop"]) + float(config["threshold_step"]) / 2, float(config["threshold_step"])),
        10,
    )
    rows = []
    group_count = 0
    for comparison, models in config["comparisons"].items():
        for algorithm in config["algorithms"]:
            for scope in list(sorted(predictions["cycle"].unique())) + ["pooled_2005_2018"]:
                base = predictions["algorithm"].eq(algorithm)
                if scope != "pooled_2005_2018":
                    base &= predictions["cycle"].eq(scope)
                core = predictions[base & predictions["model"].eq(models["core_model"])].copy()
                extended = predictions[base & predictions["model"].eq(models["extended_model"])].copy()
                pair = core.merge(extended, on=["seqn", "cycle"], suffixes=("_core", "_extended"), validate="one_to_one")
                if pair.empty:
                    raise RuntimeError(f"Empty DCA pair: {comparison} {algorithm} {scope}")
                for column in ["cvd", "analysis_weight", "strata", "psu", "cluster_index"]:
                    if not np.allclose(pair[f"{column}_core"], pair[f"{column}_extended"], equal_nan=True):
                        raise RuntimeError(f"DCA paired {column} mismatch: {comparison} {algorithm} {scope}")
                y = pair["cvd_extended"].to_numpy(dtype=int)
                weight = pair["analysis_weight_extended"].to_numpy(dtype=float)
                cluster_index = pair["cluster_index_extended"].to_numpy(dtype=int)
                core_point, core_rep = net_benefit(pair["predicted_probability_core"].to_numpy(), y, weight, cluster_index, thresholds, factors, len(clusters))
                extended_point, extended_rep = net_benefit(pair["predicted_probability_extended"].to_numpy(), y, weight, cluster_index, thresholds, factors, len(clusters))
                all_point, all_rep = net_benefit(np.ones(len(pair)), y, weight, cluster_index, thresholds, factors, len(clusters))
                none_point = np.zeros(len(thresholds))
                none_rep = np.zeros((n_replicates, len(thresholds)))
                key = {
                    "comparison": comparison,
                    "algorithm": algorithm,
                    "scope": scope,
                    "n": len(pair),
                    "events": int(y.sum()),
                    "core_model": models["core_model"],
                    "extended_model": models["extended_model"],
                    "weight_variable": pair["weight_variable_extended"].iloc[0],
                }
                rows.extend(summarize_strategy(key, "core_model", thresholds, core_point, core_rep, float(config["ci_level"])))
                rows.extend(summarize_strategy(key, "extended_model", thresholds, extended_point, extended_rep, float(config["ci_level"])))
                rows.extend(summarize_strategy(key, "treat_all", thresholds, all_point, all_rep, float(config["ci_level"])))
                rows.extend(summarize_strategy(key, "treat_none", thresholds, none_point, none_rep, float(config["ci_level"])))
                rows.extend(summarize_strategy(key, "extended_minus_core", thresholds, extended_point - core_point, extended_rep - core_rep, float(config["ci_level"])))
                group_count += 1
    result = pd.DataFrame(rows)
    result.sort_values(["comparison", "algorithm", "scope", "threshold_probability", "strategy"], inplace=True)
    atomic_csv(project_root / "results" / "tables" / "decision_curve_analysis.csv", result)
    expected_groups = 4 * 2 * 8
    expected_rows = expected_groups * len(thresholds) * 5
    status = "PASS" if all(
        [
            group_count == expected_groups,
            len(result) == expected_rows,
            result["valid_replicates"].eq(n_replicates).all(),
            np.isfinite(result[["net_benefit", "bootstrap_se", "ci_lower", "ci_upper"]].to_numpy()).all(),
            (result["ci_lower"] <= result["ci_upper"]).all(),
            set(predictions["cycle"]) == set(load_config(project_root)["development_cycles"]),
        ]
    ) else "FAIL"
    write_gate(
        project_root,
        "dca",
        {
            "module": "survey_weighted_decision_curve_analysis",
            "status": status,
            "groups_completed": group_count,
            "groups_expected": expected_groups,
            "rows_completed": len(result),
            "rows_expected": expected_rows,
            "thresholds": len(thresholds),
            "threshold_range": f"{thresholds.min():.2f}-{thresholds.max():.2f}",
            "survey_replicates": n_replicates,
            "paired_samples_and_weights": True,
            "forbidden_cycle_absent": True,
            "signature_hash": signature["signature_hash"],
        },
    )
    audit = [
        "# Decision-curve analysis audit",
        "",
        f"- Status: {status}",
        f"- Groups: {group_count}/{expected_groups}",
        f"- Thresholds: {thresholds.min():.2f} to {thresholds.max():.2f} by {float(config['threshold_step']):.2f}.",
        "- Core and extended models used identical participants, survey weights, and Rao-Wu replicate factors.",
        "- Treat-all, treat-none, and extended-minus-core net benefit were reported.",
        "- The analysis is exploratory for prevalent self-reported CVD and does not establish deployment readiness.",
        "- NHANES 2021-2023 was not accessed.",
    ]
    atomic_text(project_root / "results" / "audit" / "dca_audit.md", "\n".join(audit) + "\n")
    return status
