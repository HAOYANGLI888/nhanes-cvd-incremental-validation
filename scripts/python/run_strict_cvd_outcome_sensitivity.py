from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from survey_ci_core import (
    INCREMENTAL_METRICS,
    PERFORMANCE_METRICS,
    atomic_write_csv,
    atomic_write_text,
    design_df_for_clusters,
    incremental_vector,
    make_metric_data,
    metric_vector,
    now,
    rao_wu_factors,
    summarize_ci,
)


COMPONENTS = ["cvd_chf", "cvd_chd", "cvd_angina", "cvd_heart_attack", "cvd_stroke"]
REPLICATES = 2000
SEED = 20260721


def strict_outcome(data: pd.DataFrame) -> pd.Series:
    components = data[COMPONENTS]
    positive = components.eq(1).any(axis=1)
    complete_negative = components.notna().all(axis=1) & components.eq(0).all(axis=1)
    return pd.Series(np.where(positive, 1.0, np.where(complete_negative, 0.0, np.nan)), index=data.index)


def incremental_from_metrics(core: np.ndarray, extended: np.ndarray) -> np.ndarray:
    return np.array(
        [
            extended[0] - core[0],
            extended[1] - core[1],
            core[2] - extended[2],
            core[3] - extended[3],
            core[4],
            extended[4],
            core[5],
            extended[5],
            abs(core[4]) - abs(extended[4]),
            abs(core[5] - 1.0) - abs(extended[5] - 1.0),
        ],
        dtype=float,
    )


def analyze_period(root: Path, period: str, prediction_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pred = pd.read_csv(prediction_path, low_memory=False)
    pred = pred[pred["algorithm"].eq("elastic_net") & pred["model"].isin(["model0_core", "model1_renal"])].copy()
    outcome = pd.read_csv(
        root / "data/interim/stage2_harmonized_audit_dataset.csv",
        usecols=["seqn", "cycle", "cvd", *COMPONENTS],
        low_memory=False,
    )
    outcome["strict_cvd"] = strict_outcome(outcome)
    outcome = outcome[["seqn", "cycle", "cvd", "strict_cvd"]].drop_duplicates(["seqn", "cycle"])
    pred = pred.merge(outcome, on=["seqn", "cycle"], how="left", suffixes=("", "_source"), validate="many_to_one")
    pred = pred[pred["strict_cvd"].notna()].copy()
    pred["cvd"] = pred["strict_cvd"].astype(int)
    core = pred[pred["model"].eq("model0_core")].sort_values(["cycle", "seqn"], kind="mergesort").reset_index(drop=True)
    renal = pred[pred["model"].eq("model1_renal")].sort_values(["cycle", "seqn"], kind="mergesort").reset_index(drop=True)
    if not core[["seqn", "cycle"]].equals(renal[["seqn", "cycle"]]):
        raise RuntimeError(f"Strict-outcome paired participants differ in {period}.")
    for column in ["cvd", "analysis_weight", "strata", "psu"]:
        if not np.array_equal(core[column].to_numpy(), renal[column].to_numpy(), equal_nan=True):
            raise RuntimeError(f"Strict-outcome paired {column} differs in {period}.")

    design = core[["cycle", "strata", "psu"]].drop_duplicates().sort_values(["cycle", "strata", "psu"]).reset_index(drop=True)
    design["cluster_index"] = np.arange(len(design), dtype=np.int32)
    counts = design.groupby(["cycle", "strata"])["psu"].nunique()
    if (counts < 2).any():
        raise RuntimeError(f"Singleton survey strata in {period}: {counts[counts < 2].to_dict()}")
    core = core.merge(design, on=["cycle", "strata", "psu"], how="left", validate="many_to_one")
    renal = renal.merge(design, on=["cycle", "strata", "psu"], how="left", validate="many_to_one")
    strata_indices = [g["cluster_index"].to_numpy(dtype=np.int32) for _, g in design.groupby(["cycle", "strata"], sort=False)]
    groups = []
    keys = []
    for model, frame in [("model0_core", core), ("model1_renal", renal)]:
        groups.append(
            make_metric_data(
                frame["cvd"].to_numpy(),
                frame["predicted_probability"].to_numpy(),
                frame["analysis_weight"].to_numpy(),
                frame["cluster_index"].to_numpy(),
            )
        )
        keys.append(
            {
                "period": period,
                "algorithm": "elastic_net",
                "model": model,
                "outcome_definition": "strict: any positive; otherwise all five components observed and negative",
                "n": len(frame),
                "events": int(frame["cvd"].sum()),
                "weight_variable": str(frame["weight_variable"].iloc[0]),
            }
        )
    points = np.vstack([metric_vector(group) for group in groups])
    increments = incremental_from_metrics(points[0], points[1]).reshape(1, -1)
    perf_rep = np.full((REPLICATES, 2, len(PERFORMANCE_METRICS)), np.nan)
    inc_rep = np.full((REPLICATES, 1, len(INCREMENTAL_METRICS)), np.nan)
    for replicate_id in range(REPLICATES):
        factors = rao_wu_factors(strata_indices, len(design), SEED, replicate_id)
        perf_rep[replicate_id, 0] = metric_vector(groups[0], factors)
        perf_rep[replicate_id, 1] = metric_vector(groups[1], factors)
        inc_rep[replicate_id, 0] = incremental_from_metrics(perf_rep[replicate_id, 0], perf_rep[replicate_id, 1])
    config = {
        "ci_level": 0.95,
        "ci_method": "percentile",
        "method": "rao_wu_rescaled_bootstrap",
        "replicates": REPLICATES,
    }
    dfs = np.array([design_df_for_clusters(design, group.cluster_index) for group in groups])
    performance = summarize_ci(perf_rep, points, keys, PERFORMANCE_METRICS, dfs, config)
    inc_key = [{
        "period": period,
        "comparison": "renal",
        "algorithm": "elastic_net",
        "core_model": "model0_core",
        "extended_model": "model1_renal",
        "outcome_definition": "strict: any positive; otherwise all five components observed and negative",
        "n": len(core),
        "events": int(core["cvd"].sum()),
        "weight_variable": str(core["weight_variable"].iloc[0]),
    }]
    incremental = summarize_ci(inc_rep, increments, inc_key, INCREMENTAL_METRICS, np.array([dfs[0]]), config)
    summary = {
        "period": period,
        "n_strict": len(core),
        "events_strict": int(core["cvd"].sum()),
        "excluded_partial_noncase": int(
            outcome.loc[outcome["strict_cvd"].isna() & outcome["cvd"].eq(0), ["seqn", "cycle"]]
            .merge(
                pd.read_csv(prediction_path, usecols=["seqn", "cycle", "model", "algorithm"])
                .query("algorithm == 'elastic_net' and model == 'model0_core'")[["seqn", "cycle"]],
                on=["seqn", "cycle"],
                how="inner",
            )
            .drop_duplicates()
            .shape[0]
        ),
    }
    return performance, incremental, summary


def run(root: Path) -> int:
    root = root.resolve()
    periods = [
        ("development 2005-2018 out-of-cycle", root / "results/predictions/stage3a_outer_predictions.csv"),
        ("temporal 2021-2023", root / "results/predictions/stage3b_temporal_predictions.csv"),
    ]
    performance_frames = []
    incremental_frames = []
    summaries = []
    for period, path in periods:
        perf, inc, summary = analyze_period(root, period, path)
        performance_frames.append(perf)
        incremental_frames.append(inc)
        summaries.append(summary)
    performance = pd.concat(performance_frames, ignore_index=True)
    incremental = pd.concat(incremental_frames, ignore_index=True)
    atomic_write_csv(root / "results/tables/strict_cvd_outcome_sensitivity_performance.csv", performance)
    atomic_write_csv(root / "results/tables/strict_cvd_outcome_sensitivity_incremental.csv", incremental)
    atomic_write_csv(root / "results/audit/strict_cvd_outcome_sensitivity_counts.csv", pd.DataFrame(summaries))
    display_metrics = {
        "delta_AUROC": "Delta AUROC",
        "delta_PR_AUC": "Delta PR-AUC",
        "delta_Brier_improvement": "Brier improvement",
        "delta_log_loss_improvement": "Log-loss improvement",
    }
    display_rows = []
    for period, group in incremental[incremental["metric"].isin(display_metrics)].groupby("period", sort=False):
        first = group.iloc[0]
        row = {
            "Period": period,
            "Strict n (events)": f"{int(first['n'])} ({int(first['events'])})",
        }
        values = group.set_index("metric")
        for metric, label in display_metrics.items():
            item = values.loc[metric]
            row[f"{label} (95% CI)"] = f"{item['point_estimate']:.5f} ({item['ci_lower']:.5f} to {item['ci_upper']:.5f})"
        display_rows.append(row)
    pd.DataFrame(display_rows).to_csv(
        root / "results/manuscript_tables/post_stage3a/Supplementary_Table_S8_strict_CVD_outcome_sensitivity.csv",
        index=False,
    )
    passed = (
        performance["valid_replicate_fraction"].ge(0.95).all()
        and incremental["valid_replicate_fraction"].ge(0.95).all()
        and len(performance) == 2 * 2 * len(PERFORMANCE_METRICS)
        and len(incremental) == 2 * len(INCREMENTAL_METRICS)
    )
    gate = pd.DataFrame([{
        "module": "strict_cvd_outcome_definition_sensitivity",
        "status": "PASS" if passed else "FAIL",
        "algorithm": "elastic_net",
        "periods": 2,
        "bootstrap_replicates": REPLICATES,
        "minimum_valid_replicate_fraction": min(performance["valid_replicate_fraction"].min(), incremental["valid_replicate_fraction"].min()),
    }])
    atomic_write_csv(root / "results/audit/strict_cvd_outcome_sensitivity_gate.csv", gate)
    lines = [
        "# Strict CVD outcome-definition sensitivity",
        "",
        f"- Status: **{'PASS' if passed else 'FAIL'}**",
        f"- Completed: {now()}",
        "- Primary Elastic Net predictions were unchanged; no refitting, retuning, or recalibration was performed.",
        "- Cases required any positive self-report component.",
        "- Noncases required all five components observed and negative; partial non-positive response patterns were excluded.",
        "- Survey uncertainty used 2,000 Rao-Wu rescaled bootstrap replicates.",
        "",
    ]
    for item in summaries:
        lines.append(
            f"- {item['period']}: n={item['n_strict']}; events={item['events_strict']}; excluded partial noncases={item['excluded_partial_noncase']}."
        )
    atomic_write_text(root / "results/audit/strict_cvd_outcome_sensitivity_audit.md", "\n".join(lines) + "\n")
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the locked models under a strict complete-negative CVD definition.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    return run(Path(args.project_root))


if __name__ == "__main__":
    raise SystemExit(main())
