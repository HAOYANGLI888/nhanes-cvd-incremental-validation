from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, t

from post_stage3a_common import atomic_csv, atomic_text, load_config, module_signature, validate_upstream, write_gate


def reml_tau2(effect: np.ndarray, variance: np.ndarray) -> float:
    spread = float(np.var(effect, ddof=1)) if len(effect) > 1 else 0.0
    upper = max(1e-10, spread * 100.0, float(np.max(variance)) * 100.0)

    def objective(tau2: float) -> float:
        total_variance = variance + tau2
        weight = 1.0 / total_variance
        mean = float(np.sum(weight * effect) / np.sum(weight))
        return 0.5 * (
            float(np.sum(np.log(total_variance)))
            + math.log(float(np.sum(weight)))
            + float(np.sum(weight * (effect - mean) ** 2))
        )

    result = minimize_scalar(objective, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-14})
    tau2 = float(result.x) if result.success else 0.0
    if objective(0.0) <= objective(tau2) + 1e-12:
        return 0.0
    return max(0.0, tau2)


def pool_group(group: pd.DataFrame, ci_level: float) -> dict[str, float]:
    effect = group["point_estimate"].to_numpy(dtype=float)
    se = group["bootstrap_se"].to_numpy(dtype=float)
    variance = se**2
    k = len(effect)
    if k < 3 or not np.isfinite(effect).all() or not np.isfinite(variance).all() or (variance <= 0).any():
        raise RuntimeError("Invalid cycle inputs for random-effects meta-analysis.")
    tau2 = reml_tau2(effect, variance)
    random_weight = 1.0 / (variance + tau2)
    pooled = float(np.sum(random_weight * effect) / np.sum(random_weight))
    q_hk = float(np.sum(random_weight * (effect - pooled) ** 2) / (k - 1))
    q_modified = max(1.0, q_hk)
    se_modified = math.sqrt(q_modified / float(np.sum(random_weight)))
    alpha = 1.0 - ci_level
    critical = float(t.ppf(1.0 - alpha / 2.0, df=k - 1))
    lower = pooled - critical * se_modified
    upper = pooled + critical * se_modified
    p_value = float(2.0 * t.sf(abs(pooled / se_modified), df=k - 1)) if se_modified > 0 else float("nan")
    fixed_weight = 1.0 / variance
    fixed_mean = float(np.sum(fixed_weight * effect) / np.sum(fixed_weight))
    q = float(np.sum(fixed_weight * (effect - fixed_mean) ** 2))
    q_p = float(chi2.sf(q, df=k - 1))
    i2 = max(0.0, (q - (k - 1)) / q) * 100.0 if q > 0 else 0.0
    prediction_se = math.sqrt(tau2 + se_modified**2)
    prediction_critical = float(t.ppf(1.0 - alpha / 2.0, df=max(1, k - 2)))
    return {
        "k_cycles": k,
        "pooled_effect_REML": pooled,
        "modified_HK_se": se_modified,
        "ci_lower": lower,
        "ci_upper": upper,
        "p_value": p_value,
        "tau2_REML": tau2,
        "tau_REML": math.sqrt(tau2),
        "Cochran_Q": q,
        "Q_df": k - 1,
        "Q_p_value": q_p,
        "I2_percent": i2,
        "HK_q": q_hk,
        "modified_HK_q": q_modified,
        "prediction_interval_lower": pooled - prediction_critical * prediction_se,
        "prediction_interval_upper": pooled + prediction_critical * prediction_se,
    }


def run_meta_analysis(project_root: Path) -> str:
    validate_upstream(project_root)
    config = load_config(project_root)["meta_analysis"]
    signature = module_signature(
        project_root,
        "meta_analysis",
        [
            "scripts/python/meta_analysis_module.py",
            "results/tables/cycle_specific_incremental_value_survey_ci.csv",
            "results/audit/survey_ci_completion_gate.csv",
        ],
    )
    source = pd.read_csv(project_root / "results" / "tables" / "cycle_specific_incremental_value_survey_ci.csv")
    source = source[source["metric"].isin(config["metrics"])].copy()
    rows = []
    for keys, group in source.groupby(["comparison", "algorithm", "metric"], sort=True):
        if group["cycle"].nunique() != 7:
            raise RuntimeError(f"Meta-analysis group does not contain seven cycles: {keys}")
        row = dict(zip(["comparison", "algorithm", "metric"], keys))
        row.update(pool_group(group.sort_values("cycle"), float(config["ci_level"])))
        row["between_cycle_variance_method"] = config["between_cycle_variance"]
        row["confidence_interval_method"] = config["confidence_interval"]
        row["ci_level"] = config["ci_level"]
        rows.append(row)
    result = pd.DataFrame(rows)
    atomic_csv(project_root / "results" / "tables" / "cycle_random_effects_meta_analysis.csv", result)
    finite_columns = ["pooled_effect_REML", "modified_HK_se", "ci_lower", "ci_upper", "tau2_REML", "I2_percent"]
    status = "PASS" if all(
        [
            len(result) == 32,
            result["k_cycles"].eq(7).all(),
            np.isfinite(result[finite_columns].to_numpy()).all(),
            (result["ci_lower"] <= result["ci_upper"]).all(),
        ]
    ) else "FAIL"
    write_gate(
        project_root,
        "cycle_meta_analysis",
        {
            "module": "random_effects_cycle_meta_analysis",
            "status": status,
            "rows_completed": len(result),
            "rows_expected": 32,
            "cycles_per_analysis": 7,
            "metrics": "|".join(config["metrics"]),
            "tau2_method": config["between_cycle_variance"],
            "ci_method": config["confidence_interval"],
            "forbidden_cycle_absent": True,
            "signature_hash": signature["signature_hash"],
        },
    )
    audit = [
        "# Random-effects cycle meta-analysis audit",
        "",
        f"- Status: {status}",
        "- Seven disjoint NHANES development cycles were treated as the meta-analytic units.",
        "- Sampling variances came from the 2,000-replicate survey bootstrap.",
        "- Between-cycle variance used REML.",
        "- Primary inference used modified Hartung-Knapp with k-1 degrees of freedom.",
        "- NHANES 2021-2023 was not accessed.",
    ]
    atomic_text(project_root / "results" / "audit" / "cycle_meta_analysis_audit.md", "\n".join(audit) + "\n")
    return status
