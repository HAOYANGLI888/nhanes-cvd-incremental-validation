from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import FancyBboxPatch

from survey_ci_core import rao_wu_factors


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "interim" / "stage2_harmonized_audit_dataset.csv"
PRED_PATH = ROOT / "results" / "predictions" / "stage3b_temporal_predictions.csv"
PERF_PATH = ROOT / "results" / "tables" / "stage3b_temporal_model_performance_survey_ci.csv"
TABLE_ROOT = ROOT / "results" / "manuscript_tables" / "post_stage3a"
FIG_ROOT = ROOT / "results" / "manuscript_figures" / "post_stage3a"
SOURCE_ROOT = FIG_ROOT / "source_data"
AUDIT_ROOT = ROOT / "results" / "audit"
REPORTING_ROOT = ROOT / "reporting"
DEV_CYCLES = [
    "2005-2006",
    "2007-2008",
    "2009-2010",
    "2011-2012",
    "2013-2014",
    "2015-2016",
    "2017-2018",
]
EXTERNAL_CYCLE = "2021-2023"


# Submission-figure contract
# Core conclusion: the frozen Elastic Net models remain well calibrated in the later
# NHANES cycle, while uncertainty is explicitly design-aware and wider in the fasting
# sample. The participant flow makes every locked exclusion and analysis domain visible.
# Archetype: quantitative grid (calibration) and schematic-led flow (participants).
# Target/output: BMC Cardiovascular Disorders, 170 mm width, editable SVG/PDF plus
# 600-dpi TIFF and 300-dpi PNG.
# Reviewer risk: cross-sectional outcome, smaller fasting domain, and apparent agreement
# in grouped calibration must not be described as recalibration or clinical readiness.

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7.0
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_hashes() -> dict[str, str]:
    manifest = pd.read_csv(AUDIT_ROOT / "final_model_freeze_manifest.csv")
    return {row.model_key: sha256(ROOT / row.model_path) for row in manifest.itertuples()}


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & weights.gt(0)
    if not mask.any():
        return float("nan")
    x = values.loc[mask].astype(float).to_numpy()
    w = weights.loc[mask].astype(float).to_numpy()
    return float(np.sum(w * x) / np.sum(w))


def weighted_sd(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & weights.gt(0)
    if not mask.any():
        return float("nan")
    x = values.loc[mask].astype(float).to_numpy()
    w = weights.loc[mask].astype(float).to_numpy()
    mean = np.sum(w * x) / np.sum(w)
    return float(np.sqrt(np.sum(w * (x - mean) ** 2) / np.sum(w)))


def weighted_quantile(values: pd.Series, weights: pd.Series, probabilities: list[float]) -> list[float]:
    mask = values.notna() & weights.notna() & weights.gt(0)
    if not mask.any():
        return [float("nan")] * len(probabilities)
    x = values.loc[mask].astype(float).to_numpy()
    w = weights.loc[mask].astype(float).to_numpy()
    order = np.argsort(x, kind="mergesort")
    x, w = x[order], w[order]
    cumulative = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return [float(np.interp(q, cumulative, x)) for q in probabilities]


def fmt_mean_sd(data: pd.DataFrame, variable: str, weight: str) -> str:
    mean = weighted_mean(data[variable], data[weight])
    sd = weighted_sd(data[variable], data[weight])
    return f"{mean:.1f} ({sd:.1f})"


def fmt_median_iqr(data: pd.DataFrame, variable: str, weight: str, digits: int = 1) -> str:
    q1, median, q3 = weighted_quantile(data[variable], data[weight], [0.25, 0.50, 0.75])
    return f"{median:.{digits}f} ({q1:.{digits}f}-{q3:.{digits}f})"


def fmt_percent(data: pd.DataFrame, indicator: pd.Series, weight: str) -> str:
    value = weighted_mean(indicator.astype(float), data[weight])
    return f"{100 * value:.1f}"


def analysis_mask(frame: pd.DataFrame, cycles: list[str], weight: str) -> pd.Series:
    # Stage 3A used EPS=1e-12 to exclude NHANES placeholder/tiny weights. The
    # separately locked Stage 3B eligibility rule used positive (>0) weights.
    threshold = 0.0 if cycles == [EXTERNAL_CYCLE] else 1.0e-12
    return (
        frame["cycle"].isin(cycles)
        & frame["age"].ge(20)
        & (~frame["pregnancy_code"].eq(1).fillna(False))
        & frame["cvd"].notna()
        & frame["strata"].notna()
        & frame["psu"].notna()
        & frame[weight].notna()
        & frame[weight].gt(threshold)
    )


def group_columns(frame: pd.DataFrame, weight: str) -> list[tuple[str, pd.DataFrame]]:
    labels = []
    for period_label, cycles in [("Development 2005-2018", DEV_CYCLES), ("Temporal 2021-2023", [EXTERNAL_CYCLE])]:
        base = frame.loc[analysis_mask(frame, cycles, weight)].copy()
        for outcome, outcome_label in [(0.0, "No CVD"), (1.0, "CVD")]:
            group = base.loc[base["cvd"].eq(outcome)].copy()
            labels.append((f"{period_label}: {outcome_label}", group))
    return labels


def build_baseline_table(frame: pd.DataFrame) -> pd.DataFrame:
    mec_groups = group_columns(frame, "combined_mec_weight")
    fasting_groups = group_columns(frame, "combined_fasting_weight")
    columns = [name for name, _ in mec_groups]
    rows: list[dict[str, str]] = []

    def add_row(characteristic: str, values: list[str]) -> None:
        rows.append({"Characteristic": characteristic, **dict(zip(columns, values))})

    add_row("MEC examination sample", ["" for _ in mec_groups])
    add_row("Unweighted n", [f"{len(g):,}" for _, g in mec_groups])
    add_row("Age, years", [fmt_mean_sd(g, "age", "combined_mec_weight") for _, g in mec_groups])
    add_row("Female, %", [fmt_percent(g, g["sex"].eq("Female"), "combined_mec_weight") for _, g in mec_groups])
    race_levels = ["Mexican American", "Other Hispanic", "Non-Hispanic White", "Non-Hispanic Black", "Non-Hispanic Asian", "Other/Multi"]
    for level in race_levels:
        add_row(f"Race/ethnicity: {level}, %", [fmt_percent(g, g["race_ethnicity"].eq(level), "combined_mec_weight") for _, g in mec_groups])
    education_levels = ["<9th grade", "9-11th grade", "High school/GED", "Some college/AA", "College graduate+"]
    for level in education_levels:
        add_row(f"Education: {level}, %", [fmt_percent(g, g["education"].eq(level), "combined_mec_weight") for _, g in mec_groups])
    add_row("Education missing, %", [fmt_percent(g, g["education"].isna(), "combined_mec_weight") for _, g in mec_groups])
    for level in ["Never", "Former", "Current"]:
        add_row(f"Smoking: {level}, %", [fmt_percent(g, g["smoking"].eq(level), "combined_mec_weight") for _, g in mec_groups])
    add_row("Smoking missing, %", [fmt_percent(g, g["smoking"].isna(), "combined_mec_weight") for _, g in mec_groups])
    add_row("Body mass index, kg/m2", [fmt_mean_sd(g, "bmi", "combined_mec_weight") for _, g in mec_groups])
    add_row("Systolic blood pressure, mm Hg", [fmt_mean_sd(g, "mean_sbp", "combined_mec_weight") for _, g in mec_groups])
    add_row("Diabetes, %", [fmt_percent(g, g["diabetes"].eq(1), "combined_mec_weight") for _, g in mec_groups])
    add_row("Total cholesterol, mg/dL", [fmt_mean_sd(g, "total_cholesterol", "combined_mec_weight") for _, g in mec_groups])
    add_row("HDL cholesterol, mg/dL", [fmt_mean_sd(g, "hdl_cholesterol", "combined_mec_weight") for _, g in mec_groups])
    add_row("eGFR, mL/min/1.73 m2", [fmt_mean_sd(g, "egfr", "combined_mec_weight") for _, g in mec_groups])
    add_row("UACR, mg/g", [fmt_median_iqr(g, "uacr", "combined_mec_weight", 1) for _, g in mec_groups])
    add_row("SIRI", [fmt_median_iqr(g, "siri", "combined_mec_weight", 2) for _, g in mec_groups])

    add_row("Fasting subsample", ["" for _ in fasting_groups])
    add_row("Unweighted fasting n", [f"{len(g):,}" for _, g in fasting_groups])
    add_row("Waist circumference, cm", [fmt_mean_sd(g, "waist", "combined_fasting_weight") for _, g in fasting_groups])
    add_row("Fasting glucose, mg/dL", [fmt_mean_sd(g, "fasting_glucose", "combined_fasting_weight") for _, g in fasting_groups])
    add_row("Triglycerides, mg/dL", [fmt_median_iqr(g, "triglycerides_harmonized", "combined_fasting_weight", 1) for _, g in fasting_groups])
    add_row("TyG-WC", [fmt_median_iqr(g, "tyg_wc", "combined_fasting_weight", 1) for _, g in fasting_groups])
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_ROOT / "Main_Table_1_weighted_baseline_characteristics.csv", index=False, encoding="utf-8-sig")
    return result


def build_missingness_table(frame: pd.DataFrame) -> pd.DataFrame:
    with (ROOT / "config" / "frozen_variable_specification.yml").open(encoding="utf-8") as handle:
        features = yaml.safe_load(handle)
    rows = []
    for period, cycles in [("Development 2005-2018", DEV_CYCLES), ("Temporal 2021-2023", [EXTERNAL_CYCLE])]:
        for model in ["model0_core", "model1_renal", "model2_metabolic", "model3_inflammatory", "model4_combined"]:
            weight = "combined_fasting_weight" if model in {"model2_metabolic", "model4_combined"} else "combined_mec_weight"
            sample = frame.loc[analysis_mask(frame, cycles, weight)]
            for predictor in features[model]:
                missing = int(sample[predictor].isna().sum())
                rows.append({
                    "period": period,
                    "model": model,
                    "weight_variable": weight,
                    "predictor": predictor,
                    "analysis_sample_n": len(sample),
                    "missing_n": missing,
                    "missing_percent": round(100 * missing / len(sample), 2),
                })
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_ROOT / "Supplementary_Table_S7_predictor_missingness.csv", index=False, encoding="utf-8-sig")
    return result


def flow_counts(frame: pd.DataFrame, label: str, cycles: list[str]) -> list[dict[str, object]]:
    d = frame.loc[frame["cycle"].isin(cycles)].copy()
    masks: list[tuple[str, pd.Series]] = []
    current = pd.Series(True, index=d.index)
    masks.append(("Public-use examination records", current.copy()))
    current &= d["age"].ge(20)
    masks.append(("Age 20 years or older", current.copy()))
    current &= ~d["pregnancy_code"].eq(1).fillna(False)
    masks.append(("Not pregnant under locked rule", current.copy()))
    current &= d["cvd"].notna()
    masks.append(("Ascertainable self-reported CVD", current.copy()))
    current &= d["strata"].notna() & d["psu"].notna()
    masks.append(("Valid masked stratum and PSU", current.copy()))
    threshold = 0.0 if cycles == [EXTERNAL_CYCLE] else 1.0e-12
    mec = current & d["combined_mec_weight"].notna() & d["combined_mec_weight"].gt(threshold)
    fasting = current & d["combined_fasting_weight"].notna() & d["combined_fasting_weight"].gt(threshold)
    masks.append(("Final MEC analysis sample", mec.copy()))
    masks.append(("Final fasting analysis sample", fasting.copy()))
    rows = []
    previous = None
    for step, mask in masks:
        count = int(mask.sum())
        excluded = "" if previous is None else previous - count
        events = int(d.loc[mask, "cvd"].sum()) if d.loc[mask, "cvd"].notna().all() and count else ""
        rows.append({"period": label, "step": step, "included_n": count, "excluded_since_previous": excluded, "cvd_events": events})
        if step != "Final MEC analysis sample":
            previous = count
    return rows


def draw_flowchart(flow: pd.DataFrame) -> None:
    colors = {"Development 2005-2018": "#DCE6F2", "Temporal 2021-2023": "#E5E0EC"}
    fig, ax = plt.subplots(figsize=(6.69, 5.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    columns = [("Development 2005-2018", 0.27), ("Temporal 2021-2023", 0.73)]
    sequential = [
        "Public-use examination records",
        "Age 20 years or older",
        "Not pregnant under locked rule",
        "Ascertainable self-reported CVD",
        "Valid masked stratum and PSU",
    ]
    y_values = [0.88, 0.72, 0.56, 0.40, 0.24]
    for period, x in columns:
        ax.text(x, 0.98, period, ha="center", va="top", fontsize=8.2, fontweight="bold")
        source = flow.loc[flow["period"].eq(period)].set_index("step")
        for i, (step, y) in enumerate(zip(sequential, y_values)):
            n = int(source.loc[step, "included_n"])
            exc = source.loc[step, "excluded_since_previous"]
            suffix = "" if exc == "" else f"\nExcluded since prior step: {int(exc):,}"
            box = FancyBboxPatch((x - 0.18, y - 0.055), 0.36, 0.11, boxstyle="round,pad=0.008,rounding_size=0.01", linewidth=0.8, edgecolor="#4D4D4D", facecolor=colors[period])
            ax.add_patch(box)
            ax.text(x, y, f"{step}\nn = {n:,}{suffix}", ha="center", va="center", fontsize=6.2, linespacing=1.15)
            if i < len(y_values) - 1:
                ax.annotate("", xy=(x, y_values[i + 1] + 0.06), xytext=(x, y - 0.06), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#606060"))
        for dx, step in [(-0.105, "Final MEC analysis sample"), (0.105, "Final fasting analysis sample")]:
            n = int(source.loc[step, "included_n"])
            events = int(source.loc[step, "cvd_events"])
            box = FancyBboxPatch((x + dx - 0.095, 0.035), 0.19, 0.115, boxstyle="round,pad=0.007,rounding_size=0.01", linewidth=1.0, edgecolor="#0F4D92", facecolor="white")
            ax.add_patch(box)
            label = "MEC sample" if "MEC" in step else "Fasting sample"
            ax.text(x + dx, 0.0925, f"{label}\nn = {n:,}\nCVD events = {events:,}", ha="center", va="center", fontsize=6.2, fontweight="bold")
            ax.annotate("", xy=(x + dx, 0.155), xytext=(x, 0.18), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#606060"))
    fig.tight_layout(pad=0.35)
    save_figure(fig, FIG_ROOT / "Figure_1_participant_flow")


def build_flow(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(flow_counts(frame, "Development 2005-2018", DEV_CYCLES) + flow_counts(frame, "Temporal 2021-2023", [EXTERNAL_CYCLE]))
    result.to_csv(SOURCE_ROOT / "Figure_1_participant_flow_source_data.csv", index=False, encoding="utf-8-sig")
    draw_flowchart(result)
    return result


def build_temporal_performance_table() -> pd.DataFrame:
    perf = pd.read_csv(PERF_PATH)
    metric_labels = {
        "survey_weighted_AUROC": "AUROC (95% CI)",
        "survey_weighted_PR_AUC": "PR-AUC (95% CI)",
        "weighted_Brier": "Brier score (95% CI)",
        "weighted_log_loss": "Log loss (95% CI)",
        "calibration_intercept": "Calibration intercept (95% CI)",
        "calibration_slope": "Calibration slope (95% CI)",
        "observed_expected_ratio": "O/E ratio (95% CI)",
    }
    model_order = ["model0_core", "model1_renal", "model0_paired_metabolic", "model2_metabolic", "model3_inflammatory", "model0_paired_combined", "model4_combined"]
    rows = []
    for algorithm in ["elastic_net", "xgboost"]:
        for model in model_order:
            group = perf.loc[perf["algorithm"].eq(algorithm) & perf["model"].eq(model)]
            if group.empty:
                raise RuntimeError(f"Missing temporal performance group: {algorithm} {model}")
            first = group.iloc[0]
            row = {
                "Algorithm": "Elastic Net" if algorithm == "elastic_net" else "XGBoost",
                "Model": first["model_label"],
                "Target sample": first["target_population"],
                "n (events)": f"{int(first['n']):,} ({int(first['events']):,})",
            }
            indexed = group.set_index("metric")
            for metric, label in metric_labels.items():
                item = indexed.loc[metric]
                row[label] = f"{item['point_estimate']:.3f} ({item['ci_lower']:.3f} to {item['ci_upper']:.3f})"
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_ROOT / "Supplementary_Table_S6_full_temporal_performance.csv", index=False, encoding="utf-8-sig")
    return result


def weighted_deciles(probability: np.ndarray, weight: np.ndarray) -> np.ndarray:
    order = np.argsort(probability, kind="mergesort")
    cumulative = np.cumsum(weight[order]) / np.sum(weight)
    bins_sorted = np.minimum((cumulative * 10).astype(int), 9)
    bins = np.empty(len(probability), dtype=int)
    bins[order] = bins_sorted
    return bins


def calibration_source() -> pd.DataFrame:
    predictions = pd.read_csv(PRED_PATH)
    selected = predictions.loc[
        predictions["algorithm"].eq("elastic_net")
        & predictions["model"].isin(["model0_core", "model1_renal", "model4_combined"])
    ].copy()
    clusters = selected[["cycle", "strata", "psu"]].drop_duplicates().sort_values(["cycle", "strata", "psu"], kind="mergesort").reset_index(drop=True)
    clusters["cluster_index"] = np.arange(len(clusters), dtype=np.int32)
    selected = selected.merge(clusters, on=["cycle", "strata", "psu"], how="left", validate="many_to_one")
    strata_indices = [g["cluster_index"].to_numpy(dtype=np.int32) for _, g in clusters.groupby(["cycle", "strata"], sort=False)]
    config = yaml.safe_load((ROOT / "config" / "stage3b_temporal_validation.yml").read_text(encoding="utf-8"))
    seed = int(config["uncertainty"]["seed"])
    n_replicates = int(config["uncertainty"]["replicates"])
    factor_matrix = np.vstack([rao_wu_factors(strata_indices, len(clusters), seed, r) for r in range(n_replicates)])
    rows = []
    for model in ["model0_core", "model1_renal", "model4_combined"]:
        group = selected.loc[selected["model"].eq(model)].sort_values("seqn", kind="mergesort").reset_index(drop=True)
        p = group["predicted_probability"].to_numpy(dtype=float)
        y = group["cvd"].to_numpy(dtype=float)
        w = group["analysis_weight"].to_numpy(dtype=float)
        cluster_index = group["cluster_index"].to_numpy(dtype=int)
        bins = weighted_deciles(p, w)
        for bin_id in range(10):
            mask = bins == bin_id
            point_pred = float(np.sum(w[mask] * p[mask]) / np.sum(w[mask]))
            point_obs = float(np.sum(w[mask] * y[mask]) / np.sum(w[mask]))
            replicate_obs = np.empty(n_replicates)
            replicate_pred = np.empty(n_replicates)
            cluster_factors = factor_matrix[:, cluster_index[mask]]
            replicate_weights = cluster_factors * w[mask][None, :]
            denominator = replicate_weights.sum(axis=1)
            valid = denominator > 0
            replicate_obs[:] = np.nan
            replicate_pred[:] = np.nan
            replicate_obs[valid] = (replicate_weights[valid] @ y[mask]) / denominator[valid]
            replicate_pred[valid] = (replicate_weights[valid] @ p[mask]) / denominator[valid]
            rows.append({
                "model": model,
                "model_label": group["model_label"].iloc[0],
                "target_population": group["target_population"].iloc[0],
                "decile": bin_id + 1,
                "unweighted_n": int(mask.sum()),
                "events": int(y[mask].sum()),
                "weighted_mean_predicted_probability": point_pred,
                "weighted_observed_prevalence": point_obs,
                "observed_ci_lower": float(np.nanquantile(replicate_obs, 0.025)),
                "observed_ci_upper": float(np.nanquantile(replicate_obs, 0.975)),
                "predicted_ci_lower": float(np.nanquantile(replicate_pred, 0.025)),
                "predicted_ci_upper": float(np.nanquantile(replicate_pred, 0.975)),
                "n_replicates_valid": int(np.isfinite(replicate_obs).sum()),
            })
    result = pd.DataFrame(rows)
    result.to_csv(SOURCE_ROOT / "Figure_5_temporal_calibration_source_data.csv", index=False, encoding="utf-8-sig")
    return result


def draw_calibration(source: pd.DataFrame) -> None:
    perf = pd.read_csv(PERF_PATH)
    titles = {
        "model0_core": "Core model (MEC)",
        "model1_renal": "Renal extension (MEC)",
        "model4_combined": "Combined extension (fasting)",
    }
    colors = {"model0_core": "#484878", "model1_renal": "#3775BA", "model4_combined": "#B64342"}
    max_value = max(0.5, float(source[["weighted_mean_predicted_probability", "observed_ci_upper"]].max().max()))
    max_value = min(1.0, math.ceil(max_value * 10) / 10)
    fig, axes = plt.subplots(1, 3, figsize=(6.69, 2.45), sharex=True, sharey=True)
    for panel, (ax, model) in enumerate(zip(axes, titles)):
        data = source.loc[source["model"].eq(model)].sort_values("decile")
        x = data["weighted_mean_predicted_probability"].to_numpy()
        y = data["weighted_observed_prevalence"].to_numpy()
        yerr = np.vstack([y - data["observed_ci_lower"].to_numpy(), data["observed_ci_upper"].to_numpy() - y])
        ax.plot([0, max_value], [0, max_value], ls="--", lw=0.9, color="#8F8F8F", label="Ideal")
        ax.errorbar(x, y, yerr=yerr, fmt="o-", ms=3.2, lw=1.2, elinewidth=0.8, capsize=2, color=colors[model], markeredgecolor="white", markeredgewidth=0.5, label="Grouped calibration")
        ax.set_title(titles[model], fontsize=7.6, fontweight="bold", pad=5)
        ax.set_xlim(0, max_value)
        ax.set_ylim(0, max_value)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=6.3, width=0.7, length=2.5)
        p = perf.loc[perf["algorithm"].eq("elastic_net") & perf["model"].eq(model)].set_index("metric")
        intercept = p.loc["calibration_intercept"]
        slope = p.loc["calibration_slope"]
        oe = p.loc["observed_expected_ratio"]
        annotation = (
            f"Intercept {intercept.point_estimate:.2f}\n"
            f"Slope {slope.point_estimate:.2f}\n"
            f"O/E {oe.point_estimate:.2f}"
        )
        ax.text(0.04, 0.96, annotation, transform=ax.transAxes, ha="left", va="top", fontsize=5.9, color="#333333")
        ax.text(-0.16, 1.04, chr(ord("a") + panel), transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    axes[0].set_ylabel("Survey-weighted observed prevalence", fontsize=6.8)
    for ax in axes:
        ax.set_xlabel("Survey-weighted mean predicted probability", fontsize=6.8)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.87, bottom=0.19, wspace=0.24)
    save_figure(fig, FIG_ROOT / "Figure_5_temporal_calibration", tight=False)


def save_figure(fig: plt.Figure, base: Path, tight: bool = True) -> None:
    kwargs = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(base.with_suffix(".svg"), **kwargs)
    fig.savefig(base.with_suffix(".pdf"), **kwargs)
    fig.savefig(base.with_suffix(".png"), dpi=300, **kwargs)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"}, **kwargs)
    plt.close(fig)


def write_contract() -> None:
    text = """# Submission figure contract

## Figure 1: participant flow

- Core conclusion: all locked eligibility steps and the MEC-versus-fasting domain split are transparent for both development and temporal samples.
- Archetype: schematic-led participant flow.
- Backend: Python only.
- Output: 170 mm maximum width; SVG, PDF, 600-dpi TIFF, and 300-dpi PNG.
- Statistics/source: unweighted counts and CVD events derived from the locked harmonised dataset; source CSV accompanies the figure.
- Reviewer risk: the fasting sample is a domain-specific subsample, not attrition after the MEC model.

## Figure 5: temporal calibration

- Core conclusion: the frozen primary Elastic Net core, renal, and combined models show broadly concordant grouped predictions and observed prevalence in NHANES 2021-2023 without recalibration.
- Archetype: quantitative three-panel grid.
- Evidence hierarchy: core model, isolated renal extension, combined fasting extension.
- Backend: Python only.
- Statistics/source: fixed survey-weighted probability deciles; observed prevalence with 95% percentile intervals from 2,000 Rao-Wu rescaled bootstrap replicates; model-level intercept, slope, and O/E are copied from the locked Stage 3B inference table.
- Image integrity: vector text remains editable; no raster source images or selective image adjustment.
- Reviewer risk: grouped calibration can conceal within-bin departures and does not establish clinical utility or transport outside NHANES.
"""
    (REPORTING_ROOT / "submission_figure_contract.md").write_text(text, encoding="utf-8")


def write_audit(before: dict[str, str], after: dict[str, str], baseline: pd.DataFrame, flow: pd.DataFrame, temporal: pd.DataFrame, calibration: pd.DataFrame, missingness: pd.DataFrame) -> None:
    dev = flow.loc[flow["period"].eq("Development 2005-2018")].set_index("step")
    ext = flow.loc[flow["period"].eq("Temporal 2021-2023")].set_index("step")
    checks = {
        "frozen_model_hashes_unchanged": before == after,
        "baseline_table_created": len(baseline) >= 25,
        "development_mec_n_37482": int(dev.loc["Final MEC analysis sample", "included_n"]) == 37482,
        "development_fasting_n_15956": int(dev.loc["Final fasting analysis sample", "included_n"]) == 15956,
        "temporal_mec_n_7766": int(ext.loc["Final MEC analysis sample", "included_n"]) == 7766,
        "temporal_fasting_n_3397": int(ext.loc["Final fasting analysis sample", "included_n"]) == 3397,
        "full_temporal_groups_14": len(temporal) == 14,
        "calibration_models_3": calibration["model"].nunique() == 3,
        "calibration_deciles_30": len(calibration) == 30,
        "calibration_replicates_2000": calibration["n_replicates_valid"].min() == 2000,
        "predictor_missingness_reported": len(missingness) > 0,
        "figure_1_exports_complete": all((FIG_ROOT / f"Figure_1_participant_flow.{extn}").exists() for extn in ["svg", "pdf", "png", "tiff"]),
        "figure_5_exports_complete": all((FIG_ROOT / f"Figure_5_temporal_calibration.{extn}").exists() for extn in ["svg", "pdf", "png", "tiff"]),
        "no_shap_added": not any(ROOT.rglob("*shap*")),
    }
    path = AUDIT_ROOT / "submission_additions_completion_gate.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check", "status"])
        for key, passed in checks.items():
            writer.writerow([key, "PASS" if passed else "FAIL"])
        writer.writerow(["overall", "PASS" if all(checks.values()) else "FAIL"])
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise RuntimeError(f"Submission additions gate failed: {failed}")


def main() -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTING_ROOT.mkdir(parents=True, exist_ok=True)
    before = frozen_hashes()
    frame = pd.read_csv(DATA_PATH, low_memory=False)
    baseline = build_baseline_table(frame)
    missingness = build_missingness_table(frame)
    flow = build_flow(frame)
    temporal = build_temporal_performance_table()
    calibration = calibration_source()
    draw_calibration(calibration)
    write_contract()
    after = frozen_hashes()
    write_audit(before, after, baseline, flow, temporal, calibration, missingness)
    print("submission_additions=PASS")


if __name__ == "__main__":
    main()
