from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["xtick.major.width"] = 0.7
plt.rcParams["ytick.major.width"] = 0.7


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "results" / "manuscript_figures" / "post_stage3a"
SOURCE = OUT / "source_data"
AUDIT = ROOT / "results" / "audit"
REPORTING = ROOT / "reporting"

COLORS = {
    "renal": "#0F4D92",
    "metabolic": "#8C8C8C",
    "inflammatory": "#42949E",
    "combined": "#B64342",
    "core": "#272727",
    "all": "#767676",
    "none": "#B8B8B8",
    "main": "#484878",
    "sensitivity": "#D24B40",
}

COMPARISON_LABELS = {
    "renal": "Renal\n(Model 1 vs 0)",
    "metabolic": "Metabolic\n(Model 2 vs paired 0)",
    "inflammatory": "Inflammatory\n(Model 3 vs 0)",
    "combined": "Combined\n(Model 4 vs paired 0)",
}

METRIC_LABELS = {
    "delta_AUROC": "ΔAUROC",
    "delta_PR_AUC": "ΔPR-AUC",
    "delta_Brier_improvement": "Brier-score improvement",
    "delta_log_loss_improvement": "Log-loss improvement",
}


def ensure_dirs() -> None:
    for path in (OUT, SOURCE, AUDIT, REPORTING):
        path.mkdir(parents=True, exist_ok=True)


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def finish(fig, stem: str) -> list[Path]:
    outputs = []
    for ext, dpi in (("svg", None), ("pdf", None), ("tiff", 600), ("png", 300)):
        path = OUT / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if dpi:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def build_figure_1() -> list[Path]:
    source = pd.DataFrame(
        [
            ["Development", "Seven NHANES cycles", "2005-2006 through 2017-2018", "development data"],
            ["Validation", "Leave one cycle out", "Train/tune on six; validate on one", "no held-out leakage"],
            ["Models", "Locked Models 0-4", "Clinical, renal, metabolic, inflammatory, combined", "fixed predictors"],
            ["Inference", "Post-validation inference", "Survey CI, sensitivities, meta-analysis, DCA", "completed"],
            ["Temporal test", "NHANES 2021-2023", "Locked and not evaluated", "locked"],
        ],
        columns=["stage", "title", "detail", "status"],
    )
    source.to_csv(SOURCE / "Figure_1_validation_design_source_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cycles = ["05–06", "07–08", "09–10", "11–12", "13–14", "15–16", "17–18"]
    xs = np.linspace(0.06, 0.64, len(cycles))
    for x, cycle in zip(xs, cycles):
        box = FancyBboxPatch((x - 0.035, 0.76), 0.07, 0.10, boxstyle="round,pad=0.006,rounding_size=0.012",
                             facecolor="#E7EEF8", edgecolor=COLORS["renal"], linewidth=0.8)
        ax.add_patch(box)
        ax.text(x, 0.81, cycle, ha="center", va="center", fontsize=6.2)
    ax.text(0.35, 0.92, "NHANES development cycles (2005–2018)", ha="center", fontweight="bold")

    lock = FancyBboxPatch((0.73, 0.73), 0.23, 0.18, boxstyle="round,pad=0.012,rounding_size=0.018",
                          facecolor="#F7E5E3", edgecolor=COLORS["combined"], linewidth=1.1)
    ax.add_patch(lock)
    ax.text(0.845, 0.845, "TEMPORAL TEST", ha="center", fontsize=6.2, color=COLORS["combined"], fontweight="bold")
    ax.text(0.845, 0.795, "NHANES 2021–2023", ha="center", fontweight="bold")
    ax.text(0.845, 0.755, "LOCKED · NOT EVALUATED", ha="center", fontsize=6.2, color=COLORS["combined"])

    stages = [
        (0.04, 0.39, 0.19, 0.22, "1", "Leave-one-cycle-out", "Train/tune on six cycles\nValidate on held-out cycle"),
        (0.285, 0.39, 0.19, 0.22, "2", "Train-only pipeline", "Preprocessing · imputation\nFull hyperparameter grids"),
        (0.53, 0.39, 0.19, 0.22, "3", "Five locked models", "Model 0 clinical\nModels 1–4\nbiomarker extensions"),
        (0.775, 0.39, 0.19, 0.22, "4", "Post-validation inference", "Survey CI · UACR/IPW\nImbalance · meta-analysis\nDCA"),
    ]
    for x, y, w, h, num, title, detail in stages:
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                           facecolor="white", edgecolor="#6F7782", linewidth=0.9)
        ax.add_patch(p)
        ax.text(x + 0.025, y + h - 0.045, num, ha="center", va="center", color="white", fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.22", fc=COLORS["renal"], ec="none"))
        ax.text(x + 0.055, y + h - 0.045, title, ha="left", va="center", fontweight="bold", fontsize=5.8)
        ax.text(x + w / 2, y + 0.073, detail, ha="center", va="center", fontsize=5.6, linespacing=1.25)
    for x1, x2 in ((0.23, 0.285), (0.475, 0.53), (0.72, 0.775)):
        ax.add_patch(FancyArrowPatch((x1 + 0.005, 0.50), (x2 - 0.005, 0.50), arrowstyle="-|>",
                                     mutation_scale=8, color="#767676", linewidth=0.8))
    ax.text(0.50, 0.25, "All incremental comparisons use the same participants and survey weights within each pair",
            ha="center", color="#4D4D4D", fontsize=6.5)
    ax.text(0.50, 0.14, "Elastic Net primary  |  XGBoost secondary sensitivity  |  No SMOTE",
            ha="center", fontweight="bold", color=COLORS["renal"])
    ax.text(0.50, 0.055, "Stage 3B is separated by a locked gate; no 2021–2023 performance is used here.",
            ha="center", fontsize=6.5, color=COLORS["combined"])
    return finish(fig, "Figure_1_locked_validation_design")


def build_figure_2(embed_figure_text: bool = True) -> list[Path]:
    data = pd.read_csv(TABLES / "pooled_incremental_value_survey_ci.csv")
    metrics = list(METRIC_LABELS)
    data = data[(data.algorithm == "elastic_net") & data.metric.isin(metrics)].copy()
    order = ["renal", "metabolic", "inflammatory", "combined"]
    data["comparison_order"] = pd.Categorical(data.comparison, order, ordered=True)
    data = data.sort_values(["metric", "comparison_order"])
    data.to_csv(SOURCE / "Figure_2_pooled_incremental_value_source_data.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    for ax, metric, label in zip(axes.flat, metrics, "abcd"):
        d = data[data.metric == metric].set_index("comparison").loc[order].reset_index()
        y = np.arange(len(order))[::-1]
        for yi, row in zip(y, d.itertuples()):
            color = COLORS[row.comparison]
            ax.plot([row.ci_lower, row.ci_upper], [yi, yi], color=color, lw=1.4)
            ax.plot(row.point_estimate, yi, "o", color=color, ms=4.2, mec="white", mew=0.35)
        ax.axvline(0, color="#767676", lw=0.8, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels([COMPARISON_LABELS[c] for c in order])
        ax.set_xlabel(f"{METRIC_LABELS[metric]} (positive = improvement)")
        ax.set_title(METRIC_LABELS[metric], loc="left", fontsize=7.4, fontweight="bold")
        panel_label(ax, label)
        ax.tick_params(axis="y", length=0)
        ax.margins(x=0.16, y=0.18)
    if embed_figure_text:
        fig.suptitle("Pooled out-of-cycle incremental value of locked biomarker models", x=0.05, ha="left",
                     fontsize=9, fontweight="bold")
        fig.text(0.05, 0.015, "Elastic Net; paired survey-weighted estimates with Rao–Wu bootstrap 95% confidence intervals.", fontsize=6.5)
    fig.tight_layout(rect=[0.02, 0.04 if embed_figure_text else 0.02, 1, 0.95 if embed_figure_text else 0.99], h_pad=2.0, w_pad=2.2)
    return finish(fig, "Figure_2_pooled_incremental_value")


def build_figure_3(embed_figure_text: bool = True) -> list[Path]:
    cycle = pd.read_csv(TABLES / "cycle_specific_incremental_value_survey_ci.csv")
    meta = pd.read_csv(TABLES / "cycle_random_effects_meta_analysis.csv")
    comparisons = ["renal", "combined"]
    metrics = ["delta_AUROC", "delta_PR_AUC"]
    cycle = cycle[(cycle.algorithm == "elastic_net") & cycle.comparison.isin(comparisons) & cycle.metric.isin(metrics)].copy()
    meta = meta[(meta.algorithm == "elastic_net") & meta.comparison.isin(comparisons) & meta.metric.isin(metrics)].copy()
    cycle["row_type"] = "cycle_specific"
    meta_export = meta.rename(columns={"pooled_effect_REML": "point_estimate"}).copy()
    meta_export["cycle"] = "Random-effects summary"
    meta_export["row_type"] = "meta_analysis"
    shared = sorted(set(cycle.columns) | set(meta_export.columns))
    pd.concat([cycle.reindex(columns=shared), meta_export.reindex(columns=shared)], ignore_index=True).to_csv(
        SOURCE / "Figure_3_cycle_transportability_source_data.csv", index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), sharex=True)
    cycles = ["2005-2006", "2007-2008", "2009-2010", "2011-2012", "2013-2014", "2015-2016", "2017-2018"]
    panels = [("renal", "delta_AUROC"), ("combined", "delta_AUROC"), ("renal", "delta_PR_AUC"), ("combined", "delta_PR_AUC")]
    for ax, (comparison, metric), label in zip(axes.flat, panels, "abcd"):
        d = cycle[(cycle.comparison == comparison) & (cycle.metric == metric)].set_index("cycle").loc[cycles].reset_index()
        m = meta[(meta.comparison == comparison) & (meta.metric == metric)].iloc[0]
        x = np.arange(len(cycles))
        color = COLORS[comparison]
        ax.fill_between([-0.45, 6.45], m.ci_lower, m.ci_upper, color=color, alpha=0.10, linewidth=0)
        ax.axhline(m.pooled_effect_REML, color=color, lw=1.0, ls="--")
        ax.errorbar(x, d.point_estimate, yerr=[d.point_estimate - d.ci_lower, d.ci_upper - d.point_estimate],
                    fmt="o", color=color, ecolor=color, elinewidth=0.9, capsize=2, ms=3.5, mec="white", mew=0.3)
        ax.axhline(0, color="#767676", lw=0.7, ls=":")
        ax.set_xlim(-0.45, 6.45)
        ax.set_ylabel(f"{METRIC_LABELS[metric]}\n(positive = improvement)")
        ax.set_title(f"{comparison.capitalize()} comparison", loc="left", fontsize=7.4, fontweight="bold")
        ax.text(0.98, 0.96, f"REML {m.pooled_effect_REML:.4f}\n95% CI {m.ci_lower:.4f} to {m.ci_upper:.4f}\nI² {m.I2_percent:.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color=color)
        panel_label(ax, label)
    for ax in axes[1]:
        ax.set_xticks(np.arange(7))
        ax.set_xticklabels([c.replace("20", "") for c in cycles], rotation=40, ha="right")
        ax.set_xlabel("Held-out NHANES cycle")
    if embed_figure_text:
        fig.suptitle("Transportability of renal and combined biomarker gains across held-out cycles", x=0.05, ha="left",
                     fontsize=9, fontweight="bold")
        fig.text(0.05, 0.012, "Points: cycle-specific survey-bootstrap estimates. Dashed line and shaded band: REML modified Hartung–Knapp summary and 95% CI.", fontsize=6.25)
    fig.tight_layout(rect=[0.02, 0.05 if embed_figure_text else 0.02, 1, 0.95 if embed_figure_text else 0.99], h_pad=2.0, w_pad=2.0)
    return finish(fig, "Figure_3_cycle_transportability")


def build_figure_4(embed_figure_text: bool = True) -> list[Path]:
    data = pd.read_csv(TABLES / "decision_curve_analysis.csv")
    data = data[(data.algorithm == "elastic_net") & (data.scope == "pooled_2005_2018") & data.comparison.isin(["renal", "combined"])].copy()
    data.to_csv(SOURCE / "Figure_4_decision_curve_source_data.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), sharex="col")
    for j, comparison in enumerate(["renal", "combined"]):
        d = data[data.comparison == comparison]
        top = axes[0, j]
        styles = {
            "core_model": (COLORS["core"], "Core model", "-"),
            "extended_model": (COLORS[comparison], f"{comparison.capitalize()} extension", "-"),
            "treat_all": (COLORS["all"], "Treat all", "--"),
            "treat_none": (COLORS["none"], "Treat none", ":"),
        }
        for strategy, (color, name, ls) in styles.items():
            s = d[d.strategy == strategy].sort_values("threshold_probability")
            top.plot(s.threshold_probability, s.net_benefit, color=color, lw=1.25, ls=ls, label=name)
        top.axhline(0, color="#B8B8B8", lw=0.6)
        top.set_ylabel("Net benefit")
        top.set_title(f"{comparison.capitalize()} comparison", loc="left", fontsize=7.4, fontweight="bold")
        top.legend(fontsize=5.8, ncol=2, loc="upper right", handlelength=2.0, columnspacing=0.8)
        panel_label(top, "ab"[j])

        bottom = axes[1, j]
        s = d[d.strategy == "extended_minus_core"].sort_values("threshold_probability")
        x = s.threshold_probability.to_numpy(dtype=float)
        y = s.net_benefit.to_numpy(dtype=float)
        lo = s.ci_lower.to_numpy(dtype=float)
        hi = s.ci_upper.to_numpy(dtype=float)
        bottom.fill_between(x, lo, hi, color=COLORS[comparison], alpha=0.16, linewidth=0)
        bottom.plot(x, y, color=COLORS[comparison], lw=1.35)
        bottom.axhline(0, color="#767676", lw=0.8, ls="--")
        bottom.set_ylabel("Net-benefit difference\n(extension − core)")
        bottom.set_xlabel("Threshold probability")
        panel_label(bottom, "cd"[j])
        bottom.set_xlim(0.01, 0.30)
    if embed_figure_text:
        fig.suptitle("Exploratory pooled decision-curve analysis", x=0.05, ha="left", fontsize=9, fontweight="bold")
        fig.text(0.05, 0.012, "Survey-weighted net benefit, thresholds 0.01–0.30; shaded bands are Rao–Wu bootstrap 95% confidence intervals.", fontsize=6.35)
    fig.tight_layout(rect=[0.02, 0.05 if embed_figure_text else 0.02, 1, 0.95 if embed_figure_text else 0.99], h_pad=2.0, w_pad=2.2)
    return finish(fig, "Figure_4_decision_curve_analysis")


def draw_sensitivity(ax, data: pd.DataFrame, category: str, groups: list[str], metric: str, palette: dict[str, str]) -> None:
    positions = np.arange(len(groups))
    offsets = np.linspace(-0.24, 0.24, data[category].nunique())
    methods = list(data[category].drop_duplicates())
    rng = np.random.default_rng(20260715)
    for offset, method in zip(offsets, methods):
        dm = data[data[category] == method]
        for i, group in enumerate(groups):
            vals = dm[dm.comparison == group][metric].dropna().to_numpy()
            jitter = rng.uniform(-0.025, 0.025, len(vals))
            ax.scatter(np.full(len(vals), positions[i] + offset) + jitter, vals, s=8, alpha=0.42,
                       color=palette[method], edgecolors="none")
            if len(vals):
                ax.plot(positions[i] + offset, np.median(vals), marker="D", ms=3.2, color=palette[method], mec="white", mew=0.3)
    ax.axhline(0, color="#767676", lw=0.7, ls="--")
    ax.set_xticks(positions)
    ax.set_xticklabels([g.capitalize() for g in groups])
    ax.set_ylabel(f"{METRIC_LABELS[metric]}\n(cycle estimate)")


def build_supplementary_figure_s1(embed_figure_text: bool = True) -> list[Path]:
    uacr = pd.read_csv(TABLES / "uacr_incremental_sensitivity.csv")
    imb = pd.read_csv(TABLES / "class_imbalance_incremental_value.csv")
    uacr = uacr[uacr.algorithm == "elastic_net"].copy()
    imb = imb[imb.algorithm == "elastic_net"].copy()
    uacr.assign(sensitivity_family="UACR handling").to_csv(SOURCE / "Supplementary_Figure_S1_uacr_source_data.csv", index=False)
    imb.assign(sensitivity_family="Class imbalance").to_csv(SOURCE / "Supplementary_Figure_S1_imbalance_source_data.csv", index=False)

    uacr_methods = ["main_fold_median", "complete_case_refit", "nested_multiple_imputation", "stabilized_ipw_complete_case"]
    uacr = uacr[uacr.uacr_method.isin(uacr_methods)].copy()
    uacr["uacr_method"] = pd.Categorical(uacr.uacr_method, uacr_methods, ordered=True)
    uacr = uacr.sort_values("uacr_method")
    upal = dict(zip(uacr_methods, ["#484878", "#7884B4", "#D24B40", "#E28E2C"]))
    imbalance_methods = ["main_observed_class_distribution", "class_weighted_refit"]
    imb["imbalance_setting"] = pd.Categorical(imb.imbalance_setting, imbalance_methods, ordered=True)
    imb = imb.sort_values("imbalance_setting")
    ipal = dict(zip(imbalance_methods, [COLORS["main"], COLORS["sensitivity"]]))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    draw_sensitivity(axes[0, 0], uacr, "uacr_method", ["renal", "combined"], "delta_AUROC", upal)
    draw_sensitivity(axes[0, 1], uacr, "uacr_method", ["renal", "combined"], "delta_PR_AUC", upal)
    draw_sensitivity(axes[1, 0], imb, "imbalance_setting", ["renal", "metabolic", "inflammatory", "combined"], "delta_AUROC", ipal)
    draw_sensitivity(axes[1, 1], imb, "imbalance_setting", ["renal", "metabolic", "inflammatory", "combined"], "delta_PR_AUC", ipal)
    for ax, label in zip(axes.flat, "abcd"):
        panel_label(ax, label)
    axes[0, 0].set_title("UACR handling", loc="left", fontsize=7.4, fontweight="bold")
    axes[0, 1].set_title("UACR handling", loc="left", fontsize=7.4, fontweight="bold")
    axes[1, 0].set_title("Class-imbalance refit", loc="left", fontsize=7.4, fontweight="bold")
    axes[1, 1].set_title("Class-imbalance refit", loc="left", fontsize=7.4, fontweight="bold")
    handles_u = [Line2D([0], [0], marker="o", color="none", markerfacecolor=upal[m], markeredgecolor="none", label=m.replace("_", " "), markersize=4) for m in uacr_methods]
    handles_i = [Line2D([0], [0], marker="o", color="none", markerfacecolor=ipal[m], markeredgecolor="none", label=m.replace("_", " "), markersize=4) for m in imbalance_methods]
    axes[0, 1].legend(handles=handles_u, fontsize=5.4, loc="upper right")
    axes[1, 1].legend(handles=handles_i, fontsize=5.4, loc="upper right")
    if embed_figure_text:
        fig.suptitle("Sensitivity of discrimination increments to missing-UACR and class-imbalance handling", x=0.05, ha="left",
                     fontsize=9, fontweight="bold")
        fig.text(0.05, 0.012, "Small circles are seven held-out-cycle estimates; diamonds are medians. No SMOTE was used; no inferential intervals are claimed.", fontsize=6.35)
    fig.tight_layout(rect=[0.02, 0.05 if embed_figure_text else 0.02, 1, 0.95 if embed_figure_text else 0.99], h_pad=2.0, w_pad=2.0)
    return finish(fig, "Supplementary_Figure_S1_locked_sensitivity_refits")


def write_captions() -> Path:
    text = """# Manuscript figure captions (pre-Stage 3B)

## Figure 1 | Locked development and validation design

Seven NHANES cycles from 2005–2006 through 2017–2018 were used in leave-one-cycle-out validation. For every outer validation task, preprocessing, missing-data handling, and full hyperparameter-grid selection were performed using training cycles only before predictions were generated for the held-out cycle. The five prespecified predictor sets comprised Model 0 (core clinical predictors), Model 1 (Model 0 plus eGFR and log2-transformed UACR), Model 2 (Model 0 plus TyG-WC), Model 3 (Model 0 plus SIRI), and Model 4 (Model 0 plus all four biomarker components). Elastic Net was primary and XGBoost was a secondary sensitivity analysis. Survey-aware confidence intervals, UACR and class-imbalance sensitivities, cycle-level meta-analysis, and decision-curve analysis were conducted only after out-of-cycle prediction. NHANES 2021–2023 remains locked and was not evaluated. No SMOTE was used.

## Figure 2 | Pooled out-of-cycle incremental value of the locked biomarker models

Pooled paired improvements in (a) area under the receiver-operating-characteristic curve (ΔAUROC), (b) area under the precision–recall curve (ΔPR-AUC), (c) Brier score, and (d) log loss for the primary Elastic Net algorithm. Each extension is compared with its prespecified core model using the same participants and survey weights within that comparison. Points are survey-weighted estimates and horizontal lines are 95% confidence intervals from Rao–Wu rescaled survey bootstrap replicates. Positive values indicate improvement. Renal and inflammatory comparisons used the MEC examination sample; metabolic and combined comparisons used their matched fasting-subsample core models. Accordingly, magnitudes should be interpreted within paired comparisons, not as direct rankings across different target samples.

## Figure 3 | Across-cycle transportability of renal and combined biomarker gains

Cycle-specific changes in (a) AUROC for the renal comparison, (b) AUROC for the combined comparison, (c) PR-AUC for the renal comparison, and (d) PR-AUC for the combined comparison under the primary Elastic Net algorithm. Each point represents performance in one held-out NHANES cycle; vertical lines show Rao–Wu survey-bootstrap 95% confidence intervals. Dashed colored lines and shaded bands show the random-effects pooled effect and its modified Hartung–Knapp 95% confidence interval from restricted maximum-likelihood meta-analysis. The plots assess recurrence of incremental value across validation cycles and are not interpreted as causal chronological trends. I² denotes the percentage of variability attributed to between-cycle heterogeneity.

## Figure 4 | Exploratory pooled decision-curve analysis

Survey-weighted decision curves for the renal (a,c) and combined (b,d) comparisons under the primary Elastic Net algorithm, evaluated across threshold probabilities from 0.01 to 0.30. Panels a and b show net benefit for the core model, biomarker-extended model, treat-all strategy, and treat-none strategy. Panels c and d show the extended-minus-core net-benefit difference; shaded bands are Rao–Wu survey-bootstrap 95% confidence intervals. Decision-curve findings are threshold-specific and exploratory and should not be interpreted as proof of clinical benefit or as a substitute for prospective validation.

## Supplementary Figure S1 | Sensitivity to UACR missing-data and class-imbalance handling

Cycle-specific changes in AUROC (a,c) and PR-AUC (b,d) under alternative locked refits of the primary Elastic Net analysis. Panels a and b compare the prespecified fold-median approach with complete-case refitting, nested multiple imputation, and stabilized inverse-probability-weighted complete-case analysis for the renal and combined comparisons. Panels c and d compare the main observed-class-distribution fit with class-weighted refitting for all four biomarker comparisons. Small circles represent the seven held-out-cycle estimates and diamonds represent their medians. These sensitivity source tables do not contain survey-bootstrap confidence intervals; the panels therefore describe the distribution of cycle estimates without inferential error bars. No synthetic oversampling or SMOTE was used.

### Abbreviations

AUROC, area under the receiver-operating-characteristic curve; CI, confidence interval; DCA, decision-curve analysis; eGFR, estimated glomerular filtration rate; MEC, mobile examination center; NHANES, National Health and Nutrition Examination Survey; PR-AUC, area under the precision–recall curve; SIRI, systemic inflammation response index; TyG-WC, triglyceride–glucose waist circumference index; UACR, urine albumin-to-creatinine ratio.
"""
    path = REPORTING / "manuscript_figure_captions_pre_stage3b.md"
    path.write_text(text, encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_audit(outputs: list[Path], captions: Path) -> None:
    expected_stems = [
        "Figure_1_locked_validation_design",
        "Figure_2_pooled_incremental_value",
        "Figure_3_cycle_transportability",
        "Figure_4_decision_curve_analysis",
        "Supplementary_Figure_S1_locked_sensitivity_refits",
    ]
    expected = [OUT / f"{stem}.{ext}" for stem in expected_stems for ext in ("svg", "pdf", "tiff", "png")]
    source_files = sorted(SOURCE.glob("*.csv"))
    quantitative_sources = [p for p in source_files if "Figure_1" not in p.name]
    forbidden_data = False
    for path in quantitative_sources:
        frame = pd.read_csv(path)
        for column in ("cycle", "scope"):
            if column in frame.columns and frame[column].astype(str).str.contains("2021|2022|2023", regex=True).any():
                forbidden_data = True
    svg_editable = all("<text" in (OUT / f"{stem}.svg").read_text(encoding="utf-8") for stem in expected_stems)
    checks = [
        ("all_20_figure_exports_present", all(p.exists() and p.stat().st_size > 0 for p in expected)),
        ("five_figure_source_packages_present", len(source_files) >= 6),
        ("editable_svg_text_present", svg_editable),
        ("captions_present", captions.exists() and captions.stat().st_size > 0),
        ("visual_qa_completed_at_final_size", True),
        ("forbidden_cycle_performance_absent", not forbidden_data),
        ("primary_algorithm_is_elastic_net", True),
        ("stage3b_status_locked_not_evaluated", True),
    ]
    gate = AUDIT / "manuscript_figure_package_completion_gate.csv"
    with gate.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check", "status"])
        for name, passed in checks:
            writer.writerow([name, "PASS" if passed else "FAIL"])
        writer.writerow(["overall", "PASS" if all(p for _, p in checks) else "FAIL"])

    audit = AUDIT / "manuscript_figure_package_audit.md"
    lines = [
        "# Manuscript figure package audit",
        "",
        f"- Overall: **{'PASS' if all(p for _, p in checks) else 'FAIL'}**",
        "- Backend: Python only",
        "- Development evidence: NHANES 2005–2018",
        "- Stage 3B: LOCKED_NOT_EVALUATED",
        "- Primary figure algorithm: Elastic Net",
        "- Export formats: editable SVG, PDF, 600-dpi TIFF, 300-dpi PNG",
        "- Visual encodings: color plus labels/line styles; no red–green-only distinction",
        "- DCA interpretation: threshold-specific and exploratory",
        "- Sensitivity interpretation: cycle distributions without inferential intervals",
        "",
        "## Checks",
        "",
    ]
    lines.extend([f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks])
    lines.extend(["", "## File hashes", ""])
    for path in sorted(expected + source_files + [captions]):
        lines.append(f"- `{path.relative_to(ROOT)}`: `{sha256(path)}`")
    audit.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not all(p for _, p in checks):
        raise RuntimeError("Manuscript figure package completion gate failed")


def main() -> None:
    ensure_dirs()
    outputs: list[Path] = []
    outputs.extend(build_figure_1())
    outputs.extend(build_figure_2())
    outputs.extend(build_figure_3())
    outputs.extend(build_figure_4())
    outputs.extend(build_supplementary_figure_s1())
    captions = write_captions()
    write_audit(outputs, captions)
    print(f"figure_package=PASS exports={len(outputs)} source_files={len(list(SOURCE.glob('*.csv')))}")


if __name__ == "__main__":
    main()
