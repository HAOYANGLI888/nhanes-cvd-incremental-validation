# Manuscript figure contract (pre-Stage 3B)

## Global contract

- Core conclusion: Kidney markers provide reproducible incremental value beyond the locked clinical model, while the full biomarker model provides the largest overall gain; utility is threshold-specific and external temporal evaluation remains locked.
- Target/output: submission-grade, full-width (183 mm) manuscript figures.
- Backend: Python only (matplotlib/pandas/numpy).
- Primary export: editable SVG; secondary exports: PDF, 600-dpi TIFF, and 300-dpi PNG preview.
- Analysis boundary: NHANES 2005–2018 only. NHANES 2021–2023 remains locked and is not evaluated.
- Statistics: survey-weighted point estimates; Rao–Wu rescaled bootstrap 95% confidence intervals where available; random-effects REML meta-analysis with modified Hartung–Knapp confidence intervals; decision-curve net benefit with survey-bootstrap intervals.
- Source data: a figure-specific CSV is written for every quantitative or schematic panel.
- Image integrity: no photographic or microscopy content; no local image adjustment, cropping, or compositing.

## Figure 1 — Locked validation design

- Core conclusion: All preprocessing, imputation, and tuning are confined to training cycles before held-out-cycle validation, and Stage 3B is untouched.
- Archetype: schematic-led workflow.
- Panel map: one left-to-right workflow covering seven development cycles, train-only model development, five locked predictor sets, post-validation inference, and the locked 2021–2023 gate.
- Hero evidence: leave-one-cycle-out validation across seven NHANES cycles.
- Reviewer risk: leakage or premature access to the temporal test set.

## Figure 2 — Pooled incremental value

- Core conclusion: Renal markers improve all four primary metrics, and the combined biomarker model produces the largest pooled improvement.
- Archetype: quantitative 2 × 2 forest grid.
- Panel map: (a) ΔAUROC; (b) ΔPR-AUC; (c) Brier-score improvement; (d) log-loss improvement.
- Hero evidence: paired Elastic Net estimates on identical samples and weights within each comparison.
- Reviewer risk: metric direction and cross-sample comparison. All panels are labelled so positive values indicate improvement, and comparisons are interpreted within their paired sample only.

## Figure 3 — Across-cycle transportability

- Core conclusion: Renal and combined gains recur across held-out cycles, with random-effects summaries supporting transportability.
- Archetype: clinical 2 × 2 effect plot.
- Panel map: renal and combined ΔAUROC and ΔPR-AUC across the seven cycles, each with the random-effects estimate and confidence band.
- Hero evidence: cycle-specific survey-bootstrap estimates.
- Validation evidence: REML modified Hartung–Knapp pooled effect.
- Reviewer risk: overinterpreting heterogeneity or a chronological trend. Lines are not used to imply time-series causality.

## Figure 4 — Decision-curve analysis

- Core conclusion: The clinical utility of renal and combined extensions is threshold-specific rather than universal.
- Archetype: 2 × 2 clinical decision-curve grid.
- Panel map: (a,b) pooled net-benefit curves for renal and combined comparisons; (c,d) extended-minus-core net benefit with 95% confidence bands.
- Hero evidence: weighted net benefit over thresholds 0.01–0.30.
- Reviewer risk: treating exploratory DCA as proof of clinical benefit. Captions explicitly limit the claim.

## Supplementary Figure S1 — Locked sensitivity refits

- Core conclusion: The direction of renal and combined discrimination gains is examined across UACR handling strategies and class-imbalance refits without SMOTE.
- Archetype: quantitative sensitivity grid.
- Panel map: cycle-specific ΔAUROC and ΔPR-AUC across UACR methods, plus main-versus-class-weighted refits.
- Controls/robustness: complete-case, nested multiple imputation, stabilized IPW, main fold-median handling, observed class distribution, and class-weighted refit.
- Reviewer risk: sensitivity estimates do not carry survey-bootstrap intervals in their source table; the figure therefore shows the seven cycle estimates and medians without inferential error bars.
