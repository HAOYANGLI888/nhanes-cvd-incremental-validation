from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml


MODEL_ORDER = [
    "model0_core",
    "model1_renal",
    "model2_metabolic",
    "model3_inflammatory",
    "model4_combined",
    "model0_paired_metabolic",
    "model0_paired_combined",
]


def build_variable_dictionary(root: Path) -> pd.DataFrame:
    model_dict = pd.read_csv(root / "metadata/prespecified_model_dictionary.csv")
    formulas = pd.read_csv(root / "metadata/derived_variable_formulas.csv").set_index("analytic_name")
    rows = []
    name_map = {
        "education_or_pir": "education",
        "bmi_or_waist": "bmi",
        "systolic_blood_pressure": "mean_sbp",
        "diabetes_status": "diabetes",
    }
    for item in model_dict.itertuples(index=False):
        analytic = name_map.get(item.predictor, item.predictor)
        formula = item.formula
        handling = item.final_decision
        if analytic in formulas.index:
            formula = formulas.loc[analytic, "mathematical_expression"]
            handling = formulas.loc[analytic, "handling_rule"]
        rows.append({
            "role": "predictor",
            "analytic_name": analytic,
            "domain": item.domain,
            "raw_NHANES_variables": item.source_variables,
            "units": item.unit,
            "coding_or_formula": formula,
            "cycle_harmonization": item.availability_by_cycle,
            "sample_or_weight_requirement": "fasting subsample; WTSAF2YR" if analytic == "tyg_wc" else "2005-2018 MEC domain (WTMEC2YR); 2021-2023 phlebotomy domain (WTPH2YR); fasting weight for Models 2 and 4",
            "missing_handling": "training-fold median" if item.transformation not in ("categorical", "binary") else "training-fold explicit Missing category",
            "notes": handling,
        })
    outcome = pd.read_csv(root / "metadata/outcome_harmonization.csv")
    for item in outcome.itertuples(index=False):
        rows.append({
            "role": "outcome component",
            "analytic_name": item.component,
            "domain": "self-reported prevalent CVD",
            "raw_NHANES_variables": item.source_variables,
            "units": "binary",
            "coding_or_formula": item.coding_rule,
            "cycle_harmonization": item.availability_by_cycle,
            "sample_or_weight_requirement": "observed component response",
            "missing_handling": "special responses treated as missing",
            "notes": item.final_decision,
        })
    rows.append({
        "role": "composite outcome",
        "analytic_name": "cvd",
        "domain": "self-reported prevalent CVD",
        "raw_NHANES_variables": "MCQ160B; MCQ160C; MCQ160D; MCQ160E; MCQ160F",
        "units": "binary",
        "coding_or_formula": "1 if any observed component is positive; 0 if no observed component is positive and at least one component is observed; missing if all five are missing",
        "cycle_harmonization": "same five questions in every included cycle",
        "sample_or_weight_requirement": "adult nonpregnant analytic population",
        "missing_handling": "strict sensitivity required all five components observed for classification as a noncase",
        "notes": "Outcome components were excluded from every predictor set.",
    })
    for analytic, raw, note in [
        ("combined_mec_weight", "WTMEC2YR", "2005-2018 cycle-specific MEC weights divided by seven"),
        ("temporal_phlebotomy_weight", "WTPH2YR", "2021-2023 phlebotomy weight for Models 0, 1, and 3; no divisor"),
        ("combined_fasting_weight", "WTSAF2YR", "cycle-specific fasting-subsample weights divided by number of pooled cycles within period"),
        ("strata", "SDMVSTRA", "masked variance stratum; cycle included in the design key"),
        ("psu", "SDMVPSU", "masked variance PSU; cycle included in the design key"),
    ]:
        rows.append({
            "role": "survey design",
            "analytic_name": analytic,
            "domain": "survey design",
            "raw_NHANES_variables": raw,
            "units": "weight" if "weight" in analytic else "identifier",
            "coding_or_formula": note,
            "cycle_harmonization": (
                "2021-2023 only" if analytic == "temporal_phlebotomy_weight"
                else "2005-2018 development cycles" if analytic == "combined_mec_weight"
                else "available in every included cycle"
            ),
            "sample_or_weight_requirement": "MEC or fasting domain as specified",
            "missing_handling": "analysis weights <=1e-8 excluded, including SAS special-missing sentinels represented as tiny positive values; valid strata and PSU required",
            "notes": "Rao-Wu rescaled bootstrap respected cycle-specific strata and PSUs.",
        })
    return pd.DataFrame(rows)


def combined_manifest(root: Path) -> pd.DataFrame:
    main = pd.read_csv(root / "results/audit/final_model_freeze_manifest.csv")
    paired_path = root / "results/audit/paired_fasting_core_freeze_manifest.csv"
    paired = pd.read_csv(paired_path) if paired_path.exists() else pd.DataFrame()
    return pd.concat([main, paired], ignore_index=True, sort=False)


def hyperparameter_table(root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in manifest.itertuples(index=False):
        model_path = root / str(item.model_path)
        params = json.loads((model_path.parent / "hyperparameters.json").read_text(encoding="utf-8"))
        metadata = json.loads((model_path.parent / "training_metadata.json").read_text(encoding="utf-8"))
        rows.append({
            "model": item.model,
            "algorithm": item.algorithm,
            "training_domain": "fasting subsample" if "paired" in item.model or item.model in ("model2_metabolic", "model4_combined") else "MEC examination sample",
            "n_training": metadata.get("n_training"),
            "events_training": metadata.get("events_training"),
            "weight_variable": metadata.get("weight_variable"),
            "selected_candidate_id": metadata.get("selected_candidate_id"),
            "mean_inner_weighted_log_loss": metadata.get("mean_inner_weighted_log_loss"),
            "selected_hyperparameters": json.dumps(params, sort_keys=True),
            "model_sha256": item.model_sha256,
        })
    out = pd.DataFrame(rows)
    out["model_order"] = out["model"].map({v: i for i, v in enumerate(MODEL_ORDER)})
    return out.sort_values(["algorithm", "model_order"]).drop(columns="model_order").reset_index(drop=True)


def elastic_net_coefficients(root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subset = manifest[manifest["algorithm"].eq("elastic_net")].copy()
    subset["model_order"] = subset["model"].map({v: i for i, v in enumerate(MODEL_ORDER)})
    for item in subset.sort_values("model_order").itertuples(index=False):
        model_path = root / str(item.model_path)
        pipe = joblib.load(model_path)
        preprocess = pipe.named_steps["preprocess"]
        estimator = pipe.named_steps["model"]
        names = list(preprocess.get_feature_names_out())
        coefficients = np.asarray(estimator.coef_).reshape(-1)
        if len(names) != len(coefficients):
            raise RuntimeError(f"Coefficient-name mismatch for {item.model}")
        rows.append({
            "model": item.model,
            "term": "intercept",
            "source_predictor": "intercept",
            "coefficient": float(np.asarray(estimator.intercept_).reshape(-1)[0]),
            "coefficient_scale": "model log-odds intercept after preprocessing",
            "exp(coefficient)": float(np.exp(np.asarray(estimator.intercept_).reshape(-1)[0])),
        })
        for name, coef in zip(names, coefficients):
            clean = name.split("__", 1)[-1]
            source = clean
            scale = (
                "one-hot indicator with all levels retained (drop=None); penalized joint-model "
                "coefficient, not an unadjusted reference-category contrast"
            )
            if name.startswith("continuous__"):
                source = clean
                scale = "one training-sample standard deviation after median imputation"
            else:
                categorical = ["sex", "race_ethnicity", "education", "smoking", "diabetes"]
                source = next((c for c in categorical if clean == c or clean.startswith(c + "_")), clean)
            rows.append({
                "model": item.model,
                "term": clean,
                "source_predictor": source,
                "coefficient": float(coef),
                "coefficient_scale": scale,
                "exp(coefficient)": float(np.exp(coef)),
            })
    return pd.DataFrame(rows)


def run(root: Path) -> int:
    root = root.resolve()
    outdir = root / "results/manuscript_tables/post_stage3a"
    outdir.mkdir(parents=True, exist_ok=True)
    dictionary = build_variable_dictionary(root)
    manifest = combined_manifest(root)
    hyperparameters = hyperparameter_table(root, manifest)
    coefficients = elastic_net_coefficients(root, manifest)
    dictionary.to_csv(outdir / "Supplementary_Table_S9_variable_dictionary.csv", index=False)
    hyperparameters.to_csv(outdir / "Supplementary_Table_S10_final_hyperparameters.csv", index=False)
    coefficients.to_csv(outdir / "Supplementary_Table_S11_elastic_net_coefficients.csv", index=False)
    gate = pd.DataFrame([{
        "module": "reviewer_supplement_tables",
        "status": "PASS" if len(dictionary) >= 20 and len(hyperparameters) == 14 and len(coefficients) > 70 else "FAIL",
        "variable_dictionary_rows": len(dictionary),
        "frozen_hyperparameter_rows": len(hyperparameters),
        "elastic_net_coefficient_rows": len(coefficients),
        "paired_fasting_models_included": int(hyperparameters["model"].str.contains("paired").sum()),
    }])
    gate.to_csv(root / "results/audit/reviewer_supplement_tables_gate.csv", index=False)
    return 0 if gate.iloc[0]["status"] == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the variable dictionary and frozen-model reviewer tables.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    return run(Path(args.project_root))


if __name__ == "__main__":
    raise SystemExit(main())
