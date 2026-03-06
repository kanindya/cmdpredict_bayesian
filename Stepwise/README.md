# Stepwise

This folder contains **nested predictor models (m1–m8)** used for stepwise model comparison. Each model adds a new domain of predictors on top of the previous one, allowing you to assess the incremental predictive value of each predictor domain.

All scripts in this folder correspond to **Step 2** of the main pipeline (model fitting via 10-fold cross-validation with the Bayesian LASSO Stan model). They use the same pre-processed data and fold indices created by Step 1 in `Main analysis/`.

---

## Nested Models (m1–m8)

| Model | Predictor domain added | Predictors included |
|-------|----------------------|---------------------|
| **m1** | Age only | `time_days`, `age1` |
| **m2** | + Sex | m1 + `sex1_2` |
| **m3** | + Sociodemographics | m2 + `fh_cmd_2`, `education2_2`, `expense1_2`, `born1_2`, `location1_2`, `site1_2–6` |
| **m4** | + Lifestyle | m3 + `smoking1_2`, `alcohol2_2`, `natrium`, `fibrer`, `pa_riskvpa`, `mental_stress2_3`, `sleep_dur1_2` |
| **m5** | + Clinical (full model) | m4 + `sbp_mean`, `dbp_mean`, `waist`, `hdl`, `nonhdl`, `drug_hypertension_2`, `drug_lipid_2` |
| **m6** | Time + sex only | `time_days`, `sex1_2` |
| **m7** | Time + lifestyle only | `time_days`, `smoking1_2`, `alcohol2_2`, `natrium`, `fibrer`, `pa_riskvpa`, `mental_stress2_3`, `sleep_dur1_2` |
| **m8** | Time + clinical only | `time_days`, `sbp_mean`, `dbp_mean`, `waist`, `hdl`, `nonhdl`, `drug_hypertension_2`, `drug_lipid_2` |

> **Note:** m1–m5 are cumulative (each builds on the previous). m6–m8 are standalone domain-specific models for direct comparison.

---

## How to Use

1. Ensure Step 1 has been run from `Main analysis/` to generate the pre-processed data and fold indices.
2. Run any `cmdpredict_2_*_m{N}_lasso_v06.R` script directly. Each script:
   - Loads `logit(any)_cov_df_v06.rds` and `logit(any)_cov_fold_v06.rds` from `08. Temp/03. Paper 3/`
   - Fits the Bayesian LASSO logistic model with the specified predictor set using 10-fold CV
   - Saves results to `logit(any)_m{N}_lasso_cvresult_v06.rds`
3. To compile model-comparison results, run the corresponding Step 3 script from `Main analysis/` pointing to the relevant `.rds` output.

---

## File Naming

```
cmdpredict_2_{outcome}_{m{N}}_lasso_v{version}.R
```

Scripts exist for both `logit(any)` (binary) and `logit(multi)` (ordinal) outcomes.

---

## Purpose

Compare AUC, Brier score, and calibration across models m1–m5 to quantify how much each predictor domain improves prediction beyond demographics alone. Models m6–m8 isolate individual domains for independent evaluation.
