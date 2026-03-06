# cmdpredict_bayesian

Bayesian prediction models for cardiometabolic multimorbidity (CMD) using data from the SCAPIS cohort. The project fits Bayesian logistic regression with a LASSO prior (implemented in Stan) and evaluates prediction performance via 10-fold cross-validation.

---

## Project Overview

**Outcomes modelled:**
- `cmd_any` — Binary: any cardiometabolic disease
- `cmd_multi` — Ordinal/multinomial: number of cardiometabolic diseases

**Analysis variants (reflected in file names):**

| Tag | Meaning |
|-----|---------|
| `logit(any)` | Binary logistic regression for any CMD |
| `logit(multi)` | Ordinal/multinomial logistic regression for CMD count |
| `cov` | Complete-case analysis (covariates only, no imputation) |
| `mice` | Multiple imputation (MICE) |
| `lasso` | Bayesian LASSO prior on regression coefficients |
| `bart` | Bayesian Additive Regression Trees |

---

## Analysis Pipeline (Steps 1–5)

The core workflow runs sequentially across 5 steps. Each step is a numbered R script. Run them in order.

| Step | Script prefix | What it does |
|------|--------------|-------------|
| **1** | `cmdpredict_1_*` | Pre-processing: load SCAPIS data, define predictors, filter complete cases, create 10-fold CV indices, save processed data |
| **2** | `cmdpredict_2_*` | Model fitting: run Bayesian Stan model in 10-fold CV loop, extract fold-level predictions and posterior coefficient summaries |
| **3** | `cmdpredict_3_*` | Results compilation: AUC, Brier score, calibration slope/intercept, Youden's J threshold, ROC and calibration plots |
| **4** | `cmdpredict_4_*` | Full-sample fitting: refit the best model identified in Step #3 on the entire dataset, extract individual-level predicted probabilities with 95% credible intervals |
| **5** | `cmdpredict_5_*` | Subgroup analysis: stratify predictions by age, sex, smoking, hypertension, HDL, waist, and birthplace; compute group-level risk ratios |

---

## Folder Structure

```
cmdpredict_bayesian/
├── Main analysis/     # Steps 1–5 for all model variants (primary results)
├── Stepwise/          # Nested predictor models (m1–m8) for model building
├── Interaction/       # Interaction term models for 5 variable pairs
└── Stan/              # Stan model definition (.stan) and compiled object (.rds)
```

See the README in each subfolder for details.

---

## File Naming Convention

```
cmdpredict_{step}_{outcome}_{imputation}_{method}_v{version}.R
```

Examples:
- `cmdpredict_2_logit(any)_cov_lasso_v06.R` — Step 2, binary outcome, complete-case, LASSO
- `cmdpredict_3_logit(multi)_mice_lasso_v06.R` — Step 3, ordinal outcome, imputed, LASSO

---

## Key Predictors

| Domain | Variables |
|--------|-----------|
| Demographic | `age1`, `sex1_2`, `born1_2`, `location1_2`, `site1_*` |
| Socioeconomic | `education2_2`, `expense1_2`, `fh_cmd_2` |
| Lifestyle | `smoking1_2`, `alcohol2_2`, `natrium`, `fibrer`, `pa_riskvpa`, `mental_stress2_3`, `sleep_dur1_2` |
| Clinical | `sbp_mean`, `dbp_mean`, `waist`, `hdl`, `nonhdl`, `drug_hypertension_2`, `drug_lipid_2` |

---

## Dependencies

- R packages: `haven`, `dplyr`, `caret`, `rstan`, `pROC`, `ggplot2`, `sjmisc`, `tidyr`, `tibble`
- Stan: requires RStan and a C++ toolchain
- Input data: `05. Result/cmdpredict_python_v05.dta` (relative to SCAPIS root)
- Intermediate outputs: saved to `08. Temp/03. Paper 3/`
- Final outputs: `05. Result/`, `04. Excel/03. Paper 3/`, `07. Graph/03. Paper 3/`
