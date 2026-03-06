# Main analysis

This folder contains the primary analysis pipeline for predicting cardiometabolic multimorbidity (CMD). Scripts are numbered **1 through 5** and must be run in order. Each step builds on the outputs of the previous one.

---

## Step-by-Step Guide

### Step 1 — Pre-processing
**Scripts:** `cmdpredict_1_logit(any)_cov_v06.R`, `cmdpredict_1_logit(multi)_cov_v06.R`, and MICE variants

**What it does:**
1. Loads the SCAPIS dataset (`cmdpredict_python_v05.dta`)
2. Defines the outcome variable (`cmd_any` or `cmd_multi`) and all predictor columns
3. Removes Stata value labels and converts to numeric types
4. Filters to complete cases (no imputation in `cov` scripts)
5. Creates **10-fold cross-validation** indices (stratified random split, `set.seed(251222)`)
6. Saves the processed dataset and fold indices as `.rds` files to `08. Temp/03. Paper 3/`

**Outputs:**
- `logit(any)_cov_df_v06.rds` — cleaned data
- `logit(any)_cov_fold_v06.rds` — fold assignments

---

### Step 2 — Model Fitting (10-fold CV)
**Scripts:** `cmdpredict_2_logit(any)_cov_lasso_v06.R`, `cmdpredict_2_logit(any)_cov_bart_v06.R`, and analogues for `(multi)` and `mice`

**What it does:**
1. Loads the processed data and fold indices from Step 1
2. Loops over all 10 folds:
   - Standardises continuous predictors using **training-fold means and SDs** (applied to both train and test)
   - Prepares the Stan data list (design matrix, outcome vector, scale parameter `tau = 0.50`)
   - Runs the Bayesian LASSO logistic regression Stan model (4 chains, 2000 iterations, 1000 warmup)
   - Assesses convergence (R-hat, effective sample size)
   - Extracts posterior median predictions for test-fold observations
   - Extracts coefficient posteriors and computes odds ratios (OR, 95% credible interval)
3. Stores all fold results in a list and saves to `.rds`

**Outputs:**
- `logit(any)_cov_lasso_cvresult_v06.rds` — list of 10 fold results (predictions + coefficients)

---

### Step 3 — Results Compilation
**Scripts:** `cmdpredict_3_logit(any)_cov_lasso_v06.R`, and analogues

**What it does:**
1. Loads the cross-validation results from Step 2
2. **Convergence diagnostics:** Summarises R-hat and effective sample size across all folds
3. **Discrimination:** Computes overall AUC with 95% CI (DeLong's method) from pooled predictions
4. **Calibration:**
   - Brier score (mean squared error of predicted probabilities)
   - Calibration-in-the-large (CITL) and calibration slope via logistic recalibration
   - Calibration plot (scatter + LOESS) and binned calibration plot with 95% CIs
5. **Classification metrics** at Youden's J optimal threshold: sensitivity, specificity, PPV, NPV, accuracy, confusion matrix
6. **Performance at fixed specificity** (90%) as an alternative threshold
7. **Odds ratio summary:** Median OR and range across folds for each predictor
8. Saves ROC and calibration plots to `07. Graph/03. Paper 3/`

**Outputs:**
- Console output with all metrics
- `logit(any)_cov_lasso_ROC_v06.png` — ROC curve
- `logit(any)_cov_lasso(1)_v06.png` — calibration scatter plot
- `logit(any)_cov_lasso(2)_v06.png` — binned calibration plot

---

### Step 4 — Full-Sample Fitting and Individual Predictions
**Scripts:** `cmdpredict_4_logit(any)_cov_lasso_v06.R`, `cmdpredict_4_logit(multi)_cov_lasso_v06.R`

**What it does:**
1. Loads the raw SCAPIS data, applies the same cleaning and complete-case filtering as Step 1
2. Standardises continuous variables using the **full-sample** means and SDs
3. Fits the Bayesian LASSO Stan model on the **entire dataset** (4 chains, 2000 iterations)
4. Extracts all posterior samples for `alpha` and `beta`
5. Calculates **individual-level predicted probabilities** for every subject by applying the posterior samples manually (matrix multiplication), producing posterior mean and 95% credible interval for each person
6. Saves the Stan fit object, processed data, standardisation parameters, and individual predictions

**Outputs:**
- `logit(any)_full_stanfit_v05.rds` — full Stan fit object
- `logit(any)_full_processed_data_v05.rds` — cleaned data with IDs
- `logit(any)_full_stan_inputs_v05.rds` — standardisation parameters
- `logit(any)_cov_pred_lasso_v05.rds` — individual predictions (mean + 95% CI)
- `logit(any)_cov_prob_lasso_v05.rds` — full posterior probability matrix

---

### Step 5 — Subgroup Analysis
**Scripts:** `cmdpredict_5_logit(any)_cov_lasso_v06.R`, `cmdpredict_5_logit(multi)_cov_lasso_v06.R`

**What it does:**
1. Loads individual predictions from Step 4
2. Defines **clinical and demographic subgroups** for each subject:
   - Age group: 50–59, 60–64
   - Sex: Female / Male
   - Smoking: NonSmoker / Smoker
   - Hypertension: HyperT (SBP ≥130 or DBP ≥85 or on antihypertensives) / NoHyperT
   - HDL category (sex-specific thresholds)
   - Waist circumference category (sex-specific thresholds)
   - Birthplace: Born Sweden / Born Elsewhere
3. For each unique subgroup combination:
   - Calculates mean predicted probability (point estimate)
   - Computes **risk ratio (RR)** relative to the low-risk reference group (female, 50–59, non-smoker, no hypertension, born Sweden, normal HDL, normal waist)
   - Derives **95% credible intervals** for both absolute probability and RR using all posterior samples
4. Extracts odds ratios from the full-sample Stan fit and saves as CSV
5. Identifies top/bottom 5 highest/lowest-risk subgroups by sex
6. Calculates quintile cut-off values for risk stratification

**Outputs:**
- `logit(any)_subgroup_pred_v05.rds` / `.csv` — subgroup predictions with CIs and RRs
- `logit(any)_or_v05.csv` — odds ratios for all predictors

---

## Script Variants

Each step has multiple script variants based on:

| Suffix | Meaning |
|--------|---------|
| `logit(any)` | Binary outcome: any CMD |
| `logit(multi)` | Ordinal outcome: number of CMDs |
| `cov` | Complete-case analysis |
| `mice` | Multiple imputation (MICE) |
| `lasso` | Bayesian LASSO prior (main model) |
| `bart` | Bayesian Additive Regression Trees |

The `lasso` + `cov` variant is the **main model**. Steps 4 and 5 only exist for this variant.
