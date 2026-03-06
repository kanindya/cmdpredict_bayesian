# Interaction

This folder contains scripts that test **pairwise interaction terms** in the Bayesian LASSO logistic model. Each interaction pair is evaluated separately by adding the product term to the full predictor set and re-running the 10-fold cross-validation (Step 2).

All scripts in this folder correspond to **Step 2** of the main pipeline and use the same pre-processed data and fold indices from Step 1 in `Main analysis/`.

---

## Interaction Pairs

| Pair | Variables | Types |
|------|-----------|-------|
| **pair1** | Age × Sex | continuous × categorical |
| **pair2** | Foreign-born × Low-education | categorical × categorical |
| **pair3** | Sex × HDL | categorical × continuous |
| **pair4** | Smoking × non-HDL-C | categorical × continuous |
| **pair5** | Waist circumference × non-HDL-C | continuous × continuous |

---

## How to Use

Each script (`cmdpredict_2_*_int_pair{N}.R`) is a self-contained **Step 2 script** that:
1. Loads the pre-processed data and fold indices from `08. Temp/03. Paper 3/`
2. Constructs the interaction term:
   - Continuous × continuous: simple product (`var1 * var2`)
   - Categorical × continuous: product (`var1 * var2`)
   - Categorical × categorical: dummy-coded product
3. Appends the interaction term to the full predictor set
4. Standardises continuous variables (including the interaction term where applicable) within each fold
5. Runs the Bayesian LASSO Stan model with 10-fold CV
6. Saves results to `logit(any)_cov_lasso_int_pair{N}_cvresult_v06.rds`

To change which pair is active, edit the **CONFIGURATION BLOCK** at the top of the script and set `interaction_name`, `var1`, `var2`, `var1_type`, `var2_type`.

---

## File Naming

```
cmdpredict_{step}_{outcome}_{imputation}_{method}_v{version}_int_pair{N}.R
```

Scripts exist for:
- Steps 2 and 3 (`cmdpredict_2_*` and `cmdpredict_3_*`)
- Both `logit(any)` and `logit(multi)` outcomes

---

## Purpose

Test whether specific pairwise interactions improve model fit beyond the main-effects model. Compare AUC and calibration of interaction models against the full model from `Main analysis/` to assess clinical relevance of each interaction.
