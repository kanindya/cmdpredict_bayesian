# Stan

This folder contains the Stan model definition used across all analyses in this project.

---

## Files

| File | Description |
|------|-------------|
| `cmdpredict_single_logit_completed_lasso_v02.stan` | Stan model: Bayesian logistic regression with LASSO prior |
| `cmdpredict_single_logit_completed_lasso_v02.rds` | Pre-compiled Stan model object (speeds up repeated calls) |

---

## Model Description

The Stan model implements **single-level Bayesian logistic regression** on complete-case data with a regularising LASSO (double-exponential / Laplace) prior on the regression coefficients.

### Data block
| Variable | Description |
|----------|-------------|
| `N` | Number of complete-case training observations |
| `D` | Number of predictors |
| `DESIGN` | N × D design matrix (standardised) |
| `OUTCOME` | Binary outcome vector (0/1) |
| `PRED_N` | Number of observations to predict |
| `PRED_DESIGN` | PRED_N × D design matrix for prediction |
| `scale_global` | Scale parameter for the half-Cauchy hyperprior on `tau` (set to 0.50) |

### Parameters
| Parameter | Prior | Description |
|-----------|-------|-------------|
| `alpha` | Cauchy(0, 2.5) | Intercept |
| `beta[D]` | Double-exponential(0, tau) | Regression coefficients (LASSO shrinkage) |
| `tau` | half-Cauchy(0, scale_global) | Global shrinkage scale for the LASSO prior |

### Model
- Likelihood: `OUTCOME ~ Bernoulli_logit(alpha + DESIGN * beta)`
- The double-exponential (Laplace) prior on `beta` with shared scale `tau` implements Bayesian LASSO, shrinking small coefficients toward zero.

### Generated quantities
- `pred[PRED_N]` — posterior predicted probability for each observation in `PRED_DESIGN` (converted from log-odds via inverse-logit)

---

## Usage

The `.stan` file is called from Step 2 and Step 4 scripts via:

```r
stan(file = "03. Do file/03. Paper 3/02. R/01. Stan/cmdpredict_single_logit_completed_lasso_v02.stan",
     data = stan_data_list,
     chains = 4, iter = 2000, warmup = 1000, cores = 2,
     control = list(adapt_delta = 0.85))
```

Set `rstan_options(auto_write = TRUE)` to use the pre-compiled `.rds` and avoid recompilation on every run.
