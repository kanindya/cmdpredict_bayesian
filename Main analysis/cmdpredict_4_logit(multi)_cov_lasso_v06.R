# PROJECT : PREDICTION OF ANY CARDIOMETABOLIC MULTIMORBIDITY
# DATE    : 27 APRIL 2025
# NOTES   : STEP 4: Predictions using full sample

# Start with clean environment
rm(list = ls())

#################################################
## Load packages
#################################################

library(haven)
library(dplyr)
library(sjmisc)
library(rstan)
library(parallel)

# Set rstan options
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

#################################################
## Setup
#################################################

# --- Set working directory ---
  setwd("/Users/kanyaanindya/Documents/10. SCAPIS/") # Adjust if needed

# --- Define Stan Model File ---
  stan_model <- "03. Do file/03. Paper 3/02. R/01. Stan/cmdpredict_single_logit_completed_lasso_v02.stan"

# --- Define output file names ---
  inter_data <- "08. Temp/03. Paper 3" # intermediate_data -> inter_data
  dir.create(inter_data, showWarnings = FALSE, recursive = TRUE)

# Output files from fitting
  f_stanfit <- file.path(inter_data, "logit(multi)_full_stanfit_v05.rds")
  f_proc_data <- file.path(inter_data, "logit(multi)_full_processed_data_v05.rds") 
  f_inputs <- file.path(inter_data, "logit(multi)_full_stan_inputs_v05.rds") 

# Final prediction output files
  f_pred_rds <- file.path(inter_data, "logit(multi)_cov_pred_lasso_v05.rds") 
  f_prob_rds <- file.path(inter_data, "logit(multi)_cov_prob_lasso_v05.rds") 

#################################################
## Data loading and initial preparation
#################################################

# --- Load Data ---
  df <- read_dta("05. Result/cmdpredict_python_v05.dta")

# --- Define Variables ---
  out_col <- 'cmd_multi' # outcome_col -> out_col
  id_col <- 'subject'

# Predictors used in the model
  pred_col <- c(     'time_days',
                      'age1',
                      'sex1_2',
                      'fh_cmd_2',
                      'education2_2',
                      'expense1_2',
                      'born1_2',
                      'location1_2',
                      'site1_2',
                      'site1_3',
                      'site1_4',
                      'site1_5',
                      'site1_6',
                      'smoking1_2',
                      'alcohol2_2',
                      'natrium',
                      'fibrer',
                      'pa_riskvpa',
                      'mental_stress2_3',
                      'sleep_dur1_2',
                      'sbp_mean',
                      'dbp_mean',
                      'waist',
                      'hdl',
                      'nonhdl',
                      'drug_hypertension_2',
                      'drug_lipid_2')

# Continuous variables requiring standardization
  cont_vars <- c('time_days',
                  'age1',
                 'sbp_mean',
                 'dbp_mean',
                 'waist',
                 'hdl',
                 'nonhdl',
                 'natrium',
                 'fibrer')

# Select necessary columns
  df_selected <- df[, c(id_col, out_col, pred_col)] # 

# --- Clean Data Types & Handle Missing ---
  df_selected <- df_selected %>%
    mutate(across(everything(), ~ zap_labels(.))) %>%
    mutate(across(all_of(c(out_col, pred_col)), as.numeric)) %>%
    filter(complete.cases(.[, c(out_col, pred_col)]))

# --- Prepare Outcome Variable (convert to 0/1 integer) ---
  df_selected[[out_col]] <- as.integer(round(df_selected[[out_col]]))

# Verify outcome is binary 0/1
  out_vals <- sort(unique(df_selected[[out_col]])) # outcome_values -> out_vals
  print(paste("Outcome variable:", out_col))
  print("Summary of outcome variable:")
  print(table(df_selected[[out_col]], useNA = "ifany"))
  print(prop.table(table(df_selected[[out_col]])))

#################################################
## Standardization
#################################################

# Calculate overall means and SDs
  all_means <- sapply(df_selected[, cont_vars, drop=FALSE], mean, na.rm = TRUE)
  all_sds <- sapply(df_selected[, cont_vars, drop=FALSE], sd, na.rm = TRUE)
  all_sds[all_sds == 0 | is.na(all_sds)] <- 1

# Function to standardize
  std_val <- function(value, var_name, means, sds) { (value - means[var_name]) / sds[var_name] }

# Apply standardization
  df_std <- df_selected %>%
    mutate(across(all_of(cont_vars), ~ std_val(.x, cur_column(), all_means, all_sds)))

#################################################
## Prepare Data for Stan
#################################################

# Design Matrix (standardized)
  design_full <- as.matrix(df_std[, pred_col]) 

# Outcome Vector (integer 0/1)
  outcome_full <- df_std[[out_col]] 

# Create Stan Data List
  stan_data <- list(
    N = nrow(design_full),
    D = ncol(design_full),
    DESIGN = design_full,
    OUTCOME = outcome_full,
    PRED_N = nrow(design_full), # Use training data for prediction within Stan
    PRED_DESIGN = design_full,
    scale_global = 0.50
  )

# Check dimensions
  print(paste("N:", stan_data$N, " D:", stan_data$D))
  if(length(outcome_full) != stan_data$N) stop("Outcome length mismatch")
  if(ncol(design_full) != length(pred_col)) stop("Design matrix columns mismatch")

#################################################
## Run Stan Model on Full Dataset
#################################################

print(paste("Running Stan model:", basename(stan_model), "..."))

# --- RUN THE STAN MODEL ---
  fit_full <- stan(
    file = stan_model,
    data = stan_data,
    chains = 4,
    iter = 2000,
    warmup = 1000,
    cores = 2, 
    seed = 12345
    # control = list(adapt_delta = 0.85) # Add back if needed
  )

#################################################
## Save Stan Fit & Inputs
#################################################

print("Saving Stan fit object and inputs...")

saveRDS(fit_full, file = f_stanfit)
saveRDS(df_selected, file = f_proc_data) # Save original processed data with IDs
saveRDS(list(all_means = all_means,
             all_sds = all_sds,
             predictor_col = pred_col,
             cont_vars = cont_vars,
             id_col = id_col,
             outcome_col = out_col), 
        file = f_inputs)

print("Files saved.")

#################################################
## Extract Posterior Samples
#################################################

post_samples <- rstan::extract(fit_full, pars = c("alpha", "beta")) 
alpha_full <- post_samples$alpha
beta_full <- post_samples$beta
n_samples <- length(alpha_full)
n_predictors <- ncol(beta_full)
print(paste("Extracted", n_samples, "samples."))

#################################################
## Prepare Data for Manual Prediction 
#################################################

print("Preparing data for manual prediction...")
# --- Select Predictor Columns from original df_select ---
df_pred_ind <- df_selected %>% select(all_of(pred_col)) 

# --- Standardize Continuous Variables ---
df_std_ind <- df_pred_ind %>% mutate(across(all_of(cont_vars), ~ std_val(.x, cur_column(), all_means, all_sds))) # df_std_individual -> df_std_ind

# --- Convert to Matrix ---
x_ind <- as.matrix(df_std_ind) # x_individual -> x_ind
n_ind <- nrow(x_ind) # n_individuals -> n_ind

#################################################
## Calculate Predicted Probabilities Manually
#################################################

print(paste("Calculating manual predictions for", n_ind, "individuals..."))

alpha_matrix <- matrix(alpha_full, nrow = n_ind, ncol = n_samples, byrow = TRUE)
xbeta_t <- x_ind %*% t(beta_full) 
eta_ind <- alpha_matrix + xbeta_t 
prob_ind <- plogis(eta_ind)

#################################################
## Summarize Individual Probabilities
#################################################
print("Summarizing predictions...")

pred_matrix <- apply(prob_ind, 1, function(p) c(mean_pred = mean(p, na.rm=T), quantile(p, .025, na.rm=T), quantile(p, .975, na.rm=T)))
df_pred <- t(pred_matrix) %>% as_tibble() %>% rename(lower_ci = `2.5%`, upper_ci = `97.5%`) # df_pred_matrix -> df_pred

# --- Combine with original data # --- 
print("Combining predictions with IDs...")

df_results <- bind_cols(df_selected %>%
                          select(all_of(id_col)), df_pred) %>%
              left_join(df_selected %>%
                          select(all_of(c(id_col, out_col, pred_col))), by = id_col)

print("Finished calculating individual predictions.")

# Set option to disable scientific notation for printing
options(scipen = 999)
print("First few results:")
print(head(df_results))

#################################################
## Save compiled results
#################################################
saveRDS(df_results, file = f_pred_rds)
print(paste("Cross-validation results saved to:", f_pred_rds))

saveRDS(prob_ind, file = f_prob_rds)
print(paste("Cross-validation results saved to:", f_prob_rds))
