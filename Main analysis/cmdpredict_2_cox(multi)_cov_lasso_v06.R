# PROJECT : PREDICTION OF CARDIOMETABOLIC MULTIMORBIDITY
# DATE    : 20 APRIL 2025
# NOTES   : STEP 2: Running Bayesian LASSO Cox Model within CV (Corrected)

# Start with clean environtment
rm(list = ls())

#################################################
## Load packages
#################################################
library(dplyr)
library(rstan)
library(parallel)

#################################################
## Setup & Load Data
#################################################
# --- Set working directory ---
setwd("/Users/kanyaanindya/Documents/10. SCAPIS/")

# --- Define input/output files ---
intermediate_data <- "08. Temp/03. Paper 3"
dir.create(intermediate_data, showWarnings = FALSE, recursive = TRUE)

input_data <- file.path(intermediate_data, "cox(multi)_cov_df_v06.rds")
input_fold <- file.path(intermediate_data, "cox(multi)_cov_fold_v06.rds")
output_result <- file.path(intermediate_data, "cox(multi)_cov_lasso_cvresult_v06.rds")
stan_model <- "03. Do file/03. Paper 3/02. R/01. Stan/cmdpredict_single_cox_completed_lasso_v02.stan"

# --- Load preprocessed data and folds ---
df_selected <- readRDS(input_data)
fold_indices <- readRDS(input_fold) # Assuming a list of data.frames, each a fold

# --- Define objects ---
outcome_col <- 'cmd_multi'
time_col <- 'time_cmd_multi'
id_col <- 'subject'
# Using your latest predictor_col
predictor_col <- c( 'age1','sex1_2','fh_cmd_2','education2_2','expense1_2','born1_2','location1_2',
                    'site1_2','site1_3','site1_4','site1_5','site1_6','smoking1_2','alcohol2_2',
                    'natrium','pa_riskvpa','mental_stress2_3','sleep_dur1_2','sbp_mean','dbp_mean',
                    'waist','hdl','nonhdl','drug_hypertension_2','drug_lipid_2')
# Using your latest integer_col to define cont_vars
integer_col <- c('age1','sbp_mean','dbp_mean','waist','hdl','nonhdl','natrium')
cont_vars <- intersect(integer_col, predictor_col)

# *** IMPORTANT: Ensure your df_selected includes the time variable ***
if (!(time_col %in% names(df_selected))) {
  stop(paste("Time-to-event variable '", time_col, "' not found. Add it to Step 1."))
}

#################################################
## K-Fold Cross-Validation Loop
#################################################
k <- length(fold_indices)
cv_results <- vector("list", k)
names(cv_results) <- paste0("Fold", 1:k)

print(paste("Starting", k, "-Fold Cross-Validation for Bayesian LASSO Cox Model..."))

for (i in 1:k) {
  cat("\n--- Processing Fold:", i, "---\n")
  df_test <- fold_indices[[i]]
  df_train <- bind_rows(fold_indices[-i])
  cat("Train set size:", nrow(df_train), " Test set size:", nrow(df_test), "\n")
  
  # --- Standardization (within the loop) ---
  train_means <- sapply(df_train[, cont_vars, drop=FALSE], mean, na.rm = TRUE)
  train_sds <- sapply(df_train[, cont_vars, drop=FALSE], sd, na.rm = TRUE)
  train_sds[is.na(train_sds) | train_sds == 0] <- 1
  
  df_train_std <- df_train
  df_test_std  <- df_test
  for(col_n in cont_vars){
    if(col_n %in% names(df_train_std)) df_train_std[[col_n]] <- (df_train[[col_n]] - train_means[col_n]) / train_sds[col_n]
    if(col_n %in% names(df_test_std))  df_test_std[[col_n]]  <- (df_test[[col_n]]  - train_means[col_n]) / train_sds[col_n]
  }
  cat("Standardization applied for Fold", i, "training data.\n")
  
  # --- Prepare Data for Stan Cox Model ---
  df_train_std_sorted <- df_train_std %>% arrange(desc(.data[[time_col]]))
  
  EVENT_train <- as.integer(as.character(df_train_std_sorted[[outcome_col]]))
  TIME_train <- df_train_std_sorted[[time_col]]
  X_train <- as.matrix(df_train_std_sorted[, predictor_col])
  
  PRED_DESIGN_test <- as.matrix(df_test_std[, predictor_col])
  PRED_N_test <- nrow(PRED_DESIGN_test)
  tau_scale <- 0.50
  
  stan_data_fold <- list(
    N = nrow(X_train),
    D = ncol(X_train),
    y = TIME_train,
    event = EVENT_train,
    X = X_train,
    scale_global = tau_scale,
    PRED_N = PRED_N_test,
    PRED_DESIGN = PRED_DESIGN_test
  )
  
  # --- Run Stan Model ---
  cat("Running Stan Cox model for Fold", i, "...\n")
  fit <- stan(file = stan_model,
              data = stan_data_fold,
              chains = 4, iter = 2000, warmup = 1000,
              cores = 2, seed = 456 + i,
              control = list(adapt_delta = 0.85),
              refresh = 100)
  
  # --- Extract Results ---
  summary_output <- summary(fit)$summary
  max_rhat <- max(summary_output[, "Rhat"], na.rm = TRUE)
  min_neff <- min(summary_output[, "n_eff"], na.rm = TRUE)
  cat(paste("  Max R-hat:", round(max_rhat, 3), " Min ESS:", round(min_neff), "\n"))
  
  # --- Extract predictions (risk scores) ---
  pred_param <- "risk_score"
  pred_rows <- grep(paste0("^", pred_param, "\\["), rownames(summary_output))
  pred_median <- NA_real_
  if(length(pred_rows) > 0 && length(pred_rows) == PRED_N_test){
    pred_median <- summary_output[pred_rows, "50%"]
  } else {
    warning(paste("Risk score dimension mismatch in Fold", i))
  }
  
  fold_results_df <- data.frame(
    subject = df_test[[id_col]],
    fold = i,
    true_outcome = df_test[[outcome_col]],
    time_to_event = df_test[[time_col]],
    predicted_value = pred_median # This is a RISK SCORE
  )
  
  # --- Extract Coefficient Summaries & Calculate HRs ---
  beta_summary_df <- NULL
  beta_rows <- grep("^beta\\[", rownames(summary_output))
  if(length(beta_rows) > 0 && all(beta_rows <= nrow(summary_output))) { # <<< ADDED BOUNDARY CHECK
    beta_summary_df <- as.data.frame(summary_output[beta_rows, c("50%", "2.5%", "97.5%")])
    if(nrow(beta_summary_df) == length(predictor_col)){
      beta_summary_df$Predictor <- predictor_col
      beta_summary_df$HR_median <- exp(beta_summary_df$"50%")
      beta_summary_df$HR_LCI <- exp(beta_summary_df$"2.5%")
      beta_summary_df$HR_UCI <- exp(beta_summary_df$"97.5%")
      colnames(beta_summary_df)[1:3] <- c("beta_median", "beta_LCI", "beta_UCI")
      # Select and reorder final columns for the summary data frame
      beta_summary_df <- beta_summary_df[, c("Predictor", "beta_median", "beta_LCI", "beta_UCI", "HR_median", "HR_LCI", "HR_UCI")]
    } else {
      warning(paste("Beta coefficient mismatch in Fold", i))
      beta_summary_df <- NULL
    }
  } # <<< CLOSED THE if(length(beta_rows) > 0) BLOCK
  
  # --- Store results for the fold ---
  cv_results[[i]] <- list(
    max_rhat = max_rhat,
    min_neff = min_neff,
    predictions = fold_results_df,
    coefficients_HRs = beta_summary_df # Now contains Hazard Ratios
  )
  cat("Finished Fold", i, ". Results stored.\n")
  rm(fit, summary_output, df_train, df_test, df_train_std, df_test_std, df_train_std_sorted) # Clean up
  gc()
  
} # <<< CLOSED THE for (i in 1:k) LOOP

saveRDS(cv_results, file = output_result)
print(paste("Cross-validation results for Cox model saved to:", output_result))