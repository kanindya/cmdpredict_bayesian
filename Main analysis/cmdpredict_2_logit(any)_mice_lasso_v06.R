# PROJECT : PREDICTION OF CARDIOMETABOLIC MULTIMORBIDITY
# DATE    : 20 APRIL 2025
# NOTES   : STEP 2: MICE within CV - Memory Optimized for Predictions Only

# Start with clean environment
rm(list = ls())

#################################################
## Load packages
#################################################
library(dplyr)
library(rstan)
library(parallel)
library(mice)
# broom.mixed is NOT needed if we don't pool full stanfit objects for coefficients

#################################################
## Setup & Load Data
#################################################

# --- Set working directory ---
setwd("/Users/kanyaanindya/Documents/10. SCAPIS/") #

# --- Define input/output files ---
intermediate_data <- "08. Temp/03. Paper 3"
dir.create(intermediate_data, showWarnings = FALSE, recursive = TRUE)

input_data <- file.path(intermediate_data, "logit(any)_mice_df_v06.rds") 
input_fold <- file.path(intermediate_data, "logit(any)_mice_fold_v06.rds") 
output_result <- file.path(intermediate_data, "logit(any)_mice_lasso_cvresult_v06.rds") 
stan_model <- "03. Do file/03. Paper 3/02. R/01. Stan/cmdpredict_single_logit_completed_lasso_v02.stan"

# --- Load preprocessed data and folds (with NAs) ---
df_selected <- readRDS(input_data)
fold_indices <- readRDS(input_fold) # List of data.frames, each is a fold

# --- Define objects ---
outcome_col <- 'cmd_any'
time_col <- 'time_days'
id_col <- 'subject'
category_col <- c('sex1_2',
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
                  'pa_riskvpa',
                  'mental_stress2_3',
                  'sleep_dur1_2',
                  'drug_hypertension_2',
                  'drug_lipid_2'
)

integer_col <- c('age1',
                 'sbp_mean',
                 'dbp_mean',
                 'waist',
                 'hdl',
                 'nonhdl',
                 'natrium',
                 'fibrer')

predictor_col <- c( 'time_days',
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

all_col <- c(outcome_col, predictor_col) 
cont_vars <- intersect(integer_col, predictor_col)
m <- 10 # Number of imputations (reduced from 20 for potentially faster run, adjust as needed)

#################################################
## K-Fold Cross-Validation Loop with MICE
#################################################
k <- length(fold_indices)
cv_results <- vector("list", k)
names(cv_results) <- paste0("Fold", 1:k)

print(paste("Starting", k, "-Fold CV with MICE (m=", m, ") for predictions..."))

for (i in 1:k) {
  cat("\n--- Processing Fold:", i, "---\n")
  df_test <- fold_indices[[i]]
  df_train <- bind_rows(fold_indices[-i])
  cat("Train set size:", nrow(df_train), " Test set size:", nrow(df_test), "\n")
  
  cat("Running MICE (m=", m, ") on training data for Fold", i, "...\n")
  mice_train_data <- df_train[, all_col]
  mice_train_obj <- mice(mice_train_data, m = m, printFlag = FALSE, seed = 123 + i, maxit = 5) # maxit=5 is for speed
  
  mice_test_data <- df_test[, all_col]
  mice_test_obj <- mice.mids(mice_train_obj, newdata = mice_test_data, printFlag = FALSE)
  
  all_preds_fold <- matrix(NA, nrow = nrow(df_test), ncol = m) # Store predictions from each imputed dataset
  colnames(all_preds_fold) <- paste0("imp_", 1:m)
  rhats_imputations_fold <- numeric(m) # Store max Rhat for each imputation's Stan model
  neffs_imputations_fold <- numeric(m) # Store min ESS for each imputation's Stan model
  
  cat("Looping through", m, "imputations for Fold", i, "...\n")
  for (j in 1:m) {
    df_train_j <- complete(mice_train_obj, j)
    df_test_j <- complete(mice_test_obj, j)
    
    # Standardization
    train_means_j <- sapply(df_train_j[, cont_vars, drop=FALSE], mean, na.rm = TRUE)
    train_sds_j <- sapply(df_train_j[, cont_vars, drop=FALSE], sd, na.rm = TRUE)
    train_sds_j[is.na(train_sds_j) | train_sds_j == 0] <- 1
    
    df_train_j_std <- df_train_j
    df_test_j_std <- df_test_j
    for(col_n in cont_vars){
      if(col_n %in% names(df_train_j_std)) df_train_j_std[[col_n]] <- (df_train_j[[col_n]] - train_means_j[col_n]) / train_sds_j[col_n]
      if(col_n %in% names(df_test_j_std))  df_test_j_std[[col_n]]  <- (df_test_j[[col_n]]  - train_means_j[col_n]) / train_sds_j[col_n]
    }
    
    # Prepare Data for Stan
    OUTCOME_train_j <- as.integer(df_train_j_std[[outcome_col]])
    DESIGN_train_j <- as.matrix(df_train_j_std[, predictor_col])
    PRED_DESIGN_test_j <- as.matrix(df_test_j_std[, predictor_col])
    PRED_N_test_j <- nrow(PRED_DESIGN_test_j)
    tau_scale <- 0.50
    
    stan_data_fold_j <- list(
      N = nrow(DESIGN_train_j), D = ncol(DESIGN_train_j), DESIGN = DESIGN_train_j,
      OUTCOME = OUTCOME_train_j, PRED_N = PRED_N_test_j, PRED_DESIGN = PRED_DESIGN_test_j,
      scale_global = tau_scale
    )
    
    # Run Stan Model for imputation j
    fit_j_current <- NULL # Initialize
    summary_output_j <- NULL
    
    tryCatch({
      fit_j_current <- stan(file = stan_model, data = stan_data_fold_j,
                            chains = 4, iter = 2000, warmup = 1000, cores = 2,
                            seed = 456 + i * 100 + j, control = list(adapt_delta = 0.85),
                            refresh = 100) # Keep refresh to see progress
      
      summary_output_j <- summary(fit_j_current)$summary
    }, error = function(e) {
      warning("Stan fitting or summary failed for Fold ", i, " Imputation ", j, ": ", e$message)
    })
    
    if (!is.null(summary_output_j)) {
      pred_param <- "pred"
      pred_rows <- grep(paste0("^", pred_param, "\\["), rownames(summary_output_j))
      if (length(pred_rows) == nrow(df_test) && all(pred_rows <= nrow(summary_output_j))) {
        all_preds_fold[, j] <- summary_output_j[pred_rows, "50%"] # Store median prediction
      } else {
        warning(paste("Prediction dim mismatch in Fold", i, "Imputation", j))
        all_preds_fold[, j] <- NA
      }
      rhats_imputations_fold[j] <- max(summary_output_j[, "Rhat"], na.rm = TRUE)
      neffs_imputations_fold[j] <- min(summary_output_j[, "n_eff"], na.rm = TRUE)
    } else {
      all_preds_fold[, j] <- NA
      rhats_imputations_fold[j] <- NA
      neffs_imputations_fold[j] <- NA
    }
    rm(fit_j_current, summary_output_j) # Remove large Stan object immediately
  } # End of inner loop (j over m imputations)
  
  # Pool predictions by averaging across imputations
  pred_pooled <- rowMeans(all_preds_fold, na.rm = TRUE)
  
  # Summarize convergence across imputations FOR THIS FOLD
  avg_max_rhat_fold <- mean(rhats_imputations_fold, na.rm = TRUE)
  avg_min_neff_fold <- mean(neffs_imputations_fold, na.rm = TRUE)
  cat(paste("Fold", i, "Pooled - Avg Max R-hat:", round(avg_max_rhat_fold, 3),
            " Avg Min ESS:", round(avg_min_neff_fold), "\n"))
  
  # Store results for the fold using pooled predictions
  fold_results_df <- data.frame(
    subject = df_test[[id_col]],
    fold = i,
    true_outcome = df_test[[outcome_col]], # Original true outcome from test set
    predicted_value = pred_pooled
  )
  
  # Store only what's needed for Step 3 (AUC and Calib)
  cv_results[[i]] <- list(
    avg_max_rhat_fold = avg_max_rhat_fold, # Store average diagnostics for the fold
    avg_min_neff_fold = avg_min_neff_fold,
    predictions = fold_results_df         # Store pooled predictions
    # NO coefficients_ORs, NO full posterior samples, NO train_means/sds
  )
  
  rm(mice_train_obj, mice_test_obj, all_preds_fold, rhats_imputations_fold, neffs_imputations_fold,
     fold_results_df, df_train, df_test, mice_train_data, mice_test_data)
  gc(verbose = FALSE)
  
  cat("Finished Fold", i, ". Pooled predictions stored.\n")
} # End of outer k-fold loop (i)

#################################################
## Save Compiled CV Results (Predictions & Diagnostics Only)
#################################################
cat("\nSaving final compiled CV results (predictions & summary diagnostics)...\n")
saveRDS(cv_results, file = output_result)
print(paste("Cross-validation results (predictions only) saved to:", output_result))
cat("Script finished successfully.\n")