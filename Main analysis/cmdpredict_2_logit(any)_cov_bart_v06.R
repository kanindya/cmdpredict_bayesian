# PROJECT : PREDICTION OF CARDIOMETABOLIC MULTIMORBIDITY
# DATE    : 20 APRIL 2025
# NOTES   : STEP 2B: Running BART model within CV (with Diagnostics)

# Start with clean environment
rm(list = ls())

#################################################
## Load packages
#################################################
library(dplyr)
library(dbarts)
library(parallel)

#################################################
## Setup & Load Data
#################################################

# --- Set working directory ---
setwd("/Users/kanyaanindya/Documents/10. SCAPIS/") # Adjust if needed

# --- Define input/output files ---
intermediate_data <- "08. Temp/03. Paper 3"
dir.create(intermediate_data, showWarnings = FALSE, recursive = TRUE)

input_data <- file.path(intermediate_data, "logit(any)_cov_df_v06.rds")
input_fold <- file.path(intermediate_data, "logit(any)_cov_fold_v06.rds")
output_result_bart <- file.path(intermediate_data, "logit(any)_cov_bart_v06.rds")

# --- Load preprocessed data and folds ---
fold_data_list <- readRDS(input_fold)

# --- Define objects (Must match those used to create df_selected in Step 1) ---
out_col <- 'cmd_any'
id_col <- 'subject'
integer_col <- c('time_days',
                 'age1',
                 'sbp_mean',
                 'dbp_mean',
                 'waist',
                 'hdl',
                 'nonhdl',
                 'natrium')

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

cont_vars <- intersect(integer_col, predictor_col)

#################################################
## K-Fold Cross-Validation Loop for BART
#################################################
k <- length(fold_data_list)
cv_results_bart <- vector("list", k)
names(cv_results_bart) <- paste0("Fold", 1:k)

print(paste("Starting", k, "-Fold Cross-Validation for BART..."))

for (i in 1:k) {
  cat("\n--- Processing Fold:", i, "for BART ---\n")
  
  df_test  <- fold_data_list[[i]]
  df_train <- bind_rows(fold_data_list[-i])
  cat("Train set size:", nrow(df_train), " Test set size:", nrow(df_test), "\n")
  
  # --- Standardization ---
  train_means <- sapply(df_train[, cont_vars, drop=FALSE], mean, na.rm = TRUE)
  train_sds <- sapply(df_train[, cont_vars, drop=FALSE], sd, na.rm = TRUE)
  train_sds[is.na(train_sds) | train_sds == 0] <- 1
  std_val_func <- function(data_col, col_mean, col_sd) (data_col - col_mean) / col_sd
  
  df_train_std <- df_train
  df_test_std  <- df_test
  for(col_n in cont_vars){
    if(col_n %in% names(df_train_std)) df_train_std[[col_n]] <- std_val_func(df_train[[col_n]], train_means[col_n], train_sds[col_n])
    if(col_n %in% names(df_test_std))  df_test_std[[col_n]]  <- std_val_func(df_test[[col_n]],  train_means[col_n], train_sds[col_n])
  }
  cat("Standardization applied for Fold", i, "\n")
  
  # --- Prepare Data for BART ---
  x_train_bart <- as.matrix(df_train_std[, predictor_col])
  y_train_bart <- as.numeric(as.character(df_train_std[[out_col]]))
  x_test_bart  <- as.matrix(df_test_std[, predictor_col])
  
  # *** DIAGNOSTIC CHECK 1: Inspect y_train_bart ***
  cat("Fold", i, "- Summary of y_train_bart (outcome for BART):\n")
  print(summary(y_train_bart))
  print(table(y_train_bart, useNA = "ifany"))
  if(!all(y_train_bart %in% c(0,1) & !is.na(y_train_bart))){
    warning(paste("Fold", i, "- y_train_bart does NOT strictly contain 0s and 1s! This is likely the problem."))
  }
  
  # *** DIAGNOSTIC CHECK 2: Check for non-finite values in predictors ***
  if(any(!is.finite(x_train_bart))) warning(paste("Fold", i, "- x_train_bart contains non-finite values (NA, NaN, Inf)!"))
  if(any(!is.finite(x_test_bart)))  warning(paste("Fold", i, "- x_test_bart contains non-finite values (NA, NaN, Inf)!"))
  
  
  # --- Run BART Model ---
  cat("Running BART model for Fold", i, "...\n")
  bart_pred_probabilities <- rep(NA_real_, nrow(df_test)) # Initialize
  fit_bart <- NULL # Initialize to handle potential errors
  tryCatch({
    fit_bart <- dbarts::bart(
      x.train = x_train_bart,
      y.train = y_train_bart, # Ensure this is numeric 0/1
      x.test = x_test_bart,
      ntree = 50,
      ndpost = 1000,
      nskip = 250,
      verbose = FALSE,
      keeptrees = FALSE,
      keeptrainfits = FALSE,
      seed = 789 + i
    )
    
    # This is the critical part for getting correct probabilities
    if (!is.null(fit_bart$yhat.test)) {
      # fit_bart$yhat.test for binary y.train contains samples on the PROBIT SCALE (linear predictor)
      # We need to transform them to the PROBABILITY SCALE using pnorm()
      
      # DIAGNOSTIC CHECK 3: Inspect raw BART yhat.test samples (PROBIT SCALE)
      cat("Fold", i, "- Summary of raw BART yhat.test samples (PROBIT SCALE):\n")
      print(summary(c(fit_bart$yhat.test))) # Values can be negative or positive
      
      # Apply inverse probit (pnorm) to each sample for each observation
      prob_samples_bart <- pnorm(fit_bart$yhat.test) # This transforms to 0-1 probability scale
      
      # DIAGNOSTIC CHECK 3.5: Inspect transformed probability samples
      cat("Fold", i, "- Summary of BART probability samples (after pnorm, should be 0-1):\n")
      print(summary(c(prob_samples_bart))) # These values MUST be between 0 and 1
      
      # Now take the mean of these correctly scaled probability samples
      bart_pred_probabilities <- colMeans(prob_samples_bart)
      
      # DIAGNOSTIC CHECK 4: Inspect final mean BART probabilities
      cat("Fold", i, "- Summary of final mean BART probabilities (bart_pred_probabilities):\n")
      print(summary(bart_pred_probabilities))
      if(any(bart_pred_probabilities < 0, na.rm=TRUE) || any(bart_pred_probabilities > 1, na.rm=TRUE)){
        warning(paste("Fold", i, "- bart_pred_probabilities are OUTSIDE [0,1] range! This is still unexpected after pnorm(). Investigate further."))
      }
    } else {
      warning(paste("BART predictions (yhat.test) not found for Fold", i))
      # bart_pred_probabilities remains NA_real_ as initialized
    }
  }, error = function(e) {
    warning(paste("BART fitting/prediction failed for Fold", i, ":", e$message))
    # bart_pred_probabilities remains NA_real_ as initialized
  })
  
  # --- Store BART predictions ---
  # (This part of your script was already correct)
  bart_fold_df <- data.frame(
    subject = df_test[[id_col]],
    fold = i,
    true_outcome = df_test[[out_col]],
    predicted_value = bart_pred_probabilities # This now uses the correctly transformed probabilities
  )
  
  cv_results_bart[[i]] <- list(predictions = bart_fold_df)
  cat("Finished Fold", i, "for BART. Predictions stored.\n")
  rm(fit_bart, df_train, df_test, df_train_std, df_test_std, x_train_bart, y_train_bart, x_test_bart, prob_samples_bart) # Added prob_samples_bart to rm
  gc()
  
} # End of k-fold loop

saveRDS(cv_results_bart, file = output_result_bart)
print(paste("BART Cross-validation predictions saved to:", output_result_bart))
print("Step 2B (BART Only CV) finished.")