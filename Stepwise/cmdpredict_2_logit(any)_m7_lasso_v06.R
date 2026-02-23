# PROJECT : PREDICTION OF CARDIOMETABOLIC MULTIMORBIDITY 
# DATE  : 20 APRIL 2025
# NOTES : STEP 2: Runnning the model (logit, no covariates)

# Start with clean environtment
rm(list = ls())

#################################################
## Load packages
#################################################

library(dplyr)
library(rstan)
library(parallel) # Ensure parallel is loaded if using cores > 1

#################################################
## Setup & Load Data
#################################################

# --- Set working directory ---
  setwd("/Users/kanyaanindya/Documents/10. SCAPIS/") #

# --- Define input/output files ---
# Directory for intermediate data
  intermediate_data <- "08. Temp/03. Paper 3"
  dir.create(intermediate_data, showWarnings = FALSE, recursive = TRUE) # Added recursive = TRUE

  input_data <- file.path(intermediate_data, "logit(any)_cov_df_v06.rds")
  input_fold <- file.path(intermediate_data, "logit(any)_cov_fold_v06.rds")
  output_result <- file.path(intermediate_data, "logit(any)_m7_lasso_cvresult_v06.rds")
  stan_model <- "03. Do file/03. Paper 3/02. R/01. Stan/cmdpredict_single_logit_completed_lasso_v02.stan"

# --- Load preprocessed data and folds ---
  df_selected <- readRDS(input_data)
  fold_indices <- readRDS(input_fold)

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
  
  integer_col <- c('time_days',
                   'age1',
                   'sbp_mean',
                   'dbp_mean',
                   'waist',
                   'hdl',
                   'nonhdl',
                   'natrium',
                   'fibrer')
  
  predictor_col <- c( 'time_days',
                      'smoking1_2',
                      'alcohol2_2',
                      'natrium',
                      'fibrer',
                      'pa_riskvpa',
                      'mental_stress2_3',
                      'sleep_dur1_2')
  
#################################################
## K-Fold Cross-Validation Loop
#################################################
# --- List to store results ---
  k <- length(fold_indices)
  cv_results <- vector("list", k)
  names(cv_results) <- paste0("Fold", 1:k)

# --- Define key parameters for trace plots  ---
  key_pars_trace <- c("alpha")
  idx_age <- which(predictor_col == "age1"); idx_smoke <- which(predictor_col == "smoking1_2")
  idx_sbp <- which(predictor_col == "sbp_mean"); idx_waist <- which(predictor_col == "waist"); idx_hdl <- which(predictor_col == "hdl")
  key_beta_indices <- c(idx_age, idx_smoke, idx_sbp, idx_waist, idx_hdl)
  key_beta_indices <- key_beta_indices[key_beta_indices > 0]
  if(length(key_beta_indices) > 0) { key_pars_trace <- c(key_pars_trace, paste0("beta[", key_beta_indices, "]")) }
  
  print(paste("Starting", k, "-Fold Cross-Validation (logit, no covariates)..."))

  for (i in 1:k) {
    cat("\n--- Processing Fold:", i, "---\n")
    
    # --- Define Train/Test sets for this fold ---
    df_test <- fold_indices[[i]]
    df_train <- bind_rows(fold_indices[-i])
    
    cat("Train set size:", nrow(df_train), " Test set size:", nrow(df_test), "\n")
    
    #################################################
    ## Standardization
    #################################################
    
    # --- Standardization (within the loop) ---
    valid_integer_col <- intersect(integer_col, names(df_train)[sapply(df_train, is.numeric)])
    train_means <- sapply(df_train[, valid_integer_col, drop=FALSE], mean, na.rm = TRUE)
    train_sds <- sapply(df_train[, valid_integer_col, drop=FALSE], sd, na.rm = TRUE)
    train_sds[is.na(train_sds) | train_sds == 0] <- 1 # Handle NA SDs (from single value columns) and zero SDs
      
      df_train <- df_train %>%
        mutate(across(all_of(valid_integer_col),
                      ~ (. - train_means[cur_column()]) / train_sds[cur_column()]))
      df_test <- df_test %>%
        mutate(across(all_of(valid_integer_col),
                      ~ (. - train_means[cur_column()]) / train_sds[cur_column()]))
      
      cat("Standardization applied to:", paste(valid_integer_col, collapse=", "), "based on Fold", i, "training data.\n")
    
    #################################################
    ## STAN
    #################################################
      
    # --- Prepare Data for Stan ---
      OUTCOME_train <- as.integer(as.character(df_train[[outcome_col]]))
      DESIGN_train <- as.matrix(df_train[, predictor_col])
      PRED_DESIGN_test <- as.matrix(df_test[, predictor_col])
      PRED_N_test <- nrow(PRED_DESIGN_test)
      tau_scale <- 0.50
      
      # Create Stan Data List
        stan_data_fold <- list(
          N = nrow(DESIGN_train),
          D = ncol(DESIGN_train),
          DESIGN = DESIGN_train,
          OUTCOME = OUTCOME_train, # Should be 0s and 1s
          PRED_N = PRED_N_test,
          PRED_DESIGN = PRED_DESIGN_test,
          scale_global = tau_scale
        )
  
    # --- Run Stan Model ---
      cat("Running Stan model for Fold", i, "...\n")
      fit <- stan(file = stan_model,      
                  data = stan_data_fold,
                  chains = 4,             
                  iter = 2000,            
                  warmup = 1000,          
                  cores = 2, # 
                  seed = 456 + i,         
                  control = list(adapt_delta = 0.85) #
      )
      # --- Extract Results ---
      summary_output <- summary(fit)$summary
      
      # --- Assess Convergence for this Fold ---
      cat("Assessing convergence...\n")
      max_rhat <- max(summary_output[, "Rhat"], na.rm = TRUE)
      min_neff <- min(summary_output[, "n_eff"], na.rm = TRUE)
      cat(paste("  Max R-hat:", round(max_rhat, 3), " Min ESS:", round(min_neff), "\n"))
      
      # Generate trace plot using summary_output to check pars
      pars_in_fit <- rownames(summary_output)
      current_key_pars <- intersect(key_pars_trace, pars_in_fit)
      if ("tau" %in% pars_in_fit) { current_key_pars <- c(current_key_pars, "tau") }
      if (length(current_key_pars) > 0) {
        print(stan_trace(fit, pars = current_key_pars, inc_warmup = FALSE))
      } else {
        cat("  No key parameters found for trace plot.\n")
      }
      
    # --- Extract predictions ---
      pred_param <- "pred" 
      pred_rows <- grep(paste0("^", pred_param, "\\["), rownames(summary_output))
      pred_median <- summary_output[pred_rows, "50%"]
      
    # Store results
      fold_results_df <- data.frame(
        subject = df_test[[id_col]],
        fold = i,
        true_outcome = df_test[[outcome_col]], # Original 0/1 outcome
        predicted_value = pred_median # Predicted probability of outcome = 1
      )
      
    # --- Extract Coefficient Summaries & Calculate ORs ---
      beta_summary_df <- NULL # Initialize
      beta_rows <- grep("^beta\\[", rownames(summary_output))
      
    # Extract median, 2.5% and 97.5% quantiles (credible interval on log-odds scale)
      beta_summary_df <- as.data.frame(summary_output[beta_rows, c("50%", "2.5%", "97.5%")])
      
    # Add predictor names (ensure predictor_col is defined correctly and matches beta order)
      beta_summary_df$Predictor <- predictor_col
      
    # Calculate ORs and their 95% CIs
      beta_summary_df$OR_median <- exp(beta_summary_df$"50%")
      beta_summary_df$OR_LCI <- exp(beta_summary_df$"2.5%")
      beta_summary_df$OR_UCI <- exp(beta_summary_df$"97.5%")
      
    # Reorder columns for clarity
      beta_summary_df <- beta_summary_df[, c("Predictor", "50%", "2.5%", "97.5%", "OR_median", "OR_LCI", "OR_UCI")]
      colnames(beta_summary_df)[2:4] <- c("beta_median", "beta_LCI", "beta_UCI") # Rename log-odds cols
  
    # --- Store both results for the fold ---
    # Store results as a list containing both predictions and coeffs/ORs
      posterior_extract <- rstan::extract(fit, pars = c("alpha", "beta"))
      
      cv_results[[i]] <- list(
        max_rhat = max_rhat,
        min_neff = min_neff,
        posterior = posterior_extract, 
        train_means = train_means,
        train_sds = train_sds,
        predictions = fold_results_df,
        coefficients_ORs = beta_summary_df # Will be NULL if no betas found
      )
      cat("Finished Fold", i, ". Results stored.\n")
  
  } # End of k-fold loop

#################################################
## Save Compiled Results
#################################################
saveRDS(cv_results, file = output_result)
print(paste("Cross-validation results saved to:", output_result))
