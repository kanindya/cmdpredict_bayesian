# PROJECT : PREDICTION OF CARDIOMETABOLIC MULTIMORBIDITY
# DATE  : 20 APRIL 2025 # Adjusted to May 4, 2025 based on context
# NOTES : STEP 1: Pre-processing (logit, covariates) - MICE Adapted

# Start with clean environtment
rm(list = ls())

#################################################
## Load packages
#################################################

library(haven)
library(dplyr)
library(sjmisc)
library(corrplot)
library(ggcorrplot)
library(mice) # <<< MICE Change: Load mice package

#################################################
## Setup
#################################################

# --- Set working directory ---
setwd("/Users/kanyaanindya/Documents/10. SCAPIS/")

#################################################
## Data loading and initial preparation
#################################################

# --- Load Data ---
df <- read_dta("05. Result/cmdpredict_python_v05.dta")

# --- Define Variables ---
outcome_col <- 'cmd_multi'
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

# Select necessary columns
  df_selected <- df[, c(id_col, outcome_col, predictor_col)]

# Check collinearity (on original data before imputation)
  df_col <- df_selected[, c(predictor_col)]
  df_col <- data.frame(lapply(df_col, as.numeric))
  cor_matrix <- cor(df_col, method="spearman", use = "pairwise.complete.obs") 

# --- Clean Data Types ---
  df_selected <- df_selected %>%
    mutate(across(all_of(predictor_col), ~ zap_labels(.))) %>% # Zap labels only on predictors first
    mutate(across(all_of(c(outcome_col, predictor_col)), as.numeric)) # Ensure all relevant cols are numeric

# Prepare Outcome Variable (Ensure it's integer 0/1 for Stan later)
  df_selected[[outcome_col]] <- as.integer(round(df_selected[[outcome_col]]))
  values <- sort(unique(df_selected[[outcome_col]]))
  print(paste("Unique integer values in", outcome_col, ":", paste(values, collapse=", ")))
  # Optional: Check if outcome is indeed 0/1
  if (!all(values %in% c(0, 1))) {
    warning(paste("Outcome column '", outcome_col, "' contains values other than 0 or 1 after conversion. Check data integrity.", sep=""))
  }

#################################################
## K-Fold indices on FULL data
#################################################
# Set K
  k <- 10
  n_obs <- nrow(df_selected) # <<< MICE Change: Use full N

# Cross-validation splitting on the full dataset
  set.seed(251222)
  
# Create fold assignments first
  fold_assignments <- sample(rep(1:k, length.out = n_obs))
  df_selected$fold_id <- fold_assignments # add fold ID to split
  fold_indices <- split(df_selected, df_selected$fold_id) 
  
# Remove the temporary fold_id column from each list element
  fold_indices <- lapply(fold_indices, function(df) 
    { df$fold_id <- NULL; df })
  names(fold_indices) <- paste0("Fold", 1:k)

# Check number of cases (outcome=1) for each fold (on potentially incomplete data)
  sapply(fold_indices, function(fold) sum(fold[[outcome_col]] == 1, na.rm = TRUE))
  
#################################################
## Save processed data (with NAs) and fold indices
#################################################
# Directory for intermediate data
  intermediate_data <-"08. Temp/03. Paper 3"
  dir.create(intermediate_data, showWarnings = FALSE, recursive = TRUE)

# Save output
# All data (now contains NAs)
  output_data <- file.path(intermediate_data, "logit(multi)_mice_df_v06.rds") 
  saveRDS(df_selected, file = output_data)

# Fold indices (list of dataframes with NAs)
  output_fold <- file.path(intermediate_data, "logit(multi)_mice_fold_v06.rds")
  saveRDS(fold_indices, file = output_fold)

print("Step 1 finished. Data with NAs and corresponding folds saved.")