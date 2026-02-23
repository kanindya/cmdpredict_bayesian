# PROJECT : PREDICTION OF CARDIOMETABOLIC MULTIMORBIDITY 
# DATE  : 20 APRIL 2025
# NOTES : STEP 1: Pre-processing (logit, covariates)

# Start with clean environtment
rm(list = ls())

#################################################
## Load packages
#################################################

library(haven)
library(dplyr)
library(caret)
library(sjmisc)
library(corrplot)
library(ggcorrplot)

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
                   'fibrer',
                   'time_days'
                   )
  
  predictor_col <- c( 'age1',
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
  df_selected <- df[, c(id_col, outcome_col, time_col, predictor_col)]
  
# Check collinearity
  df_col <- df_selected[, c(predictor_col)]
  df_col <- data.frame(lapply(df_col, as.numeric))
  cor_matrix <- cor(df_col, method="spearman", use = "pairwise.complete.obs")
  ggcorrplot(cor_matrix)

# --- Clean Data Types ---
# Remove labels and ensure numeric types
  df_selected <- df_selected %>%
    mutate(across(everything(), ~ zap_labels(.))) %>%
    mutate(across(all_of(c(outcome_col, predictor_col)), as.numeric))

# Filter for Complete Cases 
  df_selected <- df_selected[complete.cases(df_selected[, c(outcome_col, predictor_col)]), ]

# Prepare Outcome Variable
  df_selected[[outcome_col]] <- as.integer(round(df_selected[[outcome_col]]))
  
  values <- sort(unique(df_selected[[outcome_col]]))
  print(paste("Unique integer values in", outcome_col, "(complete cases):", paste(values, collapse=", ")))
  df_selected[[outcome_col]] <- factor(df_selected[[outcome_col]],
                                       levels = values)

  print(paste("Class of", outcome_col, "after conversion:"))
  print(class(df_selected[[outcome_col]]))
  print("Summary:")
  print(summary(df_selected[[outcome_col]]))

#################################################
## K-Fold indices
#################################################
# Set K
  k <- 10
  n_obs <- nrow(df_selected)
  
# Cross-validation
  set.seed(251222)
  fold_indices <- split(df_selected, sample(rep(1:k, length.out = n_obs)))
  names(fold_indices) <- paste0("Fold", 1:k)

# Check number of cases for each fold
  sapply(fold_indices, function(fold) sum(fold[[outcome_col]] == 1))

#################################################
## Save processed data and fold indices
#################################################
# Directory for intermediate data
  intermediate_data <-"08. Temp/03. Paper 3"
  dir.create(intermediate_data, showWarnings = FALSE)
  
# Save output
  # All data
  output_data <- file.path(intermediate_data, "logit(multi)_cov_df_v06.rds")
  saveRDS(df_selected, file = output_data)
  
  # Fold indices
  output_fold <- file.path(intermediate_data, "logit(multi)_cov_fold_v06.rds")
  saveRDS(fold_indices, file = output_fold)
