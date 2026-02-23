# PROJECT : PREDICTION OF CARDIOMETABOLIC MULTIMORBIDITY
# DATE    : 14 APRIL 2025
# NOTES   : STEP 3: Compile results from Bayesian LASSO Cox Model CV

# Start with clean environtment
rm(list = ls())

#################################################
## Load packages
#################################################
library(dplyr)
library(pROC)
library(ggplot2)
library(caret) # For confusionMatrix

#################################################
## Setup & Load Data
#################################################

# --- Set working directory ---
setwd("/Users/kanyaanindya/Documents/10. SCAPIS/")

# --- Define Input File ---
intermediate_data <- "08. Temp/03. Paper 3"
# <<< CHANGED to load the COX model results from Step 2 >>>
input <- file.path(intermediate_data, "cox(multi)_cov_lasso_cvresult_v06.rds")

# --- Load Results ---
if (!file.exists(input)) {
  stop(paste("Input file not found:", input, "\nPlease ensure Step 2 for the Cox model has been run successfully."))
}
cv_results <- readRDS(input)

# --- Compile Predictions ---
all_pred <- bind_rows(lapply(cv_results, function(res) res$predictions))

if (nrow(all_pred) == 0 || all(is.na(all_pred$predicted_value))) {
  stop("No valid predictions found in cv_results. Check the output of Step 2 for the Cox model.")
}

#################################################
## Diagnostic
#################################################

all_max_rhats <- sapply(cv_results, function(res) res$max_rhat)
all_min_neffs <- sapply(cv_results, function(res) res$min_neff)
print("--- Convergence Summary Across Folds (Cox Model) ---")
print("Max R-hats:")
print(summary(all_max_rhats))
print("Min Effective Sample Sizes:")
print(summary(all_min_neffs))

#################################################
## AUC & 95% CI for AUC (using Risk Score)
#################################################
print("--- Calculating AUC for Cox Model Risk Scores ---")
all_pred$true_outcome_factor <- factor(all_pred$true_outcome, levels = c(0, 1))
roc_obj <- NULL; auc_value <- NA

if(length(unique(na.omit(all_pred$true_outcome_factor))) == 2){
  roc_obj <- roc(response = all_pred$true_outcome_factor,
                 predictor = all_pred$predicted_value, # This is the Cox RISK SCORE
                 direction = "<", na.rm = TRUE)
  auc_value <- auc(roc_obj)
  print(paste("Overall AUC (from risk scores):", round(auc_value, 3)))
  
  auc_ci <- ci.auc(roc_obj)
  print(paste("95% CI for AUC (DeLong):",
              round(auc_ci[1], 3), "-", round(auc_ci[3], 3)))
} else {
  print("Cannot calculate AUC: outcome variable does not have two levels.")
}


#################################################
## MSE (Brier Score) - NOTE: This is less meaningful for risk scores
#################################################
# MSE/Brier score is designed for probabilities (0-1). Applying it to an unbounded
# risk score is not standard practice. We will comment this section out.
# print("--- Calculating MSE (Brier Score) ---")
# all_pred$true_outcome_numeric <- as.numeric(as.character(all_pred$true_outcome_factor))
# mse_value <- mean((all_pred$true_outcome_numeric - all_pred$predicted_value)^2, na.rm = TRUE)
# print(paste("Mean Squared Error (Brier Score):", format(mse_value, digits = 4, nsmall = 4)))


#################################################
## Classification Metrics at Youden's J Optimal Threshold (using Risk Score)
#################################################
print("--- Calculating Metrics at Youden's J Optimal Threshold (from Risk Score) ---")

if (!is.null(roc_obj) && inherits(roc_obj, "roc")) {
  coords_youden <- coords(roc_obj, "best",
                          ret = c("threshold", "sensitivity", "specificity",
                                  "ppv", "npv", "accuracy"),
                          best.method = "youden", transpose = FALSE)
  
  if (nrow(coords_youden) > 0) {
    best_threshold <- coords_youden$threshold[1] # This is a threshold on the RISK SCORE scale
    # ... (rest of the metric extraction is the same) ...
    youden_j_value <- coords_youden$sensitivity[1] + coords_youden$specificity[1] - 1
    
    print(paste("Optimal Risk Score Threshold (Youden's J):", format(best_threshold, digits = 4)))
    print(paste("Youden's J Index Value:", format(youden_j_value, digits = 4)))
    
    all_pred$predicted_class_youden <- factor(
      ifelse(all_pred$predicted_value >= best_threshold, 1, 0), # Classify based on risk score
      levels = c(0, 1)
    )
    
    if ("1" %in% all_pred$true_outcome_factor) {
      print(paste("--- Confusion Matrix (Threshold:", format(best_threshold, digits = 4), ") ---"))
      print(table(Predicted_at_Youden_Thresh = all_pred$predicted_class_youden, Actual = all_pred$true_outcome_factor))
      
      conf_matrix_youden <- tryCatch({
        confusionMatrix(data = all_pred$predicted_class_youden,
                        reference = all_pred$true_outcome_factor,
                        positive = "1")
      }, error = function(e) { NULL })
      if (!is.null(conf_matrix_youden)) print(conf_matrix_youden)
    } else {
      print("Positive class '1' not found in true_outcome_factor.")
    }
  } else {
    print("Could not determine optimal threshold using Youden's J.")
  }
} else {
  print("ROC object not available or invalid, skipping Youden's J and related metrics.")
}


#################################################
## ROC Curve Plot (using Risk Score)
#################################################
print("--- Generating ROC Curve Plot (Cox Model) ---")
if (!is.null(roc_obj) && inherits(roc_obj, "roc")) {
  auc_ci_formatted <- paste0("AUC = ", format(auc_value, digits = 3, nsmall = 3),
                             " (95% CI: ", round(auc_ci[1], 3), " - ", round(auc_ci[3], 3), ")")
  roc_plot <- ggroc(roc_obj, colour = 'darkred', size = 1) + # Changed color
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey50") +
    annotate("text", x = 0.55, y = 0.15, label = auc_ci_formatted, hjust = 0, size = 3.5) +
    xlab("1 - Specificity (False Positive Rate)") +
    ylab("Sensitivity (True Positive Rate)") +
    ggtitle("Cross-Validated ROC Curve (Cox Model)") + # Updated title
    theme_bw() +
    coord_fixed()
  print(roc_plot)
  roc_plot_filename <- "07. Graph/03. Paper 3/cox(multi)_cov_lasso_ROC_v06.png" # Updated filename
  ggsave(filename = roc_plot_filename, plot = roc_plot, width = 5, height = 5, units = "in", dpi = 300)
  print(paste("ROC Curve plot saved to:", roc_plot_filename))
} else {
  print("ROC object not available, skipping ROC plot generation.")
}

#################################################
## Calibration Plot (NOTE: For Risk Scores, this is less direct)
#################################################
print("--- Generating Calibration-Style Plot (Cox Model) ---")
# NOTE: A true calibration plot for a Cox model compares predicted survival probabilities
# to observed Kaplan-Meier survival rates. This plot is a simplified visual check to see if
# higher risk scores correspond to a higher proportion of events.
cal_plot <- all_pred %>%
  mutate(true_outcome_numeric = as.numeric(as.character(true_outcome_factor))) %>%
  ggplot(aes(x = predicted_value, y = true_outcome_numeric)) + # x is RISK SCORE
  geom_smooth(method = "loess", color = "red", se = TRUE) + # Use smoother to see the trend
  scale_y_continuous(name = "Observed Outcome Proportion", limits = c(0, 1), breaks = c(0, 1)) +
  scale_x_continuous(name = "Predicted Risk Score (Log-Hazard Ratio)") +
  ggtitle("Calibration-Style Plot for Cox Model Risk Score") +
  theme_minimal()
print(cal_plot)

ggsave(filename = "07. Graph/03. Paper 3/cox(multi)_cov_lasso_calib_v06.png", # Updated filename
       plot = cal_plot,
       width = 7, height = 5, units = "in", dpi = 300)
print("Calibration plot saved.")

#################################################
## Compile Coefficients (HAZARD RATIOS)
#################################################
print("--- Summarizing Hazard Ratios (HRs) Across Folds ---")
# Use coefficients_HRs as saved from the modified Step 2
all_hr <- bind_rows(lapply(1:length(cv_results), function(i) {
  res <- cv_results[[i]]$coefficients_HRs
  if (!is.null(res) && nrow(res) > 0) {
    res$Fold <- i
    return(res)
  } else {
    return(NULL)
  }
}))

if (!is.null(all_hr) && nrow(all_hr) > 0) {
  # <<< CHANGED to match the predictor_col from the Cox Step 2 script >>>
  order <- c('age1','sex1_2','fh_cmd_2','education2_2','expense1_2','born1_2','location1_2',
             'site1_2','site1_3','site1_4','site1_5','site1_6','smoking1_2','alcohol2_2',
             'natrium','pa_riskvpa','mental_stress2_3','sleep_dur1_2','sbp_mean','dbp_mean',
             'waist','hdl','nonhdl','drug_hypertension_2','drug_lipid_2')
  
  hr_summary <- all_hr %>% # Renamed or -> hr
    filter(!is.na(Predictor)) %>%
    group_by(Predictor) %>%
    summarise(
      median_HR = median(HR_median, na.rm = TRUE), # OR -> HR
      min_HR = min(HR_median, na.rm = TRUE),
      max_HR = max(HR_median, na.rm = TRUE),
      median_HR_LCI = median(HR_LCI, na.rm = TRUE),
      median_HR_UCI = median(HR_UCI, na.rm = TRUE),
      n_folds = n(),
      .groups = 'drop'
    ) %>%
    mutate(Predictor = factor(Predictor, levels = intersect(order, unique(Predictor)))) %>%
    arrange(Predictor) %>%
    select(Predictor, median_HR, min_HR, max_HR, median_HR_LCI, median_HR_UCI, n_folds)
  
  print(as.data.frame(hr_summary))
} else {
  print("No coefficient/HR summaries to process.")
}

print("Script 3 finished.")