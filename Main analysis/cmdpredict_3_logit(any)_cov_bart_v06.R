# PROJECT : PREDICTION OF CARDIOMETABOLIC MULTIMORBIDITY
# DATE    : 14 APRIL 2025 # Or current date
# NOTES   : STEP 3: Compile and Evaluate BART CV Results

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
input <- file.path(intermediate_data, "logit(any)_cov_bart_v06.rds") 

# --- Load Results ---
cv_results <- readRDS(input) # This now contains cv_results_bart from Step 2B

# --- Compile Predictions ---
# The structure should be similar: cv_results[[i]]$predictions contains the predictions
all_pred <- bind_rows(lapply(cv_results, function(res) {
  # Ensure the prediction column is named 'predicted_value' if it was different in bart_fold_df
  # In Step 2B, bart_fold_df had 'predicted_value', so this should be fine.
  if (!is.null(res$predictions)) {
    return(res$predictions)
  } else {
    return(NULL) # Handle case where a fold might have failed for BART
  }
}))

# Stop if no predictions compiled (e.g., all BART folds failed)
if (nrow(all_pred) == 0) {
  stop("No BART predictions found in cv_results. Check Step 2B output.")
}
# Filter out rows where BART predictions might be NA (if any fold failed partially)
all_pred <- all_pred %>% filter(!is.na(predicted_value))
if (nrow(all_pred) == 0) {
  stop("All BART predictions are NA. Check Step 2B output.")
}

#################################################
## AUC & 95% CI for AUC (for BART)
#################################################
print("--- BART Model Performance: AUC ---")
all_pred$true_outcome_factor <- factor(all_pred$true_outcome, levels = c(0, 1))

roc_obj_bart <- NULL # Initialize
if(length(unique(na.omit(all_pred$true_outcome_factor))) == 2 && sum(!is.na(all_pred$predicted_value)) > 0) {
  roc_obj_bart <- roc(response = all_pred$true_outcome_factor,
                      predictor = all_pred$predicted_value, # This is BART's prediction
                      levels = levels(all_pred$true_outcome_factor),
                      direction = "<", na.rm = TRUE)
  auc_value_bart <- auc(roc_obj_bart)
  print(paste("BART Overall AUC:", round(auc_value_bart, 3)))
  
  auc_ci_bart <- ci.auc(roc_obj_bart)
  print(paste("BART 95% CI for AUC (DeLong):",
              round(auc_ci_bart[1], 3), "-", round(auc_ci_bart[3], 3)))
} else {
  print("Cannot calculate BART AUC: not enough outcome levels or valid predictions.")
  auc_value_bart <- NA # So roc_obj_bart remains NULL if this block isn't run fully
}


#################################################
## MSE (Brier Score) - Threshold Independent (for BART)
#################################################
print("--- BART Model: Calculating MSE (Brier Score) ---")
all_pred$true_outcome_numeric <- as.numeric(as.character(all_pred$true_outcome_factor))
mse_value_bart <- mean((all_pred$true_outcome_numeric - all_pred$predicted_value)^2, na.rm = TRUE)
print(paste("BART Mean Squared Error (Brier Score):", format(mse_value_bart, digits = 4, nsmall = 4)))

#################################################
## Classification Metrics at Youden's J Optimal Threshold (for BART)
#################################################
print("--- BART Model: Calculating Metrics at Youden's J Optimal Threshold ---")

if (!is.null(roc_obj_bart) && inherits(roc_obj_bart, "roc")) {
  coords_youden_bart <- coords(roc_obj_bart, "best",
                               ret = c("threshold", "sensitivity", "specificity",
                                       "ppv", "npv", "accuracy"),
                               best.method = "youden", transpose = FALSE)
  
  if (nrow(coords_youden_bart) > 0) {
    best_threshold_bart <- coords_youden_bart$threshold[1]
    # ... (extract other metrics as before) ...
    youden_j_value_bart <- coords_youden_bart$sensitivity[1] + coords_youden_bart$specificity[1] - 1
    print(paste("BART Optimal Threshold (Youden's J):", format(best_threshold_bart, digits = 4)))
    print(paste("BART Youden's J Index Value:", format(youden_j_value_bart, digits = 4)))
    
    all_pred$predicted_class_youden_bart <- factor(
      ifelse(all_pred$predicted_value >= best_threshold_bart, 1, 0),
      levels = c(0, 1)
    )
    
    if ("1" %in% all_pred$true_outcome_factor) {
      print(paste("--- BART Confusion Matrix (Threshold:", format(best_threshold_bart, digits = 4), ") ---"))
      print(table(Predicted_BART_Youden = all_pred$predicted_class_youden_bart, Actual = all_pred$true_outcome_factor))
      conf_matrix_youden_bart <- tryCatch({
        confusionMatrix(data = all_pred$predicted_class_youden_bart,
                        reference = all_pred$true_outcome_factor,
                        positive = "1")
      }, error = function(e) { NULL })
      if (!is.null(conf_matrix_youden_bart)) print(conf_matrix_youden_bart)
    } else {
      print("Positive class '1' not found in true_outcome_factor for BART.")
    }
  } else {
    print("BART: Could not determine optimal threshold using Youden's J.")
  }
} else {
  print("BART ROC object not available or invalid for Youden's J metrics.")
}

#################################################
## Choose own threshold
#################################################

coords_spec90 <- coords(roc_obj_bart, x = 0.90, input = "specificity", ret = "all", transpose = FALSE)

# Print main performance metrics
cat("Specificity:", coords_spec90$specificity, "\n")
cat("Sensitivity:", coords_spec90$sensitivity, "\n")
cat("Accuracy:", coords_spec90$accuracy, "\n")
cat("Misclassification rate:", round(coords_spec90$`1-accuracy`, 4), "\n")

#################################################
## ROC Curve Plot (for BART)
#################################################
print("--- BART Model: Generating ROC Curve Plot ---")
if (!is.null(roc_obj_bart) && inherits(roc_obj_bart, "roc") && !is.na(auc_value_bart)) {
  auc_value_text_bart <- paste0("AUC = ", format(auc_value_bart, digits = 3, nsmall = 3),
                                " (95% CI: ", round(auc_ci_bart[1], 3), " - ", round(auc_ci_bart[3], 3), ")")
  roc_plot_bart <- ggroc(roc_obj_bart, colour = 'forestgreen', size = 1) + # Changed color
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey50") +
    annotate("text", x = 0.55, y = 0.15, label = auc_value_text_bart, hjust = 0, size = 3.5) +
    xlab("1 - Specificity (False Positive Rate)") +
    ylab("Sensitivity (True Positive Rate)") +
    ggtitle("BART Cross-Validated ROC Curve") + # Updated title
    theme_bw() +
    coord_fixed()
  print(roc_plot_bart)
  roc_plot_filename_bart <- "07. Graph/03. Paper 3/logit(any)_cov_bart_v06.png" # New filename
  ggsave(filename = roc_plot_filename_bart, plot = roc_plot_bart, width = 5, height = 5, units = "in", dpi = 300)
  print(paste("BART ROC Curve plot saved to:", roc_plot_filename_bart))
} else {
  print("BART ROC object not available or AUC invalid, skipping ROC plot generation.")
}

##################################################
## Calibration Plot
#################################################
# 1st graph
print("--- Generating Calibration Plot ---")
cal_plot <- all_pred %>%
  mutate(true_outcome = as.numeric(as.character(true_outcome))) %>%
  ggplot(aes(x = predicted_value, y = true_outcome)) +
  geom_jitter(width = 0.005, height = 0.05, size = 0.5, alpha = 0.1) + # Adjust/remove if needed
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
  geom_smooth(method = "loess", color = "blue", se = TRUE) +
  scale_y_continuous(name = "Observed cardiometabolic disease", breaks = c(0, 1)) +
  scale_x_continuous(name = "Predicted probability") +
  theme_minimal() %>%
  print()
print(cal_plot)

ggsave(filename = "07. Graph/03. Paper 3/logit(any)_cov_bart(1)_v06.png",
       plot = cal_plot,
       width = 7, height = 5, units = "in", dpi = 300)
print("Calibration plot saved.")


# 2nd graph
# ---- Data for calibration ----
cal_dat <- all_pred %>%
  transmute(
    p  = pmin(pmax(predicted_value, 1e-6), 1 - 1e-6),
    y  = as.integer(true_outcome_factor == "1"),
    lp = qlogis(p)
  )

# ---- Bin-level observed rates (8 bins, Wald 95% CI) ----
cal_bins <- cal_dat %>%
  mutate(bin = ntile(p, 10)) %>%
  group_by(bin) %>%
  summarise(
    p_mean = mean(p),
    y_mean = mean(y),
    n      = n(),
    se     = sqrt(pmax(y_mean * (1 - y_mean) / n, 0)),
    lwr    = pmax(0, y_mean - 1.96 * se),
    upr    = pmin(1, y_mean + 1.96 * se),
    .groups = "drop"
  )

# ---- Plot controls (data-driven range) ----
p_lo <- quantile(cal_dat$p, 0.01, na.rm = TRUE)
p_hi <- quantile(cal_dat$p, 0.99, na.rm = TRUE)
p_lo <- max(p_lo, 1e-6)                  # avoid 0 on logit
if (p_hi <= p_lo) {                      # safety for degenerate cases
  p_lo <- max(min(cal_dat$p, na.rm = TRUE), 1e-6)
  p_hi <- max(cal_dat$p, na.rm = TRUE)
}
x_max <- p_hi * 1.05                     # small padding
y_max <- x_max                           # keep 45° line meaningful
hist_frac <- 0.15
bw        <- max((p_hi - p_lo) / 20, 1e-5)         # ~20 bins over data range
cap_width <- max((p_hi - p_lo) * 0.02, 1e-5)       # ~2% of range

# ---- Logistic recalibration & 95% band (Wald on link) ----
fit_recal <- glm(y ~ lp, family = binomial(), data = cal_dat)

p_grid <- seq(p_lo, p_hi, length.out = 400)
grid   <- data.frame(p = p_grid, lp = qlogis(p_grid))
pr     <- predict(fit_recal, newdata = grid, type = "link", se.fit = TRUE)
grid$phat <- plogis(pr$fit)
grid$lwr  <- plogis(pr$fit - 1.96 * pr$se.fit)
grid$upr  <- plogis(pr$fit + 1.96 * pr$se.fit)

# ---- Plot ----
cal_blue <- "#063970"

cal_plot <- ggplot() +
  geom_histogram(
    data = subset(cal_dat, p >= p_lo & p <= x_max),
    aes(x = p, y = after_stat(ncount) * (hist_frac * y_max)),
    binwidth = bw, boundary = p_lo, closed = "left",
    fill = "grey70", color = "grey60", alpha = 0.8
  ) +
  # (Removed the geom_ribbon band)
  geom_abline(intercept = 0, slope = 1, linetype = "dashed",
              color = "grey55", linewidth = 0.5) +
  geom_point(data = cal_bins, aes(p_mean, y_mean),
             shape = 16, size = 2.6, color = cal_blue) +
  geom_errorbar(data = cal_bins, aes(p_mean, ymin = lwr, ymax = upr),
                width = cap_width, color = cal_blue) +
  coord_equal(xlim = c(0, x_max), ylim = c(0, y_max), expand = FALSE) +
  labs(x = "Predicted probability", y = "Observed event rate") +
  theme_gray(base_size = 12)

print(cal_plot)


# Save
ggsave("07. Graph/03. Paper 3/logit(any)_cov_bart(2)_v06.png",
       plot = cal_plot, width = 5, height = 5, units = "in", dpi = 300)

#################################################
## Slope and intercept
#################################################

## Calibration-in-the-large (CITL) and Slope 
# CITL: fit intercept with the model's linear predictor as an offset
fit_citl  <- glm(y ~ 1 + offset(lp), family = binomial(), data = cal_dat)
citl      <- coef(fit_citl)[1]
citl_se   <- sqrt(vcov(fit_citl)[1, 1])
citl_ci   <- citl + c(-1, 1) * 1.96 * citl_se   # Wald 95% CI (quick & standard)

# Slope: regress outcome on the model's linear predictor
fit_slope <- glm(y ~ lp, family = binomial(), data = cal_dat)
slope     <- coef(fit_slope)["lp"]
slope_se  <- sqrt(vcov(fit_slope)["lp", "lp"])
slope_ci  <- slope + c(-1, 1) * 1.96 * slope_se
slope_int <- coef(fit_slope)[1]                  # intercept from slope model (ideally ~0)

# Pretty print
fmt <- function(x) formatC(x, digits = 3, format = "f")
cat("CITL (ideal 0): ", fmt(citl), "  95% CI [", fmt(citl_ci[1]), ", ", fmt(citl_ci[2]), "]\n", sep = "")
cat("Slope (ideal 1): ", fmt(slope), "  95% CI [", fmt(slope_ci[1]), ", ", fmt(slope_ci[2]), "]",
    "  (intercept from slope model: ", fmt(slope_int), ")\n", sep = "")

## Brier
p <- cal_dat$p
y <- cal_dat$y

brier <- mean((y - p)^2)
# simple SE & CI via asymptotics (OK for large n)
brier_se <- sqrt(var((y - p)^2) / length(y))
brier_ci <- brier + c(-1, 1) * 1.96 * brier_se

# Brier Skill Score vs prevalence baseline
pi_hat    <- mean(y)
brier_ref <- pi_hat * (1 - pi_hat)
bss       <- 1 - brier / brier_ref

cat("Brier score: ", fmt(brier), "  95% CI [", fmt(brier_ci[1]), ", ", fmt(brier_ci[2]), "]\n", sep = "")
cat("Brier Skill Score (vs prevalence): ", fmt(bss), "\n", sep = "")

print("Script 3 finished.")