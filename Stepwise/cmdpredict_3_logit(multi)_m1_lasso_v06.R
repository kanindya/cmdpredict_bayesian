# PROJECT : PREDICTION OF CARDIOMETABOLIC MULTIMORBIDITY
# DATE    : 14 APRIL 2025
# NOTES : STEP 3: Compile results (logit, no covariates)

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
input <- file.path(intermediate_data, "logit(multi)_m1_lasso_cvresult_v06.rds")

# --- Load Results ---
cv_results <- readRDS(input)

# --- Compile Predictions ---
all_pred <- bind_rows(lapply(cv_results, function(res) res$predictions))

#################################################
## Diagnostic
#################################################

all_max_rhats <- sapply(cv_results, function(res) res$max_rhat)
all_min_neffs <- sapply(cv_results, function(res) res$min_neff)
print("--- Convergence Summary Across Folds ---")
print("Max R-hats:")
print(summary(all_max_rhats))
print("Min Effective Sample Sizes:")
print(summary(all_min_neffs))

#################################################
## AUC & 95% CI for AUC
#################################################
all_pred$true_outcome_factor <- factor(all_pred$true_outcome, levels = c(0, 1))
roc_obj <- roc(response = all_pred$true_outcome_factor,
               predictor = all_pred$predicted_value,
               levels = levels(all_pred$true_outcome_factor),
               direction = "<", na.rm = TRUE)
auc_value <- auc(roc_obj)
print(paste("Overall AUC:", round(auc_value, 3)))

# --- Calculate and print 95% CI for AUC ---
# By default, ci.auc uses DeLong's method for unpaired ROC curves
auc_ci <- ci.auc(roc_obj) # Returns a vector: [lower_bound, AUC, upper_bound]
print(paste("95% CI for AUC (DeLong):",
            round(auc_ci[1], 3), "-", round(auc_ci[3], 3)))
# Note: auc_ci[2] would be the same as auc_value

#################################################
## MSE (Brier Score) - Threshold Independent
#################################################
# ... (rest of your script continues as before) ...
print("--- Calculating MSE (Brier Score) ---")
all_pred$true_outcome_numeric <- as.numeric(as.character(all_pred$true_outcome_factor))
mse_value <- mean((all_pred$true_outcome_numeric - all_pred$predicted_value)^2, na.rm = TRUE)
print(paste("Mean Squared Error (Brier Score):", format(mse_value, digits = 4, nsmall = 4)))


#################################################
## Classification Metrics at Youden's J Optimal Threshold
#################################################
print("--- Calculating Metrics at Youden's J Optimal Threshold ---")

if (!is.null(roc_obj) && inherits(roc_obj, "roc")) {
  coords_youden <- coords(roc_obj, "best",
                          ret = c("threshold", "sensitivity", "specificity",
                                  "ppv", "npv", "accuracy"),
                          best.method = "youden", transpose = FALSE)
  
  if (nrow(coords_youden) > 0) {
    best_threshold        <- coords_youden$threshold[1]
    sensitivity_at_best   <- coords_youden$sensitivity[1]
    specificity_at_best   <- coords_youden$specificity[1]
    ppv_at_best           <- coords_youden$ppv[1]
    npv_at_best           <- coords_youden$npv[1]
    accuracy_at_best      <- coords_youden$accuracy[1]
    youden_j_value        <- sensitivity_at_best + specificity_at_best - 1
    misclass_at_best      <- 1 - accuracy_at_best
    
    print(paste("Optimal Threshold (Youden's J):", format(best_threshold, digits = 4)))
    print(paste("Youden's J Index Value at this threshold:", format(youden_j_value, digits = 4)))
    # print(paste("  Sensitivity at this threshold:", format(sensitivity_at_best, digits = 4))) # Covered by confusionMatrix
    # print(paste("  Specificity at this threshold:", format(specificity_at_best, digits = 4))) # Covered by confusionMatrix
    # print(paste("  PPV at this threshold:", format(ppv_at_best, digits = 4))) # Covered by confusionMatrix
    # print(paste("  NPV at this threshold:", format(npv_at_best, digits = 4))) # Covered by confusionMatrix
    # print(paste("  Accuracy at this threshold:", format(accuracy_at_best, digits = 4))) # Covered by confusionMatrix
    print(paste("  Misclassification Rate at this threshold:", format(misclass_at_best, digits = 4)))
    
    
    all_pred$predicted_class_youden <- factor(
      ifelse(all_pred$predicted_value >= best_threshold, 1, 0),
      levels = c(0, 1)
    )
    
    if ("1" %in% all_pred$true_outcome_factor) {
      print(paste("--- Confusion Matrix & Detailed Statistics (Threshold:", format(best_threshold, digits = 4), ") ---"))
      print("Counts (Predicted vs Actual):")
      print(table(Predicted_at_Youden_Thresh = all_pred$predicted_class_youden, Actual = all_pred$true_outcome_factor))
      
      conf_matrix_youden <- tryCatch({
        confusionMatrix(data = all_pred$predicted_class_youden,
                        reference = all_pred$true_outcome_factor,
                        positive = "1")
      }, error = function(e) {
        warning("Could not generate confusion matrix with Youden's J threshold. Error: ", e$message)
        NULL
      })
      if (!is.null(conf_matrix_youden)) print(conf_matrix_youden)
    } else {
      print("Positive class '1' not found in true_outcome_factor. Cannot calculate full confusion matrix.")
    }
  } else {
    print("Could not determine optimal threshold using Youden's J (coords returned no rows).")
  }
} else {
  print("ROC object ('roc_obj') not available or invalid, skipping Youden's J and related metrics.")
}

#################################################
## Choose own threshold
#################################################

coords_spec90 <- coords(roc_obj, x = 0.90, input = "specificity", ret = "all", transpose = FALSE)

# Print main performance metrics
cat("Specificity:", coords_spec90$specificity, "\n")
cat("Sensitivity:", coords_spec90$sensitivity, "\n")
cat("Accuracy:", coords_spec90$accuracy, "\n")
cat("Misclassification rate:", round(coords_spec90$`1-accuracy`, 4), "\n")


#################################################
## ROC Curve Plot
#################################################
print("--- Generating ROC Curve Plot ---")
if (!is.null(roc_obj) && inherits(roc_obj, "roc")) {
  # Updated AUC text to include CI
  auc_ci_formatted <- paste0("AUC = ", format(auc_value, digits = 3, nsmall = 3),
                             " (95% CI: ", round(auc_ci[1], 3), " - ", round(auc_ci[3], 3), ")")
  roc_plot <- ggroc(roc_obj, colour = 'darkblue', size = 1) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey50") +
    annotate("text", x = 0.55, y = 0.15, label = auc_ci_formatted, hjust = 0, size = 3.5) + # Adjusted x for longer text
    xlab("1 - Specificity (False Positive Rate)") +
    ylab("Sensitivity (True Positive Rate)") +
    ggtitle("Cross-Validated ROC Curve") +
    theme_bw() +
    coord_fixed()
  print(roc_plot)
  roc_plot_filename <- "07. Graph/03. Paper 3/logit(multi)_m1_lasso_ROC_v06.png"
  ggsave(filename = roc_plot_filename, plot = roc_plot, width = 5, height = 5, units = "in", dpi = 300)
  print(paste("ROC Curve plot saved to:", roc_plot_filename))
} else {
  print("ROC object not available, skipping ROC plot generation.")
}

#################################################
## Calibration Plot
#################################################
print("--- Generating Calibration Plot ---")

cal_plot <- all_pred %>%
  mutate(true_outcome = as.numeric(as.character(true_outcome))) %>%
  ggplot(aes(x = predicted_value, y = true_outcome)) +
  geom_jitter(width = 0.005, height = 0.05, size = 0.5, alpha = 0.1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
  geom_smooth(method = "gam", formula = y ~ x, color = "blue", se = TRUE) +
  scale_y_continuous(name = "Observed cardiometabolic multimorbidity", breaks = c(0, 1)) +
  scale_x_continuous(name = "Predicted cardiometabolic multimorbidity risk") +
  theme_minimal() %>%
  print()

ggsave(filename = "07. Graph/03. Paper 3/logit(multi)_m1_lasso_v06.png",
       plot = cal_plot,
       width = 7, height = 5, units = "in", dpi = 300)
print("Calibration plot saved.")

#################################################
## Compile Coefficients/ORs
#################################################
# ... (ORs summary code remains the same) ...
all_or <- bind_rows(lapply(1:length(cv_results), function(i) {
  res <- cv_results[[i]]$coefficients_ORs
  if (!is.null(res) && nrow(res) > 0) {
    res$Fold <- i
    return(res)
  } else {
    return(NULL)
  }
}))

if (!is.null(all_or) && nrow(all_or) > 0) {
  order <- c( 'time_days',
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
  
  or_summary <- all_or %>%
    filter(!is.na(Predictor)) %>%
    group_by(Predictor) %>%
    summarise(
      median_OR = median(OR_median, na.rm = TRUE),
      min_OR = min(OR_median, na.rm = TRUE),
      max_OR = max(OR_median, na.rm = TRUE),
      median_OR_LCI = median(OR_LCI, na.rm = TRUE),
      median_OR_UCI = median(OR_UCI, na.rm = TRUE),
      n_folds = n(),
      .groups = 'drop'
    ) %>%
    mutate(Predictor = factor(Predictor, levels = intersect(order, unique(Predictor)))) %>%
    arrange(Predictor) %>%
    select(Predictor, median_OR, min_OR, max_OR, median_OR_LCI, median_OR_UCI, n_folds)
  
  print("--- Summary of OR Distribution (Custom Order) ---")
  print(as.data.frame(or_summary))
} else {
  print("No coefficient/OR summaries to process.")
}


#################################################
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
  scale_y_continuous(name = "Observed cardiometabolic multimorbidity", breaks = c(0, 1)) +
  scale_x_continuous(name = "Predicted probability") +
  theme_minimal() %>%
  print()
print(cal_plot)

ggsave(filename = "07. Graph/03. Paper 3/logit(multi)_m1_lasso(1)_v06.png",
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
  mutate(bin = ntile(p, 8)) %>%
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
  labs(x = "Predicted probability", y = "Observed probability") +
  theme_gray(base_size = 12)

print(cal_plot)


# Save
ggsave("07. Graph/03. Paper 3/logit(multi)_m1_lasso(2)_v06.png",
       plot = cal_plot, width = 6, height = 6, units = "in", dpi = 300)

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