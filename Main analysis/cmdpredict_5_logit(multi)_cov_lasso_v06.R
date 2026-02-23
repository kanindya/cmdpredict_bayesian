# PROJECT : PREDICTION OF MULTI CARDIOMETABOLIC MULTIMORBIDITY
# DATE    : 27 APRIL 2025
# NOTES   : STEP 5: Subgroup Analysis (Short Names, Cleaned)

#################################################
## Load packages
#################################################

library(dplyr)
library(tidyr)
library(tibble)
# library(haven) # Likely needed only if loading raw .dta again

#################################################
## Setup - Define File Paths
#################################################

# --- Set working directory ---
setwd("/Users/kanyaanindya/Documents/10. SCAPIS/") # Adjust if needed

# --- Define Input/Output file names ---
inter_data <- "08. Temp/03. Paper 3"
results_dir <- "05. Result"
excel_dir <- "04. Excel/03. Paper 3"
dir.create(excel_dir, showWarnings = FALSE, recursive = TRUE)

# Input files (from Step 4)
f_pred_rds <- file.path(inter_data, "logit(multi)_cov_pred_lasso_v05.rds")
f_prob <- file.path(inter_data, "logit(multi)_cov_prob_lasso_v05.rds") 
f_stanfit <- file.path(inter_data, "logit(multi)_full_stanfit_v05.rds")
f_inputs <- file.path(inter_data, "logit(multi)_full_stan_inputs_v05.rds")

# Output files for Step 5
f_subgrp_rds <- file.path(results_dir, "logit(multi)_subgroup_pred_v05.rds") 
f_subgrp_csv <- file.path(excel_dir, "logit(multi)_subgroup_pred_v05.csv") 
f_or_csv <- file.path(excel_dir, "logit(multi)_or_v05.csv") 

#################################################
## Load Data from Step 4
#################################################
print("Loading results from Step 4...")
if (!file.exists(f_pred_rds) || !file.exists(f_prob)) stop("Input files not found.")
df_results <- readRDS(f_pred_rds)
prob_ind <- readRDS(f_prob)
if (nrow(df_results) != nrow(prob_ind)) stop("Loaded data row mismatch.")
if (!"subject" %in% names(df_results)) stop("df_results needs 'subject' ID column")
print("Data loaded.")

#################################################
## Define Subgroups & Align Data
#################################################
print("Defining subgroups and aligning data...")
df_grp <- df_results %>%
  mutate(
    age_grp = case_when( age1 >= 50 & age1 < 60 ~ "50-59", age1 >= 60 & age1 < 65 ~ "60-64", TRUE ~ "Other" ),
    sex_lbl = ifelse(sex1_2 == 0, "Female", "Male"),
    smoke_lbl = ifelse(smoking1_2 == 0, "NonSmoker", "Smoker"),
    htn_lbl = case_when(
      sbp_mean >= 130 | dbp_mean >= 85 | drug_hypertension_2 == 1 ~ "HyperT",
      sbp_mean < 130 & dbp_mean < 85 & drug_hypertension_2 == 0  ~ "NoHyperT", TRUE ~ NA_character_),
    hdl_cat = case_when( 
      sex1_2 == 0 & hdl < 1.29  ~ "HDL <50mg/dL (F)", sex1_2 == 0 & hdl >= 1.29 ~ "HDL ≥50mg/dL (F)",
      sex1_2 == 1 & hdl < 1.03  ~ "HDL <40mg/dL (M)", sex1_2 == 1 & hdl >= 1.03 ~ "HDL ≥40mg/dL (M)",
      TRUE ~ NA_character_ ),
    waist_cat = case_when(
      sex1_2 == 0 & waist < 88  ~ "Waist <88cm (F)",  sex1_2 == 0 & waist >= 88 ~ "Waist ≥88cm (F)",
      sex1_2 == 1 & waist < 102 ~ "Waist <102cm (M)", sex1_2 == 1 & waist >= 102~ "Waist ≥102cm (M)",
      TRUE ~ NA_character_ ),
    born_lbl = ifelse(born1_2 == 0, "Born Sweden", "Born Elsewhere") # Assuming 0=Sweden(Ref)
  ) %>%
  filter(age_grp %in% c("50-59", "60-64") & !is.na(htn_lbl) & !is.na(hdl_cat) & !is.na(waist_cat) & !is.na(born_lbl))

indices_kept <- match(df_grp$subject, df_results$subject)
prob_ind_filt <- prob_ind[indices_kept, , drop = FALSE]
rm(prob_ind, df_results); gc()
print("Subgroups defined and data aligned.")

#################################################
## Assign Strata Group Label to Each Individual
#################################################
print("Assigning unique group labels to each individual...")

df_grp_with_labels <- df_grp %>%
  mutate(
    group_number = group_indices(., age_grp, sex_lbl, smoke_lbl, htn_lbl, born_lbl, hdl_cat, waist_cat),
    group_label = paste(group_number)
  )

# Optional: Reorder columns for clarity, putting the new labels first
df_grp_with_labels <- df_grp_with_labels %>%
  select(subject, group_label, age_grp, sex_lbl, smoke_lbl, htn_lbl, born_lbl, hdl_cat, waist_cat, everything(), -group_number)

# --- Display the first few rows of the result ---
print("--- Individual data with assigned group labels (first 10 rows) ---")
print(head(df_grp_with_labels, 10))

#################################################
## Calculate Point Estimates (Mean Prob, RR)
#################################################
print("Calculating point estimates...")
sgrp_sum <- df_grp %>%
  group_by(age_grp, sex_lbl, smoke_lbl, htn_lbl, born_lbl, hdl_cat, waist_cat) %>% # Added born_lbl
  summarise(count = n(), avg_prob = mean(mean_pred, na.rm = TRUE), .groups = 'drop')

# --- Reference Group Definitions for BINARY HDL/Waist ---
ref_age<-"50-59"; ref_smoke<-"NonSmoker"; ref_htn<-"NoHyperT"; ref_born <- "Born Sweden" # Added ref_born
ref_sex_f<-"Female"; ref_hdl_f<-"HDL ≥50mg/dL (F)"; ref_waist_f<-"Waist <88cm (F)"
ref_sex_m<-"Male";   ref_hdl_m<-"HDL ≥40mg/dL (M)"; ref_waist_m<-"Waist <102cm (M)"

ref_prob_f <- sgrp_sum %>% filter(sex_lbl==ref_sex_f & age_grp==ref_age & smoke_lbl==ref_smoke & htn_lbl==ref_htn & born_lbl==ref_born & hdl_cat==ref_hdl_f & waist_cat==ref_waist_f) %>% pull(avg_prob)
ref_prob_m <- sgrp_sum %>% filter(sex_lbl==ref_sex_m & age_grp==ref_age & smoke_lbl==ref_smoke & htn_lbl==ref_htn & born_lbl==ref_born & hdl_cat==ref_hdl_m & waist_cat==ref_waist_m) %>% pull(avg_prob)

sgrp_res <- sgrp_sum %>% mutate(rr = case_when(sex_lbl==ref_sex_f & !is.na(ref_prob_f) & ref_prob_f > 0 ~ avg_prob/ref_prob_f, sex_lbl==ref_sex_m & !is.na(ref_prob_m) & ref_prob_m > 0 ~ avg_prob/ref_prob_m, TRUE~NA_real_))
rm(sgrp_sum)

#################################################
## Calculate 95% CIs for Abs Prob and RR
#################################################
print("Calculating 95% CIs...")
grouping_vars <- c("subject", "age_grp", "sex_lbl", "smoke_lbl", "htn_lbl", "born_lbl", "hdl_cat", "waist_cat") # Added born_lbl
prob_long <- bind_cols(df_grp %>% select(all_of(grouping_vars)), as_tibble(prob_ind_filt, .name_repair="minimal")) %>%
  pivot_longer(cols = !all_of(grouping_vars), names_to = "sample_id", values_to = "prob")
rm(prob_ind_filt, df_grp); gc()

sgrp_samp <- prob_long %>%
  group_by(age_grp, sex_lbl, smoke_lbl, htn_lbl, born_lbl, hdl_cat, waist_cat, sample_id) %>% # Added born_lbl
  summarise(avg_prob_sample = mean(prob, na.rm = TRUE), .groups = "drop_last")
rm(prob_long); gc()

ref_samp_f <- sgrp_samp %>% filter(sex_lbl==ref_sex_f & age_grp==ref_age & smoke_lbl==ref_smoke & htn_lbl==ref_htn & born_lbl==ref_born & hdl_cat==ref_hdl_f & waist_cat==ref_waist_f) %>% ungroup() %>% select(sample_id, refprob_f=avg_prob_sample)
ref_samp_m <- sgrp_samp %>% filter(sex_lbl==ref_sex_m & age_grp==ref_age & smoke_lbl==ref_smoke & htn_lbl==ref_htn & born_lbl==ref_born & hdl_cat==ref_hdl_m & waist_cat==ref_waist_m) %>% ungroup() %>% select(sample_id, refprob_m=avg_prob_sample)

all_cis <- sgrp_samp %>%
  ungroup() %>%
  left_join(ref_samp_f, by = "sample_id") %>% left_join(ref_samp_m, by = "sample_id") %>%
  mutate(rr_samp = case_when(
    sex_lbl==ref_sex_f & !is.na(refprob_f) & refprob_f > 0 ~ avg_prob_sample/refprob_f,
    sex_lbl==ref_sex_m & !is.na(refprob_m) & refprob_m > 0 ~ avg_prob_sample/refprob_m, TRUE ~ NA_real_)) %>%
  group_by(age_grp, sex_lbl, smoke_lbl, htn_lbl, born_lbl, hdl_cat, waist_cat) %>% # Added born_lbl
  summarise(abs_lowci=quantile(avg_prob_sample,0.025,na.rm=T), abs_upci=quantile(avg_prob_sample,0.975,na.rm=T),
            rr_lowci=quantile(rr_samp,0.025,na.rm=T), rr_upci=quantile(rr_samp,0.975,na.rm=T), .groups="drop")
rm(sgrp_samp, ref_samp_f, ref_samp_m); gc()

sgrp_final <- sgrp_res %>% left_join(all_cis, by = c("age_grp","sex_lbl","smoke_lbl","htn_lbl","born_lbl","hdl_cat","waist_cat")) # Added born_lbl
rm(sgrp_res, all_cis); gc()
print("CI calculation finished.")

#################################################
## Calculate Odds Ratios from Model Coefficients
#################################################
print("Calculating Odds Ratios...")
fit_full <- readRDS(f_stanfit); inputs <- readRDS(f_inputs)
pred_col <- inputs$predictor_col; cont_vars <- inputs$cont_vars; all_sds <- inputs$all_sds
beta_full <- rstan::extract(fit_full, pars = "beta")$beta; rm(fit_full, inputs)

# --- Define conversion factor ---
mmol_to_mgdl <- 38.67 

or_list <- list()
for (i in 1:length(pred_col)) {
  pred_name <- pred_col[i]; beta_samps <- beta_full[, i]
  med_b <- median(beta_samps,na.rm=T); q_b <- quantile(beta_samps,c(0.025,0.975),na.rm=T)
  or_med<-NA; or_lowci<-NA; or_upci<-NA; or_med_sd<-NA; or_low_sd<-NA; or_up_sd<-NA; interp<-NA
  
  if (pred_name %in% cont_vars) {
    sd_orig <- all_sds[pred_name]
    
    # --- MODIFICATION: Check for hdl/nonhdl ---
    if (pred_name %in% c("hdl", "nonhdl")) {
      interp <- "OR/(mg/dL)" # Set new interpretation
      
      # Calculate log(OR) per 1 mmol/L
      log_or_per_mmol_L_med   <- med_b / sd_orig
      log_or_per_mmol_L_low   <- q_b[1] / sd_orig
      log_or_per_mmol_L_high  <- q_b[2] / sd_orig
      
      # Convert to log(OR) per 1 mg/dL
      log_or_per_mg_dL_med  <- log_or_per_mmol_L_med / mmol_to_mgdl
      log_or_per_mg_dL_low  <- log_or_per_mmol_L_low / mmol_to_mgdl
      log_or_per_mg_dL_high <- log_or_per_mmol_L_high / mmol_to_mgdl
      
      # Exponentiate
      or_med   <- exp(log_or_per_mg_dL_med)
      or_lowci <- exp(log_or_per_mg_dL_low)
      or_upci  <- exp(log_or_per_mg_dL_high)
      
    } else {
      # Original calculation for other continuous vars (e.g., waist, sbp)
      interp <- "OR/unit"
      or_med   <- exp(med_b / sd_orig)
      or_lowci <- exp(q_b[1] / sd_orig)
      or_upci  <- exp(q_b[2] / sd_orig)
    }
    # --- END MODIFICATION ---
    
    # OR per SD remains the same regardless of unit conversion
    or_med_sd<-exp(med_b); or_low_sd<-exp(q_b[1]); or_up_sd<-exp(q_b[2])
    
  } else {
    interp <- "OR (1vs0)"; or_med<-exp(med_b); or_lowci<-exp(q_b[1]); or_upci<-exp(q_b[2])
  }
  
  or_list[[pred_name]] <- tibble(Pred=pred_name, Interp=interp, OR=or_med, OR_L=or_lowci, OR_U=or_upci,
                                 OR_SD=or_med_sd, OR_L_SD=or_low_sd, OR_U_SD=or_up_sd)
}
or_tbl <- bind_rows(or_list); rm(or_list, beta_full, pred_col, cont_vars, all_sds); gc()
print(or_tbl %>% mutate(across(where(is.numeric),~round(.x,3))))
write.csv(or_tbl, file=f_or_csv, row.names=F, na="")
print(paste("Odds Ratios saved to CSV:", f_or_csv))

#################################################
## Optional: Identify Missing Combinations
#################################################
print("Identifying missing combinations...")
age_levels<-c("50-59","60-64"); sex_levels<-c("Female","Male"); smoke_levels<-c("NonSmoker","Smoker"); htn_levels<-c("HyperT","NoHyperT")
born_levels <- c("Born Sweden", "Born Elsewhere") # New
hdl_lvlf<-c("HDL <50mg/dL (F)", "HDL ≥50mg/dL (F)"); hdl_lvlm<-c("HDL <40mg/dL (M)", "HDL ≥40mg/dL (M)") # New HDL levels
waist_lvlf<-c("Waist <88cm (F)", "Waist ≥88cm (F)"); waist_lvlm<-c("Waist <102cm (M)", "Waist ≥102cm (M)") # New Waist levels

allcomb_f<-tidyr::crossing(age_grp=age_levels,sex_lbl="Female",smoke_lbl=smoke_levels,htn_lbl=htn_levels,born_lbl=born_levels,hdl_cat=hdl_lvlf,waist_cat=waist_lvlf) # Added born_lbl
allcomb_m<-tidyr::crossing(age_grp=age_levels,sex_lbl="Male",smoke_lbl=smoke_levels,htn_lbl=htn_levels,born_lbl=born_levels,hdl_cat=hdl_lvlm,waist_cat=waist_lvlm) # Added born_lbl
all_comb<-bind_rows(allcomb_f,allcomb_m)
exist_comb<-sgrp_final %>% select(age_grp,sex_lbl,smoke_lbl,htn_lbl,born_lbl,hdl_cat,waist_cat) %>% distinct() # Added born_lbl
miss_comb<-anti_join(all_comb,exist_comb,by=names(exist_comb))
print(paste(nrow(miss_comb),"missing combinations found."))
if(nrow(miss_comb)>0) write.csv(miss_comb,file=f_miss_csv,row.names=F)
rm(list=c("all_comb","allcomb_f","allcomb_m","exist_comb","miss_comb","age_levels","sex_levels","smoke_levels","htn_levels","born_levels","hdl_lvlf","hdl_lvlm","waist_lvlf","waist_lvlm")); gc()

#################################################
## Optional: Identify Top/Bottom 5 Groups by Sex
#################################################
print("Identifying top/bottom 5 groups and saving...")
print("--- Top 5 Highest Probability Groups (by Sex) ---")
sgrp_final %>% group_by(sex_lbl) %>% slice_max(order_by=avg_prob,n=5) %>% ungroup() %>% print(n=20)
print("--- Top 5 Lowest Probability Groups (by Sex) ---")
sgrp_final %>% group_by(sex_lbl) %>% slice_min(order_by=avg_prob,n=5) %>% ungroup() %>% print(n=20)

#################################################
## Save Final Subgroup Summary Table & Display
#################################################
print("Saving final subgroup summary table...")
saveRDS(sgrp_final, file = f_subgrp_rds)
write.csv(sgrp_final, file = f_subgrp_csv, row.names = FALSE, na = "")
print(paste("Results saved to RDS:", f_subgrp_rds))
print(paste("Results saved to CSV:", f_subgrp_csv))

print("--- Final Results Table (First 20 Rows) ---")
print(as.data.frame(sgrp_final %>%
                      select(age_grp:count, avg_prob, abs_lowci, abs_upci, rr, rr_lowci, rr_upci) %>%
                      arrange(sex_lbl, age_grp, htn_lbl, smoke_lbl, hdl_cat, waist_cat) %>%
                      head(20)))

#################################################
## Percentile
#################################################
# Calculate the 5 quintile breaks
quintile_breaks <- quantile(sgrp_final$avg_prob, 
                            probs = c(0, 0.2, 0.4, 0.6, 0.8, 1.0), 
                            na.rm = TRUE)

# Print the cutoff values to your console
print("--- Cutoff values for 5 color groups (Quintiles) ---")
print(quintile_breaks)

print("Step 5 finished.")


