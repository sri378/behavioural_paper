
pkgs <- c("lavaan", "semTools", "dplyr", "readr", "openxlsx",
          "ggplot2", "tidyr", "stringr")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p, repos="https://cran.r-project.org")
}
suppressPackageStartupMessages({
  library(lavaan); library(semTools); library(dplyr)
  library(readr);  library(openxlsx); library(ggplot2)
  library(tidyr);  library(stringr)
})

# PATHS
BASE <- "."
args       <- commandArgs(trailingOnly = FALSE)
script_dir <- dirname(sub("--file=", "", args[grep("--file=", args)]))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "scripts"
BASE       <- file.path(script_dir, "..")

DATA_PATH  <- file.path(BASE, "data", "processed", "analysis_ready_clean.csv")
TABLE_DIR  <- file.path(BASE, "outputs", "tables")
FIG_DIR    <- file.path(BASE, "outputs", "figures")

dir.create(TABLE_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIG_DIR,   showWarnings = FALSE, recursive = TRUE)

cat("=" , strrep("=", 59), "\n")
cat("LOADING DATA\n")
cat("=" , strrep("=", 59), "\n")

df <- read_csv(DATA_PATH, show_col_types = FALSE)
cat("Rows:", nrow(df), "  Cols:", ncol(df), "\n")
cat("Group sizes:\n")
print(table(df$MOD_label))

# All items available
COG_ITEMS    <- c("C1","C2","C3","C4","C5")
REG_ITEMS    <- c("R1","R2","R3","R4","R5")
NORM_ITEMS   <- c("N1","N2","N3","N4","N5","N6")
INTENT_ITEMS <- c("I1","I2","I3","I4","I5","I6")

ordered_items <- c(COG_ITEMS, REG_ITEMS, c("N1","N2","N3","N4","N5"), INTENT_ITEMS)

# STEP 1: INITIAL CFA (all items, WLSMV)
cat("\n", strrep("=", 60), "\n")
cat("STEP 1: INITIAL CFA — ALL ITEMS\n")
cat(strrep("=", 60), "\n")

cfa_init_model <- '
  L_COG  =~ C1 + C2 + C3 + C4 + C5
  L_REG  =~ R1 + R2 + R3 + R4 + R5
  L_NORM =~ N1 + N2 + N3 + N4 + N5
  L_INT  =~ I1 + I2 + I3 + I4 + I5 + I6
'

fit_cfa_init <- cfa(
  cfa_init_model, data = df, std.lv = TRUE,
  ordered = ordered_items, estimator = "WLSMV"
)

cat("\nFit indices (initial CFA):\n")
fit_idx_init <- fitMeasures(fit_cfa_init,
  c("cfi","tli","rmsea","rmsea.ci.lower","rmsea.ci.upper","srmr","wrmr"))
print(round(fit_idx_init, 3))

cat("\nStandardised loadings:\n")
loadings_init <- parameterEstimates(fit_cfa_init, standardized = TRUE) %>%
  filter(op == "=~") %>%
  select(lhs, rhs, std.all, pvalue) %>%
  rename(Factor = lhs, Item = rhs, StdLoading = std.all, p = pvalue)
print(as.data.frame(loadings_init), row.names = FALSE)

# Identify weak items (loading < 0.4)
weak_items <- loadings_init %>% filter(StdLoading < 0.40)
if (nrow(weak_items) > 0) {
  cat("\n⚠ Weak loadings (<0.40):\n")
  print(weak_items)
} else {
  cat("\n✓ All loadings ≥ 0.40\n")
}

# Modification indices
mi_init <- modindices(fit_cfa_init, sort. = TRUE, maximum.number = 15)
cat("\nTop modification indices:\n")
print(mi_init[1:15,])
# STEP 2: REFINED CFA — trim weak items, handle MIs
cat("\n", strrep("=", 60), "\n")
cat("STEP 2: REFINED CFA\n")
cat(strrep("=", 60), "\n")


cfa_refined_model <- '
  L_COG  =~ C1 + C2 + C3 + C4 + C5
  L_REG  =~ R1 + R2 + R3 + R4 + R5
  L_NORM =~ N1 + N2 + N3 + N4 + N5
  L_INT  =~ I1 + I2 + I3 + I4 + I5 + I6
'


fit_cfa_ref <- cfa(
  cfa_refined_model, data = df, std.lv = TRUE,
  ordered = ordered_items, estimator = "WLSMV"
)

cat("\nFit indices (refined CFA):\n")
fit_idx_ref <- fitMeasures(fit_cfa_ref,
  c("cfi","tli","rmsea","rmsea.ci.lower","rmsea.ci.upper","srmr","wrmr"))
print(round(fit_idx_ref, 3))

cat("\nStd loadings (refined):\n")
loadings_ref <- parameterEstimates(fit_cfa_ref, standardized = TRUE) %>%
  filter(op == "=~") %>%
  select(lhs, rhs, std.all, pvalue) %>%
  rename(Factor=lhs, Item=rhs, StdLoading=std.all, p=pvalue)
print(as.data.frame(loadings_ref), row.names = FALSE)

# Composite reliability (AVE)
cat("\nComposite Reliability & AVE:\n")
rel_out <- tryCatch(reliability(fit_cfa_ref), error = function(e) NULL)
if (!is.null(rel_out)) print(rel_out)

# Save CFA loadings
write.csv(loadings_ref,
  file.path(TABLE_DIR, "cfa_loadings_refined.csv"), row.names = FALSE)
cat("✓ Loadings saved\n")

cat("\n", strrep("=", 60), "\n")
cat("STEP 3: FULL SEM (COG + REG + NORM → Intention)\n")
cat(strrep("=", 60), "\n")

sem_main <- '
  # Measurement model
  L_COG  =~ C1 + C2 + C3 + C4 + C5
  L_REG  =~ R1 + R2 + R3 + R4 + R5
  L_NORM =~ N1 + N2 + N3 + N4 + N5
  L_INT  =~ I1 + I2 + I3 + I4 + I5 + I6

  # Structural paths (barriers → entrepreneurial intention)
  L_INT ~ L_COG + L_REG + L_NORM
'

fit_sem <- sem(
  sem_main, data = df, std.lv = TRUE,
  ordered = ordered_items, estimator = "WLSMV"
)

cat("\nSEM fit indices:\n")
sem_fit <- fitMeasures(fit_sem,
  c("cfi","tli","rmsea","rmsea.ci.lower","rmsea.ci.upper","srmr","wrmr"))
print(round(sem_fit, 3))

cat("\nStructural paths:\n")
sem_paths <- parameterEstimates(fit_sem, standardized = TRUE) %>%
  filter(op == "~") %>%
  select(lhs, rhs, est, se, z, pvalue, std.all) %>%
  rename(Outcome=lhs, Predictor=rhs, Beta=est, SE=se,
         z_val=z, p=pvalue, Std_Beta=std.all)
print(sem_paths)

write.csv(sem_paths,
  file.path(TABLE_DIR, "sem_structural_paths.csv"), row.names = FALSE)

# Rank predictors by absolute standardized beta
cat("\n── Predictor Ranking by |Std Beta| ──\n")
sem_paths %>%
  arrange(desc(abs(Std_Beta))) %>%
  select(Predictor, Std_Beta, p) %>%
  print()


cat("\n", strrep("=", 60), "\n")
cat("STEP 4: MEASUREMENT INVARIANCE (EduYes vs EduNo)\n")
cat(strrep("=", 60), "\n")

fit_config <- cfa(
  cfa_refined_model, data = df, group = "MOD_label",
  std.lv = TRUE, ordered = ordered_items, estimator = "WLSMV",
  group.equal = c()
)


fit_metric <- cfa(
  cfa_refined_model, data = df, group = "MOD_label",
  std.lv = TRUE, ordered = ordered_items, estimator = "WLSMV",
  group.equal = c("loadings")
)


fit_scalar <- cfa(
  cfa_refined_model, data = df, group = "MOD_label",
  std.lv = TRUE, ordered = ordered_items, estimator = "WLSMV",
  group.equal = c("loadings", "thresholds")
)

# Model comparison
cat("\nInvariance model fit summary:\n")
inv_compare <- compareFit(fit_config, fit_metric, fit_scalar)
summary(inv_compare)

# RMSEA/CFI for each
for (label in c("Configural","Metric","Scalar")) {
  mdl <- switch(label, Configural=fit_config, Metric=fit_metric, Scalar=fit_scalar)
  idx <- fitMeasures(mdl, c("cfi","rmsea","srmr"))
  cat(sprintf("  %-12s CFI=%.3f  RMSEA=%.3f  SRMR=%.3f\n",
              label, idx["cfi"], idx["rmsea"], idx["srmr"]))
}

# Delta CFI check: ΔcFI < .010 → acceptable invariance
cfi_config <- fitMeasures(fit_config, "cfi")
cfi_metric <- fitMeasures(fit_metric, "cfi")
cfi_scalar <- fitMeasures(fit_scalar, "cfi")
cat(sprintf("\n  ΔCFI Configural→Metric : %.4f (threshold < .010)\n",
            abs(cfi_config - cfi_metric)))
cat(sprintf("  ΔCFI Metric→Scalar     : %.4f (threshold < .010)\n",
            abs(cfi_metric - cfi_scalar)))

inv_summary <- data.frame(
  Model     = c("Configural","Metric","Scalar"),
  CFI       = round(c(cfi_config, cfi_metric, cfi_scalar), 4),
  RMSEA     = round(c(
    fitMeasures(fit_config,"rmsea"),
    fitMeasures(fit_metric,"rmsea"),
    fitMeasures(fit_scalar,"rmsea")), 4),
  delta_CFI = round(c(NA,
    abs(cfi_config - cfi_metric),
    abs(cfi_metric - cfi_scalar)), 4)
)
print(inv_summary)
write.csv(inv_summary,
  file.path(TABLE_DIR, "measurement_invariance.csv"), row.names = FALSE)
cat("✓ Invariance table saved\n")

cat("\n", strrep("=", 60), "\n")
cat("STEP 5: MULTI-GROUP SEM — MODERATION ANALYSIS\n")
cat(strrep("=", 60), "\n")

sem_mg_model <- '
  L_COG  =~ C1 + C2 + C3 + C4 + C5
  L_REG  =~ R1 + R2 + R3 + R4 + R5
  L_NORM =~ N1 + N2 + N3 + N4 + N5
  L_INT  =~ I1 + I2 + I3 + I4 + I5 + I6
  L_INT ~ L_COG + L_REG + L_NORM
'

# Free model (unconstrained structural paths per group)
fit_mg_free <- sem(
  sem_mg_model, data = df, group = "MOD_label",
  std.lv = TRUE, ordered = ordered_items, estimator = "WLSMV",
  group.equal = c("loadings", "thresholds")
)

# Constrained model (equal structural paths = no moderation)
fit_mg_constrained <- sem(
  sem_mg_model, data = df, group = "MOD_label",
  std.lv = TRUE, ordered = ordered_items, estimator = "WLSMV",
  group.equal = c("loadings", "thresholds", "regressions")
)

cat("\nMulti-group SEM: Structural paths by group\n")
mg_paths <- parameterEstimates(fit_mg_free, standardized = TRUE) %>%
  filter(op == "~") %>%
  select(lhs, rhs, group, std.all, pvalue) %>%
  rename(Outcome=lhs, Predictor=rhs, Group=group,
         Std_Beta=std.all, p=pvalue) %>%
  mutate(Group = ifelse(Group == 1, "EduYes", "EduNo"))
print(mg_paths)

write.csv(mg_paths,
  file.path(TABLE_DIR, "multigroup_paths.csv"), row.names = FALSE)

# Chi-square difference test (constrained vs free paths)
cat("\nChi-square difference test (constrained vs free structural paths):\n")
chi_diff <- tryCatch(
  lavTestScore(fit_mg_free),
  error = function(e) NULL
)

# LRT using lavTestLRT
lrt_result <- tryCatch(
  lavTestLRT(fit_mg_free, fit_mg_constrained, method = "satorra.bentler.2010"),
  error = function(e) {
    cat("  Note: Using standard LRT\n")
    lavTestLRT(fit_mg_free, fit_mg_constrained)
  }
)
print(lrt_result)

lrt_df <- as.data.frame(lrt_result)
write.csv(lrt_df,
  file.path(TABLE_DIR, "multigroup_lrt.csv"), row.names = TRUE)
cat("\n", strrep("=", 60), "\n")
cat("STEP 6: VISUALISING MODERATION\n")
cat(strrep("=", 60), "\n")

plot_data <- mg_paths %>%
  mutate(
    Predictor = recode(Predictor,
      "L_COG"  = "Cognitive",
      "L_REG"  = "Regulative",
      "L_NORM" = "Normative")
  )

p_forest <- ggplot(plot_data, aes(x = Std_Beta, y = Predictor,
                                   color = Group, shape = Group)) +
  geom_point(size = 5, position = position_dodge(width = 0.5)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  scale_color_manual(values = c("EduYes" = "#2c7bb6", "EduNo" = "#d7191c"),
                     labels = c("EduYes" = "Has done Entrepreneurship Module",
                                "EduNo"  = "Has NOT done Module")) +
  scale_shape_manual(values = c("EduYes" = 16, "EduNo" = 17),
                     labels = c("EduYes" = "Has done Entrepreneurship Module",
                                "EduNo"  = "Has NOT done Module")) +
  labs(
    title    = "Multi-Group SEM: Moderation by Entrepreneurial Education",
    subtitle = "Standardised path coefficients (β) to Entrepreneurial Intention",
    x        = "Standardised Path Coefficient (β)",
    y        = "Barrier Type",
    color    = "Education Group",
    shape    = "Education Group"
  ) +
  theme_bw(base_size = 13) +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold"))

ggsave(file.path(FIG_DIR, "02_moderation_forest.png"), p_forest,
       width = 9, height = 5, dpi = 200)
cat("✓ Forest plot saved\n")

mg_wide <- mg_paths %>%
  select(Predictor, Group, Std_Beta) %>%
  pivot_wider(names_from = Group, values_from = Std_Beta) %>%
  mutate(
    Difference = EduYes - EduNo,
    Stronger_in = ifelse(abs(EduYes) > abs(EduNo), "EduYes", "EduNo")
  )

cat("\n── Group Comparison ──\n")
print(mg_wide)
write.csv(mg_wide,
  file.path(TABLE_DIR, "moderation_comparison.csv"), row.names = FALSE)

cat("\n", strrep("=", 60), "\n")
cat("STEP 8: FULL FIT SUMMARY FOR PAPER\n")
cat(strrep("=", 60), "\n")

models_list <- list(
  "CFA_Init"    = fit_cfa_init,
  "CFA_Refined" = fit_cfa_ref,
  "SEM_Full"    = fit_sem,
  "MG_Free"     = fit_mg_free,
  "MG_Const"    = fit_mg_constrained
)

fit_summary <- lapply(names(models_list), function(nm) {
  idx <- fitMeasures(models_list[[nm]],
    c("chisq","df","cfi","tli","rmsea","rmsea.ci.lower","rmsea.ci.upper","srmr"))
  c(Model = nm, round(idx, 3))
})
fit_summary_df <- as.data.frame(do.call(rbind, fit_summary))
print(fit_summary_df)
write.csv(fit_summary_df,
  file.path(TABLE_DIR, "all_model_fit_summary.csv"), row.names = FALSE)
cat("✓ Fit summary saved\n")

cat("\n", strrep("=", 60), "\n")
cat("✓ R SEM ANALYSIS COMPLETE\n")
cat(strrep("=", 60), "\n")
cat("  Outputs:\n")
cat("    tables/cfa_loadings_refined.csv\n")
cat("    tables/sem_structural_paths.csv\n")
cat("    tables/measurement_invariance.csv\n")
cat("    tables/multigroup_paths.csv\n")
cat("    tables/multigroup_lrt.csv\n")
cat("    tables/moderation_comparison.csv\n")
cat("    tables/all_model_fit_summary.csv\n")
cat("    figures/02_moderation_forest.png\n")
