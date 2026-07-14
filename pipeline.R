## =============================================================
## pipeline.R
## A.N.D.R.E. — Adaptive Neural Dixon-coles Regularized Ensemble
## Modelo predictivo Mundial 2026
## Autor: @s7ex.j
## =============================================================

if (!dir.exists("output")) dir.create("output")

scripts <- c(
  "scripts/01_data_prep.R",
  "scripts/02_elo_ratings.R",
  "scripts/03_dixon_coles.R",
  "scripts/04_feature_engineering.R",
  "scripts/05_model_xgboost.R",
  "scripts/06_model_bayesian_mcmc.R",
  "scripts/07_model_deep_learning.R",
  "scripts/08_ensemble.R",
  "scripts/09_evaluation.R",
  "scripts/09b_evaluation_worldcup.R",
  "scripts/09c_significance_test.R",
  "scripts/09d_significance_full.R",
  "scripts/09e_calibration_diagnostics.R",
  "scripts/10_build_team_state.R"
)

for (s in scripts) {
  cat("\n====================================================\n>>", s, "\n====================================================\n")
  source(s)
}

cat("\nPipeline completo.\n")
