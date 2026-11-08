## =============================================================
## pipeline.R
## A.N.D.R.E. — Adaptive Neural Dixon-coles Regularized Ensemble
## Modelo predictivo Mundial 2026
## Autor: @s7ex.j
##
## Corre todos los scripts en orden. Cada uno se ejecuta en un entorno
## limpio (sin variables de los anteriores): se comunican solo por los
## archivos de data/ y output/, así que cualquiera se puede correr solo.
##
## El modelo bayesiano (06) es opcional: se salta si faltan brms/rstan,
## si no hay compilador de C++ (Rtools en Windows) o si se define la
## variable de entorno ANDRE_SKIP_BAYES=1. Si falla al correr, el
## pipeline sigue: no entra al ensamble, solo se reporta aparte.
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
  "scripts/09f_draws_and_segments.R",
  "scripts/10_build_team_state.R"
)
OPTIONAL <- "scripts/06_model_bayesian_mcmc.R"

bayes_available <- function() {
  if (identical(Sys.getenv("ANDRE_SKIP_BAYES"), "1")) return("ANDRE_SKIP_BAYES=1")
  for (p in c("brms", "rstan")) {
    if (!requireNamespace(p, quietly = TRUE)) return(paste("falta el paquete", p))
  }
  if (requireNamespace("pkgbuild", quietly = TRUE) &&
      !isTRUE(suppressMessages(pkgbuild::has_build_tools(debug = FALSE)))) {
    return("no hay compilador de C++ (en Windows: instala Rtools)")
  }
  TRUE
}

t_start <- Sys.time()
for (s in scripts) {
  cat("\n====================================================\n>>", s,
      "\n====================================================\n")
  if (s == OPTIONAL) {
    ok <- bayes_available()
    if (!isTRUE(ok)) {
      cat("Se omite el modelo bayesiano:", ok, "\n")
      # sin esto, 08 tomaría predicciones bayesianas viejas de otra corrida
      if (file.exists("data/preds_bayes.csv")) file.remove("data/preds_bayes.csv")
      next
    }
    res <- tryCatch({ source(s, local = new.env()); TRUE },
                    error = function(e) {
                      cat("El modelo bayesiano falló y se omite:", conditionMessage(e), "\n")
                      if (file.exists("data/preds_bayes.csv")) file.remove("data/preds_bayes.csv")
                      FALSE
                    })
    next
  }
  source(s, local = new.env())
}

cat("\nPipeline completo en", round(as.numeric(Sys.time() - t_start, units = "mins"), 1), "min.\n")
