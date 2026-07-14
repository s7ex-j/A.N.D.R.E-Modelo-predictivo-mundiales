## =============================================================
## 06_model_bayesian_mcmc.R
## Modelo bayesiano jerárquico de Poisson (estilo Baio & Blangiardo,
## 2010) ajustado vía MCMC (Stan/brms). A diferencia de Dixon-Coles
## (MLE + ridge ad-hoc), aquí la regularización de selecciones con
## pocos partidos surge NATURALMENTE del "partial pooling" jerárquico:
## attack_i, defense_i ~ Normal(mu, sigma) con mu/sigma estimados
## de los datos. Salida: goles esperados con INTERVALOS DE CREDIBILIDAD
## (no solo un punto), y probabilidades 1X2 vía simulación posterior.
## =============================================================
suppressMessages({
  library(data.table)
  library(brms)
  library(rstan)
})
options(mc.cores = parallel::detectCores())

DATA_DIR <- "data"
model_dt <- fread(file.path(DATA_DIR, "model_features.csv"))
model_dt[, date := as.Date(date)]

# Ventana reciente para mantener el MCMC tratable en tiempo razonable
# (en tu máquina, con más tiempo/cores, puedes usar el dataset completo)
WINDOW_FROM <- as.Date("2024-09-01")
SPLIT_DATE  <- as.Date("2025-10-01")

dt <- model_dt[date >= WINDOW_FROM & date < as.Date("2026-01-01")]
train <- dt[date < SPLIT_DATE]
test  <- dt[date >= SPLIT_DATE]
cat("Train:", nrow(train), " | Test:", nrow(test), "\n")

# ---- Formato largo: cada partido -> 2 filas (goles del local, goles visita) ----
to_long <- function(d) {
  rbindlist(list(
    d[, .(match_id, date, team = home_team, opponent = away_team,
          goals = home_score, is_home = 1)],
    d[, .(match_id, date, team = away_team, opponent = home_team,
          goals = away_score, is_home = 0)]
  ))
}
train_long <- to_long(train)
train_long[, team := factor(team)]
train_long[, opponent := factor(opponent)]

## ---------------------------------------------------------------
## Modelo: goals ~ is_home + (1 | team) + (1 | opponent)
## team    -> fuerza de ataque (partial pooling)
## opponent-> fuerza de defensa (partial pooling)
## ---------------------------------------------------------------
priors <- c(
  prior(normal(0, 1),   class = "b"),
  prior(normal(0, 0.6), class = "sd", group = "team"),
  prior(normal(0, 0.6), class = "sd", group = "opponent")
)

t0 <- Sys.time()
fit_bayes <- brm(
  goals ~ is_home + (1 | team) + (1 | opponent),
  data = train_long, family = poisson(),
  prior = priors,
  chains = 1, iter = 400, warmup = 200, seed = 42,
  refresh = 0, backend = "rstan"
)
cat("Tiempo de ajuste MCMC:", round(as.numeric(Sys.time()-t0, units="mins"),2), "min\n")
print(summary(fit_bayes)$fixed)

## ---------------------------------------------------------------
## Predicción posterior para partidos de test: simulamos S=2000
## marcadores por partido y derivamos P(H)/P(D)/P(A) + goles esperados
## con incertidumbre.
## ---------------------------------------------------------------
predict_all <- function(test_dt, n_draws = 200) {
  nd_home <- data.frame(team = test_dt$home_team, opponent = test_dt$away_team, is_home = 1)
  nd_away <- data.frame(team = test_dt$away_team, opponent = test_dt$home_team, is_home = 0)
  # una sola llamada vectorizada: filas = draws, columnas = partidos
  lh <- posterior_epred(fit_bayes, newdata = nd_home, allow_new_levels = TRUE, ndraws = n_draws)
  la <- posterior_epred(fit_bayes, newdata = nd_away, allow_new_levels = TRUE, ndraws = n_draws)
  gh <- matrix(rpois(length(lh), lh), nrow = n_draws)
  ga <- matrix(rpois(length(la), la), nrow = n_draws)
  data.table(
    p_home = colMeans(gh > ga), p_draw = colMeans(gh == ga), p_away = colMeans(gh < ga),
    lambda_home = colMeans(lh), lambda_away = colMeans(la),
    lambda_home_lo = apply(lh, 2, quantile, 0.05), lambda_home_hi = apply(lh, 2, quantile, 0.95)
  )
}

cat("\nPrediciendo", nrow(test), "partidos de test con incertidumbre posterior...\n")
preds <- predict_all(test)
preds <- cbind(test[, .(match_id, date, home_team, away_team, result)], preds)
setnames(preds, c("p_home","p_draw","p_away","lambda_home","lambda_away",
                   "lambda_home_lo","lambda_home_hi"),
         paste0("bayes_", c("p_home","p_draw","p_away","lambda_home","lambda_away",
                             "lambda_home_lo","lambda_home_hi")))

pred_class <- fifelse(preds$bayes_p_home >= preds$bayes_p_away & preds$bayes_p_home >= preds$bayes_p_draw, "H",
              fifelse(preds$bayes_p_away >= preds$bayes_p_home & preds$bayes_p_away >= preds$bayes_p_draw, "A", "D"))
cat("Accuracy Bayes MCMC (test):", round(mean(pred_class == preds$result), 4), "\n")

fwrite(preds, file.path(DATA_DIR, "preds_bayes.csv"))
saveRDS(fit_bayes, "output/model_bayes_mcmc.rds")
cat("Predicciones bayesianas guardadas en data/preds_bayes.csv\n")
