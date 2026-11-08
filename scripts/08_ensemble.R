## =============================================================
## 08_ensemble.R
## Ensamble y stacking (meta-learner glmnet).
##
## Diseño (v2): la calibración isotónica y el meta-learner se ajustan con
## las predicciones OUT-OF-FOLD de META_YEARS (walk-forward, generadas en
## 05 y 07), que están en el periodo de entrenamiento. Así el test set
## completo (2023+) queda fuera de muestra para TODOS los modelos y se
## evalúan sobre los mismos partidos.
##
## (Antes la calibración y el stacking se ajustaban con el primer 70% del
## test, lo que dejaba solo el 30% final para evaluar; ver bitácora §5.)
##
##  1. Probabilidades Dixon-Coles (de los lambdas walk-forward).
##  2. Calibración isotónica por modelo, ajustada en META_YEARS.
##  3. Promedio simple de las probabilidades calibradas.
##  4. Stacking: glmnet multinomial entrenado en META_YEARS.
##  5. Guarda calibración y orden de columnas para la API.
## El bayesiano (06) se agrega al archivo de salida solo para evaluarlo
## aparte; no entra al promedio ni al stacking (cubre una ventana corta).
## =============================================================
suppressMessages({ library(data.table); library(glmnet) })
source("scripts/00_config.R")

model_dt <- fread(file.path(DATA_DIR, "model_features.csv"))
model_dt[, date := as.Date(date)]
META_START <- as.Date(sprintf("%d-01-01", min(META_YEARS)))

## ---------------------------------------------------------------
## (1) Probabilidades Dixon-Coles vía Poisson bivariado + tau
## ---------------------------------------------------------------
dc_probs <- function(lh, la, rho, max_goals = 10) {
  tau1 <- function(x, y, l1, l2, rho) {
    ifelse(x == 0 & y == 0, 1 - l1*l2*rho,
    ifelse(x == 0 & y == 1, 1 + l1*rho,
    ifelse(x == 1 & y == 0, 1 + l2*rho,
    ifelse(x == 1 & y == 1, 1 - rho, 1))))
  }
  n <- length(lh)
  p_home <- p_draw <- p_away <- numeric(n)
  for (x in 0:max_goals) for (y in 0:max_goals) {
    p <- dpois(x, lh) * dpois(y, la) * pmax(tau1(x, y, lh, la, rho), 1e-8)
    if (x > y) p_home <- p_home + p
    else if (x == y) p_draw <- p_draw + p
    else p_away <- p_away + p
  }
  tot <- p_home + p_draw + p_away
  data.table(dc_p_home = p_home/tot, dc_p_draw = p_draw/tot, dc_p_away = p_away/tot)
}

base <- model_dt[date >= META_START]
base <- cbind(base[, .(match_id, date, home_team, away_team, result)],
              dc_probs(base$lambda_home, base$lambda_away, base$dc_rho))
base[, split := fifelse(date < SPLIT_DATE, "meta", "test")]

## ---------------------------------------------------------------
## (2) Unir predicciones de nn y xgb: out-of-fold (meta) + test
## ---------------------------------------------------------------
read_preds <- function(test_file, oof_file, prefix) {
  cols <- c("match_id", paste0(prefix, c("_p_home", "_p_draw", "_p_away")))
  te <- fread(file.path(DATA_DIR, test_file))[, ..cols]
  of <- fread(file.path(DATA_DIR, oof_file))[, ..cols]
  rbind(of, te)
}
ens <- merge(base, read_preds("preds_nnet.csv", "preds_nnet_oof.csv", "nn"), by = "match_id")
ens <- merge(ens, read_preds("preds_xgboost.csv", "preds_xgboost_oof.csv", "xgb"), by = "match_id")
setorder(ens, date, match_id)
cat("Partidos para calibrar y entrenar el stacking (", paste(range(META_YEARS), collapse = "-"),
    "):", ens[split == "meta", .N], " | test:", ens[split == "test", .N], "\n")

pred_class <- function(pa, pd, ph) c("A","D","H")[max.col(cbind(pa, pd, ph), ties.method = "first")]
tst <- ens$split == "test"
cat("Accuracy Dixon-Coles (test):",
    round(mean(pred_class(ens$dc_p_away, ens$dc_p_draw, ens$dc_p_home)[tst] == ens$result[tst]), 4), "\n")

## ---------------------------------------------------------------
## (3) Calibración isotónica por modelo, ajustada SOLO con META_YEARS
## ---------------------------------------------------------------
fit_rows <- which(ens$split == "meta")
calibration_store <- list()   # breakpoints para aplicar la misma calibración en vivo (API)

calibrate_col <- function(col, outcome_binary) {
  ok <- fit_rows[!is.na(col[fit_rows])]
  iso <- isoreg(col[ok], outcome_binary[ok])
  # isoreg() devuelve $x en el orden ORIGINAL y $yf ordenado según $ord:
  # hay que usar iso$x[iso$ord] para que calcen (bug real, ver bitácora §1.1).
  sorted_x <- iso$x[iso$ord]
  calib_fun <- approxfun(sorted_x, iso$yf, rule = 2, ties = mean)
  list(vals = calib_fun(col), iso = list(x = sorted_x, yf = iso$yf))
}

for (pfx in ENSEMBLE_MODELS) {
  cols <- paste0(pfx, c("_p_home", "_p_draw", "_p_away"))
  r_h <- calibrate_col(ens[[cols[1]]], as.integer(ens$result == "H"))
  r_d <- calibrate_col(ens[[cols[2]]], as.integer(ens$result == "D"))
  r_a <- calibrate_col(ens[[cols[3]]], as.integer(ens$result == "A"))
  h_v <- r_h$vals; d_v <- r_d$vals; a_v <- r_a$vals
  # Las 3 clases se calibran por separado: se renormaliza para que sumen 1.
  # Si las 3 quedan en 0 para una fila (raro), esa fila pasa a 1/3 cada una.
  tot <- h_v + d_v + a_v
  degenerate <- is.na(tot) | tot == 0
  if (any(degenerate)) {
    h_v[degenerate] <- d_v[degenerate] <- a_v[degenerate] <- 1/3; tot[degenerate] <- 1
    cat("  (", pfx, ":", sum(degenerate), "filas con calibración degenerada -> 1/3)\n")
  }
  set(ens, j = paste0(pfx, "_p_home_cal"), value = h_v / tot)
  set(ens, j = paste0(pfx, "_p_draw_cal"), value = d_v / tot)
  set(ens, j = paste0(pfx, "_p_away_cal"), value = a_v / tot)
  calibration_store[[pfx]] <- list(home = r_h$iso, draw = r_d$iso, away = r_a$iso)
}
cat("Calibración isotónica (ajustada en", paste(range(META_YEARS), collapse = "-"), ") aplicada a:",
    paste(ENSEMBLE_MODELS, collapse = ", "), "\n")
saveRDS(calibration_store, file.path(OUT_DIR, "calibration_functions.rds"))

## ---------------------------------------------------------------
## (4) Promedio simple de las probabilidades calibradas
## ---------------------------------------------------------------
model_probs <- as.vector(t(outer(ENSEMBLE_MODELS, c("_p_home_cal", "_p_draw_cal", "_p_away_cal"), paste0)))
ens[, avg_p_home := rowMeans(.SD), .SDcols = grep("_p_home_cal$", model_probs, value = TRUE)]
ens[, avg_p_draw := rowMeans(.SD), .SDcols = grep("_p_draw_cal$", model_probs, value = TRUE)]
ens[, avg_p_away := rowMeans(.SD), .SDcols = grep("_p_away_cal$", model_probs, value = TRUE)]
cat("Accuracy ENSAMBLE (promedio calibrado, test):",
    round(mean(pred_class(ens$avg_p_away, ens$avg_p_draw, ens$avg_p_home)[tst] == ens$result[tst]), 4), "\n")

## ---------------------------------------------------------------
## (5) Stacking: glmnet multinomial (ridge) entrenado en META_YEARS
## ---------------------------------------------------------------
meta <- ens[split == "meta"]
set.seed(42)
cvfit <- cv.glmnet(as.matrix(meta[, ..model_probs]),
                   factor(meta$result, levels = c("A","D","H")),
                   family = "multinomial", alpha = 0)
pred_stack <- predict(cvfit, as.matrix(ens[, ..model_probs]), s = "lambda.min", type = "response")[, , 1]
# Extraer por NOMBRE de columna, no por posición (evita el bug de orden A/D/H, bitácora §1.4)
ens[, stack_p_away := pred_stack[, "A"]]
ens[, stack_p_draw := pred_stack[, "D"]]
ens[, stack_p_home := pred_stack[, "H"]]
cat("Accuracy ENSAMBLE (stacking glmnet, test completo):",
    round(mean(pred_class(ens$stack_p_away, ens$stack_p_draw, ens$stack_p_home)[tst] == ens$result[tst]), 4), "\n")

## ---------------------------------------------------------------
## (6) Bayesiano: solo para evaluarlo aparte (no entra al ensamble)
## ---------------------------------------------------------------
bayes_path <- file.path(DATA_DIR, "preds_bayes.csv")
if (file.exists(bayes_path)) {
  bayes <- fread(bayes_path)[, .(match_id, bayes_p_home, bayes_p_draw, bayes_p_away)]
  ens <- merge(ens, bayes, by = "match_id", all.x = TRUE)
  cat("-> Bayesiano agregado para evaluación aparte (", sum(!is.na(ens$bayes_p_home) & ens$split == "test"),
      "partidos del test con predicción).\n")
}

## ---------------------------------------------------------------
## Guardar: solo el test (las filas meta ya cumplieron su función)
## ---------------------------------------------------------------
out <- ens[split == "test"]
out[, split := NULL]
setorder(out, date, match_id)
fwrite(out, file.path(DATA_DIR, "ensemble_predictions.csv"))
saveRDS(cvfit, file.path(OUT_DIR, "model_stacking.rds"))
saveRDS(model_probs, file.path(OUT_DIR, "stack_colnames.rds"))   # orden exacto que espera el stacker

cat("\nGuardado: data/ensemble_predictions.csv (", nrow(out), "partidos del test)\n")
cat("Guardado: output/stack_colnames.rds (orden:", paste(model_probs, collapse = ", "), ")\n")
