## =============================================================
## 08_ensemble.R — VERSIÓN CORREGIDA / AMPLIADA
## Cambios:
##  1. Incorpora preds_bayes.csv al ensamble (si existe) — antes se
##     calculaba pero nunca se usaba.
##  2. Calibración isotónica por modelo ANTES de promediar/stackear
##     (mejora Brier/log-loss sin tocar el ranking de accuracy).
##  3. Guarda stack_colnames.rds para que la API sepa el orden EXACTO
##     de columnas que espera el meta-learner.
## =============================================================
suppressMessages({ library(data.table); library(glmnet) })

DATA_DIR <- "data"
OUT_DIR  <- "output"
model_dt <- fread(file.path(DATA_DIR, "model_features.csv"))
model_dt[, date := as.Date(date)]
SPLIT_DATE <- as.Date("2023-01-01")
test <- model_dt[date >= SPLIT_DATE]

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

test <- cbind(test, dc_probs(test$lambda_home, test$lambda_away, test$dc_rho))
pred_class_dc <- c("A","D","H")[apply(test[, .(dc_p_away, dc_p_draw, dc_p_home)], 1, which.max)]
cat("Accuracy Dixon-Coles (baseline puro):",
    round(mean(pred_class_dc == test$result), 4), "\n")

## ---------------------------------------------------------------
## (2) Cargar predicciones de los demás modelos
## ---------------------------------------------------------------
nn <- fread(file.path(DATA_DIR, "preds_nnet.csv"))
ens <- merge(test[, .(match_id, date, home_team, away_team, result,
                       dc_p_home, dc_p_draw, dc_p_away)],
             nn[, .(match_id, nn_p_home, nn_p_draw, nn_p_away)], by = "match_id")

xgb_path <- file.path(DATA_DIR, "preds_xgboost.csv")
if (file.exists(xgb_path)) {
  xgb <- fread(xgb_path)
  ens <- merge(ens, xgb[, .(match_id, xgb_p_home, xgb_p_draw, xgb_p_away)],
               by = "match_id", all.x = TRUE)
  cat("-> Predicciones de XGBoost incorporadas al ensamble.\n")
} else {
  cat("-> preds_xgboost.csv no encontrado. Se omite del ensamble.\n")
}

## ---- NUEVO: incorporar el modelo bayesiano MCMC ----
bayes_path <- file.path(DATA_DIR, "preds_bayes.csv")
if (file.exists(bayes_path)) {
  bayes <- fread(bayes_path)
  ens <- merge(ens, bayes[, .(match_id, bayes_p_home, bayes_p_draw, bayes_p_away)],
               by = "match_id", all.x = TRUE)
  cat("-> Predicciones bayesianas MCMC incorporadas al ensamble (",
      sum(!is.na(ens$bayes_p_home)), "de", nrow(ens),
      "filas -- recuerda que 06 solo cubre la ventana sep-2024/ene-2026).\n")
} else {
  cat("-> preds_bayes.csv no encontrado (corre 06_model_bayesian_mcmc.R). Se omite.\n")
}

## ---------------------------------------------------------------
## (2b) NUEVO: calibración isotónica por modelo base
## Ajusta cada probabilidad p_home/p_draw/p_away de cada modelo con
## una regresión isotónica 1-D (monótona) contra el resultado real,
## usando SOLO el primer 70% cronológico del test (para no usar el
## mismo tramo que luego se usa como holdout del stacking).
## ---------------------------------------------------------------
setorder(ens, date)
cut_cal <- floor(0.7 * nrow(ens))
cal_fit_dat <- ens[1:cut_cal]

model_probs_raw <- grep("_p_home$|_p_draw$|_p_away$", names(ens), value = TRUE)
prefixes <- unique(gsub("_p_(home|draw|away)$", "", model_probs_raw))

calibration_store <- list()  # NUEVO: guardamos los breakpoints para usarlos en vivo

calibrate_col <- function(col, outcome_binary, fit_rows) {
  ok <- !is.na(col[fit_rows]) & !is.na(outcome_binary[fit_rows])
  if (sum(ok) < 30) return(list(vals = col, iso = NULL))
  iso <- isoreg(col[fit_rows][ok], outcome_binary[fit_rows][ok])
  # BUG REAL encontrado al probar con datos de verdad: isoreg() devuelve $x y $y
  # en el ORDEN ORIGINAL de entrada (sin ordenar), pero $yf (los valores ajustados)
  # SÍ vienen en orden ascendente de x (alineados con $ord, el vector de permutación
  # que ordena $x). Pasarle iso$x tal cual a approxfun() -- sin ordenar -- produce
  # una "calibración" sin sentido (ni monótona ni relacionada con el score real).
  # Fix: usar iso$x[iso$ord] (= x ordenado), que sí calza con iso$yf.
  sorted_x <- iso$x[iso$ord]
  calib_fun <- approxfun(sorted_x, iso$yf, rule = 2)
  list(vals = calib_fun(col), iso = list(x = sorted_x, yf = iso$yf))
}

for (pfx in prefixes) {
  hc <- paste0(pfx, "_p_home"); dc_ <- paste0(pfx, "_p_draw"); ac <- paste0(pfx, "_p_away")
  if (!all(c(hc, dc_, ac) %in% names(ens))) next
  h_bin <- as.integer(ens$result == "H")
  d_bin <- as.integer(ens$result == "D")
  a_bin <- as.integer(ens$result == "A")
  r_h <- calibrate_col(ens[[hc]], h_bin, seq_len(cut_cal))
  r_d <- calibrate_col(ens[[dc_]], d_bin, seq_len(cut_cal))
  r_a <- calibrate_col(ens[[ac]], a_bin, seq_len(cut_cal))
  ens[[paste0(pfx, "_p_home_cal")]] <- r_h$vals
  ens[[paste0(pfx, "_p_draw_cal")]] <- r_d$vals
  ens[[paste0(pfx, "_p_away_cal")]] <- r_a$vals
  # renormalizar para que sumen 1 -- CON SALVAGUARDA: si la calibración isotónica
  # (ajustada independientemente por clase) llega a mandar las 3 probabilidades a 0
  # para la misma fila (raro pero real -- pasó en 10-19 filas de 3,686 con datos
  # reales), tot=0 produce NaN (0/0) que después revienta el which.max(). Se
  # reemplaza esa fila puntual por "sin información" (1/3, 1/3, 1/3).
  h_v <- ens[[paste0(pfx,"_p_home_cal")]]; d_v <- ens[[paste0(pfx,"_p_draw_cal")]]; a_v <- ens[[paste0(pfx,"_p_away_cal")]]
  tot <- h_v + d_v + a_v
  degenerate <- is.na(tot) | tot == 0
  if (any(degenerate)) {
    h_v[degenerate] <- 1/3; d_v[degenerate] <- 1/3; a_v[degenerate] <- 1/3; tot[degenerate] <- 1
    cat("  (", pfx, ": ", sum(degenerate), "filas con calibración degenerada -> fallback 1/3)\n")
  }
  ens[[paste0(pfx,"_p_home_cal")]] <- h_v / tot
  ens[[paste0(pfx,"_p_draw_cal")]] <- d_v / tot
  ens[[paste0(pfx,"_p_away_cal")]] <- a_v / tot
  calibration_store[[pfx]] <- list(home = r_h$iso, draw = r_d$iso, away = r_a$iso)
}
cat("Calibración isotónica aplicada a:", paste(prefixes, collapse=", "), "\n")
saveRDS(calibration_store, file.path(OUT_DIR, "calibration_functions.rds"))
cat("Guardado: output/calibration_functions.rds (para aplicar la MISMA calibración en vivo)\n")

## ---------------------------------------------------------------
## (3a) Ensamble por promedio simple (usa versiones CALIBRADAS si existen)
## ---------------------------------------------------------------
model_probs <- if (any(grepl("_cal$", names(ens)))) {
  grep("_p_(home|draw|away)_cal$", names(ens), value = TRUE)
} else {
  grep("_p_home$|_p_draw$|_p_away$", names(ens), value = TRUE)
}
home_cols <- grep("_p_home", model_probs, value = TRUE)
draw_cols <- grep("_p_draw", model_probs, value = TRUE)
away_cols <- grep("_p_away", model_probs, value = TRUE)

ens[, avg_p_home := rowMeans(.SD, na.rm = TRUE), .SDcols = home_cols]
ens[, avg_p_draw := rowMeans(.SD, na.rm = TRUE), .SDcols = draw_cols]
ens[, avg_p_away := rowMeans(.SD, na.rm = TRUE), .SDcols = away_cols]
pred_class_avg <- c("A","D","H")[apply(ens[, .(avg_p_away, avg_p_draw, avg_p_home)], 1, which.max)]
cat("Accuracy ENSAMBLE (promedio, calibrado):",
    round(mean(pred_class_avg == ens$result), 4), "\n")

## ---------------------------------------------------------------
## (3b) Stacking (meta-learner multinomial glmnet) sobre probs CALIBRADAS
## ---------------------------------------------------------------
## NUEVO: glmnet no acepta NA. El modelo bayesiano solo cubre una ventana
## (sep-2024 a ene-2026), así que fuera de esa ventana sus columnas *_cal
## quedan NA. Las imputamos como "sin información" (1/3, 1/3, 1/3) --
## el ridge (alpha=0) del stacker ya se encarga de darle poco peso a una
## columna que no aporta señal real en esas filas.
for (col in model_probs) {
  if (any(is.na(ens[[col]]))) ens[is.na(get(col)), (col) := 1/3]
}

cut <- floor(0.7 * nrow(ens))
meta_train <- ens[1:cut]; meta_test <- ens[(cut+1):nrow(ens)]

X_meta_train <- as.matrix(meta_train[, ..model_probs])
X_meta_test  <- as.matrix(meta_test[, ..model_probs])
y_meta_train <- factor(meta_train$result, levels = c("A","D","H"))

cvfit <- cv.glmnet(X_meta_train, y_meta_train, family = "multinomial", alpha = 0)
pred_stack <- predict(cvfit, X_meta_test, s = "lambda.min", type = "response")[,,1]
pred_class_stack <- c("A","D","H")[apply(pred_stack, 1, which.max)]
cat("Accuracy ENSAMBLE (stacking glmnet, sobre 30% final del test):",
    round(mean(pred_class_stack == meta_test$result), 4), "\n")

ens[, stack_split := c(rep("meta_train", nrow(meta_train)), rep("meta_test", nrow(meta_test)))]
pred_stack_all <- predict(cvfit, as.matrix(ens[, ..model_probs]), s = "lambda.min", type = "response")[,,1]
# Extraer por NOMBRE de columna, no por posición (evita el bug de orden A/D/H)
ens[, stack_p_away := pred_stack_all[, "A"]]
ens[, stack_p_draw := pred_stack_all[, "D"]]
ens[, stack_p_home := pred_stack_all[, "H"]]

fwrite(ens, file.path(DATA_DIR, "ensemble_predictions.csv"))
saveRDS(cvfit, file.path(OUT_DIR, "model_stacking.rds"))

## ---- NUEVO: guardar el orden exacto de columnas que espera el stacker ----
saveRDS(model_probs, file.path(OUT_DIR, "stack_colnames.rds"))

cat("\nGuardado: data/ensemble_predictions.csv\n")
cat("Guardado: output/stack_colnames.rds (orden:", paste(model_probs, collapse=", "), ")\n")
