## =============================================================
## 05_model_xgboost.R
## XGBoost: (a) clasificación multiclase 1X2  (b) regresión Poisson
## de goles esperados por equipo. Usa como features las variables
## intermedias de Dixon-Coles + Elo + forma + contexto.
##
## Validación SIEMPRE temporal (walk-forward), nunca k-fold aleatorio:
## un fold aleatorio mezcla partidos de 2022 en el entrenamiento para
## predecir partidos de 2010, y eso filtra información del futuro al
## elegir hiperparámetros.
##
##  1. Tuning: para cada configuración de la grilla y cada año Y de
##     META_YEARS, entrena con todo lo anterior a Y y mide log-loss en Y
##     (con early stopping). Gana la configuración con menor log-loss
##     promedio; n_rounds final = mediana de las mejores iteraciones.
##  2. Predicciones out-of-fold de META_YEARS (para calibrar y entrenar
##     el stacking en 08 sin tocar el test).
##  3. Modelo final con todo el train -> predicciones del test.
##
## Requiere install.packages("xgboost").
## =============================================================
suppressMessages({
  library(data.table)
  library(xgboost)
})
source("scripts/00_config.R")

set.seed(42)
model_dt <- fread(file.path(DATA_DIR, "model_features.csv"))
model_dt[, date := as.Date(date)]
setorder(model_dt, date)
feature_cols <- FEATURE_COLS

train <- model_dt[date < SPLIT_DATE]
test  <- model_dt[date >= SPLIT_DATE]
cat("Train:", nrow(train), " | Test:", nrow(test), "\n")

X_of <- function(d) as.matrix(d[, ..feature_cols])
year_start <- function(y) as.Date(sprintf("%d-01-01", y))

## Mejor iteración tras early stopping (compatible con xgboost 1.x a 3.x)
best_iter <- function(m) {
  bi <- attributes(m)$early_stop$best_iteration
  if (is.null(bi)) bi <- m$best_iteration
  if (is.null(bi)) bi <- as.integer(xgb.attr(m, "best_iteration")) + 1L
  as.integer(bi)
}
best_score <- function(m) {
  bs <- attributes(m)$early_stop$best_score
  if (is.null(bs)) bs <- m$best_score
  as.numeric(bs)
}
## predict() devuelve matriz n x 3 en xgboost >= 2; vector plano en versiones viejas
as_prob_matrix <- function(raw) {
  pm <- if (is.matrix(raw)) raw else matrix(raw, ncol = 3, byrow = TRUE)
  colnames(pm) <- c("p_away", "p_draw", "p_home")
  pm
}

## ---------------------------------------------------------------
## (1) Tuning con validación walk-forward
## ---------------------------------------------------------------
base_params <- list(objective = "multi:softprob", num_class = 3,
                    eval_metric = "mlogloss", subsample = 0.8,
                    colsample_bytree = 0.8, lambda = 2, alpha = 0.5)
grid <- list(
  list(eta = 0.03, max_depth = 3, min_child_weight = 10),
  list(eta = 0.03, max_depth = 4, min_child_weight = 10),
  list(eta = 0.03, max_depth = 5, min_child_weight = 20)
)

fit_fold <- function(params, Y) {
  tr <- train[date < year_start(Y)]
  va <- train[date >= year_start(Y) & date < year_start(Y + 1)]
  dtr <- xgb.DMatrix(X_of(tr), label = tr$y_result)
  dva <- xgb.DMatrix(X_of(va), label = va$y_result)
  set.seed(42)
  m <- xgb.train(params, dtr, nrounds = 1500, evals = list(val = dva),
                 early_stopping_rounds = 40, verbose = 0)
  list(best_iter = best_iter(m), logloss = best_score(m))
}

tuning <- rbindlist(lapply(seq_along(grid), function(g) {
  params <- c(base_params, grid[[g]])
  rbindlist(lapply(META_YEARS, function(Y) {
    r <- fit_fold(params, Y)
    data.table(config = g, year = Y, best_iter = r$best_iter, logloss = r$logloss)
  }))
}))
tuning_summary <- tuning[, .(logloss_prom = round(mean(logloss), 4),
                             n_rounds = as.integer(round(median(best_iter)))), by = config]
tuning_summary <- cbind(tuning_summary, rbindlist(grid))
cat("\nTuning walk-forward (log-loss promedio en", paste(range(META_YEARS), collapse = "-"), "):\n")
print(tuning_summary)

best_cfg <- tuning_summary[which.min(logloss_prom)]
params_cls <- c(base_params, grid[[best_cfg$config]])
best_nrounds_cls <- best_cfg$n_rounds
cat("Configuración elegida:", best_cfg$config, "| n_rounds:", best_nrounds_cls, "\n")
fwrite(tuning_summary, file.path(DATA_DIR, "xgb_tuning.csv"))

## ---------------------------------------------------------------
## (2) Predicciones out-of-fold de META_YEARS (para 08)
## ---------------------------------------------------------------
oof <- rbindlist(lapply(META_YEARS, function(Y) {
  tr <- train[date < year_start(Y)]
  va <- train[date >= year_start(Y) & date < year_start(Y + 1)]
  set.seed(42)
  m <- xgb.train(params_cls, xgb.DMatrix(X_of(tr), label = tr$y_result),
                 nrounds = best_nrounds_cls, verbose = 0)
  p <- as_prob_matrix(predict(m, xgb.DMatrix(X_of(va))))
  data.table(match_id = va$match_id, date = va$date, fold = Y,
             xgb_p_away = p[, "p_away"], xgb_p_draw = p[, "p_draw"], xgb_p_home = p[, "p_home"])
}))
fwrite(oof, file.path(DATA_DIR, "preds_xgboost_oof.csv"))
cat("Predicciones out-of-fold:", nrow(oof), "partidos de", paste(range(META_YEARS), collapse = "-"), "\n")

## ---------------------------------------------------------------
## (3) Modelo final (todo el train) -> test
## ---------------------------------------------------------------
dtrain_cls <- xgb.DMatrix(X_of(train), label = train$y_result)
dtest_cls  <- xgb.DMatrix(X_of(test),  label = test$y_result)
set.seed(42)
model_cls <- xgb.train(params_cls, dtrain_cls, nrounds = best_nrounds_cls, verbose = 0)

pred_cls <- as_prob_matrix(predict(model_cls, dtest_cls))
pred_class <- apply(pred_cls, 1, which.max) - 1
acc <- mean(pred_class == test$y_result)
cat("Accuracy XGBoost (test 2023+):", round(acc, 4), "\n")

imp <- xgb.importance(model = model_cls)
print(head(imp, 10))

## ---------------------------------------------------------------
## Regresión: goles esperados home / away (objetivo Poisson).
## n_rounds con early stopping sobre el último año del train
## (validación temporal), luego reentreno con todo el train.
## ---------------------------------------------------------------
fit_goals_model <- function(target_col, label) {
  params_reg <- list(objective = "count:poisson", eval_metric = "poisson-nloglik",
                     eta = 0.03, max_depth = 3, subsample = 0.8,
                     colsample_bytree = 0.8, min_child_weight = 15, lambda = 3)
  last <- max(META_YEARS)
  tr <- train[date < year_start(last)]; va <- train[date >= year_start(last)]
  set.seed(42)
  m_val <- xgb.train(params_reg, xgb.DMatrix(X_of(tr), label = tr[[target_col]]), nrounds = 1500,
                     evals = list(val = xgb.DMatrix(X_of(va), label = va[[target_col]])),
                     early_stopping_rounds = 40, verbose = 0)
  set.seed(42)
  m <- xgb.train(params_reg, xgb.DMatrix(X_of(train), label = train[[target_col]]),
                 nrounds = best_iter(m_val), verbose = 0)
  pred <- predict(m, xgb.DMatrix(X_of(test)))
  cat(label, "- MAE test:", round(mean(abs(pred - test[[target_col]])), 3), "\n")
  list(model = m, pred = pred)
}

goals_home <- fit_goals_model("home_score", "Goles LOCAL")
goals_away <- fit_goals_model("away_score", "Goles VISITA")

## ---------------------------------------------------------------
## Guardar predicciones para el ensamble
## ---------------------------------------------------------------
out <- data.table(
  match_id = test$match_id, date = test$date,
  home_team = test$home_team, away_team = test$away_team,
  xgb_p_away = pred_cls[, "p_away"], xgb_p_draw = pred_cls[, "p_draw"], xgb_p_home = pred_cls[, "p_home"],
  xgb_goals_home = goals_home$pred, xgb_goals_away = goals_away$pred
)
fwrite(out, file.path(DATA_DIR, "preds_xgboost.csv"))
saveRDS(model_cls, file.path(OUT_DIR, "model_xgb_classifier.rds"))
cat("\nPredicciones XGBoost guardadas en data/preds_xgboost.csv\n")
