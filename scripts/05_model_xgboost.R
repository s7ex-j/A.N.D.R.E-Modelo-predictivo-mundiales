## =============================================================
## 05_model_xgboost.R
## XGBoost: (a) clasificación multiclase 1X2  (b) regresión Poisson
## de goles esperados por equipo. Usa como features las variables
## intermedias de Dixon-Coles + Elo + forma + contexto.
##
## NOTA: requiere install.packages("xgboost") -- no disponible en
## el sandbox de Claude (sin acceso a CRAN), pero sí en tu RStudio.
## =============================================================
suppressMessages({
  library(data.table)
  library(xgboost)
})

set.seed(42)
DATA_DIR <- "data"
model_dt <- fread(file.path(DATA_DIR, "model_features.csv"))
model_dt[, date := as.Date(date)]
setorder(model_dt, date)

feature_cols <- c("elo_home_pre","elo_away_pre","elo_diff","elo_diff_abs",
                   "lambda_home","lambda_away","dc_rho",
                   "home_form_pts","away_form_pts","form_pts_diff",
                   "home_form_gf","away_form_gf","form_gf_diff",
                   "home_form_ga","away_form_ga",
                   "home_days_rest","away_days_rest","rest_diff",
                   "h2h_home_wr","h2h_n_prev","importance","neutral")

# ---- Split temporal (NUNCA aleatorio: evita fuga de información futura) ----
SPLIT_DATE <- as.Date("2023-01-01")
train <- model_dt[date < SPLIT_DATE]
test  <- model_dt[date >= SPLIT_DATE]
cat("Train:", nrow(train), " | Test:", nrow(test), "\n")

X_train <- as.matrix(train[, ..feature_cols])
X_test  <- as.matrix(test[, ..feature_cols])

## ---------------------------------------------------------------
## (a) Clasificación multiclase: 0=Away, 1=Draw, 2=Home
## ---------------------------------------------------------------
dtrain_cls <- xgb.DMatrix(X_train, label = train$y_result)
dtest_cls  <- xgb.DMatrix(X_test,  label = test$y_result)

params_cls <- list(
  objective = "multi:softprob", num_class = 3,
  eval_metric = "mlogloss",
  eta = 0.03, max_depth = 4, subsample = 0.8, colsample_bytree = 0.8,
  min_child_weight = 10, lambda = 2, alpha = 0.5
)

cv_cls <- xgb.cv(params_cls, dtrain_cls, nrounds = 800, nfold = 5,
                  early_stopping_rounds = 40, verbose = 0)
best_nrounds_cls <- cv_cls$early_stop$best_iteration  # xgboost >= 2.x: ya no es cv$best_iteration
cat("Mejor n_rounds (clasificación):", best_nrounds_cls, "\n")

model_cls <- xgb.train(params_cls, dtrain_cls, nrounds = best_nrounds_cls,
                        evals = list(train = dtrain_cls, test = dtest_cls),  # antes: watchlist
                        verbose = 0)

raw_pred_cls <- predict(model_cls, dtest_cls)
# xgboost >= 2.x: predict() ya devuelve una matriz (n x num_class) directamente.
# Versiones viejas devolvían un vector plano que había que reshape-ar a mano.
if (is.matrix(raw_pred_cls)) {
  pred_cls <- raw_pred_cls
} else {
  pred_cls <- matrix(raw_pred_cls, ncol = 3, byrow = TRUE)
}
colnames(pred_cls) <- c("p_away", "p_draw", "p_home")
pred_class <- apply(pred_cls, 1, which.max) - 1
acc <- mean(pred_class == test$y_result)
cat("Accuracy XGBoost (test 2023+):", round(acc, 4), "\n")

# Importancia de variables
imp <- xgb.importance(model = model_cls)
print(head(imp, 10))

## ---------------------------------------------------------------
## (b) Regresión: goles esperados home / away (objetivo Poisson)
## ---------------------------------------------------------------
fit_goals_model <- function(target_train, target_test, label) {
  dtr <- xgb.DMatrix(X_train, label = target_train)
  dte <- xgb.DMatrix(X_test,  label = target_test)
  params_reg <- list(objective = "count:poisson", eval_metric = "poisson-nloglik",
                      eta = 0.03, max_depth = 3, subsample = 0.8,
                      colsample_bytree = 0.8, min_child_weight = 15, lambda = 3)
  cv <- xgb.cv(params_reg, dtr, nrounds = 800, nfold = 5,
               early_stopping_rounds = 40, verbose = 0)
  m <- xgb.train(params_reg, dtr, nrounds = cv$early_stop$best_iteration, verbose = 0)
  pred <- predict(m, dte)
  cat(label, "- MAE test:", round(mean(abs(pred - target_test)), 3), "\n")
  list(model = m, pred = pred)
}

goals_home <- fit_goals_model(train$home_score, test$home_score, "Goles LOCAL")
goals_away <- fit_goals_model(train$away_score, test$away_score, "Goles VISITA")

## ---------------------------------------------------------------
## Guardar predicciones para el ensamble
## ---------------------------------------------------------------
out <- data.table(
  match_id = test$match_id, date = test$date,
  home_team = test$home_team, away_team = test$away_team,
  xgb_p_away = pred_cls[,1], xgb_p_draw = pred_cls[,2], xgb_p_home = pred_cls[,3],
  xgb_goals_home = goals_home$pred, xgb_goals_away = goals_away$pred
)
fwrite(out, file.path(DATA_DIR, "preds_xgboost.csv"))
saveRDS(model_cls, "output/model_xgb_classifier.rds")
cat("\nPredicciones XGBoost guardadas en data/preds_xgboost.csv\n")
