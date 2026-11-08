## =============================================================
## 07_model_deep_learning.R
## Deep Learning para clasificación 1X2.
##
## Este script tiene DOS partes:
##  (A) Una red neuronal simple con {nnet}: es la que entra al
##      ensamble y corre sin GPU ni dependencias pesadas. Genera
##      predicciones out-of-fold de META_YEARS (walk-forward, para
##      calibrar y entrenar el stacking en 08) y del test.
##  (B) Una red más profunda con {keras3}/TensorFlow, desactivada por
##      defecto (requiere install.packages("keras3");
##      keras3::install_keras()). Es una red más profunda, con dropout
##      y batch normalization, pensada para capturar interacciones no
##      lineales entre Elo, Dixon-Coles y forma reciente.
## =============================================================
suppressMessages({ library(data.table); library(nnet) })
source("scripts/00_config.R")
set.seed(42)

model_dt <- fread(file.path(DATA_DIR, "model_features.csv"))
model_dt[, date := as.Date(date)]
setorder(model_dt, date)
feature_cols <- FEATURE_COLS

train <- model_dt[date < SPLIT_DATE]
test  <- model_dt[date >= SPLIT_DATE]
year_start <- function(y) as.Date(sprintf("%d-01-01", y))

## Entrena la red con `tr` y predice `te`. La normalización se calcula
## SOLO con `tr` (crítico: usar la media del test sería fuga de información).
fit_predict_nn <- function(tr, te) {
  mu  <- sapply(tr[, ..feature_cols], mean)
  sdv <- sapply(tr[, ..feature_cols], sd)
  sc  <- function(d) scale(as.matrix(d[, ..feature_cols]), center = mu, scale = sdv)
  y   <- class.ind(factor(tr$result, levels = c("A","D","H")))  # one-hot
  set.seed(42)
  fit <- nnet(sc(tr), y, size = 12, softmax = TRUE, decay = 0.01, maxit = 300, trace = FALSE)
  p <- predict(fit, sc(te), type = "raw")
  colnames(p) <- c("nn_p_away", "nn_p_draw", "nn_p_home")
  list(fit = fit, pred = p, mu = mu, sd = sdv)
}

## ---------------------------------------------------------------
## (A1) Predicciones out-of-fold de META_YEARS (walk-forward)
## ---------------------------------------------------------------
oof <- rbindlist(lapply(META_YEARS, function(Y) {
  tr <- train[date < year_start(Y)]
  va <- train[date >= year_start(Y) & date < year_start(Y + 1)]
  r <- fit_predict_nn(tr, va)
  data.table(match_id = va$match_id, date = va$date, fold = Y, r$pred)
}))
fwrite(oof, file.path(DATA_DIR, "preds_nnet_oof.csv"))
cat("Predicciones out-of-fold:", nrow(oof), "partidos de", paste(range(META_YEARS), collapse = "-"), "\n")

## ---------------------------------------------------------------
## (A2) Modelo final (todo el train) -> test
## ---------------------------------------------------------------
final <- fit_predict_nn(train, test)
pred_nn <- final$pred
pred_class_nn <- c("A","D","H")[apply(pred_nn, 1, which.max)]
cat("Accuracy Red Neuronal (nnet, 1 capa) test 2023+:",
    round(mean(pred_class_nn == test$result), 4), "\n")

out <- data.table(match_id = test$match_id, date = test$date,
                  home_team = test$home_team, away_team = test$away_team,
                  pred_nn)
fwrite(out, file.path(DATA_DIR, "preds_nnet.csv"))
saveRDS(final$fit, file.path(OUT_DIR, "model_nnet.rds"))
saveRDS(list(mu = final$mu, sd = final$sd), file.path(OUT_DIR, "nnet_scaling.rds"))

## ---------------------------------------------------------------
## (B) Deep Learning con keras3 (desactivado: requiere TensorFlow)
## ---------------------------------------------------------------
if (FALSE) {
  library(keras3)
  mu <- final$mu; sdv <- final$sd
  scale_feats <- function(d) scale(as.matrix(d[, ..feature_cols]), center = mu, scale = sdv)
  X_train <- scale_feats(train); X_test <- scale_feats(test)
  y_train <- class.ind(factor(train$result, levels = c("A","D","H")))

  build_model <- function(input_dim) {
    keras_model_sequential(input_shape = input_dim) |>
      layer_dense(units = 64, activation = "relu") |>
      layer_batch_normalization() |>
      layer_dropout(0.3) |>
      layer_dense(units = 32, activation = "relu") |>
      layer_batch_normalization() |>
      layer_dropout(0.2) |>
      layer_dense(units = 16, activation = "relu") |>
      layer_dense(units = 3, activation = "softmax")   # A / D / H
  }

  dl_model <- build_model(ncol(X_train))
  dl_model |> compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )

  early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20,
                                        restore_best_weights = TRUE)

  history <- dl_model |> fit(
    X_train, y_train,
    epochs = 200, batch_size = 64, validation_split = 0.15,
    callbacks = list(early_stop), verbose = 0
  )

  pred_dl <- predict(dl_model, X_test)
  colnames(pred_dl) <- c("dl_p_away", "dl_p_draw", "dl_p_home")
  pred_class_dl <- c("A","D","H")[apply(pred_dl, 1, which.max)]
  cat("Accuracy Deep Learning (keras) test 2023+:",
      round(mean(pred_class_dl == test$result), 4), "\n")

  save_model(dl_model, file.path(OUT_DIR, "model_keras_dl.keras"))
}
