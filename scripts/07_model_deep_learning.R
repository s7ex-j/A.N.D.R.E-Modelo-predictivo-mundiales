## =============================================================
## 07_model_deep_learning.R
## Deep Learning para clasificación 1X2.
##
## Este script tiene DOS partes:
##  (A) Una red neuronal simple con {nnet} que SÍ corre en el sandbox
##      (sirve como validación rápida del enfoque y como fallback si
##      no tienes GPU/keras instalado).
##  (B) El código real de Deep Learning con {keras3}/TensorFlow que
##      debes correr en tu máquina (requiere: install.packages("keras3");
##      keras3::install_keras()). Es una red más profunda, con dropout
##      y batch normalization, pensada para capturar interacciones no
##      lineales entre Elo, Dixon-Coles y forma reciente.
## =============================================================
suppressMessages({ library(data.table); library(nnet) })
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

SPLIT_DATE <- as.Date("2023-01-01")
train <- model_dt[date < SPLIT_DATE]
test  <- model_dt[date >= SPLIT_DATE]

# ---- Normalización (crítica para redes neuronales) ----
mu <- sapply(train[, ..feature_cols], mean)
sdv <- sapply(train[, ..feature_cols], sd)
scale_feats <- function(d) scale(as.matrix(d[, ..feature_cols]), center = mu, scale = sdv)

X_train <- scale_feats(train); X_test <- scale_feats(test)
y_train <- class.ind(factor(train$result, levels = c("A","D","H")))  # one-hot

## ---------------------------------------------------------------
## (A) Red neuronal simple ejecutable ahora mismo (1 capa oculta)
## ---------------------------------------------------------------
nn_fit <- nnet(X_train, y_train, size = 12, softmax = TRUE,
               decay = 0.01, maxit = 300, trace = FALSE)

pred_nn <- predict(nn_fit, X_test, type = "raw")
colnames(pred_nn) <- c("nn_p_away", "nn_p_draw", "nn_p_home")
pred_class_nn <- c("A","D","H")[apply(pred_nn, 1, which.max)]
cat("Accuracy Red Neuronal (nnet, 1 capa) test 2023+:",
    round(mean(pred_class_nn == test$result), 4), "\n")

out <- data.table(match_id = test$match_id, date = test$date,
                   home_team = test$home_team, away_team = test$away_team,
                   pred_nn)
fwrite(out, file.path(DATA_DIR, "preds_nnet.csv"))
saveRDS(nn_fit, "output/model_nnet.rds")
saveRDS(list(mu=mu, sd=sdv), "output/nnet_scaling.rds")

## ---------------------------------------------------------------
## (B) CÓDIGO PARA TU MÁQUINA: Deep Learning real con keras3
## (no se ejecuta aquí -- requiere TensorFlow instalado localmente)
## ---------------------------------------------------------------
if (FALSE) {
  library(keras3)

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

  save_model(dl_model, "output/model_keras_dl.keras")
}
