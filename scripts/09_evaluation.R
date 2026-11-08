## =============================================================
## 09_evaluation.R
## Métricas estándar en la literatura de forecasting de fútbol
## (Constantinou & Fenton, 2012):
##  - RPS (Rank Probability Score): la métrica MÁS recomendada para
##    1X2, porque penaliza según qué tan "lejos" está la predicción
##    del resultado real (predecir 60% empate en un partido que
##    terminó en victoria por 3 goles es peor que si termina 1-0).
##  - Log-loss y Brier score: métricas de calibración probabilística.
##  - Accuracy: intuitiva pero la más débil de las 4 (ignora certeza).
## =============================================================
suppressMessages({ library(data.table) })

source("scripts/00_config.R")
ens <- fread(file.path(DATA_DIR, "ensemble_predictions.csv"))

## Todos los modelos se evalúan sobre el test set COMPLETO (2023+): desde la
## v2, la calibración y el stacking se ajustan con predicciones out-of-fold
## del periodo de entrenamiento (08), así que ningún modelo vio el test.

rps <- function(p_away, p_draw, p_home, result) {
  # Orden natural del resultado: Away(0) - Draw(1) - Home(2)
  actual <- matrix(0, nrow = length(result), ncol = 3)
  actual[cbind(seq_along(result), match(result, c("A","D","H")))] <- 1
  cum_pred   <- cbind(p_away, p_away + p_draw, p_away + p_draw + p_home)
  cum_actual <- cbind(actual[,1], actual[,1]+actual[,2], actual[,1]+actual[,2]+actual[,3])
  rowSums((cum_pred - cum_actual)^2) / 2   # /2 = num_categorias - 1
}

logloss <- function(p_away, p_draw, p_home, result) {
  p <- ifelse(result == "A", p_away, ifelse(result == "D", p_draw, p_home))
  -mean(log(pmax(p, 1e-10)))
}

brier <- function(p_away, p_draw, p_home, result) {
  actual <- matrix(0, nrow = length(result), ncol = 3)
  actual[cbind(seq_along(result), match(result, c("A","D","H")))] <- 1
  pred <- cbind(p_away, p_draw, p_home)
  mean(rowSums((pred - actual)^2))
}

accuracy <- function(p_away, p_draw, p_home, result) {
  pred_class <- c("A","D","H")[apply(cbind(p_away, p_draw, p_home), 1, which.max)]
  mean(pred_class == result)
}

evaluate_model <- function(prefix, dat = ens) {
  pa <- dat[[paste0(prefix, "_p_away")]]
  pd <- dat[[paste0(prefix, "_p_draw")]]
  ph <- dat[[paste0(prefix, "_p_home")]]
  data.table(
    modelo = prefix,
    n_partidos = nrow(dat),
    accuracy = round(accuracy(pa, pd, ph, dat$result), 4),
    log_loss = round(logloss(pa, pd, ph, dat$result), 4),
    brier    = round(brier(pa, pd, ph, dat$result), 4),
    RPS      = round(mean(rps(pa, pd, ph, dat$result)), 4)
  )
}

# Baseline "naive": siempre la frecuencia histórica (H/D/A) del periodo de
# ENTRENAMIENTO -- referencia mínima. No se usan las frecuencias del test,
# porque serían los mismos resultados que se evalúan.
model_dt <- fread(file.path(DATA_DIR, "model_features.csv"))
model_dt[, date := as.Date(date)]
base_rates <- prop.table(table(model_dt[date < SPLIT_DATE, result]))[c("A","D","H")]
ens[, naive_p_away := base_rates["A"]]
ens[, naive_p_draw := base_rates["D"]]
ens[, naive_p_home := base_rates["H"]]

modelos <- c("naive", "dc", "nn", "xgb", "avg", "stack")
modelos <- modelos[paste0(modelos, "_p_home") %in% names(ens)]
results_table <- rbindlist(lapply(modelos, evaluate_model))

## ---- Intervalos de confianza 95% (bootstrap de partidos) ----
## Se remuestrean partidos con reemplazo; la misma muestra para todos los
## modelos en cada réplica.
set.seed(42)
acc_vec <- function(m) {
  pa <- ens[[paste0(m, "_p_away")]]; pd <- ens[[paste0(m, "_p_draw")]]; ph <- ens[[paste0(m, "_p_home")]]
  list(hit = as.numeric(c("A","D","H")[max.col(cbind(pa, pd, ph), ties.method = "first")] == ens$result),
       rps = rps(pa, pd, ph, ens$result))
}
per_match <- lapply(setNames(modelos, modelos), acc_vec)
n <- nrow(ens)
boot_idx <- replicate(N_BOOT, sample.int(n, n, replace = TRUE), simplify = FALSE)
ci <- rbindlist(lapply(modelos, function(m) {
  acc_b <- vapply(boot_idx, function(i) mean(per_match[[m]]$hit[i]), numeric(1))
  rps_b <- vapply(boot_idx, function(i) mean(per_match[[m]]$rps[i]), numeric(1))
  data.table(modelo = m,
             acc_ic95 = sprintf("[%.3f, %.3f]", quantile(acc_b, 0.025), quantile(acc_b, 0.975)),
             rps_ic95 = sprintf("[%.4f, %.4f]", quantile(rps_b, 0.025), quantile(rps_b, 0.975)))
}))
results_table <- merge(results_table, ci, by = "modelo", sort = FALSE)

cat("=== Comparación de modelos (test completo, n =", nrow(ens), ",",
    as.character(min(ens$date)), "a", as.character(max(ens$date)), ") ===\n")
print(results_table)
cat("\nNota: menor RPS / log_loss / brier = mejor. Mayor accuracy = mejor.\n")
cat("IC95% por bootstrap de partidos (", N_BOOT, " réplicas). Para saber si DOS modelos\n", sep = "")
cat("difieren de verdad, mira las comparaciones pareadas de 09d (no basta con\n")
cat("que sus intervalos se solapen o no).\n")
cat("RPS es la métrica más citada en la literatura académica de forecasting\n")
cat("de fútbol (Constantinou & Fenton, 2012) porque también premia acertar\n")
cat("'casi' (ej. predecir empate y que gane el favorito por 1) frente a un\n")
cat("fallo total (predecir empate y perder goleado el favorito).\n")

fwrite(results_table, file.path(DATA_DIR, "evaluation_summary.csv"))
