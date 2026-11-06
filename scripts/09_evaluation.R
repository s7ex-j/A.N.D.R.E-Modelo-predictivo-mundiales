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

DATA_DIR <- "data"
ens <- fread(file.path(DATA_DIR, "ensemble_predictions.csv"))

## Holdout común: el meta-learner del stacking y la calibración isotónica
## se ajustan sobre el primer 70% del test set (ver 08_ensemble.R). Para que
## la comparación sea justa, TODOS los modelos se evalúan sobre el mismo 30%
## final ("meta_test"), que ninguno de ellos vio al ajustarse.
if ("stack_split" %in% names(ens)) {
  ajuste  <- ens[stack_split == "meta_train"]
  holdout <- ens[stack_split == "meta_test"]
} else {
  ajuste  <- ens
  holdout <- ens
}

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

# Baseline "naive": siempre la frecuencia histórica global (H/D/A) -- referencia mínima
## (frecuencias tomadas del tramo de ajuste, no del holdout, para no usar
##  los resultados que se van a evaluar)
base_rates <- prop.table(table(ajuste$result))[c("A","D","H")]
ens[, naive_p_away := base_rates["A"]]
ens[, naive_p_draw := base_rates["D"]]
ens[, naive_p_home := base_rates["H"]]
holdout <- if ("stack_split" %in% names(ens)) ens[stack_split == "meta_test"] else ens

results_table <- rbindlist(list(
  evaluate_model("naive", holdout),
  evaluate_model("dc",    holdout),
  evaluate_model("nn",    holdout),
  if ("xgb_p_home" %in% names(ens)) evaluate_model("xgb", holdout),
  evaluate_model("avg",   holdout),
  if ("stack_p_home" %in% names(ens)) evaluate_model("stack", holdout)
), fill = TRUE)

## Referencia secundaria: modelos base sobre el test set completo (desde
## 2023). No incluye avg ni stack, que se ajustan sobre parte de ese tramo.
full_table <- rbindlist(list(
  evaluate_model("naive"),
  evaluate_model("dc"),
  evaluate_model("nn"),
  if ("xgb_p_home" %in% names(ens)) evaluate_model("xgb")
), fill = TRUE)

cat("=== Comparación de modelos (holdout común, n =", nrow(holdout), ",",
    as.character(min(holdout$date)), "a", as.character(max(holdout$date)), ") ===\n")
print(results_table)
cat("\n=== Referencia: modelos base sobre el test completo (n =", nrow(ens), ") ===\n")
print(full_table)
cat("\nNota: menor RPS / log_loss / brier = mejor. Mayor accuracy = mejor.\n")
cat("RPS es la métrica más citada en la literatura académica de forecasting\n")
cat("de fútbol (Constantinou & Fenton, 2012) porque también premia acertar\n")
cat("'casi' (ej. predecir empate y que gane el favorito por 1) frente a un\n")
cat("fallo total (predecir empate y perder goleado el favorito).\n")

fwrite(results_table, file.path(DATA_DIR, "evaluation_summary.csv"))
fwrite(full_table,    file.path(DATA_DIR, "evaluation_summary_full_test.csv"))
