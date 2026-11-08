## =============================================================
## 09f_draws_and_segments.R
## Dos preguntas que el accuracy global esconde:
##
##  1. ¿Qué pasa con los EMPATES? Los modelos de fútbol casi nunca
##     eligen "empate" como resultado más probable, aunque ~1 de cada 4
##     partidos termine así. Aquí se mide cuántos empates predice cada
##     modelo, cuántos acierta y si la probabilidad de empate está bien
##     calibrada (promedio predicho vs. frecuencia real).
##
##  2. ¿Dónde acierta y dónde falla? Accuracy y RPS por tipo de partido
##     (amistoso, eliminatoria, torneo oficial, Mundial) y por cancha
##     (local real vs. neutral).
## =============================================================
suppressMessages({ library(data.table) })
source("scripts/00_config.R")

ens <- fread(file.path(DATA_DIR, "ensemble_predictions.csv"))
results_clean <- fread(file.path(DATA_DIR, "results_clean.csv"), encoding = "UTF-8")
ens <- merge(ens, results_clean[, .(match_id, tournament, neutral)], by = "match_id", all.x = TRUE)

modelos <- c("dc", "nn", "xgb", "avg", "stack")
modelos <- modelos[paste0(modelos, "_p_home") %in% names(ens)]

pick <- function(m, d = ens) {
  P <- cbind(d[[paste0(m, "_p_away")]], d[[paste0(m, "_p_draw")]], d[[paste0(m, "_p_home")]])
  c("A","D","H")[max.col(P, ties.method = "first")]
}
rps_of <- function(m, d) {
  pa <- d[[paste0(m, "_p_away")]]; pd <- d[[paste0(m, "_p_draw")]]; ph <- d[[paste0(m, "_p_home")]]
  y <- match(d$result, c("A","D","H"))
  ca <- cbind(pa, pa + pd); cy <- cbind(y == 1, y <= 2)
  mean(rowSums((ca - cy)^2) / 2)
}

## ---------------------------------------------------------------
## 1. Empates
## ---------------------------------------------------------------
draw_rate <- mean(ens$result == "D")
cat("=== Empates en el test (n =", nrow(ens), ") ===\n")
cat("Frecuencia real de empates:", sprintf("%.1f%%", 100 * draw_rate), "\n\n")

draws <- rbindlist(lapply(modelos, function(m) {
  pr <- pick(m)
  data.table(
    modelo = m,
    pct_predice_empate = round(100 * mean(pr == "D"), 1),
    empates_acertados = sum(pr == "D" & ens$result == "D"),
    recall_empate = round(mean(pr[ens$result == "D"] == "D"), 3),
    precision_empate = if (any(pr == "D")) round(mean(ens$result[pr == "D"] == "D"), 3) else NA_real_,
    prob_empate_media = round(mean(ens[[paste0(m, "_p_draw")]]), 3)
  )
}))
print(draws)
cat("\nprob_empate_media vs. frecuencia real: si son parecidas, la probabilidad de\n")
cat("empate está bien calibrada aunque el modelo casi nunca la elija como la más alta.\n\n")

for (m in intersect(c("xgb", "stack"), modelos)) {
  cat("--- Matriz de confusión:", m, "(filas = real, columnas = predicho) ---\n")
  print(table(real = factor(ens$result, levels = c("H","D","A")),
              predicho = factor(pick(m), levels = c("H","D","A"))))
  cat("\n")
}

## ---------------------------------------------------------------
## 2. Desglose por tipo de partido y por cancha
## ---------------------------------------------------------------
tn <- trimws(tolower(ens$tournament))
ens[, tipo := fifelse(tn == "fifa world cup", "Mundial",
              fifelse(tn == "friendly", "Amistoso",
              fifelse(grepl("qualification", tn), "Eliminatoria", "Otro torneo oficial")))]
ens[, cancha := fifelse(as.logical(neutral) %in% TRUE, "Neutral", "Con local")]

segment_table <- function(by_col) {
  rbindlist(lapply(split(ens, ens[[by_col]]), function(d) {
    rbindlist(lapply(modelos, function(m) data.table(
      segmento = d[[by_col]][1], n = nrow(d), modelo = m,
      accuracy = round(mean(pick(m, d) == d$result), 3),
      RPS = round(rps_of(m, d), 4))))
  }))
}
seg <- rbind(segment_table("tipo"), segment_table("cancha"))
wide <- dcast(seg, segmento + n ~ modelo, value.var = "accuracy")
setcolorder(wide, c("segmento", "n", modelos))
cat("=== Accuracy por segmento ===\n")
print(wide)

fwrite(draws, file.path(DATA_DIR, "evaluation_draws.csv"))
fwrite(seg, file.path(DATA_DIR, "evaluation_segments.csv"))
cat("\nGuardado: data/evaluation_draws.csv y data/evaluation_segments.csv\n")
