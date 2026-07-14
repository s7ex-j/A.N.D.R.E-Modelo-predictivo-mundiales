## =============================================================
## 09b_evaluation_worldcup.R
## Tu 09_evaluation.R mide accuracy sobre TODOS los partidos desde
## 2023 (amistosos + eliminatorias + todo). Esto mide específicamente
## sobre partidos de COPA DEL MUNDO (2018, 2022, 2026) -- que es lo
## que realmente te importa. El accuracy en Mundial puede ser distinto
## al agregado: los partidos de Mundial tienden a ser más parejos
## (todos los equipos ya clasificaron) pero también hay menos margen
## de error porque son eliminatorias directas.
## =============================================================
suppressMessages({ library(data.table) })

DATA_DIR <- "data"
ens <- fread(file.path(DATA_DIR, "ensemble_predictions.csv"))
results_clean <- fread(file.path(DATA_DIR, "results_clean.csv"))
results_clean[, date := as.Date(date)]

## Identificar qué match_id corresponden a partidos de Copa del Mundo
## (nombre real del torneo en el dataset: "FIFA World Cup" -- NO confundir
## con "Viva World Cup" ni "CONIFA World Cup qualification", que son
## torneos distintos). trimws() por si acaso hay espacios extra al
## leer en Windows.
tn <- trimws(tolower(results_clean$tournament))
wc_ids <- results_clean[tn == "fifa world cup", match_id]

if (length(wc_ids) == 0) {
  cat("AVISO: no se encontraron partidos con tournament == 'FIFA World Cup'.\n")
  cat("Valores únicos que contienen 'world cup' en tu results_clean.csv:\n")
  print(unique(results_clean$tournament[grepl("world cup", tn)]))
  stop("Revisa el nombre exacto del torneo arriba y ajusta el filtro si es distinto.")
}
ens_wc <- ens[match_id %in% wc_ids]

cat("Partidos de Copa del Mundo en el set de evaluación (2023+):", nrow(ens_wc), "\n")
if (nrow(ens_wc) == 0) {
  cat("Total de partidos de Copa del Mundo en TODA tu historia (1930-2026):", length(wc_ids), "\n")
  cat("Rango de fechas de ensemble_predictions.csv:",
      as.character(min(ens$date)), "-", as.character(max(ens$date)), "\n")
  stop("No hay partidos de Mundial dentro de esa ventana de fechas todavía. Si el ",
       "Mundial 2026 ya empezó y esto sigue en 0, revisa que resultados.csv esté ",
       "actualizado y que 08_ensemble_FIX.R se haya corrido DESPUÉS de actualizarlo.")
}

rps <- function(p_away, p_draw, p_home, result) {
  actual <- matrix(0, nrow = length(result), ncol = 3)
  actual[cbind(seq_along(result), match(result, c("A","D","H")))] <- 1
  cum_pred   <- cbind(p_away, p_away + p_draw, p_away + p_draw + p_home)
  cum_actual <- cbind(actual[,1], actual[,1]+actual[,2], actual[,1]+actual[,2]+actual[,3])
  rowSums((cum_pred - cum_actual)^2) / 2
}
logloss <- function(p_away, p_draw, p_home, result) {
  p <- ifelse(result == "A", p_away, ifelse(result == "D", p_draw, p_home))
  -mean(log(pmax(p, 1e-10)))
}
accuracy <- function(p_away, p_draw, p_home, result) {
  ok <- !is.na(p_away) & !is.na(p_draw) & !is.na(p_home)
  if (sum(ok) < 10) return(NA_real_)
  pred_class <- c("A","D","H")[apply(cbind(p_away[ok], p_draw[ok], p_home[ok]), 1, which.max)]
  mean(pred_class == result[ok])
}

evaluate_model <- function(prefix, dat) {
  pa <- dat[[paste0(prefix, "_p_away")]]
  pd <- dat[[paste0(prefix, "_p_draw")]]
  ph <- dat[[paste0(prefix, "_p_home")]]
  if (is.null(pa)) return(NULL)
  data.table(
    modelo = prefix, n_partidos = nrow(dat),
    accuracy = round(accuracy(pa, pd, ph, dat$result), 4),
    log_loss = round(logloss(pa, pd, ph, dat$result), 4),
    RPS      = round(mean(rps(pa, pd, ph, dat$result)), 4)
  )
}

modelos <- c("dc","nn","xgb","bayes","avg","stack")
wc_table <- rbindlist(lapply(modelos, function(m) evaluate_model(m, ens_wc)), fill = TRUE)

cat("\n=== Accuracy específica en partidos de COPA DEL MUNDO ===\n")
print(wc_table)
cat("\nCompara esto contra evaluation_summary.csv (todos los torneos).\n")
cat("Si el accuracy en Mundial es notablemente menor, probablemente el modelo\n")
cat("necesita más peso a la 'forma reciente' (los equipos llegan a Mundiales\n")
cat("con más descanso/preparación que en fechas FIFA normales) o más datos\n")
cat("de Mundiales pasados en el entrenamiento.\n")

fwrite(wc_table, file.path(DATA_DIR, "evaluation_worldcup.csv"))
