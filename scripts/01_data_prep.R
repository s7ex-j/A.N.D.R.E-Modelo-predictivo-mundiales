## =============================================================
## 01_data_prep.R — VERSIÓN CORREGIDA
## Único cambio: aplica team_name_map.csv para unificar selecciones
## que cambiaron de nombre (West Germany -> Germany, etc.) ANTES de
## calcular nada. Sin esto, el Elo y Dixon-Coles tratan a esas
## selecciones como equipos nuevos sin historia.
##
## NOTA sobre el mapeo: "Yugoslavia -> Serbia" y similares son
## convenciones (FIFA suele tratar a Serbia como el continuador
## administrativo de Yugoslavia/Serbia y Montenegro), pero es una
## simplificación real -- Croacia, Eslovenia, Bosnia, Macedonia del
## Norte y Montenegro también heredan parte de esa historia. Revisa
## team_name_map.csv y ajústalo si no estás de acuerdo con algún mapeo.
## =============================================================
suppressMessages({
  library(dplyr)
  library(lubridate)
  library(data.table)
})

DATA_DIR <- "data"

# ---- 1. Cargar resultados principales ----
results <- fread(file.path(DATA_DIR, "results.csv"), encoding = "UTF-8")
results[, date := as.Date(date)]
results <- results[!is.na(home_score) & !is.na(away_score)]

# ---- 1b. NUEVO: unificar nombres históricos de selecciones ----
name_map_path <- file.path(DATA_DIR, "team_name_map.csv")
if (file.exists(name_map_path) && file.info(name_map_path)$size > 0) {
  name_map <- fread(name_map_path, encoding = "UTF-8")
  if (nrow(name_map) > 0) {
    map_vec <- setNames(as.character(name_map$new_name), name_map$old_name)
    results[, home_team := fifelse(home_team %in% names(map_vec),
                                    as.character(map_vec[home_team]), home_team)]
    results[, away_team := fifelse(away_team %in% names(map_vec),
                                    as.character(map_vec[away_team]), away_team)]
    cat("Nombres unificados vía team_name_map.csv:", nrow(name_map), "reglas aplicadas.\n")
  } else {
    cat("team_name_map.csv está vacío -- se omite unificación de nombres.\n")
  }
} else {
  cat("AVISO: no se encontró data/team_name_map.csv -- se omite unificación de nombres.\n")
}

# ---- 2. Variable objetivo: resultado 1X2 ----
results[, result := fifelse(home_score > away_score, "H",
                      fifelse(home_score < away_score, "A", "D"))]
results[, goal_diff := home_score - away_score]
results[, total_goals := home_score + away_score]

# ---- 3. Peso de importancia por torneo ----
tournament_weight <- function(t) {
  t <- tolower(t)
  fifelse(grepl("world cup$", t), 1.00,
  fifelse(grepl("world cup qualification", t), 0.60,
  fifelse(grepl("confederations cup", t), 0.65,
  fifelse(grepl("euro$|copa am.rica$|african cup of nations$|afc asian cup$|gold cup$",
                t), 0.85,
  fifelse(grepl("qualification", t), 0.55,
  fifelse(grepl("nations league", t), 0.65,
  fifelse(grepl("friendly", t), 0.30,
          0.45)))))))
}
results[, importance := tournament_weight(tournament)]

# ---- 4. Confederación / neutral ----
results[, neutral := as.logical(neutral)]

# ---- 5. Orden cronológico ----
setorder(results, date)
results[, match_id := .I]

# ---- 6. Guardar dataset limpio ----
fwrite(results, file.path(DATA_DIR, "results_clean.csv"))

cat("Partidos totales:", nrow(results), "\n")
cat("Rango de fechas:", as.character(min(results$date)), "-",
    as.character(max(results$date)), "\n")
cat("Equipos únicos:", length(unique(c(results$home_team, results$away_team))), "\n")
print(table(results$result))
