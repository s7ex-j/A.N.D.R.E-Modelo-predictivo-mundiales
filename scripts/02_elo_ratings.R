## =============================================================
## 02_elo_ratings.R
## Sistema de rating Elo tipo "World Football Elo Ratings"
## (eloratings.net), adaptado. Genera elo_home / elo_away
## PRE-PARTIDO (sin fuga de información) para cada fila.
## =============================================================
suppressMessages({ library(data.table) })

DATA_DIR <- "data"
results <- fread(file.path(DATA_DIR, "results_clean.csv"))
results[, date := as.Date(date)]
setorder(results, date, match_id)

INIT_ELO   <- 1500
HOME_ADV   <- 60     # ventaja de local en puntos Elo

# K efectivo según importancia del torneo (mismo espíritu que eloratings.net:
# amistoso=20, continental/qualy=40, final de grandes copas=60)
k_from_importance <- function(imp) 20 + imp * 60

elo <- new.env()
get_elo <- function(team) {
  if (is.null(elo[[team]])) elo[[team]] <- INIT_ELO
  elo[[team]]
}

n <- nrow(results)
elo_home_pre <- numeric(n)
elo_away_pre <- numeric(n)

for (i in seq_len(n)) {
  ht <- results$home_team[i]; at <- results$away_team[i]
  Rh <- get_elo(ht); Ra <- get_elo(at)
  elo_home_pre[i] <- Rh
  elo_away_pre[i] <- Ra

  neutral <- isTRUE(results$neutral[i])
  adv <- if (neutral) 0 else HOME_ADV
  dr <- (Rh + adv) - Ra

  We <- 1 / (10^(-dr / 400) + 1)   # prob. esperada de victoria local (con empate=0.5)
  gd <- abs(results$goal_diff[i])
  G  <- if (gd <= 1) 1 else if (gd == 2) 1.5 else (11 + gd) / 8   # multiplicador World Football Elo

  W <- if (results$result[i] == "H") 1 else if (results$result[i] == "D") 0.5 else 0
  K <- k_from_importance(results$importance[i])

  delta <- K * G * (W - We)
  elo[[ht]] <- Rh + delta
  elo[[at]] <- Ra - delta
}

results[, elo_home_pre := elo_home_pre]
results[, elo_away_pre := elo_away_pre]
results[, elo_diff := elo_home_pre - elo_away_pre]

fwrite(results, file.path(DATA_DIR, "results_elo.csv"))

cat("Elo calculado para", n, "partidos.\n")
cat("\nTop 10 Elo actual (a", as.character(max(results$date)), "):\n")
final_elo <- sort(sapply(ls(elo), function(t) elo[[t]]), decreasing = TRUE)[1:10]
print(round(final_elo, 1))
