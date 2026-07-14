## =============================================================
## 04_feature_engineering.R
## Combina Elo + Dixon-Coles + forma reciente + contexto de partido
## en una única matriz de features, lista para XGBoost / Bayes / DL.
## Todo se calcula PRE-PARTIDO (walk-forward, sin fuga de datos).
## =============================================================
suppressMessages({ library(data.table) })

DATA_DIR <- "data"
dt <- fread(file.path(DATA_DIR, "results_dixoncoles.csv"))
dt[, date := as.Date(date)]
setorder(dt, date, match_id)

# ---- 1. Forma reciente (rolling, últimos N partidos, PRE-partido) ----
# Para cada equipo, en cada partido: puntos y goles de sus últimos 5 partidos
# (como local o visita), calculado ANTES del partido actual.
build_rolling_form <- function(dt, N = 5) {
  long <- rbindlist(list(
    dt[, .(match_id, date, team = home_team, opp = away_team,
           gf = home_score, ga = away_score,
           pts = fifelse(result == "H", 3, fifelse(result == "D", 1, 0)))],
    dt[, .(match_id, date, team = away_team, opp = home_team,
           gf = away_score, ga = home_score,
           pts = fifelse(result == "A", 3, fifelse(result == "D", 1, 0)))]
  ))
  setorder(long, team, date, match_id)
  long[, `:=`(
    form_pts   = shift(frollmean(pts, N, align = "right"), 1),
    form_gf    = shift(frollmean(gf,  N, align = "right"), 1),
    form_ga    = shift(frollmean(ga,  N, align = "right"), 1),
    days_rest  = as.numeric(date - shift(date, 1))
  ), by = team]
  long[, .(match_id, team, form_pts, form_gf, form_ga, days_rest)]
}

form <- build_rolling_form(dt, N = 5)

dt <- merge(dt, form, by.x = c("match_id", "home_team"),
            by.y = c("match_id", "team"), all.x = TRUE)
setnames(dt, c("form_pts", "form_gf", "form_ga", "days_rest"),
         c("home_form_pts", "home_form_gf", "home_form_ga", "home_days_rest"))

dt <- merge(dt, form, by.x = c("match_id", "away_team"),
            by.y = c("match_id", "team"), all.x = TRUE)
setnames(dt, c("form_pts", "form_gf", "form_ga", "days_rest"),
         c("away_form_pts", "away_form_gf", "away_form_ga", "away_days_rest"))

setorder(dt, date, match_id)

# ---- 2. Head-to-head histórico (racha directa, PRE-partido) ----
dt[, pair_key := paste(pmin(home_team, away_team), pmax(home_team, away_team))]
dt[, h2h_home_winrate := {
  # % de victorias históricas del home_team actual en enfrentamientos previos
  # entre este par de equipos (sin importar quién fue local en el pasado)
  n <- .N
  out <- rep(NA_real_, n)
  out
}, by = pair_key]
# (h2h calculado de forma vectorizada más abajo para performance)
dt[, h2h_idx := .I]
h2h_long <- rbindlist(list(
  dt[, .(h2h_idx, date, pair_key, team = home_team, win = as.integer(result == "H"))],
  dt[, .(h2h_idx, date, pair_key, team = away_team, win = as.integer(result == "A"))]
))
setorder(h2h_long, pair_key, team, date, h2h_idx)
h2h_long[, cum_games := seq_len(.N) - 1, by = .(pair_key, team)]
h2h_long[, cum_wins := shift(cumsum(win), 1, fill = 0), by = .(pair_key, team)]
h2h_long[, h2h_winrate := fifelse(cum_games == 0, NA_real_, cum_wins / cum_games)]

h2h_map <- h2h_long[, .(h2h_idx, team, h2h_winrate, cum_games)]
dt <- merge(dt, h2h_map, by.x = c("h2h_idx", "home_team"),
            by.y = c("h2h_idx", "team"), all.x = TRUE)
setnames(dt, c("h2h_winrate", "cum_games"), c("h2h_home_wr", "h2h_n_prev"))

# ---- 3. Features finales ----
dt[, `:=`(
  elo_diff_abs = abs(elo_diff),
  home_favored = as.integer(elo_diff > 0),
  form_pts_diff = home_form_pts - away_form_pts,
  form_gf_diff  = home_form_gf - away_form_gf,
  rest_diff = home_days_rest - away_days_rest
)]

# Target codificado para modelos (0=Away,1=Draw,2=Home) útil para xgboost multiclase
dt[, y_result := fifelse(result == "A", 0L, fifelse(result == "D", 1L, 2L))]

feature_cols <- c("elo_home_pre","elo_away_pre","elo_diff","elo_diff_abs",
                   "lambda_home","lambda_away","dc_rho",
                   "home_form_pts","away_form_pts","form_pts_diff",
                   "home_form_gf","away_form_gf","form_gf_diff",
                   "home_form_ga","away_form_ga",
                   "home_days_rest","away_days_rest","rest_diff",
                   "h2h_home_wr","h2h_n_prev","importance",
                   "neutral")

model_dt <- dt[, c("match_id","date","home_team","away_team",
                    "home_score","away_score","result","y_result",
                    feature_cols), with = FALSE]
model_dt[, neutral := as.integer(neutral)]

# ---- Imputación simple para partidos con poco historial ----
# (primeros partidos de una selección o primer cruce entre dos equipos)
model_dt[is.na(h2h_home_wr), h2h_home_wr := 0.5]
model_dt[is.na(h2h_n_prev),  h2h_n_prev  := 0]
for (col in c("home_form_pts","away_form_pts","form_pts_diff",
              "home_form_gf","away_form_gf","form_gf_diff",
              "home_form_ga","away_form_ga",
              "home_days_rest","away_days_rest","rest_diff")) {
  med <- median(model_dt[[col]], na.rm = TRUE)
  model_dt[is.na(get(col)), (col) := med]
}
model_dt <- model_dt[!is.na(lambda_home) & !is.na(lambda_away)]  # requiere Dixon-Coles

fwrite(model_dt, file.path(DATA_DIR, "model_features.csv"))
cat("Matriz de features:", nrow(model_dt), "x", ncol(model_dt), "\n")
cat("Filas completas (sin NA en features clave):",
    sum(complete.cases(model_dt[, ..feature_cols])), "\n")
print(names(model_dt))
