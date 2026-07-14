## =============================================================
## predict_match.R — VERSIÓN CORREGIDA
## Fixes vs la versión anterior:
##  1. feature_cols ahora es EXACTAMENTE el de 04_feature_engineering.R
##     (22 features, confirmado también contra 05 y 07).
##  2. xgb se carga con readRDS() (así se guardó), no xgb.load().
##  3. El stacking se lee por NOMBRE de columna ("A","D","H"), no por
##     posición -- cv.glmnet devuelve las clases en ese orden porque
##     así se entrenó factor(result, levels=c("A","D","H")).
##  4. lambda_home usa gamma condicionado a !neutral, consistente con
##     el fix de 03_dixon_coles_FIX.R (debes re-correr 03->10 primero).
## =============================================================
suppressMessages({
  library(data.table)
  library(xgboost)
  library(glmnet)
  library(nnet)
})

## Rutas relativas a la carpeta api/ (donde vive este archivo y plumber.R).
## output/ y data/ están un nivel arriba, en la raíz del proyecto.
OUT_DIR  <- "../output"
DATA_DIR <- "../data"

## ---- Cargar todo UNA VEZ al arrancar la API ----
team_state     <- readRDS(file.path(OUT_DIR, "team_state.rds"))
dc_global      <- readRDS(file.path(OUT_DIR, "dc_global.rds"))
nnet_scaling   <- readRDS(file.path(OUT_DIR, "nnet_scaling.rds"))    # requiere ajuste en 07
stack_colnames <- readRDS(file.path(OUT_DIR, "stack_colnames.rds"))  # requiere ajuste en 08 (ya en 08_ensemble_FIX.R)

xgb_model   <- readRDS(file.path(OUT_DIR, "model_xgb_classifier.rds"))  # <- FIX: readRDS, no xgb.load
nnet_model  <- readRDS(file.path(OUT_DIR, "model_nnet.rds"))
stack_model <- readRDS(file.path(OUT_DIR, "model_stacking.rds"))

## Calibración isotónica guardada por 08_ensemble_FIX.R (opcional -- si no
## existe el archivo, se sirve sin calibrar y basta con que stack_colnames
## no tenga sufijo _cal)
calibration_path <- file.path(OUT_DIR, "calibration_functions.rds")
calibration_store <- if (file.exists(calibration_path)) readRDS(calibration_path) else NULL

apply_calibration <- function(pfx, home, draw, away) {
  if (is.null(calibration_store) || is.null(calibration_store[[pfx]])) {
    return(list(home = home, draw = draw, away = away))
  }
  cal <- calibration_store[[pfx]]
  cal_val <- function(part, raw) {
    if (is.null(cal[[part]])) return(raw)
    approxfun(cal[[part]]$x, cal[[part]]$yf, rule = 2)(raw)
  }
  h <- cal_val("home", home); d <- cal_val("draw", draw); a <- cal_val("away", away)
  tot <- h + d + a
  if (is.na(tot) || tot == 0) return(list(home = 1/3, draw = 1/3, away = 1/3))
  list(home = h/tot, draw = d/tot, away = a/tot)
}

results_hist <- fread(file.path(DATA_DIR, "results_clean.csv"), encoding = "UTF-8")
results_hist[, date := as.Date(date)]

## ---- El feature_cols REAL (idéntico a 04/05/07) ----
FEATURE_COLS <- c("elo_home_pre","elo_away_pre","elo_diff","elo_diff_abs",
                   "lambda_home","lambda_away","dc_rho",
                   "home_form_pts","away_form_pts","form_pts_diff",
                   "home_form_gf","away_form_gf","form_gf_diff",
                   "home_form_ga","away_form_ga",
                   "home_days_rest","away_days_rest","rest_diff",
                   "h2h_home_wr","h2h_n_prev","importance","neutral")

## ---- Poisson bivariado Dixon-Coles (idéntico a 08_ensemble.R, max_goals=10) ----
dc_probs_one <- function(lh, la, rho, max_goals = 10) {
  tau <- function(x, y) {
    if (x==0 && y==0) return(1 - lh*la*rho)
    if (x==0 && y==1) return(1 + lh*rho)
    if (x==1 && y==0) return(1 + la*rho)
    if (x==1 && y==1) return(1 - rho)
    1
  }
  grid <- expand.grid(x = 0:max_goals, y = 0:max_goals)
  grid$p <- mapply(function(x,y) dpois(x, lh) * dpois(y, la) * max(tau(x,y), 1e-8),
                    grid$x, grid$y)
  grid$p <- grid$p / sum(grid$p)
  list(
    home = sum(grid$p[grid$x > grid$y]),
    draw = sum(grid$p[grid$x == grid$y]),
    away = sum(grid$p[grid$x < grid$y]),
    over25  = sum(grid$p[grid$x + grid$y > 2]),
    under25 = sum(grid$p[grid$x + grid$y <= 2]),
    btts_yes = sum(grid$p[grid$x > 0 & grid$y > 0]),
    btts_no  = sum(grid$p[!(grid$x > 0 & grid$y > 0)])
  )
}

## ---- Head-to-head histórico entre dos selecciones específicas ----
get_h2h <- function(ht, at) {
  h <- results_hist[(home_team == ht & away_team == at) | (home_team == at & away_team == ht)]
  if (nrow(h) == 0) return(list(h2h_home_wr = 0.5, h2h_n_prev = 0))
  h[, home_persp_win := fifelse(home_team == ht & result == "H", 1,
                          fifelse(away_team == ht & result == "A", 1, 0))]
  list(h2h_home_wr = mean(h$home_persp_win), h2h_n_prev = nrow(h))
}

## ---- Función principal ----
predict_match <- function(home_team, away_team, neutral = FALSE, importance = 1.0) {

  th <- team_state[team == home_team]
  ta <- team_state[team == away_team]
  if (nrow(th) == 0) stop(paste("Selección no encontrada en team_state:", home_team))
  if (nrow(ta) == 0) stop(paste("Selección no encontrada en team_state:", away_team))

  gamma <- dc_global$gamma
  rho   <- dc_global$rho

  ## FIX: gamma condicionado a !neutral (consistente con 03_dixon_coles_FIX.R)
  home_adv_dc <- if (neutral) 0 else gamma
  lambda_home <- exp(th$att + ta$def + home_adv_dc)
  lambda_away <- exp(ta$att + th$def)

  dc <- dc_probs_one(lambda_home, lambda_away, rho)
  h2h <- get_h2h(home_team, away_team)

  days_rest_home <- as.numeric(Sys.Date() - th$last_match_date)
  days_rest_away <- as.numeric(Sys.Date() - ta$last_match_date)

  ## NOTA: elo_home_pre/elo_away_pre son el Elo CRUDO (sin ventaja de local
  ## sumada), igual que en 02_elo_ratings.R -- la ventaja de local solo se
  ## usa internamente para actualizar el Elo, no se guarda en la feature.
  feat <- data.table(
    elo_home_pre    = th$elo,
    elo_away_pre    = ta$elo,
    elo_diff        = th$elo - ta$elo,
    elo_diff_abs    = abs(th$elo - ta$elo),
    lambda_home     = lambda_home,
    lambda_away     = lambda_away,
    dc_rho          = rho,
    home_form_pts   = th$form_pts,  away_form_pts = ta$form_pts,
    form_pts_diff   = th$form_pts - ta$form_pts,
    home_form_gf    = th$form_gf,   away_form_gf  = ta$form_gf,
    form_gf_diff    = th$form_gf - ta$form_gf,
    home_form_ga    = th$form_ga,   away_form_ga  = ta$form_ga,
    home_days_rest  = days_rest_home, away_days_rest = days_rest_away,
    rest_diff       = days_rest_home - days_rest_away,
    h2h_home_wr     = h2h$h2h_home_wr,
    h2h_n_prev      = h2h$h2h_n_prev,
    importance      = importance,
    neutral         = as.integer(neutral)
  )
  feat <- feat[, ..FEATURE_COLS]  # fuerza el mismo orden que en entrenamiento

  ## ---- XGBoost (clasificador). Columnas de salida: p_away, p_draw, p_home ----
  xgb_input <- xgb.DMatrix(as.matrix(feat))
  xgb_raw <- predict(xgb_model, xgb_input, reshape = TRUE)
  xgb_p <- list(home = unname(xgb_raw[1,3]), draw = unname(xgb_raw[1,2]), away = unname(xgb_raw[1,1]))

  ## ---- Red neuronal (misma normalización que 07_model_deep_learning.R) ----
  feat_scaled <- scale(as.matrix(feat), center = nnet_scaling$mu, scale = nnet_scaling$sd)
  nn_raw <- predict(nnet_model, feat_scaled, type = "raw")  # columnas: away, draw, home
  nn_p <- list(home = unname(nn_raw[1,3]), draw = unname(nn_raw[1,2]), away = unname(nn_raw[1,1]))

  ## ---- Meta-learner (stacking) sobre las probabilidades base CALIBRADAS ----
  dc_cal  <- apply_calibration("dc",  dc$home,     dc$draw,     dc$away)
  nn_cal  <- apply_calibration("nn",  nn_p$home,   nn_p$draw,   nn_p$away)
  xgb_cal <- apply_calibration("xgb", xgb_p$home,  xgb_p$draw,  xgb_p$away)

  base_probs_all <- data.table(
    dc_p_home_cal = dc_cal$home,   dc_p_draw_cal = dc_cal$draw,   dc_p_away_cal = dc_cal$away,
    nn_p_home_cal = nn_cal$home,   nn_p_draw_cal = nn_cal$draw,   nn_p_away_cal = nn_cal$away,
    xgb_p_home_cal = xgb_cal$home, xgb_p_draw_cal = xgb_cal$draw, xgb_p_away_cal = xgb_cal$away,
    # también dejamos las versiones sin calibrar por si stack_colnames viene
    # de una corrida sin calibración (08_ensemble.R original, no el _FIX)
    dc_p_home = dc$home,   dc_p_draw = dc$draw,   dc_p_away = dc$away,
    nn_p_home = nn_p$home, nn_p_draw = nn_p$draw, nn_p_away = nn_p$away,
    xgb_p_home = xgb_p$home, xgb_p_draw = xgb_p$draw, xgb_p_away = xgb_p$away
  )
  ## Modelos que stack_colnames pueda pedir pero que NO podemos calcular en vivo
  ## (ej. bayes_p_*_cal -- el MCMC no se recalcula por request, es muy lento).
  ## Esos se imputan como "sin información" (1/3); el ridge del stacker ya
  ## les da poco peso relativo cuando no aportan señal real.
  missing_cols <- setdiff(stack_colnames, names(base_probs_all))
  for (mc in missing_cols) base_probs_all[[mc]] <- 1/3
  base_probs <- base_probs_all[, ..stack_colnames]

  stack_raw <- predict(stack_model, as.matrix(base_probs), s = "lambda.min", type = "response")[,,1]
  # FIX: extraer por nombre de dimensión ("A","D","H"), no por posición
  stack_p <- list(home = unname(stack_raw["H"]), draw = unname(stack_raw["D"]), away = unname(stack_raw["A"]))

  list(
    home_team = home_team, away_team = away_team,
    snapshot_date = as.character(dc_global$snapshot_date),
    lambda_home = round(lambda_home, 3), lambda_away = round(lambda_away, 3),
    dc_home = round(dc$home,4), dc_draw = round(dc$draw,4), dc_away = round(dc$away,4),
    dc_over25 = round(dc$over25,4), dc_under25 = round(dc$under25,4),
    dc_btts_yes = round(dc$btts_yes,4), dc_btts_no = round(dc$btts_no,4),
    nn_home = round(nn_p$home,4), nn_draw = round(nn_p$draw,4), nn_away = round(nn_p$away,4),
    xgb_home = round(xgb_p$home,4), xgb_draw = round(xgb_p$draw,4), xgb_away = round(xgb_p$away,4),
    stack_home = round(stack_p$home,4), stack_draw = round(stack_p$draw,4), stack_away = round(stack_p$away,4)
  )
}
