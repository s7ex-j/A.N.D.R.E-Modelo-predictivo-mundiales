## =============================================================
## 00_config.R
## Parámetros compartidos por los scripts de modelado y evaluación.
## Cambiar algo aquí lo cambia en todo el pipeline (antes estaba
## copiado en cada script y era fácil que se desalinearan).
## =============================================================

DATA_DIR <- "data"
OUT_DIR  <- "output"

## Inicio del test set: todo lo anterior es entrenamiento.
SPLIT_DATE <- as.Date("2023-01-01")

## Años del periodo de entrenamiento que se predicen "out-of-fold" en
## walk-forward (para cada año Y, los modelos se entrenan con todo lo
## anterior a Y y predicen Y). Esas predicciones sirven para:
##  - elegir hiperparámetros de XGBoost con validación temporal,
##  - ajustar la calibración isotónica,
##  - entrenar el meta-learner del stacking.
## Así el test set completo queda libre para evaluar a todos los modelos.
META_YEARS <- 2019:2022

## Modelos que entran al promedio y al stacking. El bayesiano se reporta
## aparte: cubre solo una ventana corta del test (ver 06).
ENSEMBLE_MODELS <- c("dc", "nn", "xgb")

## Las 22 features (idénticas en 05, 07 y api/predict_match.R).
FEATURE_COLS <- c("elo_home_pre","elo_away_pre","elo_diff","elo_diff_abs",
                  "lambda_home","lambda_away","dc_rho",
                  "home_form_pts","away_form_pts","form_pts_diff",
                  "home_form_gf","away_form_gf","form_gf_diff",
                  "home_form_ga","away_form_ga",
                  "home_days_rest","away_days_rest","rest_diff",
                  "h2h_home_wr","h2h_n_prev","importance","neutral")

## Bootstrap para intervalos de confianza (09, 09c, 09d).
N_BOOT <- 5000
