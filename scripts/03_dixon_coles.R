## =============================================================
## 03_dixon_coles.R — VERSIÓN CORREGIDA (revisión 2)
##
## HISTORIAL DE ESTE FIX:
## Mi primer intento condicionaba gamma (ventaja de local) a que el
## partido NO fuera neutral, asumiendo que "neutral" = sin ventaja.
## Al probarlo con tus 49,493 partidos reales, el accuracy de Dixon-
## Coles CAYÓ de 0.5632 a 0.5423. Investigué por qué: en tus datos,
## incluso en partidos marcados "neutral", el equipo etiquetado
## home_team gana 44.2% / empata 22.4% / pierde 33.4% -- NO es 50/50
## como esperarías de una etiqueta verdaderamente arbitraria. Hay
## señal real ahí (probablemente sede/región/afición aunque no sea
## "cancha propia" en sentido estricto, o convención de cómo la
## fuente de datos designa "home" en un cruce neutral).
##
## CONCLUSIÓN: para el ENTRENAMIENTO histórico, es mejor dejar que
## el modelo use esa señal real (gamma incondicional, como el script
## original). El problema real no es el entrenamiento -- es que al
## SERVIR una predicción para un partido hipotético tuyo (ej. "España
## vs Bélgica" en un cuarto de final que ni siquiera se ha jugado),
## NO existe un "home_team" real -- es solo el orden en que los
## elegiste en el dropdown. Ahí SÍ no tiene sentido darle el bonus de
## gamma a quien listaste primero. Por eso: entrenamiento SIN
## condicionar (este script, igual al original), pero en
## predict_match.R SÍ se sigue forzando gamma=0 cuando neutral=TRUE
## para tus escenarios hipotéticos del Mundial 2026.
## =============================================================
suppressMessages({ library(data.table) })

DATA_DIR <- "data"
results <- fread(file.path(DATA_DIR, "results_elo.csv"))
results[, date := as.Date(date)]
setorder(results, date, match_id)

FIT_FROM   <- as.Date("2002-01-01")
XI_DECAY   <- 0.0018
REFIT_EVERY_DAYS <- 90

sub <- results[date >= FIT_FROM]
teams <- sort(unique(c(sub$home_team, sub$away_team)))
nT <- length(teams)
idx <- setNames(seq_len(nT), teams)

tau_logderivs <- function(x, y, lh, la, rho) {
  logtau <- numeric(length(x)); dlh <- numeric(length(x))
  dla <- numeric(length(x));    drho <- numeric(length(x))

  i00 <- x == 0 & y == 0
  v <- pmax(1 - lh[i00]*la[i00]*rho, 1e-8)
  logtau[i00] <- log(v); dlh[i00] <- -la[i00]*rho/v; dla[i00] <- -lh[i00]*rho/v
  drho[i00] <- -lh[i00]*la[i00]/v

  i01 <- x == 0 & y == 1
  v <- pmax(1 + lh[i01]*rho, 1e-8)
  logtau[i01] <- log(v); dlh[i01] <- rho/v; drho[i01] <- lh[i01]/v

  i10 <- x == 1 & y == 0
  v <- pmax(1 + la[i10]*rho, 1e-8)
  logtau[i10] <- log(v); dla[i10] <- rho/v; drho[i10] <- la[i10]/v

  i11 <- x == 1 & y == 1
  v <- pmax(1 - rho, 1e-8)
  logtau[i11] <- log(v); drho[i11] <- -1/v

  list(logtau = logtau, dlh = dlh, dla = dla, drho = drho)
}

L2_LAMBDA <- 0.03

neg_loglik <- function(par, dat) {
  att <- par[1:nT]; def <- par[(nT+1):(2*nT)]
  gamma <- par[2*nT+1]; rho <- par[2*nT+2]

  lh <- exp(att[dat$hi] + def[dat$ai] + gamma)
  la <- exp(att[dat$ai] + def[dat$hi])
  tt <- tau_logderivs(dat$home_score, dat$away_score, lh, la, rho)

  ll <- dat$w * (dpois(dat$home_score, lh, log = TRUE) +
                 dpois(dat$away_score, la, log = TRUE) + tt$logtau)
  penalty <- L2_LAMBDA * sum(att^2 + def^2)
  -sum(ll) + penalty
}

neg_grad <- function(par, dat) {
  att <- par[1:nT]; def <- par[(nT+1):(2*nT)]
  gamma <- par[2*nT+1]; rho <- par[2*nT+2]

  lh <- exp(att[dat$hi] + def[dat$ai] + gamma)
  la <- exp(att[dat$ai] + def[dat$hi])
  tt <- tau_logderivs(dat$home_score, dat$away_score, lh, la, rho)

  c_lh <- dat$w * ((dat$home_score - lh) + tt$dlh * lh)
  c_la <- dat$w * ((dat$away_score - la) + tt$dla * la)

  ga <- numeric(nT); gd <- numeric(nT)
  ga_h <- tapply(c_lh, dat$hi, sum); ga[as.integer(names(ga_h))] <- ga_h
  ga_a <- tapply(c_la, dat$ai, sum); ga[as.integer(names(ga_a))] <- ga[as.integer(names(ga_a))] + ga_a
  gd_a <- tapply(c_lh, dat$ai, sum); gd[as.integer(names(gd_a))] <- gd_a
  gd_h <- tapply(c_la, dat$hi, sum); gd[as.integer(names(gd_h))] <- gd[as.integer(names(gd_h))] + gd_h

  g_gamma <- sum(c_lh)
  g_rho <- sum(dat$w * tt$drho)

  grad_ll <- c(ga, gd, g_gamma, g_rho)
  grad_penalty <- c(2 * L2_LAMBDA * att, 2 * L2_LAMBDA * def, 0, 0)
  -grad_ll + grad_penalty
}

fit_dixon_coles <- function(train_dat) {
  train_dat <- copy(train_dat)
  train_dat[, hi := idx[home_team]]
  train_dat[, ai := idx[away_team]]
  train_dat <- train_dat[!is.na(hi) & !is.na(ai)]
  max_date <- max(train_dat$date)
  train_dat[, w := exp(-XI_DECAY * as.numeric(max_date - date))]

  par0 <- c(rep(0, nT), rep(0, nT), 0.2, 0.0)
  opt <- optim(par0, neg_loglik, gr = neg_grad, dat = train_dat, method = "BFGS",
               control = list(maxit = 200, reltol = 1e-9))
  list(att = opt$par[1:nT] - mean(opt$par[1:nT]),
       def = opt$par[(nT+1):(2*nT)],
       gamma = opt$par[2*nT+1], rho = opt$par[2*nT+2],
       conv = opt$convergence)
}

sub[, lambda_home := NA_real_]
sub[, lambda_away := NA_real_]
sub[, dc_rho := NA_real_]

cut_points <- seq(FIT_FROM + 365, max(sub$date), by = REFIT_EVERY_DAYS)
cat("Ajustando Dixon-Coles walk-forward en", length(cut_points), "cortes...\n")

for (k in seq_along(cut_points)) {
  cp <- cut_points[k]
  train <- sub[date < cp]
  test_end <- if (k < length(cut_points)) cut_points[k+1] else max(sub$date) + 1
  test_idx <- which(sub$date >= cp & sub$date < test_end)
  if (length(test_idx) == 0 || nrow(train) < 200) next

  fit <- tryCatch(fit_dixon_coles(train), error = function(e) NULL)
  if (is.null(fit)) next

  hi <- idx[sub$home_team[test_idx]]; ai <- idx[sub$away_team[test_idx]]
  ok <- !is.na(hi) & !is.na(ai)
  sub$lambda_home[test_idx[ok]] <- exp(fit$att[hi[ok]] + fit$def[ai[ok]] + fit$gamma)
  sub$lambda_away[test_idx[ok]] <- exp(fit$att[ai[ok]] + fit$def[hi[ok]])
  sub$dc_rho[test_idx[ok]] <- fit$rho

  if (k %% 10 == 0) cat("  corte", k, "/", length(cut_points), "-", as.character(cp), "\n")
}

fwrite(sub, file.path(DATA_DIR, "results_dixoncoles.csv"))
cat("\nListo. Cobertura lambda (no-NA):",
    round(100*mean(!is.na(sub$lambda_home)), 1), "%\n")
print(summary(sub$lambda_home))

