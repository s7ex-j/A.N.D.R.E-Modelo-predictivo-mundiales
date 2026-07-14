## =============================================================
## 10_build_team_state.R
## Corre esto UNA VEZ AL DÍA (o después de cada jornada del Mundial)
## para dejar un snapshot listo que la API pueda leer sin tener que
## re-entrenar nada en cada request.
##
## Requiere haber corrido 01_data_prep.R -> 04_feature_engineering.R
## al menos una vez (usa results_clean.csv como fuente).
##
## Genera:
##   output/team_state.rds   (data.table: 1 fila por selección)
##   output/dc_global.rds    (gamma y rho del último ajuste Dixon-Coles)
## =============================================================
suppressMessages({ library(data.table) })

DATA_DIR <- "data"
OUT_DIR  <- "output"
dir.create(OUT_DIR, showWarnings = FALSE)

results <- fread(file.path(DATA_DIR, "results_clean.csv"), encoding = "UTF-8")
results[, date := as.Date(date)]
setorder(results, date, match_id)

TODAY <- Sys.Date()

## ---------------------------------------------------------------
## 1. ELO ACTUAL (estado final del sistema Elo, no el pre-partido)
##    Reproduce exactamente la lógica de 02_elo_ratings.R pero se
##    queda con el rating FINAL de cada selección, no el histórico.
## ---------------------------------------------------------------
INIT_ELO <- 1500
HOME_ADV <- 60
k_from_importance <- function(imp) 20 + imp * 60

elo <- new.env()
get_elo <- function(team) { if (is.null(elo[[team]])) elo[[team]] <- INIT_ELO; elo[[team]] }

for (i in seq_len(nrow(results))) {
  ht <- results$home_team[i]; at <- results$away_team[i]
  Rh <- get_elo(ht); Ra <- get_elo(at)
  neutral <- isTRUE(results$neutral[i])
  adv <- if (neutral) 0 else HOME_ADV
  dr <- (Rh + adv) - Ra
  We <- 1 / (10^(-dr/400) + 1)
  gd <- abs(results$goal_diff[i])
  G  <- if (gd <= 1) 1 else if (gd == 2) 1.5 else (11 + gd) / 8
  W <- if (results$result[i] == "H") 1 else if (results$result[i] == "D") 0.5 else 0
  K <- k_from_importance(results$importance[i])
  delta <- K * G * (W - We)
  elo[[ht]] <- Rh + delta
  elo[[at]] <- Ra - delta
}
elo_now <- data.table(team = ls(elo), elo = sapply(ls(elo), function(t) elo[[t]]))

## ---------------------------------------------------------------
## 2. DIXON-COLES: UN SOLO AJUSTE CON TODA LA DATA HASTA HOY
##    (para producción no usamos walk-forward -- eso era solo para
##    medir accuracy honesto en el backtest. Aquí sí usamos toda la
##    información disponible, porque no hay fuga posible: estamos
##    prediciendo partidos que TODAVÍA no pasaron)
## ---------------------------------------------------------------
FIT_FROM <- as.Date("2002-01-01")
XI_DECAY <- 0.0018
L2_LAMBDA <- 0.03

sub <- results[date >= FIT_FROM]
teams <- sort(unique(c(sub$home_team, sub$away_team)))
nT <- length(teams)
idx <- setNames(seq_len(nT), teams)

tau_logderivs <- function(x, y, lh, la, rho) {
  logtau <- numeric(length(x)); dlh <- numeric(length(x))
  dla <- numeric(length(x));    drho <- numeric(length(x))
  i00 <- x==0 & y==0; v <- pmax(1-lh[i00]*la[i00]*rho,1e-8)
  logtau[i00]<-log(v); dlh[i00]<-(-la[i00]*rho/v); dla[i00]<-(-lh[i00]*rho/v); drho[i00]<-(-lh[i00]*la[i00]/v)
  i01 <- x==0 & y==1; v <- pmax(1+lh[i01]*rho,1e-8)
  logtau[i01]<-log(v); dlh[i01]<-rho/v; drho[i01]<-lh[i01]/v
  i10 <- x==1 & y==0; v <- pmax(1+la[i10]*rho,1e-8)
  logtau[i10]<-log(v); dla[i10]<-rho/v; drho[i10]<-la[i10]/v
  i11 <- x==1 & y==1; v <- pmax(1-rho,1e-8)
  logtau[i11]<-log(v); drho[i11]<-(-1/v)
  list(logtau=logtau, dlh=dlh, dla=dla, drho=drho)
}
neg_loglik <- function(par, dat) {
  att <- par[1:nT]; def <- par[(nT+1):(2*nT)]; gamma <- par[2*nT+1]; rho <- par[2*nT+2]
  lh <- exp(att[dat$hi]+def[dat$ai]+gamma); la <- exp(att[dat$ai]+def[dat$hi])
  tt <- tau_logderivs(dat$home_score, dat$away_score, lh, la, rho)
  ll <- dat$w * (dpois(dat$home_score, lh, log=TRUE) + dpois(dat$away_score, la, log=TRUE) + tt$logtau)
  -sum(ll) + L2_LAMBDA*sum(att^2+def^2)
}
neg_grad <- function(par, dat) {
  att <- par[1:nT]; def <- par[(nT+1):(2*nT)]; gamma <- par[2*nT+1]; rho <- par[2*nT+2]
  lh <- exp(att[dat$hi]+def[dat$ai]+gamma); la <- exp(att[dat$ai]+def[dat$hi])
  tt <- tau_logderivs(dat$home_score, dat$away_score, lh, la, rho)
  c_lh <- dat$w*((dat$home_score-lh)+tt$dlh*lh); c_la <- dat$w*((dat$away_score-la)+tt$dla*la)
  ga <- numeric(nT); gd <- numeric(nT)
  ga_h <- tapply(c_lh, dat$hi, sum); ga[as.integer(names(ga_h))] <- ga_h
  ga_a <- tapply(c_la, dat$ai, sum); ga[as.integer(names(ga_a))] <- ga[as.integer(names(ga_a))]+ga_a
  gd_a <- tapply(c_lh, dat$ai, sum); gd[as.integer(names(gd_a))] <- gd_a
  gd_h <- tapply(c_la, dat$hi, sum); gd[as.integer(names(gd_h))] <- gd[as.integer(names(gd_h))]+gd_h
  grad_ll <- c(ga, gd, sum(c_lh), sum(dat$w*tt$drho))
  -grad_ll + c(2*L2_LAMBDA*att, 2*L2_LAMBDA*def, 0, 0)
}

train_dat <- copy(sub)
train_dat[, hi := idx[home_team]]; train_dat[, ai := idx[away_team]]
train_dat <- train_dat[!is.na(hi) & !is.na(ai)]
train_dat[, w := exp(-XI_DECAY * as.numeric(TODAY - date))]

par0 <- c(rep(0,nT), rep(0,nT), 0.2, 0.0)
opt <- optim(par0, neg_loglik, gr = neg_grad, dat = train_dat, method = "BFGS",
             control = list(maxit = 300, reltol = 1e-9))

att <- opt$par[1:nT] - mean(opt$par[1:nT])
def <- opt$par[(nT+1):(2*nT)]
gamma <- opt$par[2*nT+1]
rho   <- opt$par[2*nT+2]
dc_now <- data.table(team = teams, att = att, def = def)

## ---------------------------------------------------------------
## 3. FORMA RECIENTE (últimos 5 partidos reales de cada selección)
##    y días de descanso desde su último partido
## ---------------------------------------------------------------
long <- rbindlist(list(
  results[, .(date, team = home_team, gf = home_score, ga = away_score,
              pts = fifelse(result=="H",3,fifelse(result=="D",1,0)))],
  results[, .(date, team = away_team, gf = away_score, ga = home_score,
              pts = fifelse(result=="A",3,fifelse(result=="D",1,0)))]
))
setorder(long, team, date)
form_now <- long[, .(
  form_pts = mean(tail(pts, 5)),
  form_gf  = mean(tail(gf, 5)),
  form_ga  = mean(tail(ga, 5)),
  last_match_date = max(date)
), by = team]

## ---------------------------------------------------------------
## 4. UNIR TODO Y GUARDAR
## ---------------------------------------------------------------
team_state <- merge(elo_now, dc_now, by = "team", all.x = TRUE)
team_state <- merge(team_state, form_now, by = "team", all.x = TRUE)
team_state[is.na(att), att := 0]  # selecciones sin partidos desde 2002 -> ataque/defensa neutro
team_state[is.na(def), def := 0]

saveRDS(team_state, file.path(OUT_DIR, "team_state.rds"))
saveRDS(list(gamma = gamma, rho = rho, snapshot_date = TODAY),
        file.path(OUT_DIR, "dc_global.rds"))

cat("team_state.rds guardado —", nrow(team_state), "selecciones.\n")
cat("dc_global.rds guardado — gamma:", round(gamma,3), " rho:", round(rho,3), "\n")
cat("Snapshot generado el:", as.character(TODAY), "\n")
cat("\nNota: si ya estás usando 07_model_deep_learning.R y 08_ensemble.R en\n")
cat("sus versiones _FIX (las que vienen en la auditoría), nnet_scaling.rds,\n")
cat("stack_colnames.rds y calibration_functions.rds ya se guardan solos.\n")
