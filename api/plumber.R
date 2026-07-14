## =============================================================
## plumber.R
## La API en sí. Corre esto con:
##   library(plumber)
##   pr <- plumb("plumber.R")
##   pr$run(host = "0.0.0.0", port = 8000)
##
## Deja predict_match.R en la MISMA carpeta que este archivo.
## =============================================================
library(plumber)

source("predict_match.R")   # carga modelos + team_state UNA vez al arrancar

#* Habilitar CORS para que el HTML (otro origen) pueda llamar a esta API
#* @filter cors
function(req, res) {
  res$setHeader("Access-Control-Allow-Origin", "*")
  if (req$REQUEST_METHOD == "OPTIONS") {
    res$setHeader("Access-Control-Allow-Methods", "POST,GET,OPTIONS")
    res$setHeader("Access-Control-Allow-Headers", "Content-Type")
    res$status <- 200
    return(list())
  }
  plumber::forward()
}

#* Chequeo rápido de que la API está viva
#* @serializer unboxedJSON
#* @get /health
function() {
  list(status = "ok", snapshot_date = as.character(dc_global$snapshot_date),
       n_teams = nrow(team_state))
}

#* Predicción de un partido
#* @param home_team Nombre de la selección local (ej. "Brazil")
#* @param away_team Nombre de la selección visitante (ej. "Argentina")
#* @param neutral Cancha neutral (true/false)
#* @param importance Peso del partido (0-1, Mundial = 1.0)
#* @serializer unboxedJSON
#* @post /predict
function(req, res, home_team = "", away_team = "", neutral = FALSE, importance = 1.0) {
  body <- tryCatch(jsonlite::fromJSON(req$postBody), error = function(e) list())
  ht <- body$home_team %||% home_team
  at <- body$away_team %||% away_team
  nt <- isTRUE(body$neutral %||% neutral)
  imp <- as.numeric(body$importance %||% importance)

  tryCatch({
    predict_match(ht, at, neutral = nt, importance = imp)
  }, error = function(e) {
    res$status <- 400
    list(error = conditionMessage(e))
  })
}

`%||%` <- function(a, b) if (is.null(a) || identical(a, "")) b else a
