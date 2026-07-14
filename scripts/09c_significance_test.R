## =============================================================
## 09c_significance_test.R
## ¿La diferencia de accuracy entre modelos es señal real o ruido?
## Con solo 88 partidos de Mundial, una diferencia de "61.4% vs 62.5%"
## puede ser pura casualidad de muestra chica. Esto lo prueba con
## bootstrap (remuestreo): repite 5,000 veces "toma una muestra con
## reemplazo de estos 88 partidos y mide el accuracy", y así construye
## un intervalo de confianza real para cada modelo y para la diferencia
## entre pares de modelos.
##
## Corre DESPUÉS de 09b_evaluation_worldcup.R (usa el mismo ens_wc).
## =============================================================
suppressMessages({ library(data.table) })
set.seed(42)

DATA_DIR <- "data"
ens <- fread(file.path(DATA_DIR, "ensemble_predictions.csv"))
results_clean <- fread(file.path(DATA_DIR, "results_clean.csv"))
results_clean[, date := as.Date(date)]

tn <- trimws(tolower(results_clean$tournament))
wc_ids <- results_clean[tn == "fifa world cup", match_id]
ens_wc <- ens[match_id %in% wc_ids]

if (nrow(ens_wc) < 20) {
  stop("Muy pocos partidos de Mundial (", nrow(ens_wc), ") para que el bootstrap ",
       "diga algo útil. Espera a que se jueguen más partidos.")
}

modelos_disponibles <- unique(gsub("_p_(home|draw|away)$", "",
                          grep("_p_home$", names(ens_wc), value = TRUE)))
cat("Modelos encontrados:", paste(modelos_disponibles, collapse=", "), "\n")
cat("Partidos de Mundial en la muestra:", nrow(ens_wc), "\n\n")

accuracy_of <- function(dat, prefix) {
  pa <- dat[[paste0(prefix,"_p_away")]]; pd <- dat[[paste0(prefix,"_p_draw")]]; ph <- dat[[paste0(prefix,"_p_home")]]
  if (is.null(pa)) return(NA_real_)
  ok <- !is.na(pa) & !is.na(pd) & !is.na(ph)
  if (sum(ok) < 10) return(NA_real_)
  pred <- c("A","D","H")[apply(cbind(pa[ok],pd[ok],ph[ok]), 1, which.max)]
  mean(pred == dat$result[ok])
}

N_BOOT <- 5000
n <- nrow(ens_wc)
boot_acc <- matrix(NA_real_, nrow = N_BOOT, ncol = length(modelos_disponibles),
                    dimnames = list(NULL, modelos_disponibles))

for (b in seq_len(N_BOOT)) {
  idx <- sample.int(n, n, replace = TRUE)
  boot_sample <- ens_wc[idx]
  for (m in modelos_disponibles) {
    boot_acc[b, m] <- accuracy_of(boot_sample, m)
  }
}

cat("=== Intervalo de confianza 95% del accuracy por modelo (bootstrap, n=", N_BOOT, ") ===\n", sep="")
ci_table <- rbindlist(lapply(modelos_disponibles, function(m) {
  ci <- quantile(boot_acc[, m], c(0.025, 0.5, 0.975), na.rm = TRUE)
  data.table(modelo = m, accuracy_observado = round(accuracy_of(ens_wc, m), 4),
             ci_2.5 = round(ci[1], 4), mediana_boot = round(ci[2], 4), ci_97.5 = round(ci[3], 4))
}))
print(ci_table)

cat("\n=== ¿Es real la diferencia stack vs. cada otro modelo? ===\n")
cat("(si el intervalo de la diferencia NO cruza el 0, la diferencia es estadísticamente\n")
cat(" significativa al 95%; si cruza el 0, con esta cantidad de partidos no se puede\n")
cat(" afirmar que un modelo sea mejor que otro todavía)\n\n")

if ("stack" %in% modelos_disponibles) {
  for (m in setdiff(modelos_disponibles, "stack")) {
    obs_diff <- accuracy_of(ens_wc, "stack") - accuracy_of(ens_wc, m)
    if (is.na(obs_diff)) {
      cat(sprintf("stack vs %-6s: sin datos suficientes para comparar (modelo con cobertura parcial)\n", m))
      next
    }
    diff_boot <- boot_acc[, "stack"] - boot_acc[, m]
    ci <- quantile(diff_boot, c(0.025, 0.975), na.rm = TRUE)
    significativo <- isTRUE(ci[1] > 0) || isTRUE(ci[2] < 0)
    cat(sprintf("stack vs %-6s: diferencia observada = %+.4f | IC 95%% = [%+.4f, %+.4f] | %s\n",
                m, obs_diff, ci[1], ci[2],
                if (significativo) "SIGNIFICATIVO" else "no concluyente todavía (muestra chica)"))
  }
}

fwrite(ci_table, file.path(DATA_DIR, "significance_worldcup.csv"))
cat("\nGuardado: data/significance_worldcup.csv\n")
