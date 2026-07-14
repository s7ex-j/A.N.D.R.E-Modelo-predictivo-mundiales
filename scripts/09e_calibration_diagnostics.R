## =============================================================
## 09e_calibration_diagnostics.R
## Accuracy solo mide "¿acertó la clase más probable?" -- no mide si
## las PROBABILIDADES en sí tienen sentido. Un modelo puede tener buen
## accuracy y estar pésimamente calibrado (ej. decir "90% de ganar"
## en partidos que en realidad gana solo el 60% de las veces).
##
## Esto agrupa las predicciones de "gana local" en 10 grupos según la
## probabilidad que dio el modelo, y compara contra la frecuencia real
## de victorias en cada grupo. Un modelo bien calibrado: el grupo
## "predicción ~70%" debe tener ~70% de victorias reales -- no más,
## no menos.
## =============================================================
suppressMessages({ library(data.table) })

DATA_DIR <- "data"
ens <- fread(file.path(DATA_DIR, "ensemble_predictions.csv"))

modelos_disponibles <- unique(gsub("_p_(home|draw|away)$", "",
                          grep("_p_home$", names(ens), value = TRUE)))

reliability_table <- function(dat, prefix, n_bins = 10) {
  p <- dat[[paste0(prefix, "_p_home")]]
  y <- as.integer(dat$result == "H")
  ok <- !is.na(p)
  p <- p[ok]; y <- y[ok]
  if (length(p) < 30) return(NULL)

  bins <- cut(p, breaks = quantile(p, seq(0, 1, length.out = n_bins + 1)),
              include.lowest = TRUE, labels = FALSE)
  dt <- data.table(bin = bins, p = p, y = y)
  agg <- dt[, .(n = .N, prob_predicha_prom = round(mean(p), 3),
                frecuencia_real = round(mean(y), 3)), by = bin]
  agg[, modelo := prefix]
  agg[, brecha := round(abs(prob_predicha_prom - frecuencia_real), 3)]
  setorder(agg, bin)
  agg
}

cat("=== Calibración de 'gana local' por modelo (10 bins, test completo) ===\n\n")
all_tables <- list()
for (m in modelos_disponibles) {
  rt <- reliability_table(ens, m)
  if (is.null(rt)) { cat(m, ": muy pocos datos, se omite\n\n"); next }
  cat("---", m, "---\n")
  print(rt[, .(bin, n, prob_predicha_prom, frecuencia_real, brecha)])
  cat("Brecha promedio (calibración):", round(mean(rt$brecha), 4),
      " | Brecha máxima:", round(max(rt$brecha), 4), "\n\n")
  all_tables[[m]] <- rt
}

## Ranking de qué tan bien calibrado está cada modelo (menor brecha = mejor)
resumen <- rbindlist(lapply(names(all_tables), function(m) {
  data.table(modelo = m, brecha_promedio = round(mean(all_tables[[m]]$brecha), 4))
}))
setorder(resumen, brecha_promedio)
cat("=== Ranking de calibración (menor = mejor) ===\n")
print(resumen)

fwrite(rbindlist(all_tables), file.path(DATA_DIR, "calibration_diagnostics.csv"))
fwrite(resumen, file.path(DATA_DIR, "calibration_ranking.csv"))
cat("\nGuardado: data/calibration_diagnostics.csv y data/calibration_ranking.csv\n")
