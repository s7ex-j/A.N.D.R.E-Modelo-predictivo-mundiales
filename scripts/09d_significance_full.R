## =============================================================
## 09d_significance_full.R
## Lo mismo que 09c pero sobre las 3,686 filas del test set completo
## (todos los torneos desde 2023), no solo las 88 de Mundial. Con
## 42x más partidos, el bootstrap tiene mucho más poder para detectar
## diferencias reales entre modelos -- y compara TODOS los pares,
## no solo contra el stacking.
## =============================================================
suppressMessages({ library(data.table) })
set.seed(42)

DATA_DIR <- "data"
ens <- fread(file.path(DATA_DIR, "ensemble_predictions.csv"))

modelos_disponibles <- unique(gsub("_p_(home|draw|away)$", "",
                          grep("_p_home$", names(ens), value = TRUE)))
cat("Modelos encontrados:", paste(modelos_disponibles, collapse=", "), "\n")
cat("Partidos en el test set completo:", nrow(ens), "\n\n")

accuracy_of <- function(dat, prefix) {
  pa <- dat[[paste0(prefix,"_p_away")]]; pd <- dat[[paste0(prefix,"_p_draw")]]; ph <- dat[[paste0(prefix,"_p_home")]]
  if (is.null(pa)) return(NA_real_)
  ok <- !is.na(pa) & !is.na(pd) & !is.na(ph)
  if (sum(ok) < 10) return(NA_real_)  # muy pocas filas válidas (ej. bayes fuera de su ventana)
  pred <- c("A","D","H")[apply(cbind(pa[ok],pd[ok],ph[ok]), 1, which.max)]
  mean(pred == dat$result[ok])
}

N_BOOT <- 5000
n <- nrow(ens)
boot_acc <- matrix(NA_real_, nrow = N_BOOT, ncol = length(modelos_disponibles),
                    dimnames = list(NULL, modelos_disponibles))

cat("Corriendo", N_BOOT, "remuestreos bootstrap pareados...\n")
for (b in seq_len(N_BOOT)) {
  idx <- sample.int(n, n, replace = TRUE)   # MISMOS índices para todos los modelos = pareado
  boot_sample <- ens[idx]
  for (m in modelos_disponibles) boot_acc[b, m] <- accuracy_of(boot_sample, m)
}

cat("\n=== Intervalo de confianza 95% del accuracy (test completo, n=", N_BOOT, " remuestreos) ===\n", sep="")
ci_table <- rbindlist(lapply(modelos_disponibles, function(m) {
  ci <- quantile(boot_acc[, m], c(0.025, 0.5, 0.975), na.rm = TRUE)
  data.table(modelo = m, accuracy_observado = round(accuracy_of(ens, m), 4),
             ci_2.5 = round(ci[1], 4), mediana_boot = round(ci[2], 4), ci_97.5 = round(ci[3], 4))
}))
print(ci_table)

## ---- Todos los pares, no solo vs stack ----
cat("\n=== Todas las comparaciones por pares ===\n")
pares <- combn(modelos_disponibles, 2, simplify = FALSE)
pair_table <- rbindlist(lapply(pares, function(p) {
  a <- p[1]; b <- p[2]
  obs_diff <- accuracy_of(ens, a) - accuracy_of(ens, b)
  if (is.na(obs_diff)) return(data.table(modelo_a=a, modelo_b=b, diferencia=NA, ci_2.5=NA, ci_97.5=NA, significativo=NA))
  diff_boot <- boot_acc[, a] - boot_acc[, b]
  ci <- quantile(diff_boot, c(0.025, 0.975), na.rm = TRUE)
  sig <- isTRUE(ci[1] > 0) || isTRUE(ci[2] < 0)
  data.table(modelo_a=a, modelo_b=b, diferencia=round(obs_diff,4),
             ci_2.5=round(ci[1],4), ci_97.5=round(ci[2],4), significativo=sig)
}))
print(pair_table)

fwrite(ci_table, file.path(DATA_DIR, "significance_full_ci.csv"))
fwrite(pair_table, file.path(DATA_DIR, "significance_full_pairs.csv"))
cat("\nGuardado: data/significance_full_ci.csv y data/significance_full_pairs.csv\n")
