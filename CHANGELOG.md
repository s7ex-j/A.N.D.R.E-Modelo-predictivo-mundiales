# Cambios

## [1.0.0] - 2026-11-08

Primera versión publicada. Las cifras del README salen de esta versión; el
detalle de cada corrección está en `docs/BITACORA_DECISIONES.md` (§5 y §6).

### Modelo y evaluación
- **Stacking y calibración con predicciones out-of-fold** de 2019-2022 (walk-forward
  por año). El test completo (4,098 partidos, 2023 a oct-2026) queda fuera de muestra
  para todos los modelos; antes se gastaba el 70% del test en ajustar el ensamble.
- **Tuning de XGBoost con validación temporal** en vez de k-fold aleatorio.
- **Bayesiano fuera del ensamble** y con 4 cadenas x 3,000 iteraciones: converge
  (Rhat máximo 1.003); se reporta aparte.
- **Intervalos de confianza 95%** (bootstrap) en la tabla de resultados.
- **Nuevo `09f`:** empates, matriz de confusión y accuracy por tipo de partido y cancha.
- Baseline naive con las frecuencias del entrenamiento, no del test.
- Resultado: stacking 60.6% y XGBoost 60.4% (empatados), red neuronal 59.6%,
  Dixon-Coles 56.0%, naive 47.2%. Mundial 2026 (104 partidos): XGBoost 66.3%,
  stacking 65.4% con el mejor RPS (0.155).

### Correcciones
- Evaluación sobre la misma muestra para todos los modelos (#3).
- Los goles esperados de la API salían ~4% bajos por cómo se centraba el ataque
  Dixon-Coles (bitácora §1.5).

### Reproducibilidad
- `00_config.R` con las fechas de corte, las features y los modelos del ensamble.
- `pipeline.R` corre cada script en un entorno limpio y salta el bayesiano si falta
  `brms`/`rstan` o el compilador (o con `ANDRE_SKIP_BAYES=1`).
- CI: corre el pipeline completo y prueba la API en cada push.
- `renv.lock` con las versiones exactas de los paquetes.

### Dashboard
- Lee `team_state.json` (Elo y Dixon-Coles por selección, generado por `10`) y usa
  los goles esperados del modelo en vez de la heurística Elo→Poisson.
- Action semanal que descarga los resultados nuevos y actualiza el dashboard.
- Datos del Mundial 2026 completo (hasta oct-2026).
