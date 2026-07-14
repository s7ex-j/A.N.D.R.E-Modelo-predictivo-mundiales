# Cómo conectar LAMBDA26 a tus modelos reales — guía paso a paso

# Cómo conectar LAMBDA26 a tus modelos reales — guía paso a paso

## 0. Antes que nada: lee AUDITORIA_MEJORAS.md

Ahí está el detalle de 3 bugs reales que encontré y probé con datos sintéticos
(rompían la API) más 1 problema metodológico importante (Dixon-Coles ignoraba
canchas neutrales — justo tu caso de uso). Todo ya viene corregido en los
archivos de esta carpeta, pero léelo para entender qué cambió y por qué.

## 1. Dónde va cada archivo

```
Modelo predictivo mundiales/
├── data/
│   ├── results.csv
│   └── team_name_map.csv         <- NUEVO (mapeo de nombres históricos)
├── scripts/
│   ├── 01_data_prep.R             <- REEMPLAZAR por 01_data_prep_FIX.R
│   ├── 02_elo_ratings.R           (sin cambios)
│   ├── 03_dixon_coles.R           <- REEMPLAZAR por 03_dixon_coles_FIX.R
│   ├── 04_feature_engineering.R  (sin cambios)
│   ├── 05_model_xgboost.R        (sin cambios)
│   ├── 06_model_bayesian_mcmc.R  (sin cambios)
│   ├── 07_model_deep_learning.R  (sin cambios — ya guarda nnet_scaling.rds)
│   ├── 08_ensemble.R              <- REEMPLAZAR por 08_ensemble_FIX.R
│   ├── 09_evaluation.R           (sin cambios)
│   ├── 09b_evaluation_worldcup.R  <- NUEVO
│   └── 10_build_team_state.R      <- NUEVO
├── output/                        (los .rds que generan los scripts)
├── api/
│   ├── plumber.R
│   ├── predict_match.R
│   └── Dockerfile
```

Nota: espera que `07_model_deep_learning.R` y `08_ensemble.R` ya vengan con
`saveRDS(list(mu=mu, sd=sdv), ...)` y `saveRDS(model_probs, ...)` agregados —
si usas las versiones que te di (`_FIX`), ya está incluido, no hace falta
que edites nada a mano.

## 2. Correr todo en orden (sí, todo, porque cambiaron los fundamentos)

```bash
cd "Modelo predictivo mundiales"
Rscript scripts/01_data_prep.R              # ahora unifica nombres históricos
Rscript scripts/02_elo_ratings.R
Rscript scripts/03_dixon_coles.R             # ahora neutral-aware (importante)
Rscript scripts/04_feature_engineering.R
Rscript scripts/05_model_xgboost.R
Rscript scripts/06_model_bayesian_mcmc.R
Rscript scripts/07_model_deep_learning.R
Rscript scripts/08_ensemble.R                # ahora incluye Bayes + calibración
Rscript scripts/09_evaluation.R
Rscript scripts/09b_evaluation_worldcup.R    # NUEVO: accuracy específica de Mundial
Rscript scripts/10_build_team_state.R
```

Los números de accuracy que viste antes (61.5% del stacking) van a cambiar
porque el Dixon-Coles ahora calcula distinto (correctamente) para partidos
neutrales. Compárteme el `evaluation_summary.csv` y `evaluation_worldcup.csv`
nuevos y ajusto el dashboard con los números reales actualizados.

## 3. Probar la API en tu computadora

```bash
cd "Modelo predictivo mundiales/api"
R -e "install.packages(c('plumber','data.table','xgboost','glmnet','nnet','jsonlite'))"
R -e "pr <- plumber::plumb('plumber.R'); pr\$run(port = 8000)"
```

Verifica que responde:
```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team":"Brazil","away_team":"Argentina","neutral":true,"importance":1.0}'
```

Yo probé toda la lógica de `predict_match.R` (team_state, Dixon-Coles en vivo,
calibración, stacking, red neuronal, formato del JSON) de punta a punta con
datos sintéticos en mi sandbox — corre sin errores. Lo único que NO pude
probar aquí es el XGBoost real (el paquete no está disponible en mi entorno),
así que si al correr esto te tira un error específico de esa parte, mándamelo
y lo resolvemos rápido.

## 4. Conectar el HTML a tu API local

Abre `lambda26_mundial2026.html`, activa el toggle **"Usar mi API en vivo"**,
y en el campo de URL deja `http://127.0.0.1:8000` (ya viene así por defecto).

## 5. Desplegar la API para que cualquiera la use (opcional)

- **Render.com** (free tier): "New Web Service" → conecta tu repo de GitHub
  → detecta el Dockerfile solo → dale la URL pública que te da Render
- **Railway.app**: similar, "Deploy from GitHub" → detecta el Dockerfile
- **Fly.io**: `fly launch` en la carpeta `api/` (usa el Dockerfile automáticamente)

## 6. Notas importantes

- Over/Under 2.5 y Ambos Anotan siempre salen del Dixon-Coles en vivo, no
  del stacking — tus modelos xgb/nnet solo clasifican 1X2.
- Corre `10_build_team_state.R` regularmente durante el torneo (después de
  cada jornada) para que el Elo/forma estén al día.
- Revisa `team_name_map.csv` — prioricé los 48 países del Mundial 2026 y
  sus antecesores más relevantes, pero no es exhaustivo. Algunos mapeos
  (ej. Yugoslavia→Serbia) son convenciones, no verdades absolutas — ajústalos
  si no estás de acuerdo.

