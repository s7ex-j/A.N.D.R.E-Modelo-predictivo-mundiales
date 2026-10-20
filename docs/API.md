# API y dashboard

## 1. Generar los artefactos

Corre el pipeline completo desde la raíz del proyecto:

```r
source("pipeline.R")
```

Después, durante el torneo, basta con correr `10_build_team_state.R` tras
cada jornada para actualizar Elo y forma reciente sin reentrenar.

## 2. Levantar la API en local

```bash
cd api
R -e "install.packages(c('plumber','data.table','xgboost','glmnet','nnet','jsonlite'))"
R -e "pr <- plumber::plumb('plumber.R'); pr\$run(port = 8000)"
```

Verificar:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team":"Brazil","away_team":"Argentina","neutral":true,"importance":1.0}'
```

## 3. Conectar el dashboard

Abre `Dashboard/andre_mundial2026.html`, activa **"Usar la API en vivo"** y
deja la URL en `http://127.0.0.1:8000` (valor por defecto) o pon la URL
pública de tu despliegue.

## 4. Desplegar la API (opcional)

El `Dockerfile` de `api/` funciona con:

- **Render.com:** "New Web Service" → conectar el repo → detecta el Dockerfile.
- **Railway.app:** "Deploy from GitHub" → detecta el Dockerfile.
- **Fly.io:** `fly launch` dentro de `api/`.

## 5. Notas

- Over/Under 2.5 y Ambos Anotan salen siempre del Dixon-Coles en vivo, no
  del stacking: XGBoost y la red neuronal solo clasifican 1X2.
- Con `neutral = true` la API anula la ventaja de local (ver
  [bitácora](BITACORA_DECISIONES.md#2-decisión-revertida-ventaja-de-local-en-cancha-neutral)).
- `data/team_name_map.csv` no es exhaustivo; revísalo si agregas
  selecciones con historia de cambio de nombre.
