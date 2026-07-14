# A.N.D.R.E. — Adaptive Neural Dixon-coles Regularized Ensemble

Modelo predictivo de fútbol internacional (Mundial 2026) que combina un
modelo estadístico clásico (Dixon-Coles), machine learning (XGBoost, red
neuronal) y un modelo bayesiano jerárquico (Stan/brms), unidos por un
meta-learner (stacking). Incluye una API en R (`plumber`) y un dashboard
interactivo de un solo archivo HTML con cuotas de mercado en vivo.

**Autor:** Andre ([@s7ex.j](https://instagram.com/s7ex.j))

---

## Resultados (test set: partidos desde 2023-01-01, n=3,686)

| Modelo | Accuracy | Log-loss | Brier | RPS |
|---|---|---|---|---|
| Naive (frecuencia histórica) | 47.3% | 1.053 | 0.635 | 0.229 |
| Dixon-Coles | 56.3% | 0.936 | 0.554 | 0.190 |
| Red neuronal | 59.3% | 0.873 | 0.513 | 0.170 |
| XGBoost | 60.0% | 0.861 | 0.506 | 0.168 |
| Promedio simple (calibrado) | 60.0% | 0.888 | 0.521 | 0.175 |
| **Stacking (meta-learner)** | **62.0%** | **0.836** | **0.489** | **0.160** |

Específico en partidos de **Copa del Mundo 2026** (n=88, hasta cuartos de final):
stacking 62.5% accuracy, RPS 0.167 — significativamente mejor que el promedio
simple (bootstrap, p<0.05), aunque contra XGBoost/red neuronal solos la
diferencia todavía no es concluyente con esta cantidad de partidos.

El stacking es además el modelo **mejor calibrado** de los 5 (brecha
promedio 0.022 en un diagnóstico de 10 bins) — ver `scripts/09e_calibration_diagnostics.R`.

## Estructura del repo

```
.
├── data/
│   ├── results.csv            # dataset fuente (martj42/international_results, 1872-2026)
│   └── team_name_map.csv      # unificación de nombres históricos de selecciones
├── scripts/
│   ├── 01_data_prep.R         # limpieza, target 1X2, peso de importancia, unificación de nombres
│   ├── 02_elo_ratings.R       # rating Elo dinámico pre-partido
│   ├── 03_dixon_coles.R       # Dixon-Coles walk-forward, gradiente analítico
│   ├── 04_feature_engineering.R
│   ├── 05_model_xgboost.R
│   ├── 06_model_bayesian_mcmc.R  # Stan/brms, partial pooling jerárquico
│   ├── 07_model_deep_learning.R
│   ├── 08_ensemble.R          # stacking (glmnet) + calibración isotónica
│   ├── 09_evaluation.R
│   ├── 09b_evaluation_worldcup.R  # accuracy específica en partidos de Mundial
│   ├── 09c_significance_test.R    # bootstrap, significancia (Mundial)
│   ├── 09d_significance_full.R    # bootstrap, significancia (test completo)
│   ├── 09e_calibration_diagnostics.R
│   └── 10_build_team_state.R  # snapshot diario para servir la API
├── api/
│   ├── plumber.R              # API REST (endpoints /health, /predict)
│   ├── predict_match.R        # lógica de predicción en vivo
│   └── Dockerfile             # para desplegar la API
├── dashboard/
│   └── andre_mundial2026.html # dashboard de un solo archivo (sin build step)
├── pipeline.R                 # corre todos los scripts en orden
└── docs/                      # notas de metodología y auditoría
```

`output/` (modelos entrenados, `.rds`) se genera localmente al correr el
pipeline — no se versiona (ver `.gitignore`). Igual los CSV intermedios
(`results_clean.csv`, `model_features.csv`, `ensemble_predictions.csv`,
etc.): son regenerables desde `data/results.csv` + los scripts, así que no
tiene sentido versionarlos.

## Metodología

1. **Todo es walk-forward / pre-partido.** Elo, Dixon-Coles y la forma
   reciente usan solo información anterior a cada partido — nunca el rating
   "final" de un equipo para predecir un partido de hace años.
2. **Dixon-Coles se reajusta cada 90 días**, con decaimiento temporal
   exponencial y gradiente analítico derivado a mano (~90 reajustes en
   0.3s cada uno vs. >2min con diferencias finitas).
3. **Regularización** (ridge en Dixon-Coles, partial pooling jerárquico en
   el modelo bayesiano) para que selecciones con pocos partidos no generen
   goles esperados absurdos.
4. **Split temporal, nunca aleatorio.** Todo el test set es posterior a
   todo el train set.
5. **Calibración isotónica** por modelo antes de promediar/stackear, y
   **stacking vía meta-learner (glmnet)** en vez de promedio simple — está
   medido que el stacking es tanto más preciso como mejor calibrado.
6. **Ventaja de local NO se anula en cancha neutral durante el entrenamiento**
   (decisión deliberada, no un descuido — ver `docs/` para el detalle):
   probamos anularla y el accuracy empeoró, porque incluso en partidos
   "neutrales" el equipo etiquetado `home_team` en los datos conserva señal
   real. En cambio, al **servir una predicción de un partido hipotético**
   (ej. un cruce de Mundial que todavía no se define quién es "local"), la
   API sí trata el partido como neutral sin ventaja, porque ahí no existe
   un local real.

## Cómo correr esto

```r
install.packages(c("dplyr","data.table","lubridate","glmnet","nnet","xgboost",
                    "rstan","brms","plumber","jsonlite"))
```

`rstan`/`brms` necesitan un compilador de C++ instalado (Rtools en Windows).
El modelo bayesiano (`06_model_bayesian_mcmc.R`) es opcional — el resto del
pipeline corre sin él, simplemente no se suma al ensamble.

```r
source("pipeline.R")
```

Esto corre los 14 scripts en orden y deja todo listo en `data/` y `output/`.

## API en vivo

```r
setwd("api")
library(plumber)
pr <- plumb("plumber.R")
pr$run(port = 8000)
```

```
GET  /health   -> {"status":"ok","snapshot_date":"...","n_teams":333}
POST /predict  -> {"home_team":"Brazil","away_team":"Argentina","neutral":true,"importance":1.0}
```

Para desplegarla en internet (no solo localhost), usa el `Dockerfile` incluido
con Render, Railway o Fly.io.

## Dashboard

`dashboard/andre_mundial2026.html` es autocontenido — ábrelo directo en el
navegador, no necesita servidor ni build step. Incluye:
- Selector de partido con el bracket real del Mundial 2026 (octavos → final,
  incluyendo escenarios hipotéticos para cruces aún no definidos)
- Comparación de probabilidades del modelo vs. cuotas de mercado en vivo
  ([The Odds API](https://the-odds-api.com), tier gratis)
- Cálculo de %EDGE y cuota justa por mercado (1X2, over/under 2.5, ambos anotan)
- Mapa de calor de marcadores probables, backtest real, calibración
- Vista previa móvil real (iframe con viewport propio)
- Conexión opcional a la API en vivo (arriba) para usar los modelos
  entrenados reales en vez de la heurística Elo→Poisson

## Licencia

MIT — ver `LICENSE`.

## Aviso

Este proyecto es una herramienta informativa/educativa para explorar
modelado estadístico y de machine learning aplicado a fútbol. No es
asesoría financiera ni garantía de resultados. Apostar conlleva riesgo
real de pérdida.
