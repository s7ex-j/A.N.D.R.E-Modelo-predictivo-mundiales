# A.N.D.R.E. — Adaptive Neural Dixon-coles Regularized Ensemble

Modelo predictivo de fútbol internacional (Mundial 2026) que combina un
modelo estadístico clásico (Dixon-Coles), machine learning (XGBoost, red
neuronal) y un modelo bayesiano jerárquico (Stan/brms), unidos por un
meta-learner (stacking). Incluye una API en R (`plumber`) y un dashboard
interactivo de un solo archivo HTML con cuotas de mercado en vivo.

[![Sintaxis R](https://github.com/s7ex-j/A.N.D.R.E-Modelo-predictivo-mundiales/actions/workflows/check-r.yml/badge.svg)](https://github.com/s7ex-j/A.N.D.R.E-Modelo-predictivo-mundiales/actions/workflows/check-r.yml)
[![Dashboard](https://img.shields.io/badge/dashboard-en%20vivo-2ea44f)](https://s7ex-j.github.io/A.N.D.R.E-Modelo-predictivo-mundiales/)
[![Licencia: MIT](https://img.shields.io/badge/licencia-MIT-blue.svg)](LICENSE)

<sub>**Español** · [English](README.en.md)</sub>

**Dashboard en vivo:** [s7ex-j.github.io/A.N.D.R.E-Modelo-predictivo-mundiales](https://s7ex-j.github.io/A.N.D.R.E-Modelo-predictivo-mundiales/)

**Autor:** Jharol André Vilca Ramos · [ORCID 0009-0009-7897-3439](https://orcid.org/0009-0009-7897-3439) · [@s7ex.j](https://instagram.com/s7ex.j)

---

## Resultados (holdout común: 1,230 partidos, 2025-09-09 a 2026-10-06)

Todos los modelos se evalúan sobre **los mismos partidos**: el 30% final del
test set, que ni la calibración isotónica ni el meta-learner del stacking usaron
para ajustarse. Datos de `martj42/international_results` hasta el 2026-10-06.

| Modelo | Accuracy | Log-loss | Brier | RPS |
|---|---|---|---|---|
| Naive (frecuencia histórica) | 47.9% | 1.048 | 0.632 | 0.230 |
| Dixon-Coles | 55.3% | 0.945 | 0.561 | 0.194 |
| Red neuronal | 60.6% | 0.870 | 0.509 | 0.170 |
| XGBoost | 61.4% | 0.852 | 0.500 | 0.167 |
| Promedio simple (calibrado) | 61.3% | 0.884 | 0.519 | 0.176 |
| Stacking (meta-learner) | 60.7% | 0.854 | 0.500 | 0.166 |

**Lectura:** XGBoost, red neuronal, promedio y stacking quedan **empatados
estadísticamente** (bootstrap pareado, 5,000 remuestreos: ninguna diferencia
entre ellos es significativa al 95%). Todos superan con claridad a Dixon-Coles
y al naive. El stacking tiene el mejor RPS (0.166), pero no mejora el accuracy
de XGBoost solo.

En **Copa del Mundo 2026** (104 partidos, el torneo completo): stacking y red
neuronal 65.4%, XGBoost 64.4%, promedio 63.5%, Dixon-Coles 52.9%. El stacking
tiene el mejor RPS (0.157) y supera de forma significativa solo a Dixon-Coles;
contra los demás la diferencia no es concluyente con 104 partidos.

Calibración ("gana local", 10 bins, holdout común): la red neuronal es la mejor
calibrada (brecha promedio 0.019), seguida de XGBoost (0.028) y el stacking
(0.030) — ver `scripts/09e_calibration_diagnostics.R`.

> Versiones anteriores de este README reportaban stacking 62.0% contra 60.0% de
> XGBoost. Esa comparación usaba muestras distintas (el stacking en el 30% final,
> el resto en todo el test) y no era válida; el detalle está en la
> [bitácora](docs/BITACORA_DECISIONES.md#5-el-stacking-no-era-el-mejor-comparaba-muestras-distintas).

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
├── Dashboard/
│   └── andre_mundial2026.html # dashboard de un solo archivo (sin build step)
├── pipeline.R                 # corre todos los scripts en orden
└── docs/
    ├── BITACORA_DECISIONES.md # bugs encontrados, fixes y decisiones metodológicas
    └── API.md                 # cómo levantar, conectar y desplegar la API
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
   **stacking vía meta-learner (glmnet)** en vez de promedio simple.
6. **Ventaja de local NO se anula en cancha neutral durante el entrenamiento**
   (decisión deliberada, no un descuido — ver [`docs/BITACORA_DECISIONES.md`](docs/BITACORA_DECISIONES.md)):
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

`Dashboard/andre_mundial2026.html` es autocontenido (y se publica en [GitHub Pages](https://s7ex-j.github.io/A.N.D.R.E-Modelo-predictivo-mundiales/)) — ábrelo directo en el
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

## Uso de IA

Usé un asistente de IA (Claude) durante el desarrollo, sobre todo para
auditar el pipeline, depurar la API y redactar parte del código y la
documentación. Las decisiones metodológicas y los resultados se validaron
corriendo el pipeline sobre los datos reales; los errores que esa auditoría
encontró, y una decisión que revertí tras medirla, están en
[`docs/BITACORA_DECISIONES.md`](docs/BITACORA_DECISIONES.md).

## Licencia

MIT — ver `LICENSE`.

## Aviso

Este proyecto es una herramienta informativa/educativa para explorar
modelado estadístico y de machine learning aplicado a fútbol. No es
asesoría financiera ni garantía de resultados. Apostar conlleva riesgo
real de pérdida.
