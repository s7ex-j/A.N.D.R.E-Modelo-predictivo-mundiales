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

## Resultados (test completo: 4,098 partidos, 2023-01-02 a 2026-10-06)

Todos los modelos se evalúan sobre **los mismos 4,098 partidos**, que ninguno
usó para ajustarse: la calibración y el stacking se entrenan con predicciones
*out-of-fold* de 2019-2022 (ver Metodología). Datos de
`martj42/international_results` hasta el 2026-10-06.

| Modelo | Accuracy (IC 95%) | Log-loss | Brier | RPS (IC 95%) |
|---|---|---|---|---|
| Naive (frecuencias del train) | 47.2% [45.7, 48.7] | 1.053 | 0.635 | 0.230 [0.227, 0.233] |
| Dixon-Coles | 56.0% [54.4, 57.4] | 0.938 | 0.555 | 0.191 [0.187, 0.196] |
| Red neuronal | 59.6% [58.1, 61.1] | 0.872 | 0.513 | 0.171 [0.166, 0.176] |
| XGBoost | 60.4% [58.9, 61.9] | 0.864 | 0.507 | 0.169 [0.165, 0.174] |
| Promedio simple (calibrado) | 60.1% [58.6, 61.6] | 0.872 | 0.512 | 0.171 [0.167, 0.176] |
| Stacking (meta-learner) | 60.6% [59.1, 62.1] | 0.867 | 0.509 | 0.169 [0.164, 0.174] |

**Lectura:** stacking y XGBoost quedan **empatados** (diferencia 0.2 puntos,
IC 95% de la diferencia [-0.3, +0.7]; bootstrap pareado, 5,000 remuestreos).
El stacking sí supera a la red neuronal (+1.0 punto, significativo) y todos
superan con claridad a Dixon-Coles y al naive. En otras palabras: combinar
modelos no le agrega precisión a XGBoost solo; la API sirve el stacking, que
rinde igual y además devuelve las probabilidades de cada modelo base.

En **Copa del Mundo 2026** (104 partidos): XGBoost 66.3%; stacking, red neuronal
y promedio 65.4%; Dixon-Coles 51.9%. El stacking tiene el mejor RPS (0.155) y
supera de forma significativa solo a Dixon-Coles; contra los demás la diferencia no es
concluyente con 104 partidos.

**Empates:** el 22.8% de los partidos del test terminó en empate, pero el
stacking elige "empate" solo en el 2.1% (y XGBoost en el 0.3%). No es un error
de calibración: la probabilidad media de empate que asignan (~22%) coincide con
la frecuencia real. Es que un empate casi nunca es el resultado *más* probable
de un partido. Detalle, matriz de confusión y accuracy por tipo de partido
(amistoso, eliminatoria, torneo, Mundial; local o neutral) en
`scripts/09f_draws_and_segments.R`.

Calibración ("gana local", 10 bins): todos los modelos quedan entre 0.018 y
0.027 de brecha promedio; la red neuronal es la mejor calibrada (0.018) — ver
`scripts/09e_calibration_diagnostics.R`.

El modelo **bayesiano** (`06`) se reporta aparte porque cubre solo 414 partidos
(oct-2025 a dic-2025): 59.9% de accuracy, sin diferencia significativa con los
demás en esa ventana. Converge (Rhat máximo 1.003 en 467 parámetros).

> Cifras de R 4.6.0 en Windows con las versiones de `renv.lock`; con otras
> versiones de los paquetes (sobre todo xgboost) pueden moverse unas décimas, y
> en el Mundial, por ser solo 104 partidos, uno o dos puntos. Versiones anteriores de este
> README reportaban otras cifras: ver la
> [bitácora](docs/BITACORA_DECISIONES.md) (§5 y §6) para saber por qué cambiaron.

## Estructura del repo

```
.
├── data/
│   ├── results.csv            # dataset fuente (martj42/international_results, 1872-2026)
│   └── team_name_map.csv      # unificación de nombres históricos de selecciones
├── scripts/
│   ├── 00_config.R            # parámetros compartidos (fechas de corte, features, modelos del ensamble)
│   ├── 01_data_prep.R         # limpieza, target 1X2, peso de importancia, unificación de nombres
│   ├── 02_elo_ratings.R       # rating Elo dinámico pre-partido
│   ├── 03_dixon_coles.R       # Dixon-Coles walk-forward, gradiente analítico
│   ├── 04_feature_engineering.R
│   ├── 05_model_xgboost.R     # tuning walk-forward + predicciones out-of-fold 2019-2022
│   ├── 06_model_bayesian_mcmc.R  # Stan/brms, partial pooling jerárquico (opcional, fuera del ensamble)
│   ├── 07_model_deep_learning.R
│   ├── 08_ensemble.R          # calibración isotónica + stacking (glmnet), ajustados en 2019-2022
│   ├── 09_evaluation.R        # métricas + IC 95% (bootstrap)
│   ├── 09b_evaluation_worldcup.R  # accuracy específica en partidos de Mundial
│   ├── 09c_significance_test.R    # bootstrap, significancia (Mundial)
│   ├── 09d_significance_full.R    # bootstrap, significancia (test completo)
│   ├── 09e_calibration_diagnostics.R
│   ├── 09f_draws_and_segments.R   # empates, matriz de confusión, accuracy por tipo de partido
│   └── 10_build_team_state.R  # snapshot para la API y Dashboard/team_state.json
├── api/
│   ├── plumber.R              # API REST (endpoints /health, /predict)
│   ├── predict_match.R        # lógica de predicción en vivo
│   └── Dockerfile             # para desplegar la API
├── Dashboard/
│   ├── andre_mundial2026.html # dashboard de un solo archivo (sin build step)
│   └── team_state.json        # Elo y Dixon-Coles por selección (lo actualiza una Action semanal)
├── pipeline.R                 # corre todos los scripts en orden
├── renv.lock                  # versiones exactas de los paquetes
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
4. **Split temporal, nunca aleatorio.** Todo el test set (2023+) es posterior
   a todo el train set. Los hiperparámetros de XGBoost también se eligen con
   validación temporal: para cada año de 2019 a 2022, se entrena con todo lo
   anterior y se mide en ese año (no con k-fold aleatorio, que mezclaría
   partidos futuros en el entrenamiento).
5. **Predicciones out-of-fold para el ensamble.** Esos mismos cortes anuales
   dan predicciones de 2019-2022 que ningún modelo vio al entrenarse; con ellas
   se ajustan la **calibración isotónica** de cada modelo y el **stacking**
   (meta-learner glmnet). Así el test completo queda libre para evaluar a todos
   por igual.
6. **Ventaja de local NO se anula en cancha neutral durante el entrenamiento**
   (decisión deliberada, no un descuido — ver [`docs/BITACORA_DECISIONES.md`](docs/BITACORA_DECISIONES.md)):
   probamos anularla y el accuracy empeoró, porque incluso en partidos
   "neutrales" el equipo etiquetado `home_team` en los datos conserva señal
   real. En cambio, al **servir una predicción de un partido hipotético**
   (ej. un cruce de Mundial que todavía no se define quién es "local"), la
   API sí trata el partido como neutral sin ventaja, porque ahí no existe
   un local real.

## Cómo correr esto

1. Instala R (4.5 o superior) y, solo si quieres el modelo bayesiano, el
   compilador de C++: en Windows es **Rtools** de la misma versión menor que tu
   R (R 4.5 y 4.6 usan Rtools45).
2. Abre `Modelo predictivo mundiales.Rproj` (así el directorio de trabajo es la
   raíz del repo; los scripts usan rutas relativas).
3. Instala las versiones exactas con las que se verificó el proyecto:

```r
install.packages("renv")
renv::restore()
```

   (Sin `renv`: `install.packages(c("dplyr","data.table","lubridate","glmnet","nnet",
   "xgboost","jsonlite","plumber"))`, y `"brms","rstan"` para el bayesiano.)
4. Corre todo:

```r
source("pipeline.R")
```

Corre los 15 scripts en orden, cada uno en un entorno limpio, y deja todo en
`data/` y `output/`. El modelo bayesiano (`06`) es opcional: se salta solo si
faltan `brms`/`rstan` o el compilador, o si defines `ANDRE_SKIP_BAYES=1`.

Cada push corre el pipeline completo en GitHub Actions (sin el bayesiano) y
falla si algún modelo no supera al baseline o si la API no responde.

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
- Elo y goles esperados Dixon-Coles del modelo (`team_state.json`, que una
  GitHub Action actualiza cada lunes con los resultados nuevos)
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
