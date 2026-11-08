# Bitácora de decisiones y auditoría

Registro de los errores que encontré al auditar el pipeline, cómo los
corregí y qué decisiones metodológicas tomé (incluida una que revertí
después de medirla). Las cifras intermedias de esta bitácora vienen de
corridas sobre el dataset real (`data/results.csv`, 49,493 partidos);
los números finales están en el [README](../README.md#resultados-test-set-partidos-desde-2023-01-01-n3686).

---

## 1. Bugs críticos (rompían la API sin dar error)

### 1.1 La calibración isotónica estaba rota

`isoreg()` en R devuelve `$x` en el **orden original de los datos**, pero
`$yf` (los valores ajustados) viene ordenado según `$ord`. La primera
versión hacía `approxfun(iso$x, iso$yf)` directamente, mezclando un vector
desordenado con uno ordenado: la "calibración" no tenía relación con la
probabilidad de entrada (correlación ≈ 0).

- **Impacto:** el accuracy del ensamble cayó de 59.7% / 61.5% a
  **46.8% / 48.5%**, por debajo del baseline naive (47.3%).
- **Cómo lo detecté:** al probar la API con partidos reales, todas las
  predicciones salían casi idénticas sin importar los equipos.
- **Fix:** `approxfun(iso$x[iso$ord], iso$yf)`. Reprobado con los datos
  reales y con 4 partidos del Mundial (Brazil–Argentina, Spain–Portugal,
  Morocco–France, Norway–England): las probabilidades suman 1 y varían de
  forma sensata entre partidos.

### 1.2 `feature_cols` desalineado entre entrenamiento y API

`predict_match.R` usaba 11 features; el modelo se entrena con las **22** de
`04_feature_engineering.R` (verificado también contra `05` y `07`). Ahora
la API usa exactamente la misma lista.

### 1.3 XGBoost se cargaba con la función equivocada

`05_model_xgboost.R` guarda el modelo con `saveRDS()`, pero la API lo
cargaba con `xgb.load()` (que espera el formato binario nativo). Fix:
`readRDS()`.

### 1.4 Orden de columnas del stacking

`08_ensemble.R` entrena con `factor(result, levels = c("A","D","H"))`, así
que `predict.cv.glmnet(..., type = "response")` devuelve **Away, Draw,
Home**. La API leía por posición asumiendo Home–Draw–Away, lo que habría
intercambiado "gana local" con "gana visitante". Fix: extraer por nombre
de dimensión (`"H"`, `"D"`, `"A"`).

### 1.5 Los goles esperados de la API salían ~4% bajos

`10_build_team_state.R` centraba el ataque de Dixon-Coles en 0 (`att - mean(att)`)
para que fuera legible, pero el modelo no tiene intercepto:
`lambda = exp(att + def + gamma)`. Restar la media solo al ataque bajaba todos
los goles esperados. Comparado con los partidos de 2025-2026, la API predecía
1.53 goles del local cuando el promedio real era 1.66.

- **Fix:** sumar a la defensa lo mismo que se resta al ataque; el lambda queda
  idéntico al del ajuste (1.60 vs. 1.66 real después del fix; la diferencia
  que queda es la del propio modelo, no del centrado).

---

## 2. Decisión revertida: ventaja de local en cancha neutral

**Hipótesis inicial:** Dixon-Coles suma `gamma` (ventaja de local) siempre,
incluso en partidos neutrales, donde `home_team` debería ser arbitrario.
Condicioné `gamma` a `!neutral`.

**Resultado:** el accuracy de Dixon-Coles **cayó** de 56.3% a 54.2%.

**Por qué:** en los 13,135 partidos marcados `neutral`, el equipo
etiquetado `home_team` gana el 44.2%, empata el 22.4% y pierde el 33.4%:
no es 50/50. La etiqueta conserva señal real (sede, región, afición o la
convención de la fuente de datos).

**Decisión final:**
- **Entrenamiento** (`03_dixon_coles.R`): `gamma` incondicional, para
  aprovechar esa señal.
- **Predicción** (`api/predict_match.R`): con `neutral = TRUE` se fuerza
  `gamma = 0`. En un cruce hipotético del bracket no existe un local real;
  el orden de los equipos es solo el orden del selector.

---

## 3. Mejoras metodológicas

- **Unificación de nombres históricos.** `former_names.csv` venía vacío,
  así que Elo y Dixon-Coles trataban a West Germany/Germany, etc. como
  selecciones distintas. Creé `data/team_name_map.csv`, priorizando los 48
  países del Mundial 2026 y sus antecesores. Algunos mapeos (p. ej.
  Yugoslavia → Serbia) son convenciones, no verdades absolutas.
- **Modelo bayesiano en el ensamble.** `06_model_bayesian_mcmc.R` se
  calculaba pero nunca entraba a `08_ensemble.R`; ahora se suma como modelo
  base si existe `preds_bayes.csv`. En los partidos de 2026 sale `NaN`
  porque su ventana de entrenamiento (sep-2024 a ene-2026) no los cubre;
  es esperado, no un bug.
- **Evaluación específica de Mundial** (`09b`): el accuracy agregado desde
  2023 mezcla amistosos y eliminatorias. `09b` mide solo partidos de Copa
  del Mundo, y `09c`/`09d` prueban con bootstrap si las diferencias entre
  modelos son significativas.
- **Artefactos para servir la API:** `07` guarda `nnet_scaling.rds`,
  `08` guarda `stack_colnames.rds` y las funciones de calibración, y `10`
  genera un snapshot diario (`team_state.rds`, `dc_global.rds`) para no
  reentrenar en cada request.

---

## 4. Números intermedios (sin XGBoost)

Corrida de validación con todos los fixes, antes de sumar XGBoost:

| Modelo | Accuracy | RPS |
|---|---|---|
| Naive | 47.3% | 0.229 |
| Dixon-Coles | 56.3% | 0.190 |
| Red neuronal | 59.3% | 0.170 |
| Ensamble (promedio, calibrado) | 59.4% | 0.182 |
| Stacking | 60.3% | 0.164 |

Con XGBoost incluido el stacking subía a 62.0%; esa cifra resultó no ser comparable (ver §5).

---

## 5. El stacking no era el mejor: comparaba muestras distintas

El meta-learner del stacking y la calibración isotónica se ajustan sobre el
primer 70% del test set, así que solo el 30% final está fuera de muestra para
todos los modelos. La tabla que publiqué evaluaba el stacking en ese 30%
(n = 1,106) y al resto en todo el test (n = 3,686): **comparaba muestras
distintas**, y con eso el stacking parecía el mejor (62.0% vs. 60.0% de XGBoost).

- **Fix (#3):** `09` a `09e` evalúan todos los modelos sobre el mismo holdout
  (`stack_split == "meta_test"`), y el baseline naive toma sus frecuencias del
  tramo de ajuste, no del que evalúa.
- **Resultado** (misma corrida, datos hasta oct-2026, n = 1,230): XGBoost 61.4%,
  promedio 61.3%, stacking 60.7%, red neuronal 60.6%. Ninguna diferencia entre
  esos cuatro es significativa (bootstrap pareado, 5,000 remuestreos).
- **También cae** la afirmación de que el stacking era el mejor calibrado: en el
  holdout común lo es la red neuronal (brecha 0.019 vs. 0.030).
- **Lección:** comparar modelos siempre sobre la misma muestra, y que ninguno la
  haya visto al ajustarse. Un número mejor en una muestra distinta no prueba nada.

---

## 6. v2: el test completo para evaluar, todo lo demás fuera de él

El fix de §5 dejó una comparación justa, pero sobre solo 1,230 partidos: el
70% del test se gastaba en ajustar la calibración y el meta-learner. Con tan
pocos partidos casi ninguna diferencia salía significativa. Cambios:

- **Predicciones out-of-fold walk-forward (05, 07).** Para cada año de 2019 a
  2022, XGBoost y la red se entrenan con todo lo anterior y predicen ese año.
  Son 3,597 partidos que ningún modelo vio al entrenarse.
- **Calibración y stacking con esas predicciones (08).** Ya no tocan el test,
  así que los 4,098 partidos de 2023+ sirven para evaluar a todos.
- **Tuning de XGBoost con validación temporal (05).** Antes, `xgb.cv` con
  5-fold aleatorio mezclaba partidos futuros en el entrenamiento al elegir
  `n_rounds`. Ahora una grilla pequeña (profundidad 3, 4 y 5) se evalúa con
  los mismos cortes anuales; ganó profundidad 3 (log-loss 0.8829) por muy poco.
- **El bayesiano sale del ensamble.** Cubría 414 de 4,098 partidos; en el resto
  entraba como "1/3, 1/3, 1/3", que para el stacking es ruido. Ahora se reporta
  aparte. Con 4 cadenas y 3,000 iteraciones converge: Rhat máximo 1.003 en los
  467 parámetros (antes 1.07 con 1 cadena y 400 iteraciones).
- **Resultado:** stacking 60.5% y XGBoost 60.4% en el test completo, empatados
  (IC 95% de la diferencia [-0.4, +0.6] puntos). Con 3.3 veces más partidos la
  conclusión de §5 se confirma: el ensamble no le gana a XGBoost solo.
- **Empates:** 22.8% de los partidos, pero el stacking elige empate en el 2.7%.
  La probabilidad media de empate sí está bien calibrada (~22%); es que el
  empate casi nunca es el resultado más probable. Mejorar esto no pasa por
  calibrar, sino por un modelo que trate el empate de forma explícita (roadmap).

Además: `00_config.R` reúne las fechas de corte y la lista de features (antes
copiadas en cada script), `pipeline.R` corre cada script en un entorno limpio y
sigue aunque falle el bayesiano, y la CI corre el pipeline completo en cada push.

---

## 7. Roadmap

- **Modelo específico para empates** (p. ej. Poisson bivariado con inflación de
  empates, o un umbral de decisión que elija "empate" cuando los dos equipos
  están parejos) y medir si mejora el RPS sin empeorar el accuracy.
- **Cuotas de cierre como feature o como referencia a superar:** en la
  literatura son el predictor individual más fuerte para fútbol
  (football-data.co.uk las tiene para ligas domésticas, no para selecciones;
  falta una fuente para partidos internacionales).
- **Desglose por confederación**, con un mapa selección → confederación.
- **Red profunda con keras3/TensorFlow** en lugar de la red `nnet` de una
  capa (el código ya está en `07`, desactivado).
- **Reentrenamiento automático:** la Action semanal ya actualiza los datos y el
  dashboard, pero no reentrena los modelos ni la API.
