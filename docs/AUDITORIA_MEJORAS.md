# Auditoría completa del pipeline — hallazgos y mejoras

Revisé `01_data_prep.R` a `09_evaluation.R` línea por línea contra el
`predict_match.R` que te di antes. Esto es lo que encontré, en orden
de prioridad.

---

## 🔴 CRÍTICO — rompían la API sin avisar

**1. `feature_cols` estaba mal en mi `predict_match.R` anterior.**
El real (de `04_feature_engineering.R`, confirmado también en `05` y `07`,
los 3 coinciden exactamente) es:

```
elo_home_pre, elo_away_pre, elo_diff, elo_diff_abs,
lambda_home, lambda_away, dc_rho,
home_form_pts, away_form_pts, form_pts_diff,
home_form_gf, away_form_gf, form_gf_diff,
home_form_ga, away_form_ga,
home_days_rest, away_days_rest, rest_diff,
h2h_home_wr, h2h_n_prev, importance, neutral
```
22 features, no 11 como puse antes. Ya corregido en el nuevo `predict_match.R`.

**2. El XGBoost está guardado con `saveRDS()`, no `xgb.save()`.**
Tu `05_model_xgboost.R` hace `saveRDS(model_cls, "output/model_xgb_classifier.rds")`.
Yo en la API anterior usaba `xgb.load()` — que espera el formato binario nativo,
no un `.rds`. Con eso, cargar el modelo habría tirado error apenas arrancara
la API. Corregido: ahora uso `readRDS()`.

**3. El orden de columnas del stacking (glmnet) no es Home-Draw-Away.**
`08_ensemble.R` entrena con `factor(result, levels = c("A","D","H"))`, así que
`predict.cv.glmnet(..., type="response")` devuelve las columnas en ese orden
exacto: **Away, Draw, Home** (con nombres `"A"`,`"D"`,`"H"` en el array).
Yo estaba leyendo por posición asumiendo Home-Draw-Away — habría devuelto
las probabilidades cruzadas (el % de "gana local" en realidad hubiera sido
el % de "gana visitante"). Corregido: ahora extraigo por nombre de dimensión,
no por posición.

## 🟠 IMPORTANTE — afecta la calidad real del modelo

**4. Dixon-Coles ignora la bandera `neutral` — y tu caso de uso es 100% partidos neutrales.**
En `03_dixon_coles.R`, el `gamma` (ventaja de local) se suma **siempre**,
sin importar si el partido fue en cancha neutral:
```r
sub$lambda_home[...] <- exp(fit$att[hi] + fit$def[ai] + fit$gamma)  # sin condicional
```
Esto quiere decir que en partidos históricos neutrales (muchos partidos de
Mundiales pasados, Copas América en USA, etc.) el equipo que aparece como
`home_team` en el CSV (que en cancha neutral suele ser arbitrario — el que
viene primero alfabéticamente o como lo cargó la fuente) recibe una ventaja
de gol que no le corresponde. Eso distorsiona un poco las fuerzas de ataque/
defensa aprendidas para TODAS las selecciones, y es especialmente grave
para ti porque el Mundial 2026 completo es neutral (ninguna selección juega
"en casa" salvo México/USA/Canadá como anfitriones de fase de grupos).

**Te doy el fix abajo** (`03_dixon_coles_FIX.R`): condiciona `gamma` a `!neutral`,
igual que ya hace correctamente `02_elo_ratings.R`. Requiere volver a correr
`03` → `09` porque cambian `lambda_home`, `lambda_away`, y por lo tanto todas
las features y modelos que dependen de ellas.

**5. `former_names.csv` está vacío.**
Lo vi desde el principio (0 bytes en tu zip original). Sin un mapeo de nombres
históricos (`West Germany`→`Germany`, `Czechoslovakia`→..., etc.), el Elo y
Dixon-Coles tratan a esas selecciones como equipos completamente distintos y
pierden continuidad histórica. Te doy un mapeo inicial (`team_name_map.csv`)
para que lo revises/completes — prioricé los 48 países del Mundial 2026 y
sus antecesores más relevantes.

**6. El modelo bayesiano (`06_model_bayesian_mcmc.R`) no se usa en el ensamble.**
Corriste todo ese MCMC con intervalos de credibilidad y nunca entra a
`08_ensemble.R` ni a `evaluation_summary.csv`. Es información valiosa
que se está desperdiciando. Te doy un patch para `08_ensemble.R` que lo
incorpora como modelo base adicional (si `preds_bayes.csv` existe).

**7. Evaluación solo agregada, no específica de Mundial.**
Tu `09_evaluation.R` mide accuracy sobre TODOS los partidos desde 2023
(amistosos, eliminatorias, todo junto). Para lo que realmente te importa
—Mundial— eso puede ser optimista o pesimista sin que lo sepas. Te doy
`09b_evaluation_worldcup.R`: mide accuracy/RPS específicamente en partidos
de Copa del Mundo (2018, 2022, y los ya jugados de 2026).

## 🟡 Mejora estructural — para que sea un repo "élite" de verdad

**8. Faltaba estructura de repo real para GitHub** (el pedido original del
día 1). Te doy `README.md`, `.gitignore`, `LICENSE`.

---

## Qué vas a necesitar correr (en orden, en tu RStudio)

```bash
# 1. Reemplaza 01_data_prep.R y 03_dixon_coles.R con las versiones _FIX
# 2. Reemplaza 08_ensemble.R con la versión _FIX
Rscript scripts/01_data_prep.R
Rscript scripts/02_elo_ratings.R
Rscript scripts/03_dixon_coles.R        # ahora neutral-aware
Rscript scripts/04_feature_engineering.R
Rscript scripts/05_model_xgboost.R
Rscript scripts/06_model_bayesian_mcmc.R
Rscript scripts/07_model_deep_learning.R   # + la línea nueva que guarda mu/sd
Rscript scripts/08_ensemble.R              # ahora incluye Bayes + guarda stack_colnames
Rscript scripts/09_evaluation.R
Rscript scripts/09b_evaluation_worldcup.R  # NUEVO
Rscript scripts/10_build_team_state.R
```

Los accuracy que viste antes (61.5% stacking) **van a cambiar** —posiblemente
mejoren un poco al arreglar el bug de `neutral` en Dixon-Coles y al sumar
Bayes al ensamble. Avísame los números nuevos de `evaluation_summary.csv` y
`evaluation_worldcup.csv` cuando los tengas, y ajusto el dashboard.

## Paquetes nuevos que necesitas

Ninguno además de los que ya tienes (`data.table`, `xgboost`, `glmnet`, `nnet`,
`brms`, `rstan`, `plumber`, `jsonlite`). El único añadido es `stats::isoreg`
para la calibración isotónica, que ya viene en R base.

## Roadmap opcional (si quieres seguir subiendo el nivel más adelante)

Esto no lo implementé ahora porque necesita datos que no tienes o cómputo
que no cabe en una sesión — pero si quieres, lo armamos en otra ronda:

- **Cuotas de cierre de casas de apuestas como feature**: es, en la
  literatura, el predictor individual más fuerte que existe para fútbol.
  Si consigues un histórico de cuotas (ej. football-data.co.uk lo tiene
  gratis para ligas domésticas, no para selecciones), se puede sumar como
  feature adicional.
- **keras3/TensorFlow real** en vez de la red `nnet` de una capa (ya dejaste
  el código comentado listo en `07`, solo falta que lo corras localmente).
- **CV walk-forward "en bloques"** para tunear hiperparámetros de XGBoost
  en vez de 5-fold aleatorio (reduce un poco de fuga temporal en la
  selección de hiperparámetros, aunque no en el split train/test principal).
- **Actualización automática de resultados**: un scraper que jale resultados
  nuevos cada día en vez de depender de que subas un CSV a mano.
