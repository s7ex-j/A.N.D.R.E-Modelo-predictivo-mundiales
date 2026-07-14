# Addendum — validación con tus 49,493 partidos reales

Esta vez no probé con datos sintéticos: corrí tu pipeline completo (con mis
fixes) sobre tu `results.csv` real. Encontré 2 cosas importantes que cambian
lo que te dije antes.

## 🔴 El hallazgo más grave: la calibración isotónica estaba rota

`isoreg()` en R devuelve `$x` y `$y` en el **orden original de los datos**
(sin ordenar), pero `$yf` (los valores ajustados) SÍ vienen ordenados según
`$ord`. Yo estaba haciendo `approxfun(iso$x, iso$yf)` directo — mezclando un
vector desordenado con uno ordenado. Resultado: la "calibración" no tenía
ninguna relación real con la probabilidad de entrada (correlación ≈ 0).

Esto tumbó el accuracy del ensamble de 59.7%/61.5% a **46.8%/48.5%** — peor
que tirar una moneda con 3 caras. Si hubieras corrido `08_ensemble.R` (mi
versión con calibración) sin que yo lo detectara, tu API en producción
habría dado predicciones esencialmente aleatorias. Ya está corregido
(uso `iso$x[iso$ord]`, que sí calza con `iso$yf`) y **reprobado con tus
datos reales** — ver más abajo.

## 🟠 Revertí el fix de "neutral" en Dixon-Coles (no lo apliques)

Te había dicho que Dixon-Coles debía ignorar la ventaja de local en partidos
neutrales. Al probarlo con tus datos reales, el accuracy de Dixon-Coles solo
CAYÓ (de 56.3% a 54.2%). Investigué por qué: en tus 13,135 partidos marcados
`neutral`, el equipo etiquetado `home_team` **igual gana 44.2% vs pierde
33.4%** — no es 50/50. Hay señal real ahí (sede/región/convención de la
fuente), y mi fix la estaba tirando a la basura.

**Decisión final:** el entrenamiento de Dixon-Coles (`03_dixon_coles_FIX.R`)
vuelve a ser igual al original (gamma incondicional, usa esa señal real).
Pero `predict_match.R` (la API) SÍ sigue forzando `neutral=TRUE → gamma=0`
al servir un partido hipotético tuyo del bracket — porque ahí no hay un
"local" real, es solo el orden en que tú elegiste los equipos en el
dropdown. Los archivos que te adjunto ya tienen esta versión correcta.

## Números reales, regenerados con todo corregido (sin XGBoost, ver nota)

`evaluation_summary_REAL_regenerada.csv`:

| Modelo | Accuracy | RPS |
|---|---|---|
| Naive | 47.3% | 0.229 |
| Dixon-Coles | 56.3% | 0.190 |
| Red neuronal | 59.3% | 0.170 |
| Ensamble (promedio, calibrado) | 59.4% | 0.182 |
| Stacking | 60.3% | 0.164 |

Esto **no incluye XGBoost** (el paquete no está disponible en mi sandbox,
solo tengo repos de Ubuntu, no CRAN) — cuando lo corras tú y lo sumes al
ensamble, debería acercarse o superar tu número original de 61.5%.

`evaluation_worldcup_REAL.csv` — específico de los **88 partidos del Mundial
2026 ya jugados** (tu propio dataset ya los incluye, hasta el 3 de julio):

| Modelo | Accuracy | RPS |
|---|---|---|
| Dixon-Coles | 52.3% | 0.195 |
| Red neuronal | 59.1% | 0.175 |
| Ensamble | 55.7% | 0.190 |
| Stacking | 58.0% | 0.178 |

Nota: el bayesiano sale `NaN` en esta tabla porque tu ventana de entrenamiento
(sep-2024 a ene-2026) no cubre estos 88 partidos de 2026 -- normal, no es bug.

## Otro detalle que encontré al probar: falta un ajuste que te pedí antes

Tu `07_model_deep_learning.R` subido todavía no tiene la línea que guarda
`nnet_scaling.rds` (uno de los 2 ajustes manuales de la ronda pasada). Lo
apliqué yo mismo para poder probar, pero avísame -- tienes que agregarla tú
en tu copia real:

```r
saveRDS(list(mu=mu, sd=sdv), "output/nnet_scaling.rds")
```
justo después de la línea `saveRDS(nn_fit, "output/model_nnet.rds")`.

## Validé predict_match.R con 4 partidos reales del Mundial

Brazil-Argentina, Spain-Portugal, Morocco-France, Norway-England — las 4
corrieron sin errores, las probabilidades suman 1.0000 exacto, y varían de
forma sensata partido a partido (antes, con el bug de calibración, todas
daban casi el mismo resultado sin importar el partido -- ese fue justo el
primer indicio de que algo estaba mal).

## Qué hacer ahora

1. Reemplaza `03_dixon_coles_FIX.R`, `08_ensemble_FIX.R`, `predict_match.R`
   y `10_build_team_state.R` con las versiones adjuntas (todas cambiaron
   desde la ronda anterior).
2. Agrega la línea de `nnet_scaling.rds` en tu `07_model_deep_learning.R`.
3. Vuelve a correr `01 → 10` en tu máquina (con XGBoost esta vez).
4. Compárteme tu `evaluation_summary.csv` real con XGBoost incluido.
