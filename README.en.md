# A.N.D.R.E. — Adaptive Neural Dixon-coles Regularized Ensemble

Predictive model for international football (World Cup 2026) that combines a
classic statistical model (Dixon-Coles), machine learning (XGBoost, a neural
network) and a hierarchical Bayesian model (Stan/brms), joined by a
meta-learner (stacking). It includes an R API (`plumber`) and a single-file
interactive HTML dashboard with live market odds.

[![R syntax](https://github.com/s7ex-j/A.N.D.R.E-Modelo-predictivo-mundiales/actions/workflows/check-r.yml/badge.svg)](https://github.com/s7ex-j/A.N.D.R.E-Modelo-predictivo-mundiales/actions/workflows/check-r.yml)
[![Dashboard](https://img.shields.io/badge/dashboard-live-2ea44f)](https://s7ex-j.github.io/A.N.D.R.E-Modelo-predictivo-mundiales/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

<sub>[Español](README.md) · **English**</sub>

**Live dashboard:** [s7ex-j.github.io/A.N.D.R.E-Modelo-predictivo-mundiales](https://s7ex-j.github.io/A.N.D.R.E-Modelo-predictivo-mundiales/)

**Author:** Jharol André Vilca Ramos · [ORCID 0009-0009-7897-3439](https://orcid.org/0009-0009-7897-3439) · [@s7ex.j](https://instagram.com/s7ex.j)

> The scripts, comments and decision log are in Spanish.

---

## Results (full test set: 4,098 matches, 2023-01-02 to 2026-10-06)

Every model is evaluated on **the same 4,098 matches**, which none of them used
for fitting: calibration and stacking are trained on 2019-2022 *out-of-fold*
predictions (see Methodology). Data from `martj42/international_results` up to
2026-10-06.

| Model | Accuracy (95% CI) | Log-loss | Brier | RPS (95% CI) |
|---|---|---|---|---|
| Naive (training frequencies) | 47.2% [45.7, 48.7] | 1.053 | 0.635 | 0.230 [0.227, 0.233] |
| Dixon-Coles | 56.0% [54.4, 57.4] | 0.938 | 0.555 | 0.191 [0.187, 0.196] |
| Neural network | 59.6% [58.1, 61.1] | 0.872 | 0.513 | 0.171 [0.166, 0.176] |
| XGBoost | 60.4% [58.8, 61.8] | 0.864 | 0.507 | 0.169 [0.165, 0.174] |
| Simple average (calibrated) | 60.1% [58.5, 61.6] | 0.872 | 0.513 | 0.171 [0.167, 0.176] |
| Stacking (meta-learner) | 60.5% [59.0, 62.0] | 0.868 | 0.509 | 0.169 [0.164, 0.174] |

**Reading:** stacking and XGBoost are **tied** (difference 0.1 points, 95% CI of
the difference [-0.4, +0.6]; paired bootstrap, 5,000 resamples). Stacking does
beat the neural network (+0.9 points, significant), and every model clearly
beats Dixon-Coles and the naive baseline. In other words, combining models adds
no accuracy over XGBoost alone; the API serves the stacking model, which
performs the same and also returns each base model's probabilities.

On **World Cup 2026** matches (104): stacking 67.3% (RPS 0.157), neural network
and XGBoost 65.4%, average 63.5%, Dixon-Coles 51.9%. Stacking is significantly
better only than Dixon-Coles; against the others the difference is not
conclusive with 104 matches.

**Draws:** 22.8% of test matches ended in a draw, but stacking picks "draw" in
only 2.7% of them (XGBoost in 0.3%). This is not a calibration error: the mean
draw probability they assign (~22%) matches the real frequency. A draw is just
almost never the *most* likely outcome of a match. Details, confusion matrix and
accuracy by match type (friendly, qualifier, tournament, World Cup; home or
neutral venue) in `scripts/09f_draws_and_segments.R`.

Calibration ("home win", 10 bins): every model sits between 0.018 and 0.026 mean
gap; the neural network is the best calibrated (0.018) — see
`scripts/09e_calibration_diagnostics.R`.

The **Bayesian** model (`06`) is reported separately because it covers only 414
matches (Oct-Dec 2025): 59.4% accuracy, with no significant difference from the
others in that window. It converges (max Rhat 1.003 across 467 parameters).

> Figures may move by a few tenths depending on platform and package versions;
> `renv.lock` pins the versions. Earlier versions of this README reported other
> figures: see the [decision log](docs/BITACORA_DECISIONES.md) (§5 and §6, in
> Spanish) for why they changed.

## Repository structure

```
.
├── data/
│   ├── results.csv            # source dataset (martj42/international_results, 1872-2026)
│   └── team_name_map.csv      # unifies historical national team names
├── scripts/
│   ├── 00_config.R            # shared settings (cut-off dates, features, ensemble models)
│   ├── 01_data_prep.R         # cleaning, 1X2 target, importance weight, name unification
│   ├── 02_elo_ratings.R       # dynamic pre-match Elo rating
│   ├── 03_dixon_coles.R       # walk-forward Dixon-Coles, analytical gradient
│   ├── 04_feature_engineering.R
│   ├── 05_model_xgboost.R     # walk-forward tuning + out-of-fold predictions 2019-2022
│   ├── 06_model_bayesian_mcmc.R  # Stan/brms, hierarchical partial pooling (optional, outside the ensemble)
│   ├── 07_model_deep_learning.R
│   ├── 08_ensemble.R          # isotonic calibration + stacking (glmnet), fitted on 2019-2022
│   ├── 09_evaluation.R        # metrics + 95% CI (bootstrap)
│   ├── 09b_evaluation_worldcup.R  # accuracy on World Cup matches only
│   ├── 09c_significance_test.R    # bootstrap, significance (World Cup)
│   ├── 09d_significance_full.R    # bootstrap, significance (full test)
│   ├── 09e_calibration_diagnostics.R
│   ├── 09f_draws_and_segments.R   # draws, confusion matrix, accuracy by match type
│   └── 10_build_team_state.R  # snapshot for the API and Dashboard/team_state.json
├── api/
│   ├── plumber.R              # REST API (endpoints /health, /predict)
│   ├── predict_match.R        # live prediction logic
│   └── Dockerfile             # to deploy the API
├── Dashboard/
│   ├── andre_mundial2026.html # single-file dashboard (no build step)
│   └── team_state.json        # Elo and Dixon-Coles per team (refreshed weekly by an Action)
├── pipeline.R                 # runs every script in order
├── renv.lock                  # exact package versions
└── docs/
    ├── BITACORA_DECISIONES.md # bugs found, fixes and methodological decisions
    └── API.md                 # how to run, connect to and deploy the API
```

`output/` (trained models, `.rds`) is generated locally when you run the
pipeline and is not versioned (see `.gitignore`). The same goes for the
intermediate CSVs (`results_clean.csv`, `model_features.csv`,
`ensemble_predictions.csv`, etc.): they can be regenerated from
`data/results.csv` and the scripts, so there is no point versioning them.

## Methodology

1. **Everything is walk-forward / pre-match.** Elo, Dixon-Coles and recent
   form only use information available before each match — never a team's
   "final" rating to predict a match from years ago.
2. **Dixon-Coles is refit every 90 days**, with exponential time decay and a
   hand-derived analytical gradient (~90 refits at 0.3s each vs. >2min with
   finite differences).
3. **Regularization** (ridge in Dixon-Coles, hierarchical partial pooling in
   the Bayesian model) so that teams with few matches do not produce absurd
   expected goals.
4. **Temporal split, never random.** The whole test set (2023+) comes after
   the whole train set. XGBoost hyperparameters are also chosen with temporal
   validation: for each year from 2019 to 2022, train on everything before it
   and measure on that year (not random k-fold, which would mix future matches
   into training).
5. **Out-of-fold predictions for the ensemble.** Those same yearly cuts give
   2019-2022 predictions that no model saw while training; they are used to fit
   each model's **isotonic calibration** and the **stacking** meta-learner
   (glmnet). That leaves the full test set free to evaluate every model equally.
6. **Home advantage is NOT removed on neutral ground during training**
   (a deliberate decision, not an oversight — see [`docs/BITACORA_DECISIONES.md`](docs/BITACORA_DECISIONES.md)):
   we tried removing it and accuracy got worse, because even in "neutral"
   matches the team labeled `home_team` in the data keeps real signal. On the
   other hand, when **serving a prediction for a hypothetical match** (e.g. a
   World Cup pairing where it is not yet decided who is "home"), the API does
   treat the match as neutral with no advantage, because there is no real home
   team there.

## How to run it

1. Install R (4.5 or later) and, only if you want the Bayesian model, a C++
   compiler: on Windows that is **Rtools** of the same minor version as your R
   (R 4.5 and 4.6 use Rtools45).
2. Open `Modelo predictivo mundiales.Rproj` (so the working directory is the
   repo root; the scripts use relative paths).
3. Install the exact package versions the project was verified with:

```r
install.packages("renv")
renv::restore()
```

   (Without `renv`: `install.packages(c("dplyr","data.table","lubridate","glmnet","nnet",
   "xgboost","jsonlite","plumber"))`, plus `"brms","rstan"` for the Bayesian model.)
4. Run everything:

```r
source("pipeline.R")
```

It runs the 15 scripts in order, each in a clean environment, and leaves
everything in `data/` and `output/`. The Bayesian model (`06`) is optional: it
is skipped if `brms`/`rstan` or the compiler are missing, or if you set
`ANDRE_SKIP_BAYES=1`.

Every push runs the full pipeline on GitHub Actions (without the Bayesian
model) and fails if any model does not beat the baseline or the API does not
respond.

## Live API

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

To deploy it on the internet (not just localhost), use the included
`Dockerfile` with Render, Railway or Fly.io.

## Dashboard

`Dashboard/andre_mundial2026.html` is self-contained (and published on [GitHub Pages](https://s7ex-j.github.io/A.N.D.R.E-Modelo-predictivo-mundiales/)) — open it
directly in the browser; it needs no server or build step. It includes:
- Match selector with the real World Cup 2026 bracket (round of 16 → final,
  including hypothetical scenarios for pairings not yet decided)
- Model probabilities compared with live market odds
  ([The Odds API](https://the-odds-api.com), free tier)
- %EDGE and fair odds per market (1X2, over/under 2.5, both teams to score)
- Heat map of likely scorelines, real backtest, calibration
- Real mobile preview (iframe with its own viewport)
- The model's Elo and Dixon-Coles expected goals (`team_state.json`, refreshed
  every Monday with new results by a GitHub Action)
- Optional connection to the live API (above) to use the real trained models
  instead of the Elo→Poisson heuristic

## Use of AI

I used an AI assistant (Claude) during development, mostly to audit the
pipeline, debug the API and draft part of the code and documentation. The
methodological decisions and the results were validated by running the
pipeline on the real data; the errors that audit found, and a decision I
reverted after measuring it, are in
[`docs/BITACORA_DECISIONES.md`](docs/BITACORA_DECISIONES.md).

## License

MIT — see `LICENSE`.

## Disclaimer

This project is an informational/educational tool for exploring statistical
modeling and machine learning applied to football. It is not financial
advice nor a guarantee of results. Betting carries a real risk of loss.
