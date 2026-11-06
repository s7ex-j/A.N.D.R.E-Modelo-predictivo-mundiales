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

## Results (test set: matches since 2023-01-01, n=3,686)

> **Figures under review.** In this table the stacking model was evaluated only
> on the last 30% of the test set (the stretch its meta-learner did not use),
> while the other models were evaluated on all 3,686 matches, so they are not
> compared on the same sample. The fix, which evaluates every model on the same
> holdout, is in
> [#3](https://github.com/s7ex-j/A.N.D.R.E-Modelo-predictivo-mundiales/pull/3)
> and the table will be updated once the pipeline is re-run.

| Model | Accuracy | Log-loss | Brier | RPS |
|---|---|---|---|---|
| Naive (historical frequency) | 47.3% | 1.053 | 0.635 | 0.229 |
| Dixon-Coles | 56.3% | 0.936 | 0.554 | 0.190 |
| Neural network | 59.3% | 0.873 | 0.513 | 0.170 |
| XGBoost | 60.0% | 0.861 | 0.506 | 0.168 |
| Simple average (calibrated) | 60.0% | 0.888 | 0.521 | 0.175 |
| **Stacking (meta-learner)** | **62.0%** | **0.836** | **0.489** | **0.160** |

On **World Cup 2026** matches specifically (n=88, up to the quarterfinals):
stacking 62.5% accuracy, RPS 0.167 — significantly better than the simple
average (bootstrap, p<0.05), although against XGBoost or the neural network
alone the difference is not yet conclusive with this number of matches.

Stacking is also the **best calibrated** of the 5 models (mean gap 0.022 in a
10-bin diagnostic) — see `scripts/09e_calibration_diagnostics.R`.

## Repository structure

```
.
├── data/
│   ├── results.csv            # source dataset (martj42/international_results, 1872-2026)
│   └── team_name_map.csv      # unifies historical national team names
├── scripts/
│   ├── 01_data_prep.R         # cleaning, 1X2 target, importance weight, name unification
│   ├── 02_elo_ratings.R       # dynamic pre-match Elo rating
│   ├── 03_dixon_coles.R       # walk-forward Dixon-Coles, analytical gradient
│   ├── 04_feature_engineering.R
│   ├── 05_model_xgboost.R
│   ├── 06_model_bayesian_mcmc.R  # Stan/brms, hierarchical partial pooling
│   ├── 07_model_deep_learning.R
│   ├── 08_ensemble.R          # stacking (glmnet) + isotonic calibration
│   ├── 09_evaluation.R
│   ├── 09b_evaluation_worldcup.R  # accuracy on World Cup matches only
│   ├── 09c_significance_test.R    # bootstrap, significance (World Cup)
│   ├── 09d_significance_full.R    # bootstrap, significance (full test)
│   ├── 09e_calibration_diagnostics.R
│   └── 10_build_team_state.R  # daily snapshot used to serve the API
├── api/
│   ├── plumber.R              # REST API (endpoints /health, /predict)
│   ├── predict_match.R        # live prediction logic
│   └── Dockerfile             # to deploy the API
├── Dashboard/
│   └── andre_mundial2026.html # single-file dashboard (no build step)
├── pipeline.R                 # runs every script in order
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
4. **Temporal split, never random.** The whole test set comes after the whole
   train set.
5. **Isotonic calibration** per model before averaging/stacking, and
   **stacking through a meta-learner (glmnet)** instead of a simple average.
6. **Home advantage is NOT removed on neutral ground during training**
   (a deliberate decision, not an oversight — see [`docs/BITACORA_DECISIONES.md`](docs/BITACORA_DECISIONES.md)):
   we tried removing it and accuracy got worse, because even in "neutral"
   matches the team labeled `home_team` in the data keeps real signal. On the
   other hand, when **serving a prediction for a hypothetical match** (e.g. a
   World Cup pairing where it is not yet decided who is "home"), the API does
   treat the match as neutral with no advantage, because there is no real home
   team there.

## How to run it

```r
install.packages(c("dplyr","data.table","lubridate","glmnet","nnet","xgboost",
                    "rstan","brms","plumber","jsonlite"))
```

`rstan`/`brms` need a C++ compiler (Rtools on Windows). The Bayesian model
(`06_model_bayesian_mcmc.R`) is optional — the rest of the pipeline runs
without it; it simply does not join the ensemble.

```r
source("pipeline.R")
```

This runs the 14 scripts in order and leaves everything ready in `data/` and
`output/`.

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
