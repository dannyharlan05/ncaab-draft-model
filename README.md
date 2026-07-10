# NCAAB → NBA Draft Success Predictor

Predicts whether a college prospect will become a **successful NBA player** — defined as
accumulating **4+ VORP over their first four seasons** — from their college production and
play style. Instead of one model for everyone, it groups prospects by **play style** and
fits a separate model to each, because the traits that predict a successful big are not
the ones that predict a successful guard.

## The core idea: cluster, then predict

A stretch big, a slashing wing, and a pass-first guard succeed for completely different
reasons. Pooling them into one model muddies every signal. So this project:

1. **Clusters prospects into play-style groups** using role, size, rebounding, and
   shooting profile.
2. **Trains a separate classifier per cluster**, each with its own features, so the model
   can weigh (say) rim finishing heavily for bigs and playmaking heavily for guards.

The output is a **draft rating** per prospect — the model's estimated probability that
they clear the 4-VORP success bar.

## How it works

- **Target.** A binary "hit": did the player accumulate more than 4 VORP across their
  first four NBA seasons? Success is rare, so this is an imbalanced classification
  problem, handled with class weighting (positives up-weighted).
- **Play-style clusters.** Prospects are assigned to one of three style groups from their
  role, height, rebounding rates, and three-point profile.
- **Per-cluster logistic regression.** Each cluster gets its own regularized logistic
  model, trained across multiple random seeds and averaged for stability. Logistic
  regression keeps the model interpretable — every prediction breaks down into which
  traits pushed it up or down.
- **Features.** College advanced box stats, shot-location efficiency (rim / mid / three
  makes and attempts), size/role, class year, and a **projected draft pick** — a
  *pre-draft* consensus estimate (mock-board position), used as a signal available before
  the draft rather than the actual post-draft result, so it doesn't leak the outcome.
- **Class-year adjustment.** Younger prospects (freshmen) get an upside adjustment — the
  same production means more from an 18-year-old than a senior.
- **Tools built in:** a per-player prediction breakdown (which features drove the rating),
  a top-prospects-by-year ranking, and a "find similar players" nearest-neighbor lookup.

## Results

AUROC per play-style cluster, averaged over 10 random-seed runs:

| Play-style cluster | AUROC |
|---|---|
| 0 | 0.78 |
| 1 | 0.80 |
| 2 | 0.83 |

AUROC is the metric to judge this on. Success is rare, so the models are class-weighted
toward the positive case — which shifts the decision threshold and makes threshold-based
metrics (accuracy, precision, recall) reflect that chosen operating point rather than the
model's underlying skill. AUROC is threshold-independent, so it measures the thing that
actually matters: how well the model separates future hits from misses. **0.78–0.83 is a
strong result for a problem as noisy as draft outcomes.**

The ratings are also checked against a held-out draft class (2025), comparing them to
players' actual rookie-year production (VORP), with a look at how the lottery and
highly-rated prospects panned out.

## Data

| Source | Provides |
|---|---|
| barttorvik / college stats | advanced college box production, size, class year |
| Pre-draft board | projected draft pick (pre-draft consensus estimate) |
| NBA advanced stats (VORP) | the success target |

## Running it

Open `NCAAB_Predictor.ipynb` and run it top to bottom — it loads and merges the college /
combine / NBA sources, assigns play-style clusters, trains the per-cluster models, and
scores prospects.

```bash
pip install -r requirements.txt
jupyter lab NCAAB_Predictor.ipynb
```

## Honest limitations

- **Draft prediction is a hard, noisy problem.** College and combine data explain only
  part of NBA outcomes; the model tilts the odds, it doesn't call individual players with
  certainty.
- **Small samples per cluster** mean the models are lightly regularized and best read as a
  *ranking* rather than precise probabilities.
- **The 4-VORP success line is a modeling choice** — a different threshold would reshape
  who counts as a "hit."

## Built with

Python · scikit-learn (logistic regression, t-SNE) · pandas · NumPy · matplotlib.
