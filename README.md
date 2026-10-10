# FPL Points Predictor

A machine-learning model that projects expected Fantasy Premier League points for every player over the next few gameweeks, using Opta-derived match stats, penalty/set-piece roles and bookmaker odds.

> ## Work in progress
> **This project still needs work and its projections are not that good yet.**
> Weekly FPL points are extremely noisy, and this model only explains a modest part of that noise. Treat the output as one input to your decisions (alongside fixtures, news, and your own judgement), not as a source of truth. Several parts have not been tested against real data (see [Known limitations](#known-limitations)).

---

## What it does

1. Downloads player data from the official FPL API (xG, xA, xGI, xGC, ICT, BPS, minutes, bonus, etc. per match).
2. Adds past seasons as extra training data.
3. Adds penalty / free-kick / corner taker roles.
4. Adds bookmaker odds, converted into each team's implied goals, win probability and clean-sheet probability.
5. Trains a gradient-boosted model on every historical player-match and predicts expected points for every upcoming fixture (double gameweeks are summed).
6. Also trains a goals model, giving each player's chance of scoring in the next gameweek.
7. Optionally builds the best 15-man squad within budget using an optimiser.

## Quick start

```bash
pip install -r requirements.txt

python fpl_predictor.py                      # live data, next 5 gameweeks
python fpl_predictor.py --horizon 8 --squad  # 8 gameweeks + optimal squad & captain
python fpl_predictor.py --demo               # offline test with simulated data
```

Results are saved to `predictions.csv`, and the top players, penalty takers and most likely scorers are printed to the terminal.

### Options

| Flag | Description |
|---|---|
| `--horizon N` | Gameweeks ahead to predict (default 5) |
| `--squad` | Also pick the optimal 15-man squad (2/5/5/3, max 3 per club, £100m) and a captain |
| `--seasons 2024-25 2025-26` | Past seasons used as extra training data. Pass `--seasons` with no values to disable |
| `--no-odds` | Skip bookmaker odds features |
| `--scorer-odds FILE` | Optional anytime-goalscorer odds CSV (see below) |
| `--odds-weight 0.6` | 0–1: how much to trust scorer odds versus the model |
| `--backtest-rounds 4` | Number of recent gameweeks held out for the printed backtest |
| `--refresh` | Ignore cached API data (cache lasts 6 hours) |
| `--demo` | Use simulated data, no internet needed |
| `--out FILE` | Output CSV path (default `predictions.csv`) |

### Optional: goalscorer odds

The script can blend in anytime-goalscorer odds from any source you have. Create a CSV with decimal odds (see `scorers_example.csv`):

```csv
player,team,odds
Haaland,Man City,1.65
Salah,Liverpool,2.10
```

```bash
python fpl_predictor.py --scorer-odds scorers.csv
```

There is no free bulk source for these odds, so you need to supply them yourself (paid odds APIs or manual entry). The `team` column is optional but strongly recommended to avoid ambiguous name matches. The bookmaker margin is calibrated away, so only the bookmaker's *relative* view of who is most likely to score is used. Odds apply to each player's first fixture in the next gameweek only.

## Data sources

| Source | Used for |
|---|---|
| [FPL API](https://fantasy.premierleague.com/api/bootstrap-static/) | Current-season stats (Opta-derived), fixtures, prices, injuries, penalty/set-piece roles |
| [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) | Past seasons' per-match data and team strengths |
| [football-data.co.uk](https://www.football-data.co.uk) | Historical and upcoming match odds (1X2 and over/under 2.5) |

The FPL API is unofficial and can change without notice. Please be considerate with request rates and check each source's terms of use.

## How it works

- **Features:** rolling form over the last 3, 6 and 12 matches plus season averages (computed only from *earlier* matches, so there is no leakage), opponent attack/defence strength, home/away, price, ownership rank, penalty/free-kick/corner roles, and odds-implied team goals, win probability and clean-sheet probability.
- **Model:** `HistGradientBoostingRegressor` (scikit-learn) predicting points per fixture. A second Poisson model predicts goals.
- **Odds → goals:** the bookmaker's margin is removed, then a Poisson model is fitted so that its win/draw/loss and over-2.5 probabilities match the market, giving each team's expected goals.
- **Missing odds:** football-data usually publishes odds only for the next round. For later gameweeks, odds-implied features are estimated from team strengths.
- **Availability:** FPL injury/suspension flags scale projections; doubts fade over longer horizons.
- **Squad optimiser:** mixed-integer program maximising starting-XI points (bench weighted at 0.1).

## Known limitations

- **Odds features are untested on real data.** They were only verified on simulated data, where they gave a small gain by construction. Check the backtest comparison on your first real run, and look for "unmatched team names" warnings (fix by adding to `ALIASES`).
- **Penalty flags did not improve the backtest.** Opta xG already includes penalty chances, so a player's recent xG already reflects penalty duty. The flags should help when a taker changes mid-season, but this has not been demonstrated. Past-season roles are end-of-season snapshots, which is slightly optimistic.
- **Live API download is untested end-to-end.** It was built against the documented schema, but only the archive loader and the pipeline (on simulated data) were run.
- **No lineup, rotation or press-conference information.** Minutes and availability are the biggest source of error and are handled crudely.
- **No cross-season player history.** Player IDs differ per season, so a player's previous-season form is not linked to his current one. New signings and early-season gameweeks are weak spots.
- **Players with 0 minutes this season are excluded.**
- **Team context is simplified.** Teams are mapped from the current squad list (mid-season transfers aren't tracked), and no managerial, injury-cluster or fixture-congestion effects are modelled.
- **Scorer-odds adjustment is a heuristic.** It shifts projections using a simple margin calibration and a fixed blend weight, and only covers the next gameweek.
- **Hyperparameters are untuned** and there is no probabilistic output (no confidence intervals).

## Ideas for improvement

- Minutes / "will he start?" model using rotation patterns, lineups and news
- Link player histories across seasons for better early-season projections
- Tune hyperparameters and add uncertainty estimates
- Detect and explicitly model changes in penalty-taker roles
- Add assist and clean-sheet market odds
- Multi-gameweek transfer and chip planning
- Walk-forward backtest over many gameweeks and seasons
