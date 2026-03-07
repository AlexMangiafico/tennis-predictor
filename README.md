# Tennis Predictor

A live tennis match prediction dashboard for ATP and WTA tours. Uses Elo ratings and point-by-point historical data to estimate win probability at every game state during a match.

## How it works

**Training** (`train.py`): Downloads Jeff Sackmann's historical match data and point-by-point strings. For each game boundary in every best-of-3 match, it records the match state (sets, games, serve/return win rates + point counts, Elo diff) and who ultimately won. A `HistGradientBoostingClassifier` is trained on these states.

**Live predictions** (`today.py` + `app.py`): Fetches today's ATP/WTA matches via the Tennis Abstract gateway API. For each live match, polls Hawkeye stats and runs the current game state through the model. Also shows four counterfactual probabilities: what happens if P1/P2 wins the current game or set.

**Features used by the model:**
- `elo_diff`, `surface_elo_diff` — overall and surface-specific Elo difference
- `sets_won`, `sets_lost`, `set_diff` — current set score
- `games_won_current_set`, `games_lost_current_set`, `game_diff_current_set` — current set game score
- `serving` — whether player 1 is currently serving
- `p1_serve_win_rate`, `p1_return_win_rate` — in-match point win rates (defaults to tour averages early on)
- `p1_serve_points`, `p1_return_points` — point counts (lets model discount extreme early-match stats)

## Local setup

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
git clone https://github.com/AlexMangiafico/tennis-predictor.git
cd tennis-predictor
uv sync
```

**Train the models** — downloads ~500MB of historical data and trains ATP and WTA models. Only needs to be done once (or re-run to update with new match data):

```bash
python train.py
```

**Run the dashboard:**

```bash
python app.py
```

Open `http://localhost:5001` in your browser. The dashboard shows today's ATP and WTA matches with live win probabilities and auto-refreshes every 30 seconds.

## Data sources

Historical data is downloaded at runtime from Jeff Sackmann's repositories, licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/):

- [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp)
- [JeffSackmann/tennis_wta](https://github.com/JeffSackmann/tennis_wta)
- [JeffSackmann/tennis_pointbypoint](https://github.com/JeffSackmann/tennis_pointbypoint)

Live match data is fetched from the Tennis Abstract gateway API.

Neither the raw data nor trained model files are included in this repository. Any models trained on Sackmann's data inherit the CC BY-NC-SA 4.0 non-commercial restriction.
