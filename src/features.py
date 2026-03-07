"""Build feature vectors for match outcome prediction."""

import pandas as pd


def build_live_features(
    player_elo: float,
    opponent_elo: float,
    sets_won: int,
    sets_lost: int,
    games_won_current_set: int,
    games_lost_current_set: int,
    serving: bool,
    p1_serve_win_rate: float = 0.62,
    p1_return_win_rate: float = 0.38,
    surface_elo_diff: float = 0.0,
    p1_serve_points: int = 0,
    p1_return_points: int = 0,
) -> pd.DataFrame:
    """Build a single-row feature vector for a live in-progress match."""
    row = {
        "elo_diff": player_elo - opponent_elo,
        "surface_elo_diff": surface_elo_diff,
        "sets_won": sets_won,
        "sets_lost": sets_lost,
        "set_diff": sets_won - sets_lost,
        "games_won_current_set": games_won_current_set,
        "games_lost_current_set": games_lost_current_set,
        "game_diff_current_set": games_won_current_set - games_lost_current_set,
        "serving": int(serving),
        "p1_serve_win_rate": p1_serve_win_rate,
        "p1_return_win_rate": p1_return_win_rate,
        "p1_serve_points": p1_serve_points,
        "p1_return_points": p1_return_points,
    }
    return pd.DataFrame([row])
