"""Compute per-player Elo ratings from historical match data."""

import pandas as pd

INITIAL_ELO = 1500.0
K = 32


def expected_score(r_a: float, r_b: float) -> float:
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


def compute_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Walk through matches chronologically, updating Elo after each match.

    Returns the input DataFrame with added columns:
        winner_elo_before, loser_elo_before
    """
    matches = matches.sort_values("tourney_date").reset_index(drop=True)
    ratings: dict[str, float] = {}

    winner_elo_before = []
    loser_elo_before = []

    for _, row in matches.iterrows():
        w = row["winner_id"]
        l = row["loser_id"]
        r_w = ratings.get(w, INITIAL_ELO)
        r_l = ratings.get(l, INITIAL_ELO)

        winner_elo_before.append(r_w)
        loser_elo_before.append(r_l)

        e_w = expected_score(r_w, r_l)
        ratings[w] = r_w + K * (1 - e_w)
        ratings[l] = r_l + K * (0 - (1 - e_w))

    matches["winner_elo_before"] = winner_elo_before
    matches["loser_elo_before"] = loser_elo_before
    return matches, ratings


def get_current_ratings(matches: pd.DataFrame) -> dict[str, float]:
    """Return final overall Elo ratings {player_id: elo}."""
    _, ratings = compute_elo(matches)
    return ratings


def get_surface_ratings(matches: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Compute per-surface Elo ratings.
    Returns {surface: {player_id: elo}} for Hard, Clay, Grass.
    """
    matches = matches.sort_values("tourney_date").reset_index(drop=True)
    ratings: dict[str, dict[str, float]] = {"Hard": {}, "Clay": {}, "Grass": {}}

    for _, row in matches.iterrows():
        surface = row.get("surface", "")
        if surface not in ratings:
            continue
        r = ratings[surface]
        w, l = row["winner_id"], row["loser_id"]
        r_w = r.get(w, INITIAL_ELO)
        r_l = r.get(l, INITIAL_ELO)
        e_w = expected_score(r_w, r_l)
        r[w] = r_w + K * (1 - e_w)
        r[l] = r_l + K * (0 - (1 - e_w))

    return ratings
