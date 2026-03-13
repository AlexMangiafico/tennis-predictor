#!/usr/bin/env python3
"""Download data, compute Elo, and train ATP and WTA models."""

import pickle
from pathlib import Path

from src.data import download_atp_data, download_wta_data, get_player_serve_stats, load_matches, load_wta_matches
from src.elo import get_current_ratings, get_surface_ratings
from src.model import save, train
from src.pbp import build_pbp_training_rows, download_pbp_data, load_pbp_data

DATA_DIR = Path(__file__).parent / "data"

FEATURE_COLS = [
    "elo_diff", "surface_elo_diff", "sets_won", "sets_lost", "set_diff",
    "games_won_current_set", "games_lost_current_set", "game_diff_current_set",
    "games_margin_completed_sets",
    "serving", "p1_serve_win_rate", "p1_return_win_rate",
    "p1_serve_points", "p1_return_points",
]


def train_tour(tour: str) -> None:
    label = tour.upper()
    load_fn = load_matches if tour == "atp" else load_wta_matches
    prefix = "atp" if tour == "atp" else "wta"

    print(f"\n=== {label} ===")
    print(f"Loading {label} matches and computing Elo ratings...")
    matches = load_fn()
    ratings = get_current_ratings(matches)
    surface_ratings = get_surface_ratings(matches)

    # Build name → elo lookups
    name_ratings: dict[str, float] = {}
    name_surface_ratings: dict[str, dict[str, float]] = {"Hard": {}, "Clay": {}, "Grass": {}}
    for _, row in matches.iterrows():
        for name_col, id_col in [("winner_name", "winner_id"), ("loser_name", "loser_id")]:
            name = row.get(name_col)
            pid = row.get(id_col)
            if isinstance(name, str):
                if pid in ratings:
                    name_ratings[name] = ratings[pid]
                for surf in ("Hard", "Clay", "Grass"):
                    if pid in surface_ratings[surf]:
                        name_surface_ratings[surf][name] = surface_ratings[surf][pid]

    print(f"Loading {label} point-by-point data...")
    pbp_df = load_pbp_data(tour=tour)
    print(f"  {len(pbp_df):,} matches found in pbp data")

    print("Building training features from game states...")
    df = build_pbp_training_rows(pbp_df, name_ratings, name_surface_ratings, matches)
    print(f"  {len(df):,} game-state training examples generated")

    X = df[FEATURE_COLS]
    y = df["label"]

    print(f"Training on {len(X):,} examples...")
    model = train(X, y)
    save(model, FEATURE_COLS, tour=tour)

    # Save runtime resources so the app doesn't need CSV files at startup
    lookup_cols = ["winner_name", "winner_id", "loser_name", "loser_id", "tourney_name", "surface"]
    resources = {
        "ratings": ratings,
        "surface_ratings": surface_ratings,
        "serve_stats": get_player_serve_stats(matches),
        "matches_df": matches[lookup_cols].copy(),
    }
    res_path = DATA_DIR / f"resources_{tour}.pkl"
    with open(res_path, "wb") as f:
        pickle.dump(resources, f)
    print(f"Resources saved to {res_path}")


def main():
    print("Downloading ATP match data...")
    download_atp_data()
    print("Downloading WTA match data...")
    download_wta_data()
    print("Downloading point-by-point data...")
    download_pbp_data()

    train_tour("atp")
    train_tour("wta")
    print("\nDone.")


if __name__ == "__main__":
    main()
