"""Download and load historical match data from Jeff Sackmann's tennis datasets."""

import io
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
WTA_BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"
YEARS = range(2000, 2025)


def download_atp_data(years: range = YEARS, force: bool = False) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for year in tqdm(years, desc="Downloading ATP data"):
        dest = DATA_DIR / f"atp_matches_{year}.csv"
        if dest.exists() and not force:
            continue
        url = f"{BASE_URL}/atp_matches_{year}.csv"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        dest.write_bytes(response.content)


def get_player_serve_stats(matches: pd.DataFrame, recent_years: int = 3) -> dict:
    """
    Compute per-player serve and return win rates from recent match data.
    Uses a Bayesian prior to smooth estimates for players with few matches.
    Returns {player_id: (serve_win_rate, return_win_rate)}.
    """
    SERVE_PRIOR = 0.62
    RETURN_PRIOR = 0.38
    SMOOTH = 30  # pseudo-count weight

    max_year = matches["year"].max()
    recent = matches[matches["year"] >= max_year - recent_years + 1].copy()

    stat_cols = ["winner_id", "loser_id", "w_svpt", "w_1stWon", "w_2ndWon",
                 "l_svpt", "l_1stWon", "l_2ndWon"]
    for col in stat_cols[2:]:
        recent[col] = pd.to_numeric(recent[col], errors="coerce")
    recent = recent.dropna(subset=stat_cols)

    recent["w_serve_won"] = recent["w_1stWon"] + recent["w_2ndWon"]
    recent["l_serve_won"] = recent["l_1stWon"] + recent["l_2ndWon"]
    recent["w_return_won"] = recent["l_svpt"] - recent["l_serve_won"]
    recent["l_return_won"] = recent["w_svpt"] - recent["w_serve_won"]

    w_agg = recent.groupby("winner_id").agg(
        serve_won=("w_serve_won", "sum"),
        serve_total=("w_svpt", "sum"),
        return_won=("w_return_won", "sum"),
        return_total=("l_svpt", "sum"),
    )
    l_agg = recent.groupby("loser_id").agg(
        serve_won=("l_serve_won", "sum"),
        serve_total=("l_svpt", "sum"),
        return_won=("l_return_won", "sum"),
        return_total=("w_svpt", "sum"),
    )
    l_agg.index.name = "winner_id"
    combined = w_agg.add(l_agg, fill_value=0)
    combined["swr"] = (combined["serve_won"] + SERVE_PRIOR * SMOOTH) / (combined["serve_total"] + SMOOTH)
    combined["rwr"] = (combined["return_won"] + RETURN_PRIOR * SMOOTH) / (combined["return_total"] + SMOOTH)

    return dict(zip(combined.index, zip(combined["swr"], combined["rwr"])))


def download_wta_data(years: range = YEARS, force: bool = False) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for year in tqdm(years, desc="Downloading WTA data"):
        dest = DATA_DIR / f"wta_matches_{year}.csv"
        if dest.exists() and not force:
            continue
        url = f"{WTA_BASE_URL}/wta_matches_{year}.csv"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        dest.write_bytes(response.content)


def load_matches(years: range = YEARS) -> pd.DataFrame:
    """Load all downloaded ATP match CSVs into a single DataFrame."""
    frames = []
    for year in years:
        path = DATA_DIR / f"atp_matches_{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df["year"] = year
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No match data found. Run download_atp_data() first.")
    return pd.concat(frames, ignore_index=True)


def load_wta_matches(years: range = YEARS) -> pd.DataFrame:
    """Load all downloaded WTA match CSVs into a single DataFrame."""
    frames = []
    for year in years:
        path = DATA_DIR / f"wta_matches_{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df["year"] = year
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No WTA match data found. Run download_wta_data() first.")
    return pd.concat(frames, ignore_index=True)
