"""Download and parse Jeff Sackmann's point-by-point ATP data."""

from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent / "data"
PBP_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_pointbypoint/master"
ATP_PBP_FILES = [
    "pbp_matches_atp_main_archive.csv",
    "pbp_matches_atp_main_current.csv",
]
WTA_PBP_FILES = [
    "pbp_matches_wta_main_archive.csv",
    "pbp_matches_wta_main_current.csv",
]


def download_pbp_data(force: bool = False) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for fname in ATP_PBP_FILES + WTA_PBP_FILES:
        dest = DATA_DIR / fname
        if dest.exists() and not force:
            continue
        print(f"Downloading {fname}...")
        response = requests.get(f"{PBP_BASE}/{fname}", timeout=30)
        response.raise_for_status()
        dest.write_bytes(response.content)


def load_pbp_data(tour: str = "atp") -> pd.DataFrame:
    files = ATP_PBP_FILES if tour == "atp" else WTA_PBP_FILES
    frames = []
    for fname in files:
        path = DATA_DIR / fname
        if not path.exists():
            continue
        frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        raise FileNotFoundError(f"No {tour} pbp data found. Run download_pbp_data() first.")
    return pd.concat(frames, ignore_index=True)


_POINT_CHARS = frozenset("SARD")
_SERVE_PRIOR = 0.62
_RETURN_PRIOR = 0.38


def _game_winner_is_server(game_str: str) -> bool | None:
    """
    Return True if the server won the game, False if receiver won.
    Returns None for tiebreaks (contain '/') or empty strings.
    """
    if not game_str or "/" in game_str:
        return None
    last = game_str[-1]
    if last in "SA":
        return True
    if last in "RD":
        return False
    return None


def parse_match_states(pbp_str: str, winner: int) -> list[dict]:
    """
    Parse a point-by-point string into one training row per game boundary.

    Args:
        pbp_str: The encoded point-by-point string.
        winner: 1 if server1 won the match, 2 if server2 won.

    Returns:
        List of dicts with keys: sets_won, sets_lost, set_diff,
        games_won_current_set, games_lost_current_set, game_diff_current_set,
        serving, p1_serve_win_rate, p1_return_win_rate,
        label (1 if server1 wins match, 0 if server2 wins).
    """
    if not isinstance(pbp_str, str) or not pbp_str.strip():
        return []

    label = 1 if winner == 1 else 0
    rows = []

    s1_sets = 0
    s2_sets = 0
    total_games = 0  # used to track whose turn it is to serve

    # Cumulative point stats from s1's perspective
    s1_serve_won = 0
    s1_serve_total = 0
    s1_return_won = 0
    s1_return_total = 0

    # Cumulative games won/lost across completed sets
    s1_completed_games = 0
    s2_completed_games = 0

    for set_str in pbp_str.split("."):
        s1_games = 0
        s2_games = 0

        for game_str in set_str.split(";"):
            game_str = game_str.strip()
            if not game_str:
                continue

            # server1 serves on even-numbered games (0-indexed)
            s1_serving = (total_games % 2 == 0)

            rows.append({
                "sets_won": s1_sets,
                "sets_lost": s2_sets,
                "set_diff": s1_sets - s2_sets,
                "games_won_current_set": s1_games,
                "games_lost_current_set": s2_games,
                "game_diff_current_set": s1_games - s2_games,
                "games_margin_completed_sets": s1_completed_games - s2_completed_games,
                "serving": int(s1_serving),
                "p1_serve_win_rate": s1_serve_won / s1_serve_total if s1_serve_total > 0 else _SERVE_PRIOR,
                "p1_return_win_rate": s1_return_won / s1_return_total if s1_return_total > 0 else _RETURN_PRIOR,
                "p1_serve_points": s1_serve_total,
                "p1_return_points": s1_return_total,
                "label": label,
            })

            # Accumulate point-level stats from this game (skip tiebreaks)
            if "/" not in game_str:
                for c in game_str:
                    if c not in _POINT_CHARS:
                        continue
                    if s1_serving:
                        s1_serve_total += 1
                        if c in "SA":
                            s1_serve_won += 1
                    else:
                        s1_return_total += 1
                        if c in "RD":
                            s1_return_won += 1

            server_won = _game_winner_is_server(game_str)
            if server_won is None:
                # Skip tiebreak but still advance game count
                total_games += 1
                continue

            if s1_serving:
                if server_won:
                    s1_games += 1
                else:
                    s2_games += 1
            else:
                if server_won:
                    s2_games += 1
                else:
                    s1_games += 1

            total_games += 1

        # Award set and accumulate completed games
        s1_completed_games += s1_games
        s2_completed_games += s2_games
        if s1_games > s2_games:
            s1_sets += 1
        elif s2_games > s1_games:
            s2_sets += 1

    return rows


GRAND_SLAMS = ("australian", "frenchopen", "usopen", "wimbledon")


def _is_grand_slam(tny_name: str) -> bool:
    name = tny_name.lower().replace(" ", "").replace("'", "")
    return any(slam in name for slam in GRAND_SLAMS)


def _build_tny_surface_map(matches_df) -> dict[str, str]:
    """Build {normalized_tny_name: surface} lookup from historical match data."""
    import re
    def norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())
    tny_surface = matches_df.dropna(subset=["surface"]).groupby("tourney_name")["surface"].last()
    return {norm(k): v for k, v in tny_surface.items()}


def _lookup_surface(tny_name: str, surface_map: dict[str, str]) -> str | None:
    import re
    nt = re.sub(r"[^a-z0-9]", "", str(tny_name).lower())
    for mk, surf in surface_map.items():
        if mk and (mk in nt or nt in mk):
            return surf
    return None


def build_pbp_training_rows(
    pbp_df: pd.DataFrame,
    ratings: dict,
    surface_ratings: dict | None = None,
    matches_df=None,
) -> pd.DataFrame:
    """
    Convert pbp match data into per-game-state training rows with Elo features.
    Grand slams (best-of-5) are excluded so the model only learns best-of-3 dynamics.
    Elo diff is always from server1's perspective.
    """
    import re

    before = len(pbp_df)
    pbp_df = pbp_df[~pbp_df["tny_name"].fillna("").apply(_is_grand_slam)]
    print(f"  Excluded {before - len(pbp_df):,} grand slam matches ({len(pbp_df):,} remaining)")

    def normalize(name: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", name.lower().replace("-", " "))

    norm_ratings = {normalize(k): v for k, v in ratings.items()}

    def lookup_elo(name: str) -> float | None:
        if not isinstance(name, str):
            return None
        norm = normalize(name)
        if norm in norm_ratings:
            return norm_ratings[norm]
        last = norm.split()[-1]
        found = [v for k, v in norm_ratings.items() if k.endswith(last) or k.split()[-1] == last]
        return found[0] if len(found) == 1 else None

    # Build surface Elo name lookups if provided
    surf_norm: dict[str, dict[str, float]] = {}
    tny_surface_map: dict[str, str] = {}
    if surface_ratings and matches_df is not None:
        tny_surface_map = _build_tny_surface_map(matches_df)
        for surf, pid_elo in surface_ratings.items():
            surf_norm[surf] = {normalize(k): v for k, v in pid_elo.items()}

    all_rows = []
    for _, row in pbp_df.iterrows():
        s1_elo = lookup_elo(str(row.get("server1", "")))
        s2_elo = lookup_elo(str(row.get("server2", "")))
        if s1_elo is None or s2_elo is None:
            continue

        elo_diff = s1_elo - s2_elo

        # Surface Elo diff (0.0 if unknown)
        surface_elo_diff = 0.0
        if surf_norm and tny_surface_map:
            surf = _lookup_surface(str(row.get("tny_name", "")), tny_surface_map)
            if surf and surf in surf_norm:
                sn = surf_norm[surf]
                s1n = normalize(str(row.get("server1", "")))
                s2n = normalize(str(row.get("server2", "")))
                sel1 = sn.get(s1n)
                sel2 = sn.get(s2n)
                if sel1 is not None and sel2 is not None:
                    surface_elo_diff = sel1 - sel2

        winner = row.get("winner")
        if winner not in (1, 2):
            try:
                winner = int(winner)
            except (ValueError, TypeError):
                continue

        states = parse_match_states(str(row.get("pbp", "")), winner)
        for state in states:
            state["elo_diff"] = elo_diff
            state["surface_elo_diff"] = surface_elo_diff
            all_rows.append(state)

    return pd.DataFrame(all_rows)
