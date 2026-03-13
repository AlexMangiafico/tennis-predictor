#!/usr/bin/env python3
"""
Fetch today's ATP matches and predict outcomes.

Uses ESPN's public scoreboard API (no key required).
"""

import re
import requests
import pandas as pd
from src.features import build_live_features
from src.model import load, win_probability

ESPN_ATP_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard"
ATP_GATEWAY_URL = "https://app.atptour.com/api/v2/gateway/livematches/website?scoringTournamentLevel={level}"
ATP_HEADERS = {"User-Agent": "Mozilla/5.0", "Origin": "https://www.atptour.com", "Referer": "https://www.atptour.com/"}
HAWKEYE_URL = "https://www.atptour.com/-/Hawkeye/MatchStats/{year}/{event_id}/{match_id}"
HAWKEYE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Origin": "https://www.atptour.com",
    "Referer": "https://www.atptour.com/",
}


def fetch_today_matches() -> list[dict]:
    """Return list of match dicts from ESPN scoreboard."""
    response = requests.get(ESPN_ATP_URL, timeout=10)
    response.raise_for_status()
    data = response.json()

    matches = []
    for event in data.get("events", []):
        tournament_name = event.get("name", "")
        for grouping in event.get("groupings", []):
            # Only men's singles
            slug = grouping.get("grouping", {}).get("slug", "")
            if "mens-singles" not in slug:
                continue
            for competition in grouping.get("competitions", []):
                competitors = competition.get("competitors", [])
                if len(competitors) < 2:
                    continue

                status = competition.get("status", {}).get("type", {})
                state = status.get("state", "pre")

                p1 = competitors[0]
                p2 = competitors[1]

                p1_name = p1.get("athlete", {}).get("displayName", "")
                p2_name = p2.get("athlete", {}).get("displayName", "")
                if not p1_name or not p2_name or p1_name == "TBD" or p2_name == "TBD":
                    continue

                matches.append({
                    "tournament": tournament_name,
                    "p1_name": p1_name,
                    "p2_name": p2_name,
                    "state": state,
                    "p1_linescores": p1.get("linescores", []),
                    "p2_linescores": p2.get("linescores", []),
                    "serving": p1.get("possession", False),
                    "scheduled_time": competition.get("date"),
                })

    return matches


def normalize(name: str) -> str:
    """Lowercase, replace hyphens with spaces, strip remaining punctuation."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower().replace("-", " "))


def find_player(
    ratings: dict, surface_ratings: dict, serve_stats: dict,
    matches_df: pd.DataFrame, name: str, surface: str | None = None,
) -> tuple[float, float, float, float] | None:
    """
    Find Elo, surface Elo, serve win rate, and return win rate for a player.
    Returns (elo, surface_elo, serve_win_rate, return_win_rate) or None if not found.
    surface_elo falls back to overall elo when surface data is unavailable.
    """
    norm = normalize(name)
    last = norm.split()[-1]
    w_norm = matches_df["winner_name"].fillna("").apply(normalize)
    l_norm = matches_df["loser_name"].fillna("").apply(normalize)
    w_last = w_norm.str.split().str[-1].fillna("")
    l_last = l_norm.str.split().str[-1].fillna("")

    # Pass 1: full-name substring match
    # Pass 2: exact last-word match (avoids "jovic" matching "djokovic")
    searches = [
        (w_norm.str.contains(norm, regex=False), l_norm.str.contains(norm, regex=False)),
        (w_last == last, l_last == last),
    ]
    for w_mask, l_mask in searches:
        w = matches_df[w_mask]
        l = matches_df[l_mask]
        if w.empty and l.empty:
            continue
        pid = w.iloc[-1]["winner_id"] if not w.empty else l.iloc[-1]["loser_id"]
        elo = ratings.get(pid)
        if elo is None:
            return None
        surface_elo = surface_ratings.get(surface, {}).get(pid, elo) if surface else elo
        swr, rwr = serve_stats.get(pid, (0.62, 0.38))
        return elo, surface_elo, swr, rwr

    return None


def _get_tournament_surfaces(matches_df: pd.DataFrame) -> dict[str, str]:
    """Build {normalized_tourney_name: surface} from historical match data."""
    tny_surface = matches_df.dropna(subset=["surface"]).groupby("tourney_name")["surface"].last()
    return {normalize(k): v for k, v in tny_surface.items()}


def fetch_hawkeye_stats(event_id: str | int, match_id: str, year: int = 2026) -> tuple[float, float] | None:
    """
    Fetch live cumulative serve/return win rates from the ATP Hawkeye stats API.
    Returns (p1_serve_win_rate, p1_return_win_rate) or None if unavailable.
    """
    url = HAWKEYE_URL.format(year=year, event_id=event_id, match_id=str(match_id).lower())
    try:
        r = requests.get(url, headers=HAWKEYE_HEADERS, timeout=8)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data:
            return None
        set_scores = data["Match"]["PlayerTeam"].get("SetScores") or []
        if not set_scores:
            return None
        # SetScores[0].Stats holds cumulative match-wide totals
        stats = set_scores[0].get("Stats") or {}
        pt = stats.get("PointStats") or {}
        svc = pt.get("TotalServicePointsWon") or {}
        ret = pt.get("TotalReturnPointsWon") or {}
        if not svc.get("Divisor") or not ret.get("Divisor"):
            return None
        svc_won, svc_n = svc["Dividend"], svc["Divisor"]
        ret_won, ret_n = ret["Dividend"], ret["Divisor"]
        swr = svc_won / svc_n if svc_n > 0 else 0.62
        rwr = ret_won / ret_n if ret_n > 0 else 0.38
        return swr, rwr, int(svc_n), int(ret_n)
    except Exception:
        return None


def fetch_gateway_matches(level: str) -> list[dict]:
    """Return list of match dicts from the ATP Tour gateway live API for the given tournament level."""
    url = ATP_GATEWAY_URL.format(level=level)
    response = requests.get(url, headers=ATP_HEADERS, timeout=10)
    response.raise_for_status()
    data = response.json()

    tournaments = (data.get("Data") or {}).get("LiveMatchesTournamentsOrdered") or []
    matches = []
    for tournament in tournaments:
        tny_name = tournament.get("EventTitle", "")
        event_id = tournament.get("EventId", "")
        for match in tournament.get("LiveMatches", []):
            if match.get("IsDoubles"):
                continue

            status = match.get("MatchStatus", "")
            if status == "F":
                state = "post"
            elif status == "P":
                state = "in"
            else:
                state = "pre"

            p = match.get("PlayerTeam", {})
            o = match.get("OpponentTeam", {})
            p1 = p.get("Player", {})
            p2 = o.get("Player", {})

            p1_name = f"{p1.get('PlayerFirstName', '')} {p1.get('PlayerLastName', '')}".strip()
            p2_name = f"{p2.get('PlayerFirstName', '')} {p2.get('PlayerLastName', '')}".strip()
            if not p1_name or not p2_name or p1_name == "TBD" or p2_name == "TBD":
                continue

            # Parse set scores
            p1_sets_data = p.get("SetScores") or []
            p2_sets_data = o.get("SetScores") or []

            def set_complete(s1, s2) -> bool:
                if s1 is None or s2 is None:
                    return False
                return (max(s1, s2) >= 6 and abs(s1 - s2) >= 2) or max(s1, s2) == 7

            p1_sets_won = 0
            p2_sets_won = 0
            p1_current_games = 0
            p2_current_games = 0
            p1_completed_games = 0
            p2_completed_games = 0

            for ps, os_ in zip(p1_sets_data, p2_sets_data):
                s1 = ps.get("SetScore")
                s2 = os_.get("SetScore")
                if s1 is None and s2 is None:
                    break
                if set_complete(s1, s2):
                    p1_completed_games += s1
                    p2_completed_games += s2
                    if s1 > s2:
                        p1_sets_won += 1
                    else:
                        p2_sets_won += 1
                else:
                    # This set is in progress
                    p1_current_games = s1 or 0
                    p2_current_games = s2 or 0
                    break

            matches.append({
                "tournament": tny_name,
                "p1_name": p1_name,
                "p2_name": p2_name,
                "state": state,
                "p1_sets": p1_sets_won,
                "p2_sets": p2_sets_won,
                "p1_games": p1_current_games,
                "p2_games": p2_current_games,
                "games_margin_completed_sets": p1_completed_games - p2_completed_games,
                "serving": match.get("ServerTeam") == 0,  # 0 = PlayerTeam serving
                "source": "gateway",
                "live_stats": None,
                "event_id": event_id,
                "match_id": match.get("MatchId", ""),
                "scheduled_time": match.get("MatchDate") or match.get("ScheduledTime"),
            })

    return matches


def parse_linescores(p1_scores: list, p2_scores: list) -> tuple[int, int, int, int, int]:
    """Return (p1_sets, p2_sets, p1_current_games, p2_current_games, games_margin_completed_sets)."""
    if not p1_scores:
        return 0, 0, 0, 0, 0

    def val(s):
        return int(s.get("value", 0) or 0)

    completed = list(zip(p1_scores[:-1], p2_scores[:-1])) if len(p1_scores) > 1 else []
    p1_sets = sum(1 for a, b in completed if val(a) > val(b))
    p2_sets = sum(1 for a, b in completed if val(b) > val(a))
    games_margin_completed = sum(val(a) - val(b) for a, b in completed)

    p1_games = val(p1_scores[-1])
    p2_games = val(p2_scores[-1])
    return p1_sets, p2_sets, p1_games, p2_games, games_margin_completed


def _fetch_all_matches() -> list[dict]:
    """Fetch and dedup all matches from ESPN + ATP gateway, with parallel HTTP calls."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    levels = ("1000", "500", "250", "challenger")

    with ThreadPoolExecutor(max_workers=len(levels) + 1) as pool:
        espn_future = pool.submit(fetch_today_matches)
        gateway_futures = {pool.submit(fetch_gateway_matches, lvl): lvl for lvl in levels}

        try:
            today = espn_future.result()
        except Exception as e:
            print(f"Failed to fetch ESPN matches: {e}")
            today = []

        gateway_matches = []
        seen = set()
        for future in as_completed(gateway_futures):
            try:
                for m in future.result():
                    key = frozenset({normalize(m["p1_name"]), normalize(m["p2_name"])})
                    if key not in seen:
                        seen.add(key)
                        gateway_matches.append(m)
            except Exception:
                pass

    # Fetch Hawkeye in parallel for live gateway matches only
    live = [m for m in gateway_matches if m["state"] == "in"]
    if live:
        with ThreadPoolExecutor(max_workers=len(live)) as pool:
            futures = {
                pool.submit(fetch_hawkeye_stats, m["event_id"], m["match_id"]): m
                for m in live
            }
            for future in as_completed(futures):
                futures[future]["live_stats"] = future.result()

    gateway_live_pairs = {
        frozenset({normalize(m["p1_name"]), normalize(m["p2_name"])})
        for m in gateway_matches if m["state"] == "in"
    }
    today = [
        m for m in today
        if not (m["state"] == "in" and
                frozenset({normalize(m["p1_name"]), normalize(m["p2_name"])}) in gateway_live_pairs)
    ]
    today += gateway_matches
    return today


def _resolve_surface(tny_norm: str, tny_surfaces: dict) -> str | None:
    surf = tny_surfaces.get(tny_norm)
    if surf is None:
        for k, v in tny_surfaces.items():
            if k and (k in tny_norm or tny_norm in k):
                return v
    return surf


def _build_match_row(m: dict, res: dict, surface: str | None, tour: str) -> dict | None:
    """Compute probability fields for one match using the given tour resources."""
    model      = res["model"]
    feat_cols  = res["feature_cols"]
    matches_df = res["matches_df"]
    ratings    = res["ratings"]
    surf_rat   = res["surface_ratings"]
    serve_stat = res["serve_stats"]

    p1_info = find_player(ratings, surf_rat, serve_stat, matches_df, m["p1_name"], surface)
    p2_info = find_player(ratings, surf_rat, serve_stat, matches_df, m["p2_name"], surface)
    if p1_info is None or p2_info is None:
        return None

    p1_elo, p1_sel, p1_swr, p1_rwr = p1_info
    p2_elo, p2_sel, p2_swr, p2_rwr = p2_info
    surface_elo_diff = p1_sel - p2_sel

    if m.get("source") == "gateway":
        p1_sets, p2_sets = m["p1_sets"], m["p2_sets"]
        p1_games, p2_games = m["p1_games"], m["p2_games"]
        games_margin_completed = m.get("games_margin_completed_sets", 0)
    else:
        p1_sets, p2_sets, p1_games, p2_games, games_margin_completed = parse_linescores(
            m["p1_linescores"], m["p2_linescores"]
        )

    is_live = m["state"] == "in"
    live_stats = m.get("live_stats")
    if live_stats:
        live_swr, live_rwr, live_n_serve, live_n_return = live_stats
    else:
        live_swr, live_rwr, live_n_serve, live_n_return = p1_swr, p1_rwr, 0, 0

    X_pre = build_live_features(p1_elo, p2_elo, 0, 0, 0, 0, m["serving"], p1_swr, p1_rwr, surface_elo_diff, 0, 0)[feat_cols]
    pre_prob = win_probability(model, X_pre)

    def wp(sw, sl, gw, gl, swr=live_swr, rwr=live_rwr, serving=m["serving"],
           n_serve=live_n_serve, n_return=live_n_return, gm_completed=games_margin_completed):
        X = build_live_features(p1_elo, p2_elo, sw, sl, gw, gl, serving, swr, rwr, surface_elo_diff, n_serve, n_return, gm_completed)[feat_cols]
        return win_probability(model, X)

    def set_over(g1, g2):
        if (max(g1, g2) >= 6 and abs(g1 - g2) >= 2) or max(g1, g2) == 7:
            return 1 if g1 > g2 else 2
        return 0

    if is_live:
        live_prob = wp(p1_sets, p2_sets, p1_games, p2_games)
        def wp_set(sw, sl, new_gm_completed):
            return 0.5 * wp(sw, sl, 0, 0, serving=True, gm_completed=new_gm_completed) + \
                   0.5 * wp(sw, sl, 0, 0, serving=False, gm_completed=new_gm_completed)
        # When a set completes, add the current set's game scores to the completed margin
        gm_if_p1_set = games_margin_completed + (p1_games - p2_games)
        gm_if_p2_set = games_margin_completed + (p1_games - p2_games)
        if_p1_set = 1.0 if p1_sets + 1 >= 2 else wp_set(p1_sets + 1, p2_sets, gm_if_p1_set)
        if_p2_set = 0.0 if p2_sets + 1 >= 2 else wp_set(p1_sets, p2_sets + 1, gm_if_p2_set)
        sr_if_p1_game = set_over(p1_games + 1, p2_games)
        sr_if_p2_game = set_over(p1_games, p2_games + 1)
        next_serving = not m["serving"]
        if_p1_game = (if_p1_set if sr_if_p1_game == 1 else
                      if_p2_set if sr_if_p1_game == 2 else
                      wp(p1_sets, p2_sets, p1_games + 1, p2_games, serving=next_serving))
        if_p2_game = (if_p2_set if sr_if_p2_game == 2 else
                      if_p1_set if sr_if_p2_game == 1 else
                      wp(p1_sets, p2_sets, p1_games, p2_games + 1, serving=next_serving))
    else:
        live_prob = if_p1_game = if_p2_game = if_p1_set = if_p2_set = None

    def _n(v): return None if v is None else round(v, 4)

    return {
        "tournament": m.get("tournament", ""),
        "tour": tour,
        "state": "live" if is_live else "pre",
        "scheduled_time": m.get("scheduled_time"),
        "score": f"{p1_sets}-{p2_sets}, {p1_games}-{p2_games}" if is_live else "",
        "p1_name": m["p1_name"],
        "p2_name": m["p2_name"],
        "p1_serving": m["serving"],
        "p1_pre":    round(pre_prob, 4),
        "p2_pre":    round(1 - pre_prob, 4),
        "p1_now":    _n(live_prob),
        "p2_now":    _n(1 - live_prob if live_prob is not None else None),
        "p1_p1game": _n(if_p1_game),
        "p2_p1game": _n(1 - if_p1_game if if_p1_game is not None else None),
        "p1_p2game": _n(if_p2_game),
        "p2_p2game": _n(1 - if_p2_game if if_p2_game is not None else None),
        "p1_p1set":  _n(if_p1_set),
        "p2_p1set":  _n(1 - if_p1_set if if_p1_set is not None else None),
        "p1_p2set":  _n(if_p2_set),
        "p2_p2set":  _n(1 - if_p2_set if if_p2_set is not None else None),
    }


def build_rows(atp_res: dict, wta_res: dict | None = None, live_only: bool = False) -> list[dict]:
    """
    Fetch today's matches and compute win probabilities for both tours.
    Returns rows sorted live-first, with a 'tour' field ('atp' or 'wta').
    Gateway matches are always ATP. ESPN matches are classified by which
    tour's ratings can identify both players.
    """
    atp_surfaces = _get_tournament_surfaces(atp_res["matches_df"])
    wta_surfaces = _get_tournament_surfaces(wta_res["matches_df"]) if wta_res else {}

    rows = []
    for m in _fetch_all_matches():
        if live_only and m["state"] != "in":
            continue
        if m["state"] == "post":
            continue

        tny_norm = normalize(m.get("tournament", ""))
        is_gateway = m.get("source") == "gateway"

        # Gateway matches are always ATP
        if is_gateway:
            surface = _resolve_surface(tny_norm, atp_surfaces)
            row = _build_match_row(m, atp_res, surface, "atp")
            if row:
                rows.append(row)
            continue

        # ESPN matches: try ATP first, then WTA
        atp_surface = _resolve_surface(tny_norm, atp_surfaces)
        row = _build_match_row(m, atp_res, atp_surface, "atp")
        if row:
            rows.append(row)
            continue

        if wta_res:
            wta_surface = _resolve_surface(tny_norm, wta_surfaces)
            row = _build_match_row(m, wta_res, wta_surface, "wta")
            if row:
                rows.append(row)

    rows.sort(key=lambda r: (0 if r["state"] == "live" else 1, r["tour"], r["tournament"]))
    return rows


def _load_tour_resources(tour: str) -> dict:
    import pickle
    from pathlib import Path
    from src.model import load as load_model

    model, feature_cols = load_model(tour)
    res_path = Path(__file__).parent / "data" / f"resources_{tour}.pkl"
    with open(res_path, "rb") as f:
        res = pickle.load(f)
    return {
        "model": model,
        "feature_cols": feature_cols,
        "matches_df": res["matches_df"],
        "ratings": res["ratings"],
        "surface_ratings": res["surface_ratings"],
        "serve_stats": res["serve_stats"],
    }


def main(live_only: bool = False):
    print("Loading model and historical data...")
    atp_res = _load_tour_resources("atp")
    wta_res = _load_tour_resources("wta")

    print("Fetching today's matches...\n")
    rows = build_rows(atp_res, wta_res, live_only)

    if not rows:
        print("No matches found for today.")
        return

    def pct(v): return f"{v:.1%}" if v is not None else ""

    tny_w, status_w, score_w, name_w, pct_w = 26, 5, 14, 25, 6
    blank_left = " " * (tny_w + 1 + status_w + 1 + score_w + 1)

    for r in rows:
        tny   = f"{r['tournament'][:tny_w]:<{tny_w}}"
        state_str = "LIVE" if r["state"] == "live" else "Pre"
        state = f"{state_str:^{status_w}}"
        score = f"{r['score']:<{score_w}}"
        left  = f"{tny} {state} {score} "

        if r["state"] == "live":
            p1_serve = "* " if r["p1_serving"] else "  "
            p2_serve = "  " if r["p1_serving"] else "* "
            stats_label = "" if r["has_live_stats"] else "  [season avg]"
            p1 = r["p1_name"].split()[-1]
            p2 = r["p2_name"].split()[-1]
            g1h = f"+{p1}"[:pct_w]
            g2h = f"+{p2}"[:pct_w]
            s1h = f"+{p1}"[:pct_w]
            s2h = f"+{p2}"[:pct_w]
            print(f"{left}{'Player':<{name_w+2}} {'Now':>{pct_w}}  {'— if game won by —':<{pct_w*2+2}}  {'— if set won by —':<{pct_w*2+2}}  {'Pre':>{pct_w}}{stats_label}")
            print(f"{blank_left}{'':>{name_w+2}} {'':>{pct_w}}  {g1h:>{pct_w}}  {g2h:>{pct_w}}  {s1h:>{pct_w}}  {s2h:>{pct_w}}")
            print(f"{blank_left}{p1_serve}{r['p1_name']:<{name_w}} {pct(r['p1_now']):>{pct_w}}  {pct(r['p1_p1game']):>{pct_w}}  {pct(r['p1_p2game']):>{pct_w}}  {pct(r['p1_p1set']):>{pct_w}}  {pct(r['p1_p2set']):>{pct_w}}    {pct(r['p1_pre']):>{pct_w}}")
            print(f"{blank_left}{p2_serve}{r['p2_name']:<{name_w}} {pct(r['p2_now']):>{pct_w}}  {pct(r['p2_p1game']):>{pct_w}}  {pct(r['p2_p2game']):>{pct_w}}  {pct(r['p2_p1set']):>{pct_w}}  {pct(r['p2_p2set']):>{pct_w}}    {pct(r['p2_pre']):>{pct_w}}")
        else:
            print(f"{left}{r['p1_name']:<{name_w}} {pct(r['p1_pre']):>{pct_w}}   {r['p2_name']:<{name_w}} {pct(r['p2_pre']):>{pct_w}}")
        print()


if __name__ == "__main__":
    main()
