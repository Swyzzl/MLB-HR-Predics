
#!/usr/bin/env python3
"""
build_daily_lineups_mlb_page_fallback.py

Builds daily_lineups.csv for the HR model using this fallback order:
1. Confirmed MLB lineup from the live game feed (must have 9 hitters)
2. MLB probable/starting lineup page parser from the public Gameday preview page
3. Matchups / active roster fallback (all non-pitchers on the active roster)
4. Projected lineup CSV fallback
5. Skip the team if none of the above are available
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
GAME_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
ROSTER_URL = "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
GAMEDAY_PREVIEW_URL = "https://www.mlb.com/gameday/{slug}/{game_pk}/preview"

REQUIRED_COLUMNS = [
    "batter_id",
    "batter_name",
    "batter_hand",
    "pitcher_id",
    "pitcher_name",
    "pitcher_hand",
    "team",
    "opponent",
]

PROJECTED_REQUIRED_COLUMNS = [
    "game_date",
    "team",
    "opponent",
    "batter_id",
    "batter_name",
    "batter_hand",
    "pitcher_id",
    "pitcher_name",
    "pitcher_hand",
]


def get_schedule(target_date: str) -> List[dict]:
    params = {
        "sportId": 1,
        "date": target_date,
        "hydrate": "probablePitcher,team",
    }
    response = requests.get(SCHEDULE_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    dates = payload.get("dates", [])
    if not dates:
        return []
    return dates[0].get("games", [])


def get_game_feed(game_pk: int) -> dict:
    response = requests.get(GAME_FEED_URL.format(game_pk=game_pk), timeout=30)
    response.raise_for_status()
    return response.json()


def get_team_roster(team_id: int, roster_type: str = "active") -> dict:
    params = {"rosterType": roster_type, "hydrate": "person"}
    response = requests.get(ROSTER_URL.format(team_id=team_id), params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_people_lookup(player_ids: List[int]) -> dict:
    if not player_ids:
        return {}
    ids_str = ",".join(str(pid) for pid in sorted(set(player_ids)))
    response = requests.get(PEOPLE_URL, params={"personIds": ids_str}, timeout=30)
    response.raise_for_status()
    data = response.json()
    people = data.get("people", [])
    return {int(p["id"]): p for p in people if p.get("id") is not None}


def safe_bat_side(player_blob: dict) -> Optional[str]:
    code = ((player_blob or {}).get("batSide") or {}).get("code")
    if code in {"L", "R", "S"}:
        return code
    return None


def safe_pitch_hand(player_blob: dict) -> Optional[str]:
    code = ((player_blob or {}).get("pitchHand") or {}).get("code")
    if code in {"L", "R"}:
        return code
    return None


def extract_confirmed_lineups(feed: dict) -> Tuple[List[dict], List[dict]]:
    live_data = feed.get("liveData", {})
    boxscore = live_data.get("boxscore", {})
    teams = boxscore.get("teams", {})
    game_data = feed.get("gameData", {})
    players = game_data.get("players", {})

    def parse_side(side_key: str) -> List[dict]:
        side = teams.get(side_key, {})

        # Prefer the batters list from the boxscore feed; fall back to battingOrder.
        batter_ids = side.get("batters", []) or side.get("battingOrder", [])
        if not batter_ids:
            return []

        lineup = []
        seen = set()
        pending_ids = []

        for raw_pid in batter_ids:
            player_key = f"ID{raw_pid}" if not str(raw_pid).startswith("ID") else str(raw_pid)
            player_blob = players.get(player_key, {})

            player_id = player_blob.get("id")
            full_name = player_blob.get("fullName")
            bat_side = safe_bat_side(player_blob)

            pos_type = ((player_blob.get("primaryPosition") or {}).get("type"))
            if pos_type == "Pitcher":
                continue

            if player_id is None or full_name is None:
                continue

            player_id = int(player_id)
            if player_id in seen:
                continue

            lineup.append(
                {
                    "batter_id": player_id,
                    "batter_name": full_name,
                    "batter_hand": bat_side,
                }
            )
            seen.add(player_id)

            if bat_side not in {"L", "R", "S"}:
                pending_ids.append(player_id)

        if pending_ids:
            people_lookup = get_people_lookup(pending_ids)
            for row in lineup:
                if row["batter_hand"] not in {"L", "R", "S"}:
                    person_blob = people_lookup.get(int(row["batter_id"]), {})
                    row["batter_hand"] = safe_bat_side(person_blob)

        lineup = [r for r in lineup if r["batter_hand"] in {"L", "R", "S"}]
        return lineup[:9]

    return parse_side("away"), parse_side("home")


def extract_roster_hitters(team_id: int) -> List[dict]:
    data = get_team_roster(team_id=team_id, roster_type="active")
    roster = data.get("roster", [])
    temp_rows = []
    pending_ids = []

    for entry in roster:
        position = (entry.get("position") or {}).get("type")
        person = entry.get("person") or {}
        if position == "Pitcher":
            continue
        batter_id = person.get("id")
        batter_name = person.get("fullName")
        batter_hand = safe_bat_side(person)
        temp_rows.append(
            {
                "batter_id": batter_id,
                "batter_name": batter_name,
                "batter_hand": batter_hand,
            }
        )
        if batter_id is not None and batter_hand not in {"L", "R", "S"}:
            pending_ids.append(int(batter_id))

    people_lookup = get_people_lookup(pending_ids)
    hitters = []

    for row in temp_rows:
        batter_id = row["batter_id"]
        batter_name = row["batter_name"]
        batter_hand = row["batter_hand"]
        if batter_id is None or batter_name is None:
            continue
        if batter_hand not in {"L", "R", "S"}:
            person_blob = people_lookup.get(int(batter_id), {})
            batter_hand = safe_bat_side(person_blob)
        if batter_hand not in {"L", "R", "S"}:
            continue
        hitters.append(
            {
                "batter_id": int(batter_id),
                "batter_name": batter_name,
                "batter_hand": batter_hand,
            }
        )

    return hitters


def load_projected_lineups(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=PROJECTED_REQUIRED_COLUMNS)

    missing = [c for c in PROJECTED_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Projected lineup file is missing columns: {missing}. "
            f"Expected columns: {PROJECTED_REQUIRED_COLUMNS}"
        )

    df = df.copy()
    df["game_date"] = df["game_date"].astype(str)
    df["team"] = df["team"].astype(str).str.upper()
    df["opponent"] = df["opponent"].astype(str).str.upper()
    df["batter_hand"] = df["batter_hand"].astype(str).str.upper()
    df["pitcher_hand"] = df["pitcher_hand"].astype(str).str.upper()
    return df


def get_projected_team_lineup(
    projected_df: pd.DataFrame,
    target_date: str,
    team: str,
    opponent: str,
    pitcher_id: Optional[int],
    pitcher_name: Optional[str],
    pitcher_hand: Optional[str],
) -> List[dict]:
    if projected_df.empty:
        return []

    sub = projected_df[
        (projected_df["game_date"] == str(target_date)) &
        (projected_df["team"] == str(team).upper()) &
        (projected_df["opponent"] == str(opponent).upper())
    ].copy()

    if sub.empty:
        return []

    if pitcher_id is not None:
        sub = sub[(sub["pitcher_id"].astype("Int64") == int(pitcher_id)) | (sub["pitcher_id"].isna())]
        if sub.empty:
            return []

    if pitcher_name:
        exact_name = sub[
            sub["pitcher_name"].astype(str).str.strip().str.lower() ==
            str(pitcher_name).strip().lower()
        ]
        if not exact_name.empty:
            sub = exact_name

    if pitcher_hand:
        hand_match = sub[
            sub["pitcher_hand"].astype(str).str.upper() == str(pitcher_hand).upper()
        ]
        if not hand_match.empty:
            sub = hand_match

    sub = sub.copy()
    sub["batter_id"] = sub["batter_id"].astype(int)
    keep_cols = ["batter_id", "batter_name", "batter_hand"]
    sub = sub[keep_cols].drop_duplicates().reset_index(drop=True)
    return sub.to_dict(orient="records")


def build_rows_from_lineup(
    lineup: List[dict],
    pitcher_id: int,
    pitcher_name: str,
    pitcher_hand: str,
    team: str,
    opponent: str,
) -> List[Dict]:
    rows = []
    for hitter in lineup:
        rows.append(
            {
                "batter_id": int(hitter["batter_id"]),
                "batter_name": hitter["batter_name"],
                "batter_hand": hitter["batter_hand"],
                "pitcher_id": int(pitcher_id),
                "pitcher_name": pitcher_name,
                "pitcher_hand": pitcher_hand,
                "team": team,
                "opponent": opponent,
            }
        )
    return rows


def _extract_next_data_json(html: str) -> Optional[dict]:
    patterns = [
        r'<script id="__NEXT_DATA__" type="application/json">\s*(.*?)\s*</script>',
        r'window\.__NEXT_DATA__\s*=\s*({.*?})\s*;</script>',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    return None


def _extract_player_records(obj) -> List[dict]:
    records = []
    if isinstance(obj, dict):
        pid = obj.get("id")
        name = obj.get("fullName") or obj.get("name")
        bat_side = safe_bat_side(obj)
        if pid is not None and name is not None:
            records.append(
                {
                    "batter_id": int(pid),
                    "batter_name": name,
                    "batter_hand": bat_side,
                }
            )
        for value in obj.values():
            records.extend(_extract_player_records(value))
    elif isinstance(obj, list):
        for value in obj:
            records.extend(_extract_player_records(value))
    return records


def extract_mlb_preview_lineups(game: dict, debug: bool = True) -> Tuple[List[dict], List[dict]]:
    game_pk = game.get("gamePk")
    away = game.get("teams", {}).get("away", {}).get("team", {})
    home = game.get("teams", {}).get("home", {}).get("team", {})
    away_abbr = str(away.get("abbreviation", "")).upper()
    home_abbr = str(home.get("abbreviation", "")).upper()
    away_name = str(away.get("clubName") or away.get("teamName") or away.get("name") or "").lower().replace(" ", "-")
    home_name = str(home.get("clubName") or home.get("teamName") or home.get("name") or "").lower().replace(" ", "-")
    slug = f"{away_name}-vs-{home_name}"

    if not game_pk or not away_name or not home_name:
        return [], []

    url = GAMEDAY_PREVIEW_URL.format(slug=slug, game_pk=game_pk)

    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:
        if debug:
            print(f"Game {game_pk}: MLB preview page fetch failed - {exc}")
        return [], []

    next_data = _extract_next_data_json(html)
    if next_data is None:
        if debug:
            print(f"Game {game_pk}: MLB preview page did not expose parsable page data")
        return [], []

    records = _extract_player_records(next_data)
    if not records:
        if debug:
            print(f"Game {game_pk}: MLB preview page returned no candidate hitters")
        return [], []

    pending_ids = [r["batter_id"] for r in records if r["batter_hand"] not in {"L", "R", "S"}]
    people_lookup = get_people_lookup(pending_ids)

    clean_records = []
    seen = set()
    for rec in records:
        pid = int(rec["batter_id"])
        name = rec["batter_name"]
        hand = rec["batter_hand"]
        if hand not in {"L", "R", "S"}:
            hand = safe_bat_side(people_lookup.get(pid, {}))
        if hand not in {"L", "R", "S"}:
            continue
        key = (pid, name, hand)
        if key in seen:
            continue
        seen.add(key)
        clean_records.append({"batter_id": pid, "batter_name": name, "batter_hand": hand})

    away_team_id = away.get("id")
    home_team_id = home.get("id")
    away_roster_ids = {r["batter_id"] for r in extract_roster_hitters(int(away_team_id))} if away_team_id is not None else set()
    home_roster_ids = {r["batter_id"] for r in extract_roster_hitters(int(home_team_id))} if home_team_id is not None else set()

    away_lineup = []
    home_lineup = []
    seen_away = set()
    seen_home = set()

    for r in clean_records:
        pid = r["batter_id"]
        if pid in away_roster_ids and pid not in seen_away:
            away_lineup.append(r)
            seen_away.add(pid)
        if pid in home_roster_ids and pid not in seen_home:
            home_lineup.append(r)
            seen_home.add(pid)

    return away_lineup[:9], home_lineup[:9]


def choose_team_lineup(
    *,
    confirmed_lineup: List[dict],
    mlb_page_lineup: List[dict],
    roster_lineup: List[dict],
    projected_lineup: List[dict],
    team_abbr: str,
    game_pk: int,
    debug: bool = True,
) -> Tuple[List[dict], str]:
    if len(confirmed_lineup) >= 9:
        if debug:
            print(f"Game {game_pk} {team_abbr}: using confirmed live-feed lineup")
        return confirmed_lineup, "confirmed"

    if len(mlb_page_lineup) >= 9:
        if debug:
            print(f"Game {game_pk} {team_abbr}: live-feed lineup unavailable, using MLB preview/page lineup")
        return mlb_page_lineup, "mlb_page"

    if len(roster_lineup) >= 5:
        if debug:
            print(f"Game {game_pk} {team_abbr}: lineup unavailable, using matchup roster fallback")
        return roster_lineup, "roster"

    if len(projected_lineup) >= 9:
        if debug:
            print(f"Game {game_pk} {team_abbr}: lineup unavailable, using projected lineup fallback")
        return projected_lineup, "projected"

    if debug:
        print(f"Game {game_pk} {team_abbr}: skipped - no confirmed lineup, no MLB page lineup, no usable roster fallback, no projected lineup")
    return [], "none"


def game_record_to_rows(game: dict, target_date: str, projected_df: pd.DataFrame, debug: bool = True) -> List[Dict]:
    rows: List[Dict] = []
    game_pk = game.get("gamePk")
    if game_pk is None:
        return rows

    teams = game.get("teams", {})
    away = teams.get("away", {})
    home = teams.get("home", {})

    away_team = away.get("team", {})
    home_team = home.get("team", {})

    away_team_id = away_team.get("id")
    home_team_id = home_team.get("id")
    away_abbr = str(away_team.get("abbreviation", "")).upper()
    home_abbr = str(home_team.get("abbreviation", "")).upper()

    away_pitcher = away.get("probablePitcher") or {}
    home_pitcher = home.get("probablePitcher") or {}

    if not away_pitcher or not home_pitcher:
        if debug:
            print(f"Game {game_pk}: skipped - missing probable pitcher")
        return rows

    away_pitcher_id = away_pitcher.get("id")
    away_pitcher_name = away_pitcher.get("fullName")
    home_pitcher_id = home_pitcher.get("id")
    home_pitcher_name = home_pitcher.get("fullName")

    if not all([away_pitcher_id, away_pitcher_name, home_pitcher_id, home_pitcher_name]):
        if debug:
            print(f"Game {game_pk}: skipped - incomplete probable pitcher data")
        return rows

    feed = get_game_feed(int(game_pk))
    away_confirmed, home_confirmed = extract_confirmed_lineups(feed)

    players_blob = feed.get("gameData", {}).get("players", {})
    away_pitcher_blob = players_blob.get(f"ID{away_pitcher_id}", {})
    home_pitcher_blob = players_blob.get(f"ID{home_pitcher_id}", {})

    away_pitcher_hand = safe_pitch_hand(away_pitcher_blob)
    home_pitcher_hand = safe_pitch_hand(home_pitcher_blob)

    if away_pitcher_hand is None or home_pitcher_hand is None:
        if debug:
            print(f"Game {game_pk}: skipped - could not determine pitcher handedness")
        return rows

    away_mlb_page, home_mlb_page = extract_mlb_preview_lineups(game, debug=debug)

    home_roster = extract_roster_hitters(int(home_team_id)) if home_team_id is not None else []
    home_projected = get_projected_team_lineup(
        projected_df=projected_df,
        target_date=target_date,
        team=home_abbr,
        opponent=away_abbr,
        pitcher_id=int(away_pitcher_id),
        pitcher_name=away_pitcher_name,
        pitcher_hand=away_pitcher_hand,
    )
    home_lineup, _ = choose_team_lineup(
        confirmed_lineup=home_confirmed,
        mlb_page_lineup=home_mlb_page,
        roster_lineup=home_roster,
        projected_lineup=home_projected,
        team_abbr=home_abbr,
        game_pk=int(game_pk),
        debug=debug,
    )
    if home_lineup:
        rows.extend(
            build_rows_from_lineup(
                lineup=home_lineup,
                pitcher_id=int(away_pitcher_id),
                pitcher_name=away_pitcher_name,
                pitcher_hand=away_pitcher_hand,
                team=home_abbr,
                opponent=away_abbr,
            )
        )

    away_roster = extract_roster_hitters(int(away_team_id)) if away_team_id is not None else []
    away_projected = get_projected_team_lineup(
        projected_df=projected_df,
        target_date=target_date,
        team=away_abbr,
        opponent=home_abbr,
        pitcher_id=int(home_pitcher_id),
        pitcher_name=home_pitcher_name,
        pitcher_hand=home_pitcher_hand,
    )
    away_lineup, _ = choose_team_lineup(
        confirmed_lineup=away_confirmed,
        mlb_page_lineup=away_mlb_page,
        roster_lineup=away_roster,
        projected_lineup=away_projected,
        team_abbr=away_abbr,
        game_pk=int(game_pk),
        debug=debug,
    )
    if away_lineup:
        rows.extend(
            build_rows_from_lineup(
                lineup=away_lineup,
                pitcher_id=int(home_pitcher_id),
                pitcher_name=home_pitcher_name,
                pitcher_hand=home_pitcher_hand,
                team=away_abbr,
                opponent=home_abbr,
            )
        )

    return rows


def build_daily_lineups(target_date: str, projected_path: str = "projected_lineups.csv", debug: bool = True) -> pd.DataFrame:
    games = get_schedule(target_date)
    projected_df = load_projected_lineups(projected_path)
    all_rows: List[Dict] = []

    if debug:
        print(f"Found {len(games)} scheduled games for {target_date}")

    for game in games:
        try:
            all_rows.extend(game_record_to_rows(game, target_date, projected_df, debug=debug))
        except Exception as exc:
            game_pk = game.get("gamePk", "unknown")
            print(f"Game {game_pk}: skipped - {exc}")

    frame = pd.DataFrame(all_rows, columns=REQUIRED_COLUMNS)
    if not frame.empty:
        frame = frame.drop_duplicates().sort_values(by=["team", "batter_name", "pitcher_name"]).reset_index(drop=True)
    return frame


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build daily_lineups.csv using live feed, MLB preview/page lineups, roster fallback, and projected fallback."
    )
    parser.add_argument("--date", default=date.today().isoformat(), help="Target date in YYYY-MM-DD format.")
    parser.add_argument("--output", default="daily_lineups.csv", help="Output CSV filename.")
    parser.add_argument("--projected", default="projected_lineups.csv", help="Projected lineup fallback CSV filename.")
    parser.add_argument("--quiet", action="store_true", help="Suppress debug output.")
    return parser.parse_known_args(args=args)[0]


def run_builder(
    target_date: Optional[str] = None,
    output: str = "daily_lineups.csv",
    projected: str = "projected_lineups.csv",
    show_head: bool = True,
    debug: bool = True,
) -> pd.DataFrame:
    if target_date is None:
        target_date = date.today().isoformat()

    df = build_daily_lineups(target_date=target_date, projected_path=projected, debug=debug)

    if df.empty:
        print("No lineup rows were created.")
        print("Confirmed lineups, MLB page lineups, roster fallback, and projected fallbacks all failed.")
        pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Created empty template: {output}")
        return df

    df.to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Saved {len(df)} rows to {output}")
    if show_head:
        print(df.head(20).to_string(index=False))
    return df


def main(args=None) -> None:
    parsed = parse_args(args=args)
    run_builder(
        target_date=parsed.date,
        output=parsed.output,
        projected=parsed.projected,
        show_head=True,
        debug=not parsed.quiet,
    )


if __name__ == "__main__":
    main()
