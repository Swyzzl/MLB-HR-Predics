
#!/usr/bin/env python3
"""
daily_hr_probability_model_3ab.py

Daily HR probability model using Statcast data and a generated daily_lineups.csv file.

This version:
- keeps the batter/pitcher pitch-type matchup model
- uses weighted components of:
      50% HR/BBE
      30% bat speed
      20% launch angle / exit velocity
- calibrates to a PER-AT-BAT HR probability
- converts that to a 3-AB HR probability:
      P(at least 1 HR in 3 AB) = 1 - (1 - p_ab)^3

Expected lineup input columns
-----------------------------
batter_id,batter_name,batter_hand,pitcher_id,pitcher_name,pitcher_hand,team,opponent

Typical workflow
----------------
1. Build the matchup file:
   from build_daily_lineups_matchups_fixed import run_builder
   run_builder(target_date="2026-04-02", output="daily_lineups.csv")

2. Score the hitters:
   from daily_hr_probability_model_3ab import run_model
   board = run_model(lineups_path="daily_lineups.csv", output_path="hr_predictions.csv")
"""

from __future__ import annotations

import argparse
import math
import os
import re
import unicodedata
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pybaseball import statcast_batter, statcast_pitcher

# -----------------------------
# CONFIG
# -----------------------------
SEASONS = [2025, 2026]
SEASON_WEIGHTS = {2025: 0.40, 2026: 0.60}
CACHE_DIR = "statcast_cache"

# Model weights
WEIGHT_HR_BBE = 0.55       # HR/BBE component
WEIGHT_BAT_SPEED = 0.20    # Bat speed component
WEIGHT_EV_LA = 0.25        # EV + LA component

# Smoothing priors
BATTER_PRIOR_RATE = 0.045
PITCHER_PRIOR_RATE = 0.045
PRIOR_BBE = 35.0

# Probability calibration
BASELINE_SCORE = 0.055      # approximate midpoint raw matchup score
BASELINE_PA_HR = 0.030      # baseline per-at-bat HR probability
SCORE_EXPONENT = 0.85       # spread control, lower compresses outputs
DEFAULT_ABS = 3             # convert to probability over this many at-bats

# Optional filters
MIN_USAGE_TO_KEEP = 0.12

# Optional daily Ballpark Pal Excel integration
DEFAULT_PARK_FACTORS_FILE = "Todays MLB Park Factors  Ballpark Pal.xlsx"
USE_PARK_FACTORS = True
MAX_PA_HR_PROB = 0.35
PARK_FACTOR_FALLBACK = 1.00

# -----------------------------
# Pitch normalization
# -----------------------------
PITCH_MAP = {
    "FF": "4-Seam",
    "FA": "Fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "SV": "Slurve",
    "CH": "Changeup",
    "FS": "Splitter",
    "FO": "Forkball",
    "SC": "Screwball",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "KN": "Knuckleball",
    "CS": "Slow Curve",
    "EP": "Eephus",
}

REQUIRED_LINEUP_COLS = [
    "batter_id",
    "batter_name",
    "batter_hand",
    "pitcher_id",
    "pitcher_name",
    "pitcher_hand",
    "team",
    "opponent",
]


# -----------------------------
# Helpers
# -----------------------------
def ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def season_date_range(season: int) -> tuple[str, str]:
    if season == 2026:
        return f"{season}-03-01", date.today().isoformat()
    return f"{season}-03-01", f"{season}-11-15"


def cache_path(role: str, player_id: int, season: int) -> str:
    return os.path.join(CACHE_DIR, f"{role}_{player_id}_{season}.csv")


def normalize_pitch_type(series: pd.Series) -> pd.Series:
    return series.map(PITCH_MAP).fillna(series)


def smooth_rate(hr: float, bbe: float, prior_rate: float, prior_bbe: float = PRIOR_BBE) -> float:
    return (float(hr) + prior_rate * prior_bbe) / (float(bbe) + prior_bbe)


def contact_score(ev: float, la: float) -> float:
    if pd.isna(ev) or pd.isna(la):
        return 0.0
    ev_score = np.clip((float(ev) - 85.0) / 25.0, 0.0, 1.0)
    la_score = math.exp(-((float(la) - 28.0) ** 2) / (2 * (10.0 ** 2)))
    return float(ev_score * la_score)


def bat_speed_score(bat_speed: float) -> float:
    if pd.isna(bat_speed):
        return 0.5
    return float(np.clip(((float(bat_speed) - 72.5) / 7.5) * 0.5 + 0.5, 0.0, 1.0))
    return float(np.clip((float(bat_speed) - 65.0) / 15.0, 0.0, 1.0))


def calibrated_pa_hr_probability(
    raw_score: float,
    baseline_score: float = BASELINE_SCORE,
    baseline_pa_hr: float = BASELINE_PA_HR,
    exponent: float = SCORE_EXPONENT,
) -> float:
    if baseline_score <= 0:
        raise ValueError("baseline_score must be > 0")

    relative_strength = max(float(raw_score) / baseline_score, 0.01)
    return baseline_pa_hr * (relative_strength ** exponent)


def hr_probability_over_n_abs(pa_prob: float, n_abs: int = DEFAULT_ABS) -> float:
    pa_prob = max(float(pa_prob), 0.0)
    return 1.0 - ((1.0 - pa_prob) ** int(n_abs))



def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = text.replace(".", "").replace("'", "")
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_team_abbr(value: str) -> str:
    raw = str(value or "").upper().strip()
    aliases = {
        "KCR": "KC",
        "KAN": "KC",
        "WSH": "WSN",
        "WAS": "WSN",
        "NYY": "NYY",
        "NYM": "NYM",
        "SFG": "SF",
        "SDP": "SD",
        "TBR": "TB",
        "CHW": "CWS",
        "CWS": "CWS",
        "AZ": "ARI",
        "ATH": "ATH",
    }
    return aliases.get(raw, raw)


def clamp_probability(value: float, lower: float = 0.0, upper: float = MAX_PA_HR_PROB) -> float:
    return float(min(max(float(value), lower), upper))


def load_park_factors_excel(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["team", "player", "park", "hr_factor"])

    try:
        raw = pd.read_excel(path, engine="openpyxl", header=1)
    except Exception as exc:
        print(f"Warning: could not read park factors file {path}: {exc}")
        return pd.DataFrame(columns=["team", "player", "park", "hr_factor"])

    raw.columns = [str(c).strip() for c in raw.columns]
    col_map = {}
    for col in raw.columns:
        low = col.lower()
        if low == "tm":
            col_map[col] = "team"
        elif low == "player":
            col_map[col] = "player"
        elif low == "park":
            col_map[col] = "park"
        elif low == "hr":
            col_map[col] = "hr_factor"

    required = {"team", "player", "park", "hr_factor"}
    if not required.issubset(set(col_map.values())):
        print(f"Warning: park factors file {path} is missing one of the required columns: Tm, Player, Park, HR")
        return pd.DataFrame(columns=["team", "player", "park", "hr_factor"])

    df = raw.rename(columns=col_map)[["team", "player", "park", "hr_factor"]].copy()
    df = df.dropna(subset=["team", "player", "hr_factor"]).reset_index(drop=True)

    df["team"] = df["team"].astype(str).map(normalize_team_abbr)
    df["player"] = df["player"].astype(str).map(normalize_name)
    df["park"] = df["park"].astype(str).str.strip()

    df["hr_factor"] = pd.to_numeric(df["hr_factor"], errors="coerce")
    df = df.dropna(subset=["hr_factor"]).reset_index(drop=True)

    # Keep the first occurrence per team/player in case the file has duplicates.
    df = df.drop_duplicates(subset=["team", "player"], keep="first").reset_index(drop=True)
    return df


def get_player_park_factor(
    batter_name: str,
    team: str,
    park_factors_df: Optional[pd.DataFrame],
) -> float:
    if park_factors_df is None or park_factors_df.empty:
        return PARK_FACTOR_FALLBACK

    team_key = normalize_team_abbr(team)
    player_key = normalize_name(batter_name)

    exact = park_factors_df[
        (park_factors_df["team"] == team_key) &
        (park_factors_df["player"] == player_key)
    ]
    if not exact.empty:
        return float(exact.iloc[0]["hr_factor"])

    # Fallback to team average for the sheet if a player name mismatch occurs.
    team_rows = park_factors_df[park_factors_df["team"] == team_key]
    if not team_rows.empty:
        return float(team_rows["hr_factor"].mean())

    return PARK_FACTOR_FALLBACK


def apply_park_factor(pa_prob: float, park_factor: float) -> float:
    if pd.isna(park_factor):
        return clamp_probability(pa_prob, upper=MAX_PA_HR_PROB)
    return clamp_probability(float(pa_prob) * float(park_factor), upper=MAX_PA_HR_PROB)


# -----------------------------
# Data pulling / caching
# -----------------------------
def load_or_fetch_batter(player_id: int, season: int, refresh: bool = False) -> pd.DataFrame:
    ensure_cache_dir()
    path = cache_path("batter", player_id, season)

    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)

    start_dt, end_dt = season_date_range(season)
    df = statcast_batter(start_dt, end_dt, player_id)
    if df is None:
        df = pd.DataFrame()
    df.to_csv(path, index=False)
    return df


def load_or_fetch_pitcher(player_id: int, season: int, refresh: bool = False) -> pd.DataFrame:
    ensure_cache_dir()
    path = cache_path("pitcher", player_id, season)

    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)

    start_dt, end_dt = season_date_range(season)
    df = statcast_pitcher(start_dt, end_dt, player_id)
    if df is None:
        df = pd.DataFrame()
    df.to_csv(path, index=False)
    return df


def prep_statcast_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if "pitch_type" not in out.columns:
        return pd.DataFrame()

    out["pitch_group"] = normalize_pitch_type(out["pitch_type"])
    out["is_hr"] = (out.get("events") == "home_run").astype(int)
    out["is_bbe"] = (out.get("type") == "X") & out.get("launch_speed").notna()

    # Newer Statcast exports may include bat_speed; preserve it when present.
    if "bat_speed" not in out.columns:
        out["bat_speed"] = np.nan

    return out


# -----------------------------
# Aggregation
# -----------------------------
def aggregate_batter_split(df: pd.DataFrame, pitcher_hand: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["pitch_type", "HR", "BBE", "EV", "LA", "BAT_SPEED"])

    sub = df[(df["p_throws"] == pitcher_hand) & (df["is_bbe"])].copy()
    if sub.empty:
        return pd.DataFrame(columns=["pitch_type", "HR", "BBE", "EV", "LA", "BAT_SPEED"])

    grouped = (
        sub.groupby("pitch_group", dropna=False)
        .agg(
            HR=("is_hr", "sum"),
            BBE=("is_hr", "count"),
            EV=("launch_speed", "mean"),
            LA=("launch_angle", "mean"),
            BAT_SPEED=("bat_speed", "mean"),
        )
        .reset_index()
        .rename(columns={"pitch_group": "pitch_type"})
    )
    return grouped


def aggregate_pitcher_split(df: pd.DataFrame, batter_hand: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["pitch_type", "HR_allowed", "BBE_allowed", "EV_allowed", "LA_allowed", "usage"])

    all_pitches = df[df["stand"] == batter_hand].copy()
    if all_pitches.empty:
        return pd.DataFrame(columns=["pitch_type", "HR_allowed", "BBE_allowed", "EV_allowed", "LA_allowed", "usage"])

    bbe = all_pitches[all_pitches["is_bbe"]].copy()

    perf = (
        bbe.groupby("pitch_group", dropna=False)
        .agg(
            HR_allowed=("is_hr", "sum"),
            BBE_allowed=("is_hr", "count"),
            EV_allowed=("launch_speed", "mean"),
            LA_allowed=("launch_angle", "mean"),
        )
        .reset_index()
        .rename(columns={"pitch_group": "pitch_type"})
    )

    usage = (
        all_pitches.groupby("pitch_group", dropna=False)
        .size()
        .reset_index(name="pitch_count")
        .rename(columns={"pitch_group": "pitch_type"})
    )
    usage["usage"] = usage["pitch_count"] / usage["pitch_count"].sum()

    merged = perf.merge(usage[["pitch_type", "usage"]], on="pitch_type", how="outer")
    return merged


def combine_weighted_seasons(season_frames: Dict[int, pd.DataFrame], value_cols: List[str]) -> pd.DataFrame:
    pitch_types = sorted(
        {
            p
            for df in season_frames.values()
            if not df.empty
            for p in df["pitch_type"].dropna().unique().tolist()
        }
    )

    rows = []
    for pitch in pitch_types:
        row = {"pitch_type": pitch}
        for col in value_cols:
            vals = []
            weights = []
            for season, sdf in season_frames.items():
                match = sdf[sdf["pitch_type"] == pitch]
                if match.empty:
                    continue
                value = match.iloc[0].get(col, np.nan)
                if pd.isna(value):
                    continue
                vals.append(float(value))
                weights.append(float(SEASON_WEIGHTS.get(season, 1.0)))
            row[col] = float(np.average(vals, weights=weights)) if vals else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def build_batter_profile(batter_id: int, pitcher_hand: str, refresh_cache: bool = False) -> pd.DataFrame:
    season_frames = {}
    for season in SEASONS:
        raw = load_or_fetch_batter(batter_id, season, refresh=refresh_cache)
        prepped = prep_statcast_frame(raw)
        season_frames[season] = aggregate_batter_split(prepped, pitcher_hand)
    return combine_weighted_seasons(season_frames, ["HR", "BBE", "EV", "LA", "BAT_SPEED"])


def build_pitcher_profile(pitcher_id: int, batter_hand: str, refresh_cache: bool = False) -> pd.DataFrame:
    season_frames = {}
    for season in SEASONS:
        raw = load_or_fetch_pitcher(pitcher_id, season, refresh=refresh_cache)
        prepped = prep_statcast_frame(raw)
        season_frames[season] = aggregate_pitcher_split(prepped, batter_hand)
    return combine_weighted_seasons(season_frames, ["HR_allowed", "BBE_allowed", "EV_allowed", "LA_allowed", "usage"])


# -----------------------------
# Model
# -----------------------------
def build_matchup_breakdown(batter_profile: pd.DataFrame, pitcher_profile: pd.DataFrame) -> pd.DataFrame:
    if batter_profile.empty or pitcher_profile.empty:
        return pd.DataFrame()

    df = batter_profile.merge(pitcher_profile, on="pitch_type", how="inner")
    if df.empty:
        return df

    for col in ["HR", "BBE", "HR_allowed", "BBE_allowed"]:
        df[col] = df[col].fillna(0.0)

    df["usage"] = df["usage"].fillna(0.0)
    filtered = df[df["usage"] >= MIN_USAGE_TO_KEEP].copy()

    # Fallback: if no pitch types clear the threshold, keep the top 2 pitches by usage
    if filtered.empty:
        df = df.sort_values("usage", ascending=False).head(2).copy()
    else:
        df = filtered

    if df.empty:
        return df

    usage_sum = df["usage"].sum()
    if usage_sum <= 0:
        return pd.DataFrame()
    df["usage"] = df["usage"] / usage_sum

    df["b_rate"] = df.apply(lambda r: smooth_rate(r["HR"], r["BBE"], BATTER_PRIOR_RATE), axis=1)
    df["p_rate"] = df.apply(lambda r: smooth_rate(r["HR_allowed"], r["BBE_allowed"], PITCHER_PRIOR_RATE), axis=1)
    df["outcome_component"] = (df["b_rate"] * df["p_rate"]) ** 0.75

    df["b_contact"] = df.apply(lambda r: contact_score(r["EV"], r["LA"]), axis=1)
    df["p_contact"] = df.apply(lambda r: contact_score(r["EV_allowed"], r["LA_allowed"]), axis=1)
    df["ev_la_component"] = np.sqrt(df["b_contact"] * df["p_contact"])

    # Bat speed is treated as a batter-side power input.
    df["bat_speed_component"] = df["BAT_SPEED"].apply(bat_speed_score)

    df["pitch_score"] = (
        WEIGHT_HR_BBE * df["outcome_component"]
        + WEIGHT_BAT_SPEED * df["bat_speed_component"]
        + WEIGHT_EV_LA * df["ev_la_component"]
    )
    df["weighted_pitch_score"] = df["usage"] * df["pitch_score"]

    return df.sort_values("usage", ascending=False).reset_index(drop=True)


def score_matchup(
    batter_id: int,
    batter_name: str,
    batter_hand: str,
    pitcher_id: int,
    pitcher_name: str,
    pitcher_hand: str,
    team: str,
    opponent: str,
    park_factors_df: Optional[pd.DataFrame] = None,
    refresh_cache: bool = False,
    n_abs: int = DEFAULT_ABS,
) -> Optional[dict]:
    batter_profile = build_batter_profile(batter_id, pitcher_hand, refresh_cache=refresh_cache)
    pitcher_profile = build_pitcher_profile(pitcher_id, batter_hand, refresh_cache=refresh_cache)

    if batter_profile.empty or pitcher_profile.empty:
        return None

    breakdown = build_matchup_breakdown(batter_profile, pitcher_profile)
    if breakdown.empty:
        return None

    raw_score = float(breakdown["weighted_pitch_score"].sum())
    pa_hr_probability_base = calibrated_pa_hr_probability(raw_score)

    park_factor_hr = get_player_park_factor(
        batter_name=batter_name,
        team=team,
        park_factors_df=park_factors_df,
    )
    pa_hr_probability = apply_park_factor(pa_hr_probability_base, park_factor_hr)
    hr_probability_nab = hr_probability_over_n_abs(pa_hr_probability, n_abs=n_abs)

    best_pitch = breakdown.iloc[0]["pitch_type"] if len(breakdown) > 0 else None
    top_usage = float(breakdown.iloc[0]["usage"]) if len(breakdown) > 0 else np.nan

    return {
        "date": date.today().isoformat(),
        "batter": batter_name,
        "batter_id": int(batter_id),
        "batter_hand": batter_hand,
        "team": team,
        "opponent": opponent,
        "pitcher": pitcher_name,
        "pitcher_id": int(pitcher_id),
        "pitcher_hand": pitcher_hand,
        "raw_matchup_score": raw_score,
        "hr_probability_pa_base": pa_hr_probability_base,
        "park_factor_hr": park_factor_hr,
        "hr_probability_pa": pa_hr_probability,
        f"hr_probability_{n_abs}ab": hr_probability_nab,
        "top_pitch_in_mix": best_pitch,
        "dominant_pitch_usage": top_usage,
        "pitch_types_used": int(len(breakdown)),
        "weighted_outcome_component": float((breakdown["usage"] * breakdown["outcome_component"]).sum()),
        "weighted_bat_speed_component": float((breakdown["usage"] * breakdown["bat_speed_component"]).sum()),
        "weighted_ev_la_component": float((breakdown["usage"] * breakdown["ev_la_component"]).sum()),
    }


# -----------------------------
# Runner
# -----------------------------
def load_lineups(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_LINEUP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing lineup columns: {missing}")

    out = df.copy()
    out["batter_id"] = out["batter_id"].astype(int)
    out["pitcher_id"] = out["pitcher_id"].astype(int)
    out["batter_hand"] = out["batter_hand"].astype(str).str.upper()
    out["pitcher_hand"] = out["pitcher_hand"].astype(str).str.upper()
    out["team"] = out["team"].astype(str).str.upper()
    out["opponent"] = out["opponent"].astype(str).str.upper()
    out = out.drop_duplicates().reset_index(drop=True)
    return out


def run_model(
    lineups_path: str = "daily_lineups.csv",
    output_path: Optional[str] = None,
    park_factors_path: str = DEFAULT_PARK_FACTORS_FILE,
    refresh_cache: bool = False,
    show_head: bool = True,
    n_abs: int = DEFAULT_ABS,
) -> pd.DataFrame:
    lineups = load_lineups(lineups_path)
    results = []

    if USE_PARK_FACTORS and park_factors_path:
        park_factors_df = load_park_factors_excel(park_factors_path)
        if not park_factors_df.empty:
            print(f"Loaded {len(park_factors_df)} park-factor rows from {park_factors_path}")
        else:
            print(f"No usable park factors loaded from {park_factors_path}; continuing without park adjustment.")
    else:
        park_factors_df = pd.DataFrame(columns=["team", "player", "park", "hr_factor"])

    score_cols = [
        "batter_id", "batter_name", "batter_hand",
        "pitcher_id", "pitcher_name", "pitcher_hand",
        "team", "opponent"
    ]
    lineups = lineups[score_cols].drop_duplicates().reset_index(drop=True)

    for _, row in lineups.iterrows():
        try:
            scored = score_matchup(
                batter_id=int(row["batter_id"]),
                batter_name=row["batter_name"],
                batter_hand=row["batter_hand"],
                pitcher_id=int(row["pitcher_id"]),
                pitcher_name=row["pitcher_name"],
                pitcher_hand=row["pitcher_hand"],
                team=row["team"],
                opponent=row["opponent"],
                park_factors_df=park_factors_df,
                refresh_cache=refresh_cache,
                n_abs=n_abs,
            )
            if scored is not None:
                results.append(scored)
        except Exception as exc:
            print(f"Error scoring {row['batter_name']} vs {row['pitcher_name']}: {exc}")

    board = pd.DataFrame(results)
    if board.empty:
        print("No matchup probabilities were generated.")
        return board

    sort_col = f"hr_probability_{n_abs}ab"
    board = board.sort_values(sort_col, ascending=False).reset_index(drop=True)

    if output_path is None:
        output_path = f"hr_predictions_{date.today().isoformat()}.csv"

    board.to_csv(output_path, index=False)
    print(f"Saved {len(board)} rows to {output_path}")

    if show_head:
        cols = ["batter", "team", "pitcher", "opponent", "hr_probability_pa", sort_col, "top_pitch_in_mix"]
        print(board[cols].head(20).to_string(index=False))

    return board


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily HR probability model from a daily_lineups.csv file.")
    parser.add_argument("--lineups", default="daily_lineups.csv", help="Input lineup CSV. Default: daily_lineups.csv")
    parser.add_argument("--output", default=None, help="Output predictions CSV. Default: hr_predictions_<today>.csv")
    parser.add_argument("--park-factors", default=DEFAULT_PARK_FACTORS_FILE, help=f"Path to local Ballpark Pal Excel file. Default: {DEFAULT_PARK_FACTORS_FILE}")
    parser.add_argument("--refresh-cache", action="store_true", help="Re-pull Statcast data even if cache exists.")
    parser.add_argument("--quiet", action="store_true", help="Do not print the top rows.")
    parser.add_argument("--at-bats", type=int, default=DEFAULT_ABS, help=f"Convert probability over N at-bats. Default: {DEFAULT_ABS}")
    return parser.parse_known_args(args=args)[0]


def main(args=None) -> None:
    parsed = parse_args(args=args)
    run_model(
        lineups_path=parsed.lineups,
        output_path=parsed.output,
        park_factors_path=parsed.park_factors,
        refresh_cache=parsed.refresh_cache,
        show_head=not parsed.quiet,
        n_abs=parsed.at_bats,
    )


if __name__ == "__main__":
    main()
