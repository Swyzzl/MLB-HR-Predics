#!/usr/bin/env python3
"""
daily_hr_probability_model_3ab_homepark_pullair.py

Daily HR probability model with:
- pitch-type batter/pitcher matchup scoring
- park-level HR multiplier resolved from the actual game venue / home team
- secondary batter-side power/context adjustments for Barrel% and Pull Air%
- pitcher-side adjustments for Barrel% allowed and Zone%

Expected lineup input columns
-----------------------------
Required:
    batter_id,batter_name,batter_hand,pitcher_id,pitcher_name,pitcher_hand,team,opponent

Recommended park-context columns from the revised lineup builder:
    game_date,game_pk,home_team,away_team,park_team,venue,stadium,is_home

Park-factor application logic
-----------------------------
The park factor is applied only through the actual game park context.
Priority:
    1. park_team
    2. home_team
    3. stadium
    4. venue
If none are present, the model falls back to a neutral 1.00 multiplier.

Pull Air% definition
--------------------
Pull Air% is calculated batter-side as:
    pulled air balls / all air balls
where air balls are bb_type in {fly_ball, line_drive, popup}
and "pulled" is determined from batter handedness and hit direction.
This is used as a secondary power signal, similar to Barrel%, but smaller.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import unicodedata
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pybaseball import statcast_batter, statcast_pitcher

# -----------------------------
# CONFIG
# -----------------------------
SEASONS = [2025, 2026]
SEASON_WEIGHTS = {2025: 0.80, 2026: 0.20}
CACHE_DIR = "statcast_cache"

# Core matchup weights
WEIGHT_HR_BBE = 0.55
WEIGHT_BAT_SPEED = 0.20
WEIGHT_EV_LA = 0.25

# Smoothing priors
BATTER_PRIOR_RATE = 0.045
PITCHER_PRIOR_RATE = 0.045
PRIOR_BBE = 35.0

# Probability calibration
BASELINE_SCORE = 0.055
BASELINE_PA_HR = 0.030
SCORE_EXPONENT = 0.85
DEFAULT_ABS = 3
MAX_PA_HR_PROB = 0.35

# Secondary adjustments
LEAGUE_AVG_BARREL_PCT = 10.0
LEAGUE_AVG_ZONE_PCT = 49.0
LEAGUE_AVG_PULL_AIR_PCT = 25.0

BARREL_PCT_SCALE = 0.015
ZONE_PCT_SCALE = 0.010
PULL_AIR_PCT_SCALE = 0.006

MAX_BATTER_BARREL = 18.0
MAX_PITCHER_BARREL = 12.0
MAX_BATTER_PULL_AIR = 45.0

MAX_BARREL_ADJ = 0.12
MAX_ZONE_ADJ = 0.08
MAX_PULL_AIR_ADJ = 0.10

# Pitch-mix filter
MIN_USAGE_TO_KEEP = 0.12

# Park factors
DEFAULT_PARK_FACTORS_FILE = "Todays_MLB_Park_Factors_ParkOnly.xlsx"
PARK_FACTORS_SHEET_NAME = "Park_HR_Factors"
USE_PARK_FACTORS = True
PARK_FACTOR_FALLBACK = 1.00

# Useful for rough field-direction classification when spray angle is unavailable.
HC_X_CENTER = 125.0

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

OPTIONAL_LINEUP_COLS = [
    "game_date",
    "game_pk",
    "home_team",
    "away_team",
    "park_team",
    "park",
    "venue",
    "stadium",
    "is_home",
]


# -----------------------------
# Helpers
# -----------------------------
def ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)



def season_date_range(season: int) -> Tuple[str, str]:
    if season == 2026:
        return f"{season}-03-01", date.today().isoformat()
    return f"{season}-03-01", f"{season}-11-15"



def cache_path(role: str, player_id: int, season: int) -> str:
    return os.path.join(CACHE_DIR, f"{role}_{player_id}_{season}.csv")



def normalize_pitch_type(series: pd.Series) -> pd.Series:
    mapped = series.map(PITCH_MAP)
    return mapped.where(mapped.notna(), series)



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



def normalize_stadium_name(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text



def normalize_team_abbr(value: str) -> str:
    raw = str(value or "").upper().strip()
    aliases = {
        "KCR": "KC",
        "KAN": "KC",
        "WSH": "WSN",
        "WAS": "WSN",
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



def resolve_effective_batter_hand(batter_hand: str, pitcher_hand: str) -> str:
    """
    Resolve the batter side to use for today's matchup.

    - L stays L
    - R stays R
    - S becomes the opposite of the opposing pitcher's handedness
    """
    batter = str(batter_hand or "").upper().strip()
    pitcher = str(pitcher_hand or "").upper().strip()

    if batter in {"L", "R"}:
        return batter

    if batter == "S":
        if pitcher == "R":
            return "L"
        if pitcher == "L":
            return "R"

    return batter


def clamp(value: float, low: float, high: float) -> float:
    if pd.isna(value):
        return np.nan
    return float(min(max(float(value), low), high))



def metric_adjustment(value: float, baseline: float, scale: float, cap: float) -> float:
    if pd.isna(value):
        return 0.0
    diff = float(value) - float(baseline)
    return float(min(max(diff * scale, -cap), cap))



def is_air_ball(bb_type: object) -> bool:
    return str(bb_type or "").strip().lower() in {"fly_ball", "line_drive", "popup"}



def is_pulled_ball(stand: object, hc_x: object) -> Optional[int]:
    """
    Rough pull-side classifier using Statcast hit coordinate x-position.
    For RHB, lower hc_x is more toward LF/pull side.
    For LHB, higher hc_x is more toward RF/pull side.
    Returns 1 pulled, 0 not pulled, NaN-like None if unavailable.
    """
    if pd.isna(hc_x) or pd.isna(stand):
        return None
    hand = str(stand).upper().strip()
    try:
        x = float(hc_x)
    except Exception:
        return None

    if hand == "R":
        return int(x < HC_X_CENTER)
    if hand == "L":
        return int(x > HC_X_CENTER)
    return None


# -----------------------------
# Park factors
# -----------------------------
def load_park_factors_excel(path: str, sheet_name: str = PARK_FACTORS_SHEET_NAME) -> pd.DataFrame:
    cols = ["home_team", "model_team_key", "stadium", "stadium_key", "hr_percent_effect", "hr_multiplier"]
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=cols)

    try:
        raw = pd.read_excel(path, engine="openpyxl", sheet_name=sheet_name)
    except Exception as exc:
        print(f"Warning: could not read park factors workbook {path}: {exc}")
        return pd.DataFrame(columns=cols)

    raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]
    required = {"home_team", "model_team_key", "stadium", "hr_percent_effect", "hr_multiplier"}
    if not required.issubset(set(raw.columns)):
        print(f"Warning: park factors sheet {sheet_name} is missing required columns.")
        return pd.DataFrame(columns=cols)

    df = raw[["home_team", "model_team_key", "stadium", "hr_percent_effect", "hr_multiplier"]].copy()
    df["home_team"] = df["home_team"].astype(str).map(normalize_team_abbr)
    df["model_team_key"] = df["model_team_key"].astype(str).map(normalize_team_abbr)
    df["stadium"] = df["stadium"].astype(str).str.strip()
    df["stadium_key"] = df["stadium"].map(normalize_stadium_name)
    df["hr_percent_effect"] = pd.to_numeric(df["hr_percent_effect"], errors="coerce")
    df["hr_multiplier"] = pd.to_numeric(df["hr_multiplier"], errors="coerce")

    return (
        df.dropna(subset=["model_team_key", "stadium_key", "hr_multiplier"])
        .drop_duplicates(subset=["model_team_key"], keep="first")
        .reset_index(drop=True)
    )



def resolve_park_factor(
    home_team: Optional[str],
    park_team: Optional[str],
    stadium: Optional[str],
    venue: Optional[str],
    park: Optional[str],
    park_factors_df: Optional[pd.DataFrame],
) -> Tuple[float, str, Optional[str], Optional[str], Optional[float]]:
    if park_factors_df is None or park_factors_df.empty:
        return PARK_FACTOR_FALLBACK, "fallback_1.00_no_workbook", None, None, None

    for candidate, source_name in [(park_team, "park_team_lookup"), (home_team, "home_team_lookup")]:
        if candidate and str(candidate).strip():
            team_key = normalize_team_abbr(candidate)
            match = park_factors_df[park_factors_df["model_team_key"] == team_key]
            if not match.empty:
                row = match.iloc[0]
                return (
                    float(row["hr_multiplier"]),
                    source_name,
                    str(row["model_team_key"]),
                    str(row["stadium"]),
                    float(row["hr_percent_effect"]) if pd.notna(row["hr_percent_effect"]) else np.nan,
                )

    for candidate, source_name in [(stadium, "stadium_lookup"), (venue, "venue_lookup"), (park, "park_lookup")]:
        if candidate and str(candidate).strip():
            stadium_key = normalize_stadium_name(candidate)
            match = park_factors_df[park_factors_df["stadium_key"] == stadium_key]
            if not match.empty:
                row = match.iloc[0]
                return (
                    float(row["hr_multiplier"]),
                    source_name,
                    str(row["model_team_key"]),
                    str(row["stadium"]),
                    float(row["hr_percent_effect"]) if pd.notna(row["hr_percent_effect"]) else np.nan,
                )

    return PARK_FACTOR_FALLBACK, "fallback_1.00_missing_home_park_context", None, None, None



def apply_park_factor(pa_prob: float, park_factor: float) -> float:
    if pd.isna(park_factor):
        return clamp_probability(pa_prob)
    return clamp_probability(float(pa_prob) * float(park_factor))


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

    # Defensive population of columns that are not always present.
    defaults = {
        "events": np.nan,
        "type": np.nan,
        "launch_speed": np.nan,
        "launch_angle": np.nan,
        "launch_speed_angle": np.nan,
        "zone": np.nan,
        "p_throws": np.nan,
        "stand": np.nan,
        "bat_speed": np.nan,
        "bb_type": np.nan,
        "hc_x": np.nan,
        "hc_y": np.nan,
    }
    for col, default_value in defaults.items():
        if col not in out.columns:
            out[col] = default_value

    numeric_cols = ["launch_speed", "launch_angle", "launch_speed_angle", "zone", "bat_speed", "hc_x", "hc_y"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["is_hr"] = (out["events"] == "home_run").astype(int)
    out["is_bbe"] = (out["type"] == "X") & out["launch_speed"].notna()
    out["is_barrel"] = (out["launch_speed_angle"] == 6).astype(int)
    out["in_zone"] = out["zone"].between(1, 9, inclusive="both").astype(int)
    out["zone_known"] = out["zone"].notna().astype(int)
    out["is_air_ball"] = out["bb_type"].apply(is_air_ball).astype(int)
    out["is_pulled_air_ball"] = [
        int(bool(air_flag) and (pull_flag == 1)) if pull_flag is not None else 0
        for air_flag, pull_flag in zip(
            out["is_air_ball"].tolist(),
            [is_pulled_ball(s, x) for s, x in zip(out["stand"], out["hc_x"])]
        )
    ]
    out["pulled_air_ball_known"] = [
        int(bool(air_flag) and pull_flag is not None)
        for air_flag, pull_flag in zip(
            out["is_air_ball"].tolist(),
            [is_pulled_ball(s, x) for s, x in zip(out["stand"], out["hc_x"])]
        )
    ]

    return out


# -----------------------------
# Aggregation
# -----------------------------
def aggregate_batter_split(df: pd.DataFrame, pitcher_hand: str) -> pd.DataFrame:
    if df.empty or "p_throws" not in df.columns:
        return pd.DataFrame(columns=["pitch_type", "HR", "BBE", "EV", "LA", "BAT_SPEED"])

    sub = df[(df["p_throws"] == pitcher_hand) & (df["is_bbe"])].copy()
    if sub.empty:
        return pd.DataFrame(columns=["pitch_type", "HR", "BBE", "EV", "LA", "BAT_SPEED"])

    return (
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



def aggregate_pitcher_split(df: pd.DataFrame, batter_hand: str) -> pd.DataFrame:
    if df.empty or "stand" not in df.columns:
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

    return perf.merge(usage[["pitch_type", "usage"]], on="pitch_type", how="outer")



def aggregate_batter_power_metrics(df: pd.DataFrame, pitcher_hand: str) -> dict:
    if df.empty or "p_throws" not in df.columns:
        return {"BARREL_PCT": np.nan, "PULL_AIR_PCT": np.nan}

    sub = df[df["p_throws"] == pitcher_hand].copy()
    if sub.empty:
        return {"BARREL_PCT": np.nan, "PULL_AIR_PCT": np.nan}

    bbe = sub[sub["is_bbe"]].copy()
    barrel_pct = np.nan
    if not bbe.empty:
        barrel_pct = 100.0 * float(bbe["is_barrel"].sum()) / float(len(bbe))

    air_subset = sub[(sub["is_air_ball"] == 1) & (sub["pulled_air_ball_known"] == 1)].copy()
    pull_air_pct = np.nan
    if not air_subset.empty:
        pull_air_pct = 100.0 * float(air_subset["is_pulled_air_ball"].sum()) / float(len(air_subset))

    return {"BARREL_PCT": barrel_pct, "PULL_AIR_PCT": pull_air_pct}



def aggregate_pitcher_power_metrics(df: pd.DataFrame, batter_hand: str) -> dict:
    if df.empty or "stand" not in df.columns:
        return {"BARREL_PCT_ALLOWED": np.nan, "ZONE_PCT": np.nan}

    sub = df[df["stand"] == batter_hand].copy()
    if sub.empty:
        return {"BARREL_PCT_ALLOWED": np.nan, "ZONE_PCT": np.nan}

    bbe = sub[sub["is_bbe"]].copy()
    barrel_pct_allowed = np.nan
    if not bbe.empty:
        barrel_pct_allowed = 100.0 * float(bbe["is_barrel"].sum()) / float(len(bbe))

    zone_den = int(sub["zone_known"].sum())
    zone_pct = np.nan if zone_den == 0 else (100.0 * float(sub["in_zone"].sum()) / float(zone_den))

    return {"BARREL_PCT_ALLOWED": barrel_pct_allowed, "ZONE_PCT": zone_pct}



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



def combine_weighted_metrics(metric_dicts: Dict[int, dict], keys: List[str]) -> dict:
    out = {}
    for key in keys:
        vals = []
        weights = []
        for season, metrics in metric_dicts.items():
            value = metrics.get(key, np.nan)
            if pd.isna(value):
                continue
            vals.append(float(value))
            weights.append(float(SEASON_WEIGHTS.get(season, 1.0)))
        out[key] = float(np.average(vals, weights=weights)) if vals else np.nan
    return out



def build_batter_profile(batter_id: int, pitcher_hand: str, refresh_cache: bool = False) -> Tuple[pd.DataFrame, dict]:
    season_frames = {}
    season_metrics = {}
    for season in SEASONS:
        raw = load_or_fetch_batter(batter_id, season, refresh=refresh_cache)
        prepped = prep_statcast_frame(raw)
        season_frames[season] = aggregate_batter_split(prepped, pitcher_hand)
        season_metrics[season] = aggregate_batter_power_metrics(prepped, pitcher_hand)

    return (
        combine_weighted_seasons(season_frames, ["HR", "BBE", "EV", "LA", "BAT_SPEED"]),
        combine_weighted_metrics(season_metrics, ["BARREL_PCT", "PULL_AIR_PCT"]),
    )



def build_pitcher_profile(pitcher_id: int, batter_hand: str, refresh_cache: bool = False) -> Tuple[pd.DataFrame, dict]:
    season_frames = {}
    season_metrics = {}
    for season in SEASONS:
        raw = load_or_fetch_pitcher(pitcher_id, season, refresh=refresh_cache)
        prepped = prep_statcast_frame(raw)
        season_frames[season] = aggregate_pitcher_split(prepped, batter_hand)
        season_metrics[season] = aggregate_pitcher_power_metrics(prepped, batter_hand)

    return (
        combine_weighted_seasons(season_frames, ["HR_allowed", "BBE_allowed", "EV_allowed", "LA_allowed", "usage"]),
        combine_weighted_metrics(season_metrics, ["BARREL_PCT_ALLOWED", "ZONE_PCT"]),
    )


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
    game_date: Optional[str] = None,
    game_pk: Optional[object] = None,
    home_team: Optional[str] = None,
    away_team: Optional[str] = None,
    park_team: Optional[str] = None,
    park: Optional[str] = None,
    venue: Optional[str] = None,
    stadium: Optional[str] = None,
    is_home: Optional[object] = None,
    park_factors_df: Optional[pd.DataFrame] = None,
    refresh_cache: bool = False,
    n_abs: int = DEFAULT_ABS,
) -> Optional[dict]:
    effective_batter_hand = resolve_effective_batter_hand(batter_hand, pitcher_hand)

    batter_profile, batter_metrics = build_batter_profile(batter_id, pitcher_hand, refresh_cache=refresh_cache)
    pitcher_profile, pitcher_metrics = build_pitcher_profile(pitcher_id, effective_batter_hand, refresh_cache=refresh_cache)

    if batter_profile.empty or pitcher_profile.empty:
        print(
            f"Skipped {batter_name}: batter_hand={batter_hand}, "
            f"effective_batter_hand={effective_batter_hand}, pitcher_hand={pitcher_hand}"
        )
        return None

    breakdown = build_matchup_breakdown(batter_profile, pitcher_profile)
    if breakdown.empty:
        print(
            f"Skipped {batter_name}: no matchup breakdown after merge, "
            f"effective_batter_hand={effective_batter_hand}, pitcher_hand={pitcher_hand}"
        )
        return None

    raw_score = float(breakdown["weighted_pitch_score"].sum())

    batter_barrel_pct = clamp(batter_metrics.get("BARREL_PCT", np.nan), 0.0, MAX_BATTER_BARREL)
    batter_pull_air_pct = clamp(batter_metrics.get("PULL_AIR_PCT", np.nan), 0.0, MAX_BATTER_PULL_AIR)
    pitcher_barrel_pct_allowed = clamp(pitcher_metrics.get("BARREL_PCT_ALLOWED", np.nan), 0.0, MAX_PITCHER_BARREL)
    pitcher_zone_pct = pitcher_metrics.get("ZONE_PCT", np.nan)

    batter_barrel_adj = metric_adjustment(
        batter_barrel_pct, LEAGUE_AVG_BARREL_PCT, BARREL_PCT_SCALE, MAX_BARREL_ADJ
    )
    batter_pull_air_adj = metric_adjustment(
        batter_pull_air_pct, LEAGUE_AVG_PULL_AIR_PCT, PULL_AIR_PCT_SCALE, MAX_PULL_AIR_ADJ
    )
    pitcher_barrel_adj = metric_adjustment(
        pitcher_barrel_pct_allowed, LEAGUE_AVG_BARREL_PCT, BARREL_PCT_SCALE, MAX_BARREL_ADJ
    )
    pitcher_zone_adj = metric_adjustment(
        pitcher_zone_pct, LEAGUE_AVG_ZONE_PCT, ZONE_PCT_SCALE, MAX_ZONE_ADJ
    )

    raw_score = (
        raw_score
        * (1.0 + batter_barrel_adj)
        * (1.0 + batter_pull_air_adj)
        * (1.0 + pitcher_barrel_adj)
        * (1.0 + pitcher_zone_adj)
    )

    pa_hr_probability_base = calibrated_pa_hr_probability(raw_score)

    park_factor_hr, park_factor_source, resolved_home_team, resolved_stadium, park_hr_percent_effect = resolve_park_factor(
        home_team=home_team,
        park_team=park_team,
        stadium=stadium,
        venue=venue,
        park=park,
        park_factors_df=park_factors_df,
    )

    pa_hr_probability = apply_park_factor(pa_hr_probability_base, park_factor_hr)
    hr_probability_nab = hr_probability_over_n_abs(pa_hr_probability, n_abs=n_abs)

    best_pitch = breakdown.iloc[0]["pitch_type"] if len(breakdown) > 0 else None
    top_usage = float(breakdown.iloc[0]["usage"]) if len(breakdown) > 0 else np.nan

    return {
        "date": date.today().isoformat(),
        "game_date": game_date,
        "game_pk": game_pk,
        "batter": batter_name,
        "batter_id": int(batter_id),
        "batter_hand": batter_hand,
        "effective_batter_hand": effective_batter_hand,
        "team": team,
        "opponent": opponent,
        "home_team": home_team,
        "away_team": away_team,
        "park_team": park_team,
        "input_stadium": stadium,
        "input_venue": venue,
        "is_home": is_home,
        "home_team_resolved": resolved_home_team,
        "park_resolved": resolved_stadium,
        "pitcher": pitcher_name,
        "pitcher_id": int(pitcher_id),
        "pitcher_hand": pitcher_hand,
        "raw_matchup_score": raw_score,
        "hr_probability_pa_base": pa_hr_probability_base,
        "park_factor_hr": park_factor_hr,
        "park_hr_percent_effect": park_hr_percent_effect,
        "park_factor_source": park_factor_source,
        "hr_probability_pa": pa_hr_probability,
        f"hr_probability_{n_abs}ab": hr_probability_nab,
        "top_pitch_in_mix": best_pitch,
        "dominant_pitch_usage": top_usage,
        "pitch_types_used": int(len(breakdown)),
        "batter_barrel_pct": batter_barrel_pct,
        "batter_pull_air_pct": batter_pull_air_pct,
        "pitcher_barrel_pct_allowed": pitcher_barrel_pct_allowed,
        "pitcher_zone_pct": pitcher_zone_pct,
        "batter_barrel_adj": batter_barrel_adj,
        "batter_pull_air_adj": batter_pull_air_adj,
        "pitcher_barrel_adj": pitcher_barrel_adj,
        "pitcher_zone_adj": pitcher_zone_adj,
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
    out["team"] = out["team"].astype(str).map(normalize_team_abbr)
    out["opponent"] = out["opponent"].astype(str).map(normalize_team_abbr)

    for col in OPTIONAL_LINEUP_COLS:
        if col in out.columns:
            out[col] = out[col].astype(str).replace({"nan": np.nan, "None": np.nan, "": np.nan})
            if col in {"home_team", "away_team", "park_team"}:
                out[col] = out[col].map(lambda x: normalize_team_abbr(x) if pd.notna(x) else x)

    return out.drop_duplicates().reset_index(drop=True)



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
        print(f"Loaded park-factor rows: {len(park_factors_df)}")
    else:
        park_factors_df = pd.DataFrame(columns=["home_team", "model_team_key", "stadium", "stadium_key", "hr_percent_effect", "hr_multiplier"])

    score_cols = REQUIRED_LINEUP_COLS + [c for c in OPTIONAL_LINEUP_COLS if c in lineups.columns]
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
                game_date=row.get("game_date"),
                game_pk=row.get("game_pk"),
                home_team=row.get("home_team"),
                away_team=row.get("away_team"),
                park_team=row.get("park_team"),
                park=row.get("park"),
                venue=row.get("venue"),
                stadium=row.get("stadium"),
                is_home=row.get("is_home"),
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
        cols = [
            "batter", "team", "batter_hand", "effective_batter_hand", "pitcher", "pitcher_hand", "opponent",
            "park_team", "park_resolved", "park_factor_hr", "batter_barrel_pct", "batter_pull_air_pct",
            "hr_probability_pa", sort_col
        ]
        existing_cols = [c for c in cols if c in board.columns]
        print(board[existing_cols].head(20).to_string(index=False))

    return board



def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the HR probability model with park-level factors and Pull Air% adjustment."
    )
    parser.add_argument("--lineups", default="daily_lineups.csv", help="Input lineup CSV. Default: daily_lineups.csv")
    parser.add_argument("--output", default=None, help="Output predictions CSV. Default: hr_predictions_<today>.csv")
    parser.add_argument(
        "--park-factors",
        default=DEFAULT_PARK_FACTORS_FILE,
        help=f"Path to park factors workbook. Default: {DEFAULT_PARK_FACTORS_FILE}",
    )
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
