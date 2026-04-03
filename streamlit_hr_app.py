import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="MLB HR Projections",
    page_icon="⚾",
    layout="wide"
)

CSV_PATH = "C:\Users\adria\hr_predictions.csv"

DISPLAY_COLUMNS = [
    "batter",
    "team",
    "opponent",
    "pitcher",
    "batter_hand",
    "pitcher_hand",
    "hr_probability_3ab",
]

COLUMN_LABELS = {
    "batter": "Batter",
    "team": "Team",
    "opponent": "Opponent",
    "pitcher": "Opposing Pitcher",
    "batter_hand": "Bat Hand",
    "pitcher_hand": "Pitch Hand",
    "top_pitch_in_mix": "Top Pitch",
    "dominant_pitch_usage": "Top Pitch Usage",
    "hr_probability_pa": "HR Prob / PA",
    "hr_probability_3ab": "HR Probability",
    "park_factor_hr": "Park Factor",
    "raw_matchup_score": "Matchup Score",
    "weighted_outcome_component": "Outcome Component",
    "weighted_bat_speed_component": "Bat Speed Component",
    "weighted_ev_la_component": "EV/LA Component",
}

PERCENT_COLUMNS = [
    "hr_probability_3ab",
]

DECIMAL_COLUMNS = [
    "park_factor_hr",
    "raw_matchup_score",
    "weighted_outcome_component",
    "weighted_bat_speed_component",
    "weighted_ev_la_component",
]


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    for col in PERCENT_COLUMNS + DECIMAL_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def build_display_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DISPLAY_COLUMNS if c in df.columns]
    display_df = df[cols].copy()

    rename_map = {c: COLUMN_LABELS[c] for c in cols if c in COLUMN_LABELS}
    display_df = display_df.rename(columns=rename_map)

    return display_df


def format_dataframe(df: pd.DataFrame):
    format_dict = {}
    for col in df.columns:
        original_name = next((k for k, v in COLUMN_LABELS.items() if v == col), None)
        if original_name in PERCENT_COLUMNS:
            format_dict[col] = "{:.0%}"
        elif original_name in DECIMAL_COLUMNS:
            format_dict[col] = "{:.3f}"

    return df.style.format(format_dict)


df = load_data(CSV_PATH)

st.title("⚾ Daily MLB Home Run Projections")

if "date" in df.columns and df["date"].notna().any():
    report_date = df["date"].dropna().iloc[0].date()
    st.caption(f"Projection date: {report_date}")

left, right = st.columns([2, 1])

with left:
    st.subheader("Top projected hitters")
with right:
    sort_choice = st.selectbox(
        "Sort by",
        options=[
            "hr_probability_3ab",
        ],
        index=0,
        format_func=lambda x: COLUMN_LABELS.get(x, x),
    )

st.sidebar.header("Filters")

teams = sorted(df["team"].dropna().unique().tolist()) if "team" in df.columns else []
opponents = sorted(df["opponent"].dropna().unique().tolist()) if "opponent" in df.columns else []
pitchers = sorted(df["pitcher"].dropna().unique().tolist()) if "pitcher" in df.columns else []

selected_teams = st.sidebar.multiselect("Team", teams, default=teams)
selected_opponents = st.sidebar.multiselect("Opponent", opponents, default=opponents)
selected_pitchers = st.sidebar.multiselect("Opposing pitcher", pitchers, default=[])

min_hr_3ab = st.sidebar.slider(
    "Minimum HR Probability",
    min_value=0.0,
    max_value=float(df["hr_probability_3ab"].max()) if "hr_probability_3ab" in df.columns else 1.0,
    value=0.0,
    step=0.01,
)

search_name = st.sidebar.text_input("Search batter")

filtered = df.copy()

if selected_teams and "team" in filtered.columns:
    filtered = filtered[filtered["team"].isin(selected_teams)]

if selected_opponents and "opponent" in filtered.columns:
    filtered = filtered[filtered["opponent"].isin(selected_opponents)]

if selected_pitchers and "pitcher" in filtered.columns:
    filtered = filtered[filtered["pitcher"].isin(selected_pitchers)]

if "hr_probability_3ab" in filtered.columns:
    filtered = filtered[filtered["hr_probability_3ab"] >= min_hr_3ab]

if search_name and "batter" in filtered.columns:
    filtered = filtered[
        filtered["batter"].astype(str).str.contains(search_name, case=False, na=False)
    ]

if sort_choice in filtered.columns:
    filtered = filtered.sort_values(sort_choice, ascending=False, na_position="last")

top_n = min(10, len(filtered))
top_df = filtered.head(top_n)

if len(filtered) > 0 and "hr_probability_3ab" in filtered.columns and "batter" in filtered.columns:
    top_row = filtered.sort_values("hr_probability_3ab", ascending=False).iloc[0]

    metric_cols = st.columns(3)

    with metric_cols[0]:
        st.metric("Players Shown", f"{len(filtered)}")

    with metric_cols[1]:
        st.metric("Highest HR Probability", f"{top_row['batter']}")

    with metric_cols[2]:
        st.metric("HR Probability", f"{top_row['hr_probability_3ab']:.0%}")
else:
    st.metric("Players Shown", "0")

chart_col, summary_col = st.columns([2, 1])

with chart_col:
    if len(filtered) > 0 and "batter" in filtered.columns and "hr_probability_3ab" in filtered.columns:

        top5 = filtered.sort_values("hr_probability_3ab", ascending=False).head(5)

        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.set_facecolor("#0E1117")  # dark background
        fig.patch.set_facecolor("#0E1117")

        ax.axis("off")

        # Title
        ax.text(
            0.05, 0.9,
            "TOP HR PROJECTIONS",
            fontsize=18,
            fontweight="bold",
            color="white"
        )
        ax.text(
            0.05, 0.85,
            "Top 5 hitters by HR probability",
            fontsize=10,
            color="gray"
        )
        # Loop through players
        y = 0.80
        for i, row in enumerate(top5.itertuples(), start=1):
            name = row.batter
            prob = f"{row.hr_probability_3ab:.0%}"
        
            ax.text(
                0.05, y,
                f"{i}. {name}",
                fontsize=14,
                color="white",
                va="center"
            )
        
            ax.text(
                0.95, y,
                prob,
                fontsize=14,
                color="#4FC3F7",
                ha="right",
                va="center"
            )
        
            if i < len(top5):
                ax.hlines(y - 0.028, 0.05, 0.95, color="gray", linewidth=0.4)
        
            y -= 0.075
            
        st.pyplot(fig)
        
with summary_col:
    st.markdown("### Quick notes")
    st.write(
        "- **HR Probability** represents each player's estimated chance to hit a home run in a typical 3 at-bat game sample."
    )
    st.write(
        "- The model combines **real game results**, **advanced hitting metrics**, and **matchup context** to create each projection."
    )
    st.write(
        "- Projections also account for **park and weather factors**, which can raise or lower home run likelihood depending on the game environment."
    )
st.markdown("### Projection table")
display_df = build_display_df(filtered)

st.dataframe(
    format_dataframe(display_df),
    use_container_width=True,
    height=650,
)

st.markdown("### Player detail")

if len(filtered) > 0 and "batter" in filtered.columns:
    batter_options = filtered["batter"].dropna().tolist()
    selected_batter = st.selectbox("Select a batter", batter_options)

    player_row = filtered.loc[filtered["batter"] == selected_batter].iloc[0]

    # Main metric
    st.metric("HR Probability", f"{player_row.get('hr_probability_3ab', 0):.0%}")

    # Details dictionary
    details = {
        "Team": player_row.get("team", ""),
        "Opponent": player_row.get("opponent", ""),
        "Pitcher": player_row.get("pitcher", ""),
        "Batter Hand": player_row.get("batter_hand", ""),
        "Pitcher Hand": player_row.get("pitcher_hand", ""),
    }

    st.write(details)

else:
    st.info("No players match the current filters.")
