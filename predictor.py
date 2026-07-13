"""
Match outcome predictor for the 2026 World Cup.

Two-model pattern:
  1. EVAL split   — train/test split on RAW match records FIRST, strength
                    rates computed only from the train partition, honest
                    AUC reported from the untouched test partition.
  2. PRODUCTION   — refit on ALL historical data for actual 2026 predictions,
                    since real-world inference benefits from every match we
                    have. The AUC above is what tells you how much to trust it;
                    this refit does not change that number.

Bug fixed from previous version: strength rates were previously computed via
df.sample(frac=0.8) — an INDEPENDENT random sample from the train_test_split
used for evaluation. Since the two samples didn't correspond to the same
partition, a portion of "test" matches had their own outcome contribute to
their own strength feature. Same class of leakage as Project 1, caught on
review. Fixed by splitting raw records first, then deriving both the eval
features and the eval labels from that single, consistent partition.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)

FEATURE_COLS = ["home_advantage", "home_team_strength", "away_team_strength"]

# Team name normalization — maps API names to historical dataset names
TEAM_ALIASES: dict[str, str] = {
    "USA":                      "United States",
    "US":                       "United States",
    "Korea Republic":           "South Korea",
    "IR Iran":                  "Iran",
    "Côte d'Ivoire":            "Ivory Coast",
    "Cote d'Ivoire":            "Ivory Coast",
    "Republic of Ireland":      "Republic of Ireland",
    "Congo DR":                 "DR Congo",
    "Korea DPR":                "North Korea",
    "Kyrgyz Republic":          "Kyrgyzstan",
    "Cape Verde":               "Cape Verde Islands",
    "Timor-Leste":              "East Timor",
}


def _normalize(team: str) -> str:
    return TEAM_ALIASES.get(team, team)


def _add_strength_features(frame: pd.DataFrame,
                            home_rates: pd.Series,
                            away_rates: pd.Series,
                            global_mean: float) -> pd.DataFrame:
    """Apply precomputed strength rates to a frame. Unseen teams get global_mean."""
    f = frame.copy()
    f["home_team_strength"] = f["home_team"].map(home_rates).fillna(global_mean)
    f["away_team_strength"] = f["away_team"].map(away_rates).fillna(global_mean)
    return f[FEATURE_COLS]


@st.cache_resource(show_spinner=False)
def load_model() -> tuple:
    """
    Download historical data, evaluate honestly, then refit for production.

    Returns:
        prod_model        — LogisticRegression fitted on ALL historical data
        prod_home_rates   — Series: team → home win rate (computed on ALL data)
        prod_away_rates   — Series: team → away win rate (computed on ALL data)
        prod_global_mean  — float: fallback for unknown teams
        auc               — float: HONEST held-out AUC (from the eval split,
                             never touched by the production refit)
    """
    print("  Predictor: downloading historical match data…")
    df = pd.read_csv(DATA_URL)
    df["date"] = pd.to_datetime(df["date"])

    df = df[df["date"].dt.year >= 2006].copy()
    df = df[df["tournament"] != "Friendly"].copy()
    df = df.dropna(subset=["home_score", "away_score"]).copy()

    df["home_team_win"]  = (df["home_score"] > df["away_score"]).astype(int)
    df["away_team_win"]  = (df["away_score"] > df["home_score"]).astype(int)
    df["home_advantage"] = (~df["neutral"]).astype(int)

    # ── Step 1: Honest evaluation ──────────────────────────────────────────
    # Split RAW records FIRST. Strength rates for evaluation come ONLY from
    # train_raw. Both X_train and X_test are derived from this single,
    # consistent split — no match's own outcome can leak into its own
    # feature, because train_raw and test_raw never overlap.
    train_raw, test_raw = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["home_team_win"]
    )

    eval_home_rates  = train_raw.groupby("home_team")["home_team_win"].mean()
    eval_away_rates  = train_raw.groupby("away_team")["away_team_win"].mean()
    eval_global_mean = float(eval_home_rates.mean())

    X_train = _add_strength_features(train_raw, eval_home_rates, eval_away_rates, eval_global_mean)
    X_test  = _add_strength_features(test_raw,  eval_home_rates, eval_away_rates, eval_global_mean)
    y_train = train_raw["home_team_win"]
    y_test  = test_raw["home_team_win"]

    eval_model = LogisticRegression(max_iter=1000)
    eval_model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, eval_model.predict_proba(X_test)[:, 1])
    print(f"  Predictor — honest held-out AUC: {auc:.3f} on {len(X_test)} unseen matches")

    # ── Step 2: Production refit ───────────────────────────────────────────
    # For actual 2026 predictions, use ALL historical data — maximizes
    # coverage of team strength estimates. This does NOT affect the AUC
    # reported above; that number is already locked in from the honest split.
    prod_home_rates  = df.groupby("home_team")["home_team_win"].mean()
    prod_away_rates  = df.groupby("away_team")["away_team_win"].mean()
    prod_global_mean = float(prod_home_rates.mean())

    X_all = _add_strength_features(df, prod_home_rates, prod_away_rates, prod_global_mean)
    y_all = df["home_team_win"]

    prod_model = LogisticRegression(max_iter=1000)
    prod_model.fit(X_all, y_all)

    return prod_model, prod_home_rates, prod_away_rates, prod_global_mean, auc


def predict_match(team1: str, team2: str, neutral: bool = True) -> dict:
    """
    Predict the outcome of team1 vs team2 using the production model.

    World Cup matches are at neutral venues → neutral=True always.

    Returns:
        {
            "team1": str, "team2": str,
            "team1_win": float, "draw": float, "team2_win": float,
            "favourite": str, "confidence": str,
        }
    """
    model, home_win_rates, away_win_rates, global_mean, _ = load_model()

    t1 = _normalize(team1)
    t2 = _normalize(team2)

    t1_strength = float(home_win_rates.get(t1, global_mean))
    t2_strength = float(away_win_rates.get(t2, global_mean))
    home_adv    = 0 if neutral else 1

    X = pd.DataFrame({
        "home_advantage":     [home_adv],
        "home_team_strength": [t1_strength],
        "away_team_strength": [t2_strength],
    })

    raw_proba = model.predict_proba(X)[0]
    raw_t1 = float(raw_proba[1])
    raw_t2 = float(raw_proba[0])

    draw_rate = 0.25
    t1_win = raw_t1 * (1 - draw_rate)
    t2_win = raw_t2 * (1 - draw_rate)
    draw   = draw_rate

    total  = t1_win + draw + t2_win
    t1_win /= total
    draw   /= total
    t2_win /= total

    favourite  = team1 if t1_win > t2_win else team2
    max_prob   = max(t1_win, t2_win)
    confidence = (
        "High"                 if max_prob > 0.55 else
        "Moderate"             if max_prob > 0.44 else
        "Closely contested"
    )

    return {
        "team1":      team1,
        "team2":      team2,
        "team1_win":  round(t1_win, 3),
        "draw":       round(draw,   3),
        "team2_win":  round(t2_win, 3),
        "favourite":  favourite,
        "confidence": confidence,
    }