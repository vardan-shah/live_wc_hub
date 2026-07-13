"""
Project 5 — 2026 World Cup Live Prediction Hub

Four tabs:
  🔴 Live Scores    — scores, live status, completed results
  🎯 Predictions    — ML win probabilities for every upcoming match
  🏆 Path to Final  — all remaining bracket stages with predictions
  📊 Track Record   — timestamped prediction ledger + resolved accuracy

Data: football-data.org (primary, requires free key) → openfootball (fallback)
ML:   Logistic regression, honest eval split + production refit on all data
"""

import time
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, timezone, date

load_dotenv()

from live_data import (
    get_all_knockout_matches,
    get_today_matches,
    get_upcoming_matches,
    get_completed_matches,
    is_match_live,
    has_api_key,
    STAGE_DISPLAY,
    STAGE_ORDER,
)
from predictor import load_model, predict_match
from prediction_log import log_prediction_if_new, update_results, get_track_record


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="2026 World Cup Hub",
    page_icon="⚽",
    layout="wide",
)


# ── Load ML model once per session ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_predictor():
    return load_model()

with st.spinner("Training ML predictor on historical international data…"):
    _, _, _, _, auc = get_predictor()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚽ 2026 World Cup — Live Prediction Hub")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tournament", "FIFA World Cup 2026")
c2.metric("Host nations", "USA · CAN · MEX")
c3.metric("ML model AUC", f"{auc:.3f}")
c4.metric("Data source", "football-data.org" if has_api_key() else "openfootball ↗")

if not has_api_key():
    st.info(
        "Running on **openfootball fallback** (no API key set). "
        "Completed results are available. For live in-match scores, "
        "get a free key at [football-data.org](https://www.football-data.org/client/register) "
        "and add `FOOTBALL_DATA_API_KEY=your_key` to your `.env` file.",
        icon="ℹ️",
    )

st.divider()


# ── Fetch live data ────────────────────────────────────────────────────────────
with st.spinner("Fetching live tournament data…"):
    all_matches = get_all_knockout_matches()

today_matches    = get_today_matches(all_matches)
upcoming_matches = get_upcoming_matches(all_matches)
completed        = get_completed_matches(all_matches)

# ── Backfill any predictions whose matches have now finished ──────────────────
newly_resolved = update_results(all_matches)
if newly_resolved:
    st.toast(f"📬 {newly_resolved} logged prediction(s) just resolved!", icon="🎯")


# ── Helper components ─────────────────────────────────────────────────────────

def _stage_label(stage: str) -> str:
    return STAGE_DISPLAY.get(stage, stage)


def _render_score(m: dict) -> None:
    """Render a single match card — live, finished, or upcoming."""
    stage_lbl = _stage_label(m["stage"])
    is_live   = is_match_live(m)
    finished  = m["status"] in ("FINISHED", "AWARDED")

    with st.container(border=True):
        badge_col, _, date_col = st.columns([2, 3, 2])
        with badge_col:
            if is_live:
                st.markdown("🔴 **LIVE**")
            elif finished:
                st.markdown(f"✅ *{stage_lbl}*")
            else:
                st.markdown(f"🕐 *{stage_lbl}*")

        with date_col:
            kick_off = f"{m['date']}  {m['time']} UTC" if m["time"] else m["date"]
            st.caption(kick_off)

        t1_col, mid_col, t2_col = st.columns([3, 1, 3])

        with t1_col:
            st.markdown(f"### {m['team1']}")

        with mid_col:
            if m["score1"] is not None and m["score2"] is not None:
                penalty_note = ""
                if m.get("pen1") is not None:
                    penalty_note = (
                        f"<div style='text-align:center;font-size:11px;"
                        f"color:var(--text-muted)'>({m['pen1']}–{m['pen2']} pens)</div>"
                    )
                st.markdown(
                    f"<div style='text-align:center;font-size:22px;"
                    f"font-weight:500'>{m['score1']} – {m['score2']}</div>{penalty_note}",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='text-align:center;font-size:18px;"
                    "color:var(--text-muted)'>vs</div>",
                    unsafe_allow_html=True,
                )

        with t2_col:
            st.markdown(
                f"<div style='text-align:right'><h3>{m['team2']}</h3></div>",
                unsafe_allow_html=True,
            )


def _render_prediction(m: dict) -> None:
    """Render an ML prediction card for a match, and log it for calibration."""
    t1, t2 = m["team1"], m["team2"]

    if "TBD" in (t1, t2) or not t1 or not t2:
        with st.container(border=True):
            st.caption(f"*{_stage_label(m['stage'])} · {m['date']}*")
            st.markdown("**TBD vs TBD** — predictions available once teams are confirmed.")
        return

    pred = predict_match(t1, t2, neutral=True)

    # Timestamp this prediction NOW, before the result is known.
    # Idempotent — does nothing if this match was already logged earlier.
    log_prediction_if_new(m, pred)

    with st.container(border=True):
        head_left, head_right = st.columns([3, 1])
        with head_left:
            st.markdown(
                f"**{t1}** vs **{t2}** · *{_stage_label(m['stage'])} · {m['date']}*"
            )
        with head_right:
            conf_color = {
                "High": "🟢", "Moderate": "🟡", "Closely contested": "🟠",
            }.get(pred["confidence"], "⚪")
            st.caption(f"{conf_color} {pred['confidence']}")

        st.markdown(
            f"**Model favours:** {pred['favourite']} — "
            f"*based on historical win rates since 2006*"
        )

        p1, pd_, p2 = pred["team1_win"], pred["draw"], pred["team2_win"]
        bar_col1, bar_col2, bar_col3 = st.columns([3, 2, 3])
        with bar_col1:
            st.caption(t1)
            st.progress(p1, text=f"{p1*100:.0f}%")
        with bar_col2:
            st.caption("Draw")
            st.progress(pd_, text=f"{pd_*100:.0f}%")
        with bar_col3:
            st.caption(t2)
            st.progress(p2, text=f"{p2*100:.0f}%")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔴 Live Scores", "🎯 Predictions", "🏆 Path to Final", "📊 Track Record"]
)


# ── Tab 1: Live scores ────────────────────────────────────────────────────────
with tab1:
    now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC — %d %b %Y")
    st.caption(f"Last updated: {now_utc}")

    live_now = [m for m in all_matches if is_match_live(m)]

    if live_now:
        st.subheader("🔴 In progress right now")
        for m in live_now:
            _render_score(m)
        st.divider()

    if today_matches:
        st.subheader(f"Today — {date.today().strftime('%d %B %Y')}")
        for m in today_matches:
            _render_score(m)
    else:
        st.info(
            "No knockout matches scheduled today. Check the Predictions tab for upcoming fixtures.",
            icon="📅",
        )

    if completed:
        with st.expander(f"📋 Completed knockout results ({len(completed)} matches)"):
            for m in sorted(completed, key=lambda x: x["date"], reverse=True):
                st.caption(f"*{_stage_label(m['stage'])} · {m['date']}*")
                st.markdown(f"**{m['team1']}** {m['score1']} – {m['score2']} **{m['team2']}**")


# ── Tab 2: Predictions ────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        "Win probabilities from a **logistic regression model**, evaluated on a "
        f"held-out split before being refit on all available data. Honest test-set "
        f"AUC: **{auc:.3f}**. All World Cup matches treated as neutral venue."
    )
    st.caption("📌 Every prediction below is timestamped and logged the moment it's generated — see the Track Record tab.")

    if not upcoming_matches:
        st.success(
            "The tournament is complete — no upcoming matches to predict. "
            "See the completed results in the Live Scores tab.",
            icon="🏆",
        )
    else:
        stages_seen: dict[str, list] = {}
        for m in upcoming_matches:
            stages_seen.setdefault(m["stage"], []).append(m)

        for stage in sorted(stages_seen.keys(), key=lambda s: STAGE_ORDER.get(s, 9)):
            st.subheader(_stage_label(stage))
            for m in stages_seen[stage]:
                _render_prediction(m)
            st.divider()

    with st.expander("🔬 How the predictions work"):
        st.markdown(
            """
| Feature | Description |
|---|---|
| `home_team_strength` | Team's historical win rate as home side (2006+, competitive only) |
| `away_team_strength` | Opponent's historical win rate as away side |
| `home_advantage` | 1 if home stadium, 0 if neutral — always 0 at the World Cup |

**Evaluation methodology**: the reported AUC comes from a train/test split done
on raw match records *before* any feature is computed — strength rates for
evaluation are derived only from the training partition, so no match's own
result ever leaks into its own feature. The model actually generating these
predictions is a separate refit on 100% of historical data, since production
inference benefits from every match available — that refit does not change
the AUC number above.

**Draw probability**: injected as a football-calibrated prior (~25% of
competitive matches end level), then all three outcomes are re-normalised
to sum to 1.

**Unknown teams**: any nation not in the historical dataset receives the
global historical average — a conservative fallback.
            """
        )


# ── Tab 3: Path to Final ──────────────────────────────────────────────────────
with tab3:
    st.markdown(
        "All remaining knockout stages with ML predictions. "
        "Results update automatically — refresh to pull the latest scores."
    )

    remaining_stages = sorted(
        set(m["stage"] for m in all_matches),
        key=lambda s: STAGE_ORDER.get(s, 9),
    )

    for stage in remaining_stages:
        stage_matches     = [m for m in all_matches if m["stage"] == stage]
        finished_in_stage = [m for m in stage_matches if m["status"] in ("FINISHED", "AWARDED")]
        upcoming_in_stage = [m for m in stage_matches if m not in finished_in_stage]

        st.subheader(_stage_label(stage))

        for m in finished_in_stage:
            col_a, col_b, col_c = st.columns([5, 2, 5])
            with col_a:
                st.markdown(f"✅ **{m['team1']}**")
            with col_b:
                penalty_note = f" ({m['pen1']}–{m['pen2']} pens)" if m.get("pen1") is not None else ""
                st.markdown(
                    f"<div style='text-align:center'>**{m['score1']} – {m['score2']}**{penalty_note}</div>",
                    unsafe_allow_html=True,
                )
            with col_c:
                st.markdown(f"<div style='text-align:right'>**{m['team2']}** ✅</div>", unsafe_allow_html=True)

        for m in upcoming_in_stage:
            _render_prediction(m)

        st.divider()

    if not remaining_stages:
        st.success("Tournament complete!", icon="🏆")


# ── Tab 4: Track Record ────────────────────────────────────────────────────────
with tab4:
    st.markdown(
        "Every prediction is timestamped **before** the match is played — "
        "this is real evidence of the model's accuracy, not a retrospective claim."
    )

    record = get_track_record()

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Predictions logged", record["total_logged"])
    rc2.metric("Resolved", record["resolved"])
    rc3.metric("Pending", record["pending"])
    rc4.metric(
        "Favourite accuracy",
        f"{record['accuracy']*100:.0f}%" if record["accuracy"] is not None else "—",
    )

    st.divider()

    if not record["entries"]:
        st.info(
            "No predictions logged yet — visit the Predictions or Path to Final "
            "tab to generate some. They'll appear here automatically.",
            icon="📌",
        )
    else:
        st.subheader("📋 Prediction ledger")
        for e in reversed(record["entries"]):   # Newest first
            with st.container(border=True):
                head_l, head_r = st.columns([4, 1])
                with head_l:
                    st.markdown(
                        f"**{e['team1']}** vs **{e['team2']}** · "
                        f"*{STAGE_DISPLAY.get(e['stage'], e['stage'])} · {e['date']}*"
                    )
                with head_r:
                    if e["actual_result"] is None:
                        st.caption("🔒 Pending")
                    elif e["correct"]:
                        st.caption("✅ Correct")
                    else:
                        st.caption("❌ Missed")

                logged_time = e["predicted_at"][:16].replace("T", " ")
                confidence_pct = max(e["team1_win_prob"], e["team2_win_prob"]) * 100
                st.caption(
                    f"Predicted: **{e['favourite']}** to win ({confidence_pct:.0f}%) · "
                    f"logged {logged_time} UTC"
                )

                if e["actual_result"] is not None:
                    st.markdown(
                        f"**Result:** {e['team1']} {e['actual_result']} {e['team2']} "
                        f"— winner: **{e['actual_winner']}**"
                    )

        st.caption(
            "💡 Tip: commit `predictions_log.json` to git after each session. "
            "The git commit timestamp becomes independent, tamper-evident proof "
            "these predictions were made before kickoff."
        )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚽ 2026 World Cup Hub")
    st.caption("Built with football-data.org + openfootball + scikit-learn")

    st.divider()
    st.subheader("⚙️ Settings")

    auto_refresh = st.toggle("Auto-refresh (2 min)", value=False)
    if auto_refresh:
        st.caption("Page will reload automatically every 2 minutes.")

    if st.button("🔄 Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("📊 Tournament snapshot")
    st.metric("Knockout matches total", len(all_matches))
    st.metric("Completed", len(completed))
    st.metric("Remaining", len(upcoming_matches))

    record = get_track_record()
    if record["total_logged"]:
        st.divider()
        st.subheader("🎯 Track record")
        st.metric("Logged predictions", record["total_logged"])
        if record["accuracy"] is not None:
            st.metric("Accuracy so far", f"{record['accuracy']*100:.0f}%")

    st.divider()
    st.subheader("🔗 Data sources")
    st.markdown(
        "- [football-data.org](https://football-data.org) — live scores\n"
        "- [openfootball/worldcup.json](https://github.com/openfootball/worldcup.json) — results\n"
        "- [martj42/international_results](https://github.com/martj42/international_results) — ML training"
    )

    st.divider()
    st.subheader("🗂️ Portfolio series")
    st.markdown(
        "- Project 1 — Match Outcome Predictor\n"
        "- Project 2 — xG Engine & AI Scout\n"
        "- Project 4 — World Cup RAG Chatbot\n"
        "- **Project 5 — Live 2026 Hub** ← you are here"
    )


# ── Auto-refresh logic ────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(120)
    st.cache_data.clear()
    st.rerun()