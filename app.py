"""
Project 5 — 2026 World Cup Live Prediction Hub (post-tournament edition)

Four tabs:
  🔴 Final Results  — how the knockout stage actually resolved
  🎯 Predictions    — every logged prediction, tournament complete
  🏆 Path to Final  — the full bracket as it played out
  📊 Track Record   — real-time verified vs backtested, clearly separated

Data: football-data.org (primary, requires free key) → openfootball (fallback)
ML:   Logistic regression, honest eval split + production refit on all data
"""

import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

from live_data import (
    get_all_knockout_matches,
    get_completed_matches,
    has_api_key,
    STAGE_DISPLAY,
    STAGE_ORDER,
)
from predictor import load_model, predict_match
from prediction_log import (
    log_prediction_if_new,
    update_results,
    get_track_record,
    get_calibration_summary,
)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="2026 World Cup Hub", page_icon="⚽", layout="wide")


@st.cache_resource(show_spinner=False)
def get_predictor():
    return load_model()

with st.spinner("Training ML predictor on historical international data…"):
    _, _, _, _, auc = get_predictor()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚽ 2026 World Cup — Live Prediction Hub")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tournament", "FIFA World Cup 2026")
c2.metric("Champion", "🏆 Spain")
c3.metric("ML model AUC", f"{auc:.3f}")
c4.metric("Data source", "football-data.org" if has_api_key() else "openfootball ↗")
st.divider()


# ── Fetch data ─────────────────────────────────────────────────────────────────
with st.spinner("Fetching tournament data…"):
    all_matches = get_all_knockout_matches()

completed = get_completed_matches(all_matches)

newly_resolved = update_results(all_matches)
if newly_resolved:
    st.toast(f"📬 {newly_resolved} logged prediction(s) just resolved!", icon="🎯")


# ── Helper components ─────────────────────────────────────────────────────────

def _stage_label(stage: str) -> str:
    return STAGE_DISPLAY.get(stage, stage)


def _render_score(m: dict) -> None:
    with st.container(border=True):
        badge_col, _, date_col = st.columns([2, 3, 2])
        with badge_col:
            st.markdown(f"✅ *{_stage_label(m['stage'])}*")
        with date_col:
            st.caption(f"{m['date']}  {m['time']} UTC" if m["time"] else m["date"])

        t1_col, mid_col, t2_col = st.columns([3, 1, 3])
        with t1_col:
            st.markdown(f"### {m['team1']}")
        with mid_col:
            if m["score1"] is not None:
                penalty_note = ""
                if m.get("pen1") is not None:
                    penalty_note = (
                        f"<div style='text-align:center;font-size:11px;"
                        f"color:var(--text-muted)'>({m['pen1']}–{m['pen2']} pens)</div>"
                    )
                st.markdown(
                    f"<div style='text-align:center;font-size:22px;font-weight:500'>"
                    f"{m['score1']} – {m['score2']}</div>{penalty_note}",
                    unsafe_allow_html=True,
                )
        with t2_col:
            st.markdown(f"<div style='text-align:right'><h3>{m['team2']}</h3></div>", unsafe_allow_html=True)


def _render_prediction(m: dict) -> None:
    """Render a prediction card and log it — resolving immediately if the
    match is already finished (handles first-time logging of past matches
    within the same run, rather than waiting for a second page load)."""
    t1, t2 = m["team1"], m["team2"]
    if "TBD" in (t1, t2) or not t1 or not t2:
        return

    pred = predict_match(t1, t2, neutral=True)
    was_new = log_prediction_if_new(m, pred)
    if was_new:
        # FIX: if this match already finished (e.g. logging a Quarter-final
        # for the first time weeks after it was played), resolve it right
        # away instead of leaving it null until the next rerun.
        update_results(all_matches)

    with st.container(border=True):
        head_left, head_right = st.columns([3, 1])
        with head_left:
            st.markdown(f"**{t1}** vs **{t2}** · *{_stage_label(m['stage'])} · {m['date']}*")
        with head_right:
            conf_color = {"High": "🟢", "Moderate": "🟡", "Closely contested": "🟠"}.get(pred["confidence"], "⚪")
            st.caption(f"{conf_color} {pred['confidence']}")

        st.markdown(f"**Model favoured:** {pred['favourite']} — *historical win rates since 2006*")
        p1, pd_, p2 = pred["team1_win"], pred["draw"], pred["team2_win"]
        bar1, bar2, bar3 = st.columns([3, 2, 3])
        with bar1:
            st.caption(t1); st.progress(p1, text=f"{p1*100:.0f}%")
        with bar2:
            st.caption("Draw"); st.progress(pd_, text=f"{pd_*100:.0f}%")
        with bar3:
            st.caption(t2); st.progress(p2, text=f"{p2*100:.0f}%")


def _render_ledger_entry(e: dict) -> None:
    """Shared ledger row — used in both the real-time and backtest sections."""
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
        advance_tag = "🔒 real-time" if e.get("logged_in_advance") else "🔄 backtest"
        st.caption(
            f"Predicted: **{e['favourite']}** to win ({confidence_pct:.0f}%) · "
            f"logged {logged_time} UTC · {advance_tag}"
        )
        if e["actual_result"] is not None:
            st.markdown(f"**Result:** {e['team1']} {e['actual_result']} {e['team2']} — winner: **{e['actual_winner']}**")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔴 Final Results", "🎯 Predictions", "🏆 Path to Final", "📊 Track Record"]
)

with tab1:
    now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC — %d %b %Y")
    st.caption(f"Last updated: {now_utc}")
    st.success("Tournament complete — Spain are 2026 World Cup champions.", icon="🏆")
    for m in sorted(completed, key=lambda x: STAGE_ORDER.get(x["stage"], 9)):
        _render_score(m)

with tab2:
    st.markdown(
        "Win probabilities from a **logistic regression model**, evaluated on a held-out "
        f"split before being refit on all available data. Honest test-set AUC: **{auc:.3f}**."
    )
    st.caption("📌 Predictions logged with today's date on first render were computed after their match already happened — see Track Record for which is which.")

    stages_seen: dict[str, list] = {}
    for m in all_matches:
        stages_seen.setdefault(m["stage"], []).append(m)

    for stage in sorted(stages_seen.keys(), key=lambda s: STAGE_ORDER.get(s, 9)):
        st.subheader(_stage_label(stage))
        for m in stages_seen[stage]:
            _render_prediction(m)
        st.divider()

with tab3:
    st.markdown("The full bracket as it actually resolved.")
    for stage in sorted(set(m["stage"] for m in all_matches), key=lambda s: STAGE_ORDER.get(s, 9)):
        st.subheader(_stage_label(stage))
        for m in [x for x in all_matches if x["stage"] == stage]:
            _render_score(m)
        st.divider()

with tab4:
    st.markdown(
        "Two categories live here: predictions **genuinely logged before** their match "
        "was played, and predictions **computed afterward** using the same frozen model, "
        "for additional backtesting. Only the first is real proof of foresight."
    )

    # ── Verified in real time — the gold standard, however small ──────────
    st.subheader("🔒 Verified in real time")
    st.caption("Timestamped before kickoff. This is the evidence that can't be faked after the fact.")

    advance = get_track_record(advance_only=True)
    ac1, ac2, ac3 = st.columns(3)
    ac1.metric("Predictions", advance["total_logged"])
    ac2.metric("Resolved", advance["resolved"])
    ac3.metric("Accuracy", f"{advance['accuracy']*100:.0f}%" if advance["accuracy"] is not None else "—")

    if 0 < advance["resolved"] < 5:
        st.caption(
            f"⚠️ n = {advance['resolved']} — too small to draw statistical conclusions from, "
            "but every one of these is genuine, independently verifiable evidence."
        )

    for e in reversed(advance["entries"]):
        _render_ledger_entry(e)

    st.divider()

    # ── Full backtest — larger sample, still honest, not real-time proof ──
    full = get_track_record(advance_only=False)
    backtest_entries = [e for e in full["entries"] if not e.get("logged_in_advance")]

    st.subheader("🔄 Full-tournament backtest")
    st.caption(
        "Same frozen model — trained only on pre-2026 historical data — applied to "
        "matches that had already been played by the time this ran. Supplementary "
        "validation, not real-time proof."
    )

    if full["resolved"] >= 3:
        calib = get_calibration_summary(advance_only=False)
        cal_cols = st.columns(3)
        for col, label in zip(cal_cols, CONFIDENCE_BUCKETS := ["High", "Moderate", "Closely contested"]):
            stats = calib["buckets"][label]
            with col:
                st.markdown(f"**{label}**")
                if stats["n"] == 0:
                    st.caption("No resolved predictions in this tier")
                else:
                    st.metric("Actual win rate", f"{stats['actual_accuracy']*100:.0f}%")
                    st.caption(f"n = {stats['n']} · model said {stats['avg_confidence']*100:.0f}% on average")

        if calib["brier_score"] is not None:
            st.divider()
            bc1, bc2 = st.columns([1, 2])
            bc1.metric("Brier score", f"{calib['brier_score']:.3f}")
            bc2.caption("0.00 = perfect calibration · 0.25 = coin-flip guessing · 1.00 = worst. Lower is better.")

    st.divider()
    for e in reversed(backtest_entries):
        _render_ledger_entry(e)

    st.caption(
        "💡 Commit `predictions_log.json` to git. The git commit timestamp is "
        "independent, tamper-evident proof of when each entry was written."
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚽ 2026 World Cup Hub")
    st.caption("football-data.org + openfootball + scikit-learn")
    st.divider()

    if st.button("🔄 Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("📊 Final snapshot")
    st.metric("Knockout matches", len(all_matches))
    st.metric("Champion", "🇪🇸 Spain")

    adv = get_track_record(advance_only=True)
    if adv["total_logged"]:
        st.divider()
        st.subheader("🔒 Real-time record")
        st.metric("Verified predictions", adv["total_logged"])
        if adv["accuracy"] is not None:
            st.metric("Accuracy", f"{adv['accuracy']*100:.0f}%")

    st.divider()
    st.subheader("🗂️ Portfolio series")
    st.markdown(
        "- Project 1 — Match Outcome Predictor\n"
        "- Project 2 — xG Engine & AI Scout\n"
        "- Project 4 — World Cup RAG Chatbot\n"
        "- **Project 5 — Live 2026 Hub** ← you are here"
    )