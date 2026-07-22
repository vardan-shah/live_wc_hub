# ⚽ 2026 World Cup — Live Prediction Hub

> A dashboard that tracked the FIFA World Cup 2026 knockout stage in real time, generated ML win probabilities for every match, and logged each prediction with a timestamp — then, once the tournament ended, was audited against its own logs to separate genuine real-time foresight from after-the-fact backtesting.

**Project 5** in a FIFA World Cup analytics portfolio series. **Tournament complete — Spain are 2026 World Cup champions.**

---

## Final results — verified

Two categories of evidence live in this project, and they are not the same thing.

**🔒 Verified in real time** — logged with a timestamp before kickoff:

| Match | Model favoured | Result | Call |
|---|---|---|---|
| France vs Spain (Semi-final) | Spain, 61.8% | Spain won 2–0 | ✅ Correct |
| England vs Argentina (Semi-final) | Argentina, 42.0% | Argentina won 2–1 | ✅ Correct |

**2 for 2.** Both logged a full day or more before kickoff, both correct. The sample is small — too small to claim statistical confidence — but every entry here is independently verifiable: the prediction existed before the outcome did.

**🔄 Full-tournament backtest** — same frozen model (trained only on pre-2026 historical data), applied to matches that had already been played by the time it ran:

| Match | Model favoured | Result | Call |
|---|---|---|---|
| France vs Morocco (QF) | France, 48.2% | France won 2–0 | ✅ Correct |
| Spain vs Belgium (QF) | Belgium, 42.1% | Spain won 2–1 | ❌ Missed |
| Norway vs England (QF) | England, 65.5% | England won 2–1 (AET) | ✅ Correct |
| Argentina vs Switzerland (QF) | Argentina, 38.3% | Argentina won 3–1 (AET) | ✅ Correct |
| France vs England (3rd place) | England, 56.8% | England won 6–4 | ✅ Correct |
| Spain vs Argentina (Final) | Spain, 43.2% | Spain won 1–0 | ✅ Correct |

**5 of 6 correct.** Combined with the two real-time calls: **7 of 8 knockout-stage favourites correctly identified**, with one honest miss.

**Calibration (all 8, Brier score):** 0.242 — close to the 0.25 coin-flip baseline, which is the honest nuance here: directional accuracy (87.5%) was strong, but many of the model's own stated probabilities sat close to 50/50 even when it ended up correct. High-confidence calls (>55%) went 3-for-3; the "closely contested" bucket (<44%) went 3-for-4. The pattern suggests the model was, if anything, slightly *underconfident* rather than overconfident — though every one of these bucket sizes is far too small (n = 1 to 4) to treat as a real statistical finding.

---

## Why the real-time vs backtest distinction exists

Early in this project, running the app for the first time since logging the Semi-final predictions revealed something worth being honest about: four Quarter-finals, the Third Place match, and the Final all got logged with *today's* timestamp — because the app had simply never rendered them before. The model's own computation wasn't compromised (it only ever draws on pre-2026 historical data — there's no channel through which a 2026 result could leak into it), but a timestamp generated after the fact doesn't carry the same evidentiary weight as one generated before.

The fix: every logged prediction now carries a `logged_in_advance` flag, computed automatically by comparing the logging timestamp's date against the match's own date. Nothing here relies on manual bookkeeping — the distinction is structural, not a label added after the fact. The Track Record tab shows both categories separately, on purpose, rather than blending them into one flattering number.

---

## Architecture

```
─── DATA LAYER ─────────────────────────────────────────────────────────

  football-data.org (primary, free key)     openfootball/worldcup.json (fallback)
              │                                          │
              └──────────────── live_data.py ────────────┘
                       (2-min cache · auto-fallback)

─── ML LAYER ───────────────────────────────────────────────────────────

  martj42/international_results (2006–present)
              │
              ▼
  predictor.py
    ├─ Eval split  → honest AUC (train/test on raw records, never leaked)
    └─ Prod refit  → deployed model (fit on 100% of historical data)

─── LOGGING LAYER ──────────────────────────────────────────────────────

  prediction_log.py
    ├─ log_prediction_if_new()      — timestamp on first generation, idempotent
    ├─ logged_in_advance flag       — auto-computed: predicted_at date < match date?
    ├─ update_results()             — backfill actual outcome once FINISHED
    ├─ get_track_record()           — accuracy stats, filterable to advance-only
    └─ get_calibration_summary()    — confidence-bucket breakdown + Brier score

─── APPLICATION ────────────────────────────────────────────────────────

  app.py — 🔴 Final Results · 🎯 Predictions · 🏆 Path to Final · 📊 Track Record
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Live data** | football-data.org v4 API (free tier) + openfootball JSON fallback |
| **ML** | scikit-learn `LogisticRegression`, honest eval + production refit pattern |
| **Prediction logging** | Plain-Python JSON store, self-auditing timestamp integrity |
| **Frontend** | Streamlit — `st.tabs`, `st.progress`, `st.cache_resource`, `st.toast` |
| **Environment** | `python-dotenv` |

---

## Project Structure

```
live_wc_hub/
├── app.py                # Streamlit dashboard — 4 tabs
├── predictor.py           # ML model: honest eval + production refit
├── live_data.py            # Dual-source live data fetcher
├── prediction_log.py        # Timestamped ledger + real-time/backtest split
├── predictions_log.json      # Full tournament ledger — committed to git
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Getting Started

### 1. Install

```bash
git clone https://github.com/vardan-shah/live_wc_hub.git
cd live_wc_hub
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure (optional)

```bash
cp .env.example .env
```

Get a free key at [football-data.org/client/register](https://www.football-data.org/client/register). Without one, the app runs fully on the openfootball fallback.

### 3. Launch

```bash
streamlit run app.py
```

---

## Design Decisions

**Why an honest eval split *and* a production refit?**
Evaluating and deploying the same fitted model creates a subtle temptation to let the "best" split flatter the reported metric. Splitting raw records first, computing strength features only from the training partition, and reporting AUC from an untouched test set gives an honest number. Refitting on all data afterward for actual predictions is standard practice — production inference should use every match available; the reported AUC already told you how much to trust it.

**Why distinguish real-time from backtested predictions at all?**
Because the difference is the entire point of timestamped logging. A model backtested against known outcomes will always look better than it should if presented without that context — not through malice, just through the ordinary human tendency to notice the flattering framing first. Building the distinction into the data structure itself, rather than trusting a write-up to mention it, is what makes the claim auditable by someone who has never met you.

**Why commit `predictions_log.json` to git?**
The internal timestamp is only as trustworthy as the machine that generated it. A git commit adds an independent, third-party-verifiable record that this data existed in this form at this point in time.

---

## Known Limitations

- Streamlit Community Cloud's free tier uses ephemeral storage — `predictions_log.json` may reset on app restart if deployed there. The ledger in this repo is the authoritative one.
- Draw probability is a football-calibrated prior (~25%), not learned from data — the underlying classifier is binary (favourite win / not).
- Confidence-bucket sample sizes (n = 1 to 4 per bucket) are too small to support strong calibration claims — reported as an honest observation, not a validated result.
- Team name matching relies on a manually maintained alias dictionary.

---

## Part of a larger portfolio

| # | Project | Key Technologies |
|---|---|---|
| 1 | Match Outcome Predictor | Logistic Regression, leakage-free evaluation |
| 2 | xG Engine & AI Scout | StatsBomb, XGBoost, SHAP, Claude API |
| 3 | World Cup RAG Chatbot | LangChain, ChromaDB, Groq, Llama-3.3 |
| **5** | **Live 2026 Prediction Hub** | **Live APIs, self-auditing calibration tracking, Streamlit** |

---
