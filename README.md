# ⚽ 2026 World Cup — Live Prediction Hub

> A live dashboard tracking the FIFA World Cup 2026 knockout stage with real-time scores, ML-generated win probabilities, and a **timestamped prediction ledger** that proves model calibration rather than claiming it.

**Project 5** in a FIFA World Cup analytics portfolio series.

---

## What this project demonstrates

| Skill Area | What was built |
|---|---|
| **Live Data Engineering** | Dual-source pipeline with automatic fallback (football-data.org → openfootball) |
| **ML Rigor** | Honest train/test evaluation split, decoupled from a separate production refit |
| **Model Validation** | Timestamped prediction logging — calibration evidence, not retrospective claims |
| **Full-Stack** | Four-tab Streamlit dashboard: live scores, predictions, bracket view, track record |

---

## Why the prediction ledger matters

Any model can claim to be well-calibrated *after* the fact. This project logs every prediction — team names, probabilities, confidence, and a UTC timestamp — **the moment it is generated, before the match is played.** Once a match finishes, the actual result is backfilled against that pre-existing entry. The `predictions_log.json` file is committed to git alongside the code, so the git commit history itself is independent, tamper-evident proof of when each prediction was made.

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
    ├─ log_prediction_if_new()  — timestamp on first generation, idempotent
    ├─ update_results()          — backfill actual outcome once FINISHED
    └─ get_track_record()        — accuracy stats across resolved predictions

─── APPLICATION ────────────────────────────────────────────────────────

  app.py — 🔴 Live Scores · 🎯 Predictions · 🏆 Path to Final · 📊 Track Record
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Live data** | football-data.org v4 API (free tier) + openfootball JSON fallback |
| **ML** | scikit-learn `LogisticRegression`, honest eval + production refit pattern |
| **Prediction logging** | Plain-Python JSON store, no external DB dependency |
| **Frontend** | Streamlit — `st.tabs`, `st.progress`, `st.cache_resource`, `st.toast` |
| **Environment** | `python-dotenv` |

---

## Project Structure

```
live_wc_hub/
├── app.py               # Streamlit dashboard — 4 tabs
├── predictor.py          # ML model: honest eval + production refit
├── live_data.py           # Dual-source live data fetcher
├── prediction_log.py       # Timestamped prediction ledger + calibration
├── predictions_log.json     # Generated at runtime — commit this to git
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

### 2. Configure (optional but recommended)

```bash
cp .env.example .env
```

Get a free key at [football-data.org/client/register](https://www.football-data.org/client/register) and add it to `.env`. Without a key, the app runs fully on the openfootball fallback — completed results still work, live in-match scores do not.

### 3. Launch

```bash
streamlit run app.py
```

---

## Design Decisions

**Why an honest eval split *and* a production refit?**
Evaluating and deploying the same fitted model creates a subtle temptation to let the "best" split flatter the reported metric. Splitting raw records first, computing strength features only from the training partition, and reporting AUC from an untouched test set gives an honest number. Refitting on all data afterward for actual 2026 predictions is standard practice — production inference should use every match available; the reported AUC already told you how much to trust it.

**Why JSON instead of a database for prediction logging?**
This is a single-user portfolio tool, not a multi-tenant service. A JSON file keyed by match is simpler, has zero setup cost, and — critically — is human-readable in a git diff, which makes the "predicted before the result" story visible directly in the commit history.

**Why commit `predictions_log.json` to git instead of ignoring it?**
The internal timestamp alone is only as trustworthy as the machine that generated it. A git commit adds an independent, third-party-verifiable timestamp (GitHub's own commit metadata) that the prediction existed before the outcome was known.

---

## Known Limitations

- Streamlit Community Cloud's free tier uses ephemeral storage — `predictions_log.json` may reset on app restart if deployed there. Run locally to build the ledger; deploy to cloud for a shareable live demo.
- Draw probability is a football-calibrated prior (~25%), not learned from data — logistic regression here is binary (home win / not).
- Team name matching relies on a manually maintained alias dictionary; a nation with an unusual API naming convention falls back to the global historical average rather than failing.

---

## Part of a larger portfolio

| # | Project | Key Technologies |
|---|---|---|
| 1 | Match Outcome Predictor | Logistic Regression, leakage-free evaluation |
| 2 | xG Engine & AI Scout | StatsBomb, XGBoost, SHAP, Claude API |
| 4 | World Cup RAG Chatbot | LangChain, ChromaDB, Groq, Llama-3.3 |
| **5** | **Live 2026 Prediction Hub** | **Live APIs, calibration tracking, Streamlit** |

---
