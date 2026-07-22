"""
Prediction logging and calibration tracking for the 2026 World Cup Hub.

Why this exists:
  A prediction only proves something if it was made BEFORE the result was
  known. This module timestamps every prediction the moment it is first
  generated, then backfills the actual result once a match finishes.

Critical distinction — logged_in_advance:
  If the app only runs occasionally rather than every day, a match that
  was "upcoming" the first time you ran it can become "already finished"
  by the next time you run it. When that happens, _render_prediction()
  logs it for the FIRST time with TODAY's timestamp — even though the
  match itself is long over. That entry is a legitimate backtest (the
  model never saw the 2026 result; it only uses pre-tournament historical
  data), but it is NOT proof the prediction existed before the outcome was
  known. logged_in_advance = True only when predicted_at's date is
  strictly before the match's own date. Everything else is labeled as a
  backtest, not a real-time call — this distinction is computed
  automatically, not something you have to track by hand.

Storage: local JSON file `predictions_log.json` — one entry per unique
match. Never overwrites an existing logged prediction.

No Streamlit dependency here on purpose — this module is plain Python and
testable on its own.
"""

import json
import os
from datetime import datetime, timezone

LOG_PATH = "predictions_log.json"

CONFIDENCE_BUCKETS = ["High", "Moderate", "Closely contested"]


def _match_key(m: dict) -> str:
    return f"{m['stage']}::{m['date']}::{m['team1']}_vs_{m['team2']}"


def _infer_advance_flag(entry: dict) -> bool:
    """
    For entries logged before this field existed: infer it by comparing the
    date portion of predicted_at against the match date. Strictly earlier
    date = genuinely advance. Same-day or later = NOT verifiably advance —
    kickoff time within the day is unknown, so same-day is treated
    conservatively as a backtest, not a real-time call.
    """
    predicted_date = entry["predicted_at"][:10]
    return predicted_date < entry["date"]


def _load_log() -> dict:
    if not os.path.exists(LOG_PATH):
        return {}
    try:
        with open(LOG_PATH, "r") as f:
            log = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    # Migrate any entries created before logged_in_advance existed
    migrated = False
    for entry in log.values():
        if "logged_in_advance" not in entry:
            entry["logged_in_advance"] = _infer_advance_flag(entry)
            migrated = True
    if migrated:
        _save_log(log)

    return log


def _save_log(log: dict) -> None:
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def log_prediction_if_new(match: dict, prediction: dict) -> bool:
    """
    Record a prediction ONLY if this exact match has never been logged.

    Returns True if a new entry was written, False if it already existed.
    """
    if "TBD" in (match["team1"], match["team2"]) or not match["team1"] or not match["team2"]:
        return False

    log = _load_log()
    key = _match_key(match)
    if key in log:
        return False

    now = datetime.now(timezone.utc)
    logged_in_advance = now.date().isoformat() < match["date"]

    log[key] = {
        "stage":             match["stage"],
        "date":              match["date"],
        "team1":             match["team1"],
        "team2":             match["team2"],
        "predicted_at":      now.isoformat(),
        "logged_in_advance": logged_in_advance,
        "team1_win_prob":    prediction["team1_win"],
        "draw_prob":         prediction["draw"],
        "team2_win_prob":    prediction["team2_win"],
        "favourite":         prediction["favourite"],
        "confidence":        prediction["confidence"],
        "actual_result":     None,
        "actual_winner":     None,
        "correct":           None,
    }
    _save_log(log)
    return True


def update_results(all_matches: list[dict]) -> int:
    """
    Check logged predictions against current match data. For any logged
    match that has now FINISHED, backfill the actual result.

    Returns the number of entries newly resolved in this call.
    """
    log = _load_log()
    resolved_now = 0

    finished_by_key = {
        _match_key(m): m for m in all_matches
        if m["status"] in ("FINISHED", "AWARDED")
    }

    for key, entry in log.items():
        if entry["actual_result"] is not None:
            continue
        if key not in finished_by_key:
            continue

        m = finished_by_key[key]
        s1, s2 = m["score1"], m["score2"]
        if s1 is None or s2 is None:
            continue

        if s1 > s2:
            winner = m["team1"]
        elif s2 > s1:
            winner = m["team2"]
        elif m.get("pen1") is not None:
            winner = m["team1"] if m["pen1"] > m["pen2"] else m["team2"]
        else:
            winner = "Draw"

        entry["actual_result"] = f"{s1}-{s2}"
        entry["actual_winner"] = winner
        entry["correct"]       = (winner == entry["favourite"])
        resolved_now += 1

    if resolved_now:
        _save_log(log)

    return resolved_now


def get_track_record(advance_only: bool = False) -> dict:
    """
    Compute summary statistics across logged predictions.

    Args:
        advance_only: if True, restrict to entries genuinely logged before
                      their match was played — the gold-standard subset.

    Returns:
        {
            "total_logged": int, "resolved": int, "pending": int,
            "correct": int, "accuracy": float | None,
            "entries": list[dict]  (sorted by date, oldest first)
        }
    """
    log = _load_log()
    entries = sorted(log.values(), key=lambda e: e["date"])

    if advance_only:
        entries = [e for e in entries if e.get("logged_in_advance")]

    resolved = [e for e in entries if e["actual_result"] is not None]
    correct  = [e for e in resolved if e["correct"]]
    accuracy = (len(correct) / len(resolved)) if resolved else None

    return {
        "total_logged": len(entries),
        "resolved":     len(resolved),
        "pending":      len(entries) - len(resolved),
        "correct":      len(correct),
        "accuracy":     accuracy,
        "entries":      entries,
    }


def get_calibration_summary(advance_only: bool = False) -> dict:
    """
    Break resolved predictions into confidence tiers and check whether the
    model's stated confidence matches its actual accuracy. Also computes
    the Brier score: mean squared error between the predicted probability
    of the favourite and whether it actually won (1 or 0).
        0.00 = perfect calibration · 0.25 = coin-flip guessing · 1.00 = worst

    Args:
        advance_only: restrict to genuinely advance-logged entries only.

    Returns:
        {"buckets": {...}, "brier_score": float | None, "n_resolved": int}
    """
    log = _load_log()
    resolved = [e for e in log.values() if e["actual_result"] is not None]
    if advance_only:
        resolved = [e for e in resolved if e.get("logged_in_advance")]

    buckets: dict[str, list] = {b: [] for b in CONFIDENCE_BUCKETS}
    for e in resolved:
        buckets.setdefault(e.get("confidence", "Closely contested"), []).append(e)

    bucket_stats = {}
    for label in CONFIDENCE_BUCKETS:
        entries = buckets.get(label, [])
        if not entries:
            bucket_stats[label] = {"n": 0, "avg_confidence": None, "actual_accuracy": None}
            continue
        avg_conf = sum(max(e["team1_win_prob"], e["team2_win_prob"]) for e in entries) / len(entries)
        acc      = sum(1 for e in entries if e["correct"]) / len(entries)
        bucket_stats[label] = {"n": len(entries), "avg_confidence": avg_conf, "actual_accuracy": acc}

    brier = None
    if resolved:
        squared_errors = [
            (max(e["team1_win_prob"], e["team2_win_prob"]) - (1.0 if e["correct"] else 0.0)) ** 2
            for e in resolved
        ]
        brier = sum(squared_errors) / len(squared_errors)

    return {"buckets": bucket_stats, "brier_score": brier, "n_resolved": len(resolved)}