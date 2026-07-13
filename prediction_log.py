"""
Prediction logging and calibration tracking for the 2026 World Cup Hub.

Why this exists:
  A prediction only proves something if it was made BEFORE the result was
  known. This module timestamps every prediction the moment it is first
  generated, then backfills the actual result once a match finishes. That
  turns "the model said 60%" into evidence rather than a retrospective claim.

Storage: local JSON file `predictions_log.json` — one entry per unique match
(stage + date + team pairing). Never overwrites an existing logged
prediction, even if predict_match() runs again for the same fixture on a
later page load — the original pre-match timestamp is permanent.

No Streamlit dependency here on purpose — this module is plain Python and
testable on its own.
"""

import json
import os
from datetime import datetime, timezone

LOG_PATH = "predictions_log.json"


def _match_key(m: dict) -> str:
    """Unique, stable key per match."""
    return f"{m['stage']}::{m['date']}::{m['team1']}_vs_{m['team2']}"


def _load_log() -> dict:
    if not os.path.exists(LOG_PATH):
        return {}
    try:
        with open(LOG_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_log(log: dict) -> None:
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def log_prediction_if_new(match: dict, prediction: dict) -> bool:
    """
    Record a prediction ONLY if this exact match has never been logged.

    Returns True if a new entry was written, False if it already existed
    (meaning the original pre-match timestamp was preserved, untouched).
    """
    if "TBD" in (match["team1"], match["team2"]) or not match["team1"] or not match["team2"]:
        return False

    log = _load_log()
    key = _match_key(match)

    if key in log:
        return False

    log[key] = {
        "stage":          match["stage"],
        "date":           match["date"],
        "team1":          match["team1"],
        "team2":          match["team2"],
        "predicted_at":   datetime.now(timezone.utc).isoformat(),
        "team1_win_prob": prediction["team1_win"],
        "draw_prob":      prediction["draw"],
        "team2_win_prob": prediction["team2_win"],
        "favourite":      prediction["favourite"],
        "confidence":     prediction["confidence"],
        "actual_result":  None,
        "actual_winner":  None,
        "correct":        None,
    }
    _save_log(log)
    return True


def update_results(all_matches: list[dict]) -> int:
    """
    Check logged predictions against current match data. For any logged
    match that has now FINISHED, backfill the actual result.

    Returns the number of entries newly resolved in this call — useful for
    surfacing a "X predictions just resolved" notification in the UI.
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


def get_track_record() -> dict:
    """
    Compute summary statistics across all logged predictions.

    Returns:
        {
            "total_logged": int,
            "resolved":     int,
            "pending":      int,
            "correct":      int,
            "accuracy":     float | None,
            "entries":      list[dict]  (sorted by date, oldest first)
        }
    """
    log = _load_log()
    entries = sorted(log.values(), key=lambda e: e["date"])

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