"""
Live 2026 World Cup data layer.

Primary source:  football-data.org free tier
  - 10 calls/minute rate limit
  - Requires free API key: https://www.football-data.org/client/register
  - Competition code: WC
  - Set FOOTBALL_DATA_API_KEY in .env

Fallback source: openfootball/worldcup.json on GitHub
  - No API key needed
  - Updated once daily by maintainer
  - Enough for completed results and fixture listing

The app works with EITHER source — if no API key is set, the fallback
activates automatically and all completed results are available. Live
in-match scores require the football-data.org key.
"""

import os
import requests
import streamlit as st
from datetime import date, datetime, timezone

# ── API configuration ─────────────────────────────────────────────────────────

FDORG_BASE  = "https://api.football-data.org/v4"
FDORG_COMP  = "WC"

OPENFOOTBALL_URL = (
    "https://raw.githubusercontent.com/openfootball/"
    "worldcup.json/master/2026/worldcup.json"
)

# Knockout stages we care about (excludes group stage)
KNOCKOUT_STAGES = {
    "ROUND_OF_32", "ROUND_OF_16", "QUARTER_FINALS",
    "SEMI_FINALS", "THIRD_PLACE", "FINAL",
}

# openfootball round names → standard stage codes
_ROUND_MAP: dict[str, str] = {
    "round of 32":       "ROUND_OF_32",
    "round of 16":       "ROUND_OF_16",
    "quarter-finals":    "QUARTER_FINALS",
    "quarter finals":    "QUARTER_FINALS",
    "semi-finals":       "SEMI_FINALS",
    "semi finals":       "SEMI_FINALS",
    "3rd place":         "THIRD_PLACE",
    "third place":       "THIRD_PLACE",
    "final":             "FINAL",
}

# Human-readable stage labels
STAGE_DISPLAY: dict[str, str] = {
    "ROUND_OF_32":  "Round of 32",
    "ROUND_OF_16":  "Round of 16",
    "QUARTER_FINALS": "Quarter-finals",
    "SEMI_FINALS":  "Semi-finals",
    "THIRD_PLACE":  "Third place play-off",
    "FINAL":        "Final",
}

# Ordering for sorting by tournament stage
STAGE_ORDER: dict[str, int] = {
    "ROUND_OF_32": 1, "ROUND_OF_16": 2, "QUARTER_FINALS": 3,
    "SEMI_FINALS": 4, "THIRD_PLACE": 5, "FINAL": 6,
}


# ── Public API ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def get_all_knockout_matches() -> list[dict]:
    """
    Fetch all knockout stage matches — completed and upcoming.

    Cached for 2 minutes to respect football-data.org rate limits.
    Returns list of normalised match dicts (same schema for both sources).

    Match dict schema:
        stage:    str  (e.g. "QUARTER_FINALS")
        date:     str  YYYY-MM-DD
        time:     str  HH:MM UTC
        team1:    str  (home team name)
        team2:    str  (away team name)
        score1:   int | None
        score2:   int | None
        status:   str  (SCHEDULED | LIVE | PAUSED | FINISHED | TIMED)
        source:   str
    """
    api_key = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()

    if api_key:
        matches = _fetch_fdorg(api_key)
        if matches:
            return sorted(
                matches,
                key=lambda m: (STAGE_ORDER.get(m["stage"], 9), m["date"], m["time"])
            )

    # Fallback — no key or API call failed
    matches = _fetch_openfootball()
    return sorted(
        matches,
        key=lambda m: (STAGE_ORDER.get(m["stage"], 9), m["date"], m["time"])
    )


def get_today_matches(all_matches: list[dict]) -> list[dict]:
    today = date.today().isoformat()
    return [m for m in all_matches if m["date"] == today]


def get_upcoming_matches(all_matches: list[dict]) -> list[dict]:
    today = date.today().isoformat()
    return [
        m for m in all_matches
        if m["date"] >= today and m["status"] not in ("FINISHED", "AWARDED")
    ]


def get_completed_matches(all_matches: list[dict]) -> list[dict]:
    return [m for m in all_matches if m["status"] in ("FINISHED", "AWARDED")]


def is_match_live(match: dict) -> bool:
    return match["status"] in ("IN_PLAY", "PAUSED", "LIVE")


def has_api_key() -> bool:
    return bool(os.getenv("FOOTBALL_DATA_API_KEY", "").strip())


# ── Private helpers ───────────────────────────────────────────────────────────

def _fetch_fdorg(api_key: str) -> list[dict]:
    """Fetch knockout matches from football-data.org v4."""
    try:
        resp = requests.get(
            f"{FDORG_BASE}/competitions/{FDORG_COMP}/matches",
            headers={"X-Auth-Token": api_key},
            timeout=10,
        )

        if resp.status_code == 429:
            st.warning("Rate limit hit on football-data.org — using cached data.")
            return []

        resp.raise_for_status()
        data = resp.json()

        matches = []
        for m in data.get("matches", []):
            stage = m.get("stage", "")
            if stage not in KNOCKOUT_STAGES:
                continue
            matches.append(_normalize_fdorg(m))

        return matches

    except requests.exceptions.RequestException as exc:
        st.caption(f"⚠ football-data.org unreachable ({exc.__class__.__name__}) — using fallback.")
        return []
    except Exception as exc:
        st.caption(f"⚠ Unexpected error fetching live data: {exc}")
        return []


def _fetch_openfootball() -> list[dict]:
    """Fetch knockout matches from openfootball/worldcup.json."""
    try:
        resp = requests.get(OPENFOOTBALL_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        matches = []
        for rnd in data.get("rounds", []):
            stage = _detect_stage(rnd.get("name", ""))
            if stage is None:
                continue

            for m in rnd.get("matches", []):
                score = m.get("score", {})
                ft = score.get("ft")   # [goals1, goals2] or None
                et = score.get("et")   # extra time scores
                pen = score.get("pen") # penalty scores

                # Determine winner from actual result
                if ft:
                    status = "FINISHED"
                    s1, s2 = ft[0], ft[1]
                else:
                    status = "SCHEDULED"
                    s1, s2 = None, None

                matches.append({
                    "stage":  stage,
                    "date":   m.get("date", ""),
                    "time":   m.get("time", ""),
                    "team1":  _safe_name(m, "team1"),
                    "team2":  _safe_name(m, "team2"),
                    "score1": s1,
                    "score2": s2,
                    "et1":    et[0] if et else None,
                    "et2":    et[1] if et else None,
                    "pen1":   pen[0] if pen else None,
                    "pen2":   pen[1] if pen else None,
                    "status": status,
                    "source": "openfootball",
                })

        return matches

    except requests.exceptions.RequestException as exc:
        st.error(
            f"Could not fetch match data ({exc.__class__.__name__}). "
            "Check your internet connection."
        )
        return []
    except Exception as exc:
        st.error(f"Unexpected error parsing match data: {exc}")
        return []


def _normalize_fdorg(m: dict) -> dict:
    """Convert football-data.org match dict to standard schema."""
    score = m.get("score", {})
    ft    = score.get("fullTime", {})
    utc   = m.get("utcDate", "")

    return {
        "stage":  m.get("stage", "UNKNOWN"),
        "date":   utc[:10]   if utc else "",
        "time":   utc[11:16] if len(utc) > 10 else "",
        "team1":  m.get("homeTeam", {}).get("name", "TBD"),
        "team2":  m.get("awayTeam", {}).get("name", "TBD"),
        "score1": ft.get("home"),
        "score2": ft.get("away"),
        "et1":    None,
        "et2":    None,
        "pen1":   None,
        "pen2":   None,
        "status": m.get("status", "SCHEDULED"),
        "source": "football-data.org",
    }


def _detect_stage(round_name: str) -> str | None:
    """Map openfootball round name string to stage code."""
    lower = round_name.lower().strip()
    for key, stage in _ROUND_MAP.items():
        if key in lower:
            return stage
    return None


def _safe_name(match_dict: dict, key: str) -> str:
    val = match_dict.get(key)
    if isinstance(val, dict):
        return val.get("name", "TBD")
    return str(val) if val else "TBD"
