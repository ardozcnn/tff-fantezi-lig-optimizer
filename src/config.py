"""Sabitler ve TFF Fantezi Lig kuralları."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
COOKIE_FILE = DATA_DIR / "tff_cookies.txt"
LOGIN_FILE = DATA_DIR / "tff_login.txt"
PRICES_FILE = DATA_DIR / "prices.csv"

BUDGET_M = 100.0
SQUAD = {"GK": 2, "DF": 5, "MF": 5, "FW": 3}
MAX_PER_CLUB = 3
FORMATIONS: dict[str, dict[str, int]] = {
    "4-4-2": {"GK": 1, "DF": 4, "MF": 4, "FW": 2},
    "4-5-1": {"GK": 1, "DF": 4, "MF": 5, "FW": 1},
    "4-3-3": {"GK": 1, "DF": 4, "MF": 3, "FW": 3},
    "3-5-2": {"GK": 1, "DF": 3, "MF": 5, "FW": 2},
    "3-4-3": {"GK": 1, "DF": 3, "MF": 4, "FW": 3},
    "5-4-1": {"GK": 1, "DF": 5, "MF": 4, "FW": 1},
    "5-3-2": {"GK": 1, "DF": 5, "MF": 3, "FW": 2},
    "5-2-3": {"GK": 1, "DF": 5, "MF": 2, "FW": 3},
}
BENCH_WEIGHT = 0.50
AUTOSUB_MONTE_CARLO_DRAWS = 256

RARE_RATE_KEYS = (
    "gls_pa",
    "ast_pa",
    "xg_pa",
    "xa_pa",
    "cs_rate",
    "ga_pa",
    "yc_pa",
    "rc_pa",
    "pen_save_pa",
    "pen_miss_pa",
    "og_pa",
)
COUNT_RATE_KEYS = (
    "saves_pa",
    "sot_pa",
    "shots_pa",
    "key_passes_pa",
    "bcc_pa",
    "int_p90",
    "tkl_p90",
    "dribbles_pa",
    "share_60",
    "min_per_app",
    "rating",
)
RARE_PRIOR_MATCHES = 10.0
COUNT_PRIOR_MATCHES = 4.0
FORM_PRIOR_MATCHES = 4.0
TFF_EARLY_PRIOR_MATCHES = 8.0
EXTERNAL_OPENER_PRIOR_MATCHES = 2.5
ESTABLISHED_SL_APPS = 8.0
GK_SAVES_PRIOR_MATCHES = 4.0
TEAM_CS_PRIOR_MATCHES = 8.0
EARLY_SEASON_FORM_CAP_APPS = 6.0
FIXTURE_CS_FLOOR = 0.85
FIXTURE_CS_CEILING = 1.20
SEASON_CARD_BUDGET = 10
SEASON_MATCHWEEKS = 34
CARD_STATE_FILE = DATA_DIR / "card_state.json"

AVAIL_MULT = {
    "AVAILABLE": 1.0,
    "DOUBTFUL": 0.55,
    "INJURED": 0.12,
    "SUSPENDED": 0.08,
    "UNAVAILABLE": 0.10,
    "OUT": 0.10,
}

SOT_TO_GOAL = 0.30
KEYPASS_TO_ASSIST = 0.11
BCC_TO_ASSIST = 0.22

FORM_MATCHES = 6
FORM_WEAK_APPS = 4
MIN_FORM_MINUTES_FOR_RATES = 45.0

W_FORM_DEFAULT = 0.30
W_BASE_DEFAULT = 0.70
W_FORM_WEAK = 0.15
W_BASE_WEAK = 0.85

REQUEST_DELAY_S = 0.25


def is_quiet() -> bool:
    import os
    return os.environ.get("TFF_QUIET", "1").strip().lower() not in ("0", "false", "no")

POS_MAP = {
    "GK": "GK",
    "DF": "DF",
    "FB": "DF",
    "CB": "DF",
    "LB": "DF",
    "RB": "DF",
    "WB": "DF",
    "MF": "MF",
    "DM": "MF",
    "CM": "MF",
    "AM": "MF",
    "LM": "MF",
    "RM": "MF",
    "WM": "MF",
    "FW": "FW",
    "ST": "FW",
    "LW": "FW",
    "RW": "FW",
}

GOAL_POINTS = {"GK": 10, "DF": 6, "MF": 5, "FW": 4}
ASSIST_POINTS = 3
CS_POINTS = {"GK": 4, "DF": 4, "MF": 1, "FW": 0}
MIN_60_POINTS = 1
MIN_FULL_POINTS = 2
YELLOW_PENALTY = 1
RED_PENALTY = 3
SAVE_POINTS_PER_3 = 1
CONCEDED_PENALTY_PER_2 = 1
PENALTY_SAVE_POINTS = 5
PENALTY_MISS_PENALTY = 2
OWN_GOAL_PENALTY = 2
BONUS_POINTS = (3, 2, 1)
