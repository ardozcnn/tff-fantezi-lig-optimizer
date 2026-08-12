"""Sabitler ve TFF Fantezi Lig kuralları."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
COOKIE_FILE = DATA_DIR / "tff_cookies.txt"
LOGIN_FILE = DATA_DIR / "tff_login.txt"
PRICES_FILE = DATA_DIR / "prices.csv"

# Kadro (TFF 15'li)
BUDGET_M = 100.0
SQUAD = {"GK": 2, "DF": 5, "MF": 5, "FW": 3}
MAX_PER_CLUB = 3
# İlk 11: en az 1 KL, 3 DF, 1 FV (resmi). Kalan mevki 2-5-5-3 kadroya sığmalı.
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
BENCH_WEIGHT = 0.50  # otomatik yedek girişi; 4 yedeğin hepsi her hafta girmez
NEW_SIGNING_MULT = 1.0
LOW_SAMPLE_MULT = 1.0

# TFF availabilityStatus
AVAIL_MULT = {
    "AVAILABLE": 1.0,
    "DOUBTFUL": 0.55,
    "INJURED": 0.12,
    "SUSPENDED": 0.08,
    "UNAVAILABLE": 0.10,
    "OUT": 0.10,
}

# Şut / kilit pas → gol-asist vekili (xG yoksa)
SOT_TO_GOAL = 0.30
KEYPASS_TO_ASSIST = 0.11
BCC_TO_ASSIST = 0.22

# Form penceresi
FORM_MATCHES = 6
MIN_APPS_FOR_CURRENT_BASE = 8
FORM_WEAK_APPS = 4

W_FORM_DEFAULT = 0.45
W_BASE_DEFAULT = 0.55
W_FORM_WEAK = 0.30
W_BASE_WEAK = 0.70

REQUEST_DELAY_S = 0.25


def is_quiet() -> bool:
    import os
    return os.environ.get("TFF_QUIET", "1").strip().lower() not in ("0", "false", "no")

# Mevki haritası (FBref pos → TFF)
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

# TFF gol puanları mevkiye göre
GOAL_POINTS = {"GK": 10, "DF": 6, "MF": 5, "FW": 4}
ASSIST_POINTS = 3
CS_POINTS = {"GK": 4, "DF": 4, "MF": 1, "FW": 0}
MIN_60_POINTS = 1  # herhangi bir süre: 1; 60+'dan fazla: 2
MIN_FULL_POINTS = 2
YELLOW_PENALTY = 1
RED_PENALTY = 3
SAVE_POINTS_PER_3 = 1
CONCEDED_PENALTY_PER_2 = 1  # GK/DF her 2 gol → -1
PENALTY_SAVE_POINTS = 5
PENALTY_MISS_PENALTY = 2
OWN_GOAL_PENALTY = 2
BONUS_POINTS = (3, 2, 1)  # maçın en yüksek 3 oyuncusu
