"""TFF Fantezi Lig menajer kartları için şeffaf haftalık fırsat hesabı."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .autosub import expected_squad_points
from .config import (
    BUDGET_M,
    CARD_STATE_FILE,
    SEASON_CARD_BUDGET,
    SEASON_MATCHWEEKS,
)
from .optimize import optimize_squad

CARD_USE_THRESHOLDS = {
    "Dört Dörtlük Kaptan": 12.0,
    "Tripleks Kaptan": 6.0,
    "Tüm Takım Sahaya": 8.0,
    "Hücum": 2.0,
}


def default_card_state(season: int | None = None) -> dict[str, Any]:
    return {
        "season": int(season or datetime.now(timezone.utc).year),
        "budget": SEASON_CARD_BUDGET,
        "used": 0,
        "remaining": SEASON_CARD_BUDGET,
        "weeks_left": SEASON_MATCHWEEKS,
        "history": [],
    }


def load_card_state(
    path: Path | str | None = None,
    *,
    season: int | None = None,
) -> dict[str, Any]:
    target = Path(path) if path else CARD_STATE_FILE
    state = default_card_state(season)
    if not target.exists():
        return state
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return state
    if not isinstance(payload, dict):
        return state
    state.update(payload)
    budget = int(state.get("budget") or SEASON_CARD_BUDGET)
    used = int(state.get("used") or 0)
    remaining = state.get("remaining")
    if remaining is None:
        remaining = max(0, budget - used)
    state["budget"] = budget
    state["used"] = used
    state["remaining"] = max(0, min(budget, int(remaining)))
    weeks_left = state.get("weeks_left")
    state["weeks_left"] = (
        SEASON_MATCHWEEKS
        if weeks_left is None
        else max(0, int(weeks_left))
    )
    if not isinstance(state.get("history"), list):
        state["history"] = []
    if season is not None and int(state.get("season") or 0) != int(season):
        return default_card_state(season)
    return state


def save_card_state(state: dict[str, Any], path: Path | str | None = None) -> Path:
    target = Path(path) if path else CARD_STATE_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    budget = max(0, int(payload.get("budget") or SEASON_CARD_BUDGET))
    remaining = max(
        0,
        min(
            budget,
            int(payload.get("remaining", budget - payload.get("used", 0))),
        ),
    )
    payload["budget"] = budget
    payload["remaining"] = remaining
    payload["used"] = budget - remaining
    payload["history"] = (
        payload["history"] if isinstance(payload.get("history"), list) else []
    )
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def set_cards_remaining(
    remaining: int,
    *,
    path: Path | str | None = None,
    season: int | None = None,
    weeks_left: int | None = None,
) -> dict[str, Any]:
    state = load_card_state(path, season=season)
    budget = int(state.get("budget") or SEASON_CARD_BUDGET)
    left = int(remaining)
    if not 0 <= left <= budget:
        raise ValueError(f"Kalan kart sayısı 0-{budget} aralığında olmalı.")
    state["remaining"] = left
    state["used"] = max(0, budget - left)
    if weeks_left is not None:
        state["weeks_left"] = max(0, int(weeks_left))
    if season is not None:
        state["season"] = int(season)
    save_card_state(state, path)
    return state


def record_card_use(
    card: str,
    *,
    path: Path | str | None = None,
    season: int | None = None,
    week: int | None = None,
    weeks_left: int | None = None,
) -> dict[str, Any]:
    card_name = str(card).strip()
    if not card_name:
        raise ValueError("Kaydedilecek kart adı boş olamaz.")
    state = load_card_state(path, season=season)
    remaining = int(state.get("remaining") or 0)
    if remaining <= 0:
        raise ValueError("Kaydedilecek menajer kartı hakkı kalmadı.")
    history = list(state.get("history") or [])
    history.append(
        {
            "card": card_name,
            "week": int(week) if week is not None else None,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    state["history"] = history
    state["remaining"] = remaining - 1
    state["used"] = int(state.get("budget") or SEASON_CARD_BUDGET) - int(
        state["remaining"]
    )
    if weeks_left is not None:
        state["weeks_left"] = max(0, int(weeks_left))
    if season is not None:
        state["season"] = int(season)
    save_card_state(state, path)
    return state


def opportunity_threshold(
    base: float,
    *,
    remaining: int,
    weeks_left: int,
    budget: int = SEASON_CARD_BUDGET,
) -> float:
    """Kalan hak / kalan haftaya göre fırsat maliyeti eşiği."""
    if remaining <= 0:
        return float("inf")
    pace = (SEASON_MATCHWEEKS / max(budget, 1))
    current = (max(weeks_left, 1) / max(remaining, 1))
    scarcity = current / max(pace, 1e-9)
    return float(base) * max(1.0, scarcity)


def manager_card_advice(
    result: dict[str, Any],
    available_players: pd.DataFrame,
    *,
    budget: float = BUDGET_M,
) -> list[dict[str, Any]]:
    """Kartları mevcut haftanın beklenen puan farkına göre sırala.

    Normal kaptan 2x kabul edilir. Bu nedenle Tripleks'in getirisi +1x,
    Dört Dörtlük'ün getirisi +2x kaptan puanıdır.
    """
    xi = result.get("xi")
    bench = result.get("bench")
    captain = result.get("captain") or {}
    if not isinstance(xi, pd.DataFrame) or xi.empty:
        return []
    if not isinstance(bench, pd.DataFrame):
        bench = pd.DataFrame()

    captain_pts = float(
        captain.get("pts_if_plays")
        or captain.get("projected_pts")
        or 0.0
    )
    captain_play = float(captain.get("play_probability") or 0.85)
    captain_ev = captain_pts * captain_play

    base_ev = expected_squad_points(xi, bench)
    full_bench_ev = expected_squad_points(xi, bench, full_bench=True)
    full_bench_extra = max(
        0.0,
        float(full_bench_ev["expected_pts"]) - float(base_ev["expected_pts"]),
    )

    advice = [
        {
            "card": "Dört Dörtlük Kaptan",
            "extra_pts": round(2.0 * captain_ev, 2),
            "why": f"{captain.get('display_name') or captain.get('player') or 'Kaptan'} için 4x",
        },
        {
            "card": "Tripleks Kaptan",
            "extra_pts": round(captain_ev, 2),
            "why": f"{captain.get('display_name') or captain.get('player') or 'Kaptan'} için 3x",
        },
        {
            "card": "Tüm Takım Sahaya",
            "extra_pts": round(full_bench_extra, 2),
            "why": (
                f"Otomatik yedek yerine tüm oynayan yedekler "
                f"+{full_bench_extra:.2f}p"
            ),
        },
    ]

    try:
        attack_result = optimize_squad(available_players, budget=budget + 5.0)
        normal_total = float(result.get("total_projected") or base_ev["expected_pts"] or 0.0)
        attack_total = float(attack_result.get("total_projected") or 0.0)
        advice.append(
            {
                "card": "Hücum",
                "extra_pts": round(max(0.0, attack_total - normal_total), 2),
                "why": "105M ile yeniden optimize edildi; uygulamadaki özel diziliş ayrıca doğrulanmalı",
            }
        )
    except Exception:
        pass

    advice.append(
        {
            "card": "Limitsiz Bütçe",
            "extra_pts": None,
            "why": "Transfer haftasında pahalı hedef için kullan; sabit haftalık puan bonusu yok",
        }
    )
    return sorted(
        advice,
        key=lambda row: float(row["extra_pts"]) if row["extra_pts"] is not None else -1.0,
        reverse=True,
    )


def choose_manager_card(
    advice: list[dict[str, Any]],
    *,
    remaining: int | None = None,
    weeks_left: int | None = None,
    budget: int = SEASON_CARD_BUDGET,
    card_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Haftada en fazla bir kart öner; fırsat zayıfsa tüm kartları sakla."""
    state = card_state or {}
    left = int(
        remaining
        if remaining is not None
        else state.get("remaining", SEASON_CARD_BUDGET)
    )
    weeks = int(
        weeks_left
        if weeks_left is not None
        else state.get("weeks_left", SEASON_MATCHWEEKS)
    )
    if left <= 0:
        return {
            "use": False,
            "card": "Kart kullanma",
            "extra_pts": 0.0,
            "remaining": 0,
            "weeks_left": weeks,
            "why": (
                f"Sezonluk {budget} kart hakkının tamamı kullanılmış; "
                "bu hafta kart önermiyorum."
            ),
        }

    scored: list[tuple[float, float, dict[str, Any]]] = []
    for item in advice:
        base = CARD_USE_THRESHOLDS.get(str(item.get("card") or ""))
        extra = item.get("extra_pts")
        if base and extra is not None:
            threshold = opportunity_threshold(
                base, remaining=left, weeks_left=weeks, budget=budget
            )
            scored.append((float(extra) / threshold, threshold, item))
    if not scored:
        return {
            "use": False,
            "card": "Kart kullanma",
            "extra_pts": 0.0,
            "remaining": left,
            "weeks_left": weeks,
            "why": (
                f"Bu hafta ölçülebilir kart fırsatı yok; "
                f"kalan {left}/{budget} hakkı sonraki haftaya sakla."
            ),
        }

    ratio, threshold, best = max(scored, key=lambda pair: pair[0])
    if ratio < 1.0:
        return {
            "use": False,
            "card": "Kart kullanma",
            "extra_pts": round(float(best["extra_pts"]), 2),
            "candidate": best["card"],
            "remaining": left,
            "weeks_left": weeks,
            "threshold": round(float(threshold), 2),
            "why": (
                f"En iyi aday {best['card']} (+{float(best['extra_pts']):.2f}p), "
                f"ama fırsat maliyeti eşiği +{threshold:.1f}p "
                f"(kalan {left} kart / {weeks} hafta). Bu hafta sakla."
            ),
        }
    return {
        "use": True,
        "card": best["card"],
        "extra_pts": round(float(best["extra_pts"]), 2),
        "remaining": left,
        "weeks_left": weeks,
        "threshold": round(float(threshold), 2),
        "why": (
            f"{best['why']}; +{float(best['extra_pts']):.2f}p fırsat "
            f"+{threshold:.1f}p eşiğini geçti. Kalan hak {left}/{budget}. "
            "Bu hafta yalnız bu kartı kullan."
        ),
    }
