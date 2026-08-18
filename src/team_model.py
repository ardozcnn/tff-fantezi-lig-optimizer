"""Süper Lig maç modeli: Poisson hücum/savunma + oyuncu gol payı.

AIrsenal'deki takım-oyuncu ayrımının JAX/NumPyro olmadan, TFF puanına
bağlı sade hali. Skor tahmini üretmez; beklenen gol ve clean sheet
olasılığını fantezi puanına çevirir.
"""

from __future__ import annotations

import math
import time
from typing import Any

from .config import (
    FIXTURE_ATTACK_CEILING,
    FIXTURE_ATTACK_FLOOR,
    FIXTURE_CS_CEILING,
    FIXTURE_CS_FLOOR,
    HORIZON_WEEKS,
    MATCH_MODEL_BLEND,
    SAVE_MULT_CEILING,
    SAVE_MULT_FLOOR,
)
from .names import normalize_name

LEAGUE_GOAL_RATE = 1.35
LEAGUE_CS_RATE = 0.28
HOME_ADVANTAGE = 1.28
PREV_SEASON_WEIGHT = 0.50
RECENCY_DECAY = 0.007


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def event_goals(event: dict[str, Any]) -> tuple[int, int] | None:
    home = event.get("homeScore") or {}
    away = event.get("awayScore") or {}
    hs = home.get("current")
    ag = away.get("current")
    if hs is None:
        hs = home.get("display")
    if ag is None:
        ag = away.get("display")
    try:
        return int(hs), int(ag)
    except (TypeError, ValueError):
        return None


def match_weight(timestamp: float | None, *, season_boost: float = 1.0) -> float:
    if not timestamp:
        return max(0.05, float(season_boost))
    days = max(0.0, (time.time() - float(timestamp)) / 86400.0)
    return max(0.05, float(season_boost) * math.exp(-RECENCY_DECAY * days))


def matches_from_events(
    events: list[dict[str, Any]],
    *,
    season_boost: float = 1.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events or []:
        goals = event_goals(event)
        if goals is None:
            continue
        home = str((event.get("homeTeam") or {}).get("name") or "")
        away = str((event.get("awayTeam") or {}).get("name") or "")
        home_key = normalize_name(home)
        away_key = normalize_name(away)
        if not home_key or not away_key or home_key == away_key:
            continue
        hg, ag = goals
        rows.append(
            {
                "home": home_key,
                "away": away_key,
                "home_name": home,
                "away_name": away,
                "hg": float(hg),
                "ag": float(ag),
                "weight": match_weight(
                    event.get("startTimestamp"),
                    season_boost=season_boost,
                ),
            }
        )
    return rows


def ratings_from_standings(
    metrics: dict[str, dict[str, float]],
    *,
    league_goal_rate: float = LEAGUE_GOAL_RATE,
) -> dict[str, Any]:
    if not metrics:
        return empty_ratings()
    avg = float(league_goal_rate or LEAGUE_GOAL_RATE)
    attack: dict[str, float] = {}
    defence: dict[str, float] = {}
    for team, row in metrics.items():
        key = normalize_name(team)
        if not key:
            continue
        gf = float(row.get("gf") or avg)
        ga = float(row.get("ga") or avg)
        attack[key] = _clip(gf / avg, 0.45, 2.2)
        defence[key] = _clip(ga / avg, 0.45, 2.2)
    if not attack:
        return empty_ratings()
    return {
        "attack": attack,
        "defence": defence,
        "home_adv": HOME_ADVANTAGE,
        "scale": avg,
        "league_avg": avg,
        "source": "standings",
    }


def empty_ratings() -> dict[str, Any]:
    return {
        "attack": {},
        "defence": {},
        "home_adv": HOME_ADVANTAGE,
        "scale": LEAGUE_GOAL_RATE,
        "league_avg": LEAGUE_GOAL_RATE,
        "source": "empty",
    }


def fit_poisson_ratings(matches: list[dict[str, Any]], *, iterations: int = 36) -> dict[str, Any]:
    """Bağımsız Poisson hücum/savunma (Dixon–Coles rho yok; az maçta daha kararlı)."""
    if len(matches) < 8:
        return empty_ratings()
    teams = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
    if len(teams) < 4:
        return empty_ratings()
    attack = {t: 1.0 for t in teams}
    defence = {t: 1.0 for t in teams}
    home_adv = HOME_ADVANTAGE
    for _ in range(max(8, int(iterations))):
        new_attack: dict[str, float] = {}
        for team in teams:
            num = den = 0.0
            for match in matches:
                w = float(match["weight"])
                if match["home"] == team:
                    num += w * match["hg"]
                    den += w * defence[match["away"]] * home_adv
                elif match["away"] == team:
                    num += w * match["ag"]
                    den += w * defence[match["home"]]
            new_attack[team] = (num / den) if den > 1e-9 else 1.0
        mean_a = sum(new_attack.values()) / len(new_attack)
        attack = {t: v / mean_a for t, v in new_attack.items()} if mean_a else attack

        new_def: dict[str, float] = {}
        for team in teams:
            num = den = 0.0
            for match in matches:
                w = float(match["weight"])
                if match["home"] == team:
                    num += w * match["ag"]
                    den += w * attack[match["away"]]
                elif match["away"] == team:
                    num += w * match["hg"]
                    den += w * attack[match["home"]] * home_adv
            new_def[team] = (num / den) if den > 1e-9 else 1.0
        mean_d = sum(new_def.values()) / len(new_def)
        defence = {t: v / mean_d for t, v in new_def.items()} if mean_d else defence

        home_num = home_den = 0.0
        for match in matches:
            w = float(match["weight"])
            home_num += w * match["hg"]
            home_den += w * attack[match["home"]] * defence[match["away"]]
        if home_den > 1e-9:
            home_adv = _clip(home_num / home_den, 1.08, 1.42)

    pred_h = obs_h = pred_a = obs_a = 0.0
    for match in matches:
        w = float(match["weight"])
        pred_h += w * attack[match["home"]] * defence[match["away"]] * home_adv
        obs_h += w * match["hg"]
        pred_a += w * attack[match["away"]] * defence[match["home"]]
        obs_a += w * match["ag"]
    scale_h = (obs_h / pred_h) if pred_h > 1e-9 else LEAGUE_GOAL_RATE
    scale_a = (obs_a / pred_a) if pred_a > 1e-9 else LEAGUE_GOAL_RATE
    scale = _clip(0.5 * (scale_h + scale_a), 0.85, 1.85)
    league_avg = _clip((obs_h + obs_a) / max(1e-9, 2.0 * sum(m["weight"] for m in matches)), 0.9, 1.8)
    return {
        "attack": attack,
        "defence": defence,
        "home_adv": home_adv,
        "scale": scale,
        "league_avg": league_avg,
        "source": "poisson",
        "n_matches": len(matches),
        "n_teams": len(teams),
    }


def blend_ratings(primary: dict[str, Any], fallback: dict[str, Any], *, current_weight: float) -> dict[str, Any]:
    if not primary.get("attack"):
        return fallback if fallback.get("attack") else empty_ratings()
    if not fallback.get("attack"):
        return primary
    w = _clip(current_weight, 0.0, 1.0)
    teams = set(primary["attack"]) | set(primary["defence"]) | set(fallback.get("attack") or {}) | set(
        fallback.get("defence") or {}
    )
    attack: dict[str, float] = {}
    defence: dict[str, float] = {}
    for team in teams:
        pa = float(primary["attack"].get(team) or 1.0)
        fa = float(fallback["attack"].get(team) or 1.0)
        pd_ = float(primary["defence"].get(team) or 1.0)
        fd = float(fallback["defence"].get(team) or 1.0)
        attack[team] = w * pa + (1.0 - w) * fa
        defence[team] = w * pd_ + (1.0 - w) * fd
    return {
        "attack": attack,
        "defence": defence,
        "home_adv": w * float(primary.get("home_adv") or HOME_ADVANTAGE)
        + (1.0 - w) * float(fallback.get("home_adv") or HOME_ADVANTAGE),
        "scale": w * float(primary.get("scale") or LEAGUE_GOAL_RATE)
        + (1.0 - w) * float(fallback.get("scale") or LEAGUE_GOAL_RATE),
        "league_avg": w * float(primary.get("league_avg") or LEAGUE_GOAL_RATE)
        + (1.0 - w) * float(fallback.get("league_avg") or LEAGUE_GOAL_RATE),
        "source": "blend",
        "current_weight": w,
    }


def predict_lambdas(
    ratings: dict[str, Any],
    team: str,
    opponent: str,
    *,
    home: bool,
) -> dict[str, float]:
    attack = ratings.get("attack") or {}
    defence = ratings.get("defence") or {}
    team_key = normalize_name(team)
    opp_key = normalize_name(opponent)
    att = float(attack.get(team_key) or 1.0)
    opp_def = float(defence.get(opp_key) or 1.0)
    opp_att = float(attack.get(opp_key) or 1.0)
    own_def = float(defence.get(team_key) or 1.0)
    home_adv = float(ratings.get("home_adv") or HOME_ADVANTAGE)
    scale = float(ratings.get("scale") or LEAGUE_GOAL_RATE)
    if home:
        lambda_for = att * opp_def * home_adv * scale
        lambda_against = opp_att * own_def * scale
    else:
        lambda_for = att * opp_def * scale
        lambda_against = opp_att * own_def * home_adv * scale
    lambda_for = _clip(lambda_for, 0.40, 3.20)
    lambda_against = _clip(lambda_against, 0.40, 3.20)
    league_avg = float(ratings.get("league_avg") or LEAGUE_GOAL_RATE)
    p_cs = math.exp(-lambda_against)
    attack_mult = _clip(lambda_for / max(league_avg, 0.4), FIXTURE_ATTACK_FLOOR, FIXTURE_ATTACK_CEILING)
    cs_mult = _clip(p_cs / LEAGUE_CS_RATE, FIXTURE_CS_FLOOR, FIXTURE_CS_CEILING)
    save_mult = _clip(lambda_against / max(league_avg, 0.4), SAVE_MULT_FLOOR, SAVE_MULT_CEILING)
    team_goal_rate = _clip(att * scale, 0.50, 2.60)
    return {
        "lambda_for": round(lambda_for, 4),
        "lambda_against": round(lambda_against, 4),
        "p_cs": round(_clip(p_cs, 0.05, 0.72), 4),
        "attack_mult": round(attack_mult, 4),
        "cs_mult": round(cs_mult, 4),
        "save_mult": round(save_mult, 4),
        "team_goal_rate": round(team_goal_rate, 4),
    }


def group_events_by_matchweek(
    events: list[dict[str, Any]],
    *,
    weeks: int = HORIZON_WEEKS,
) -> list[list[dict[str, Any]]]:
    buckets: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    seen: set[str] = set()
    ordered = sorted(events or [], key=lambda e: e.get("startTimestamp") or 0)
    for event in ordered:
        home = normalize_name(str((event.get("homeTeam") or {}).get("name") or ""))
        away = normalize_name(str((event.get("awayTeam") or {}).get("name") or ""))
        if not home or not away:
            continue
        if home in seen or away in seen:
            if current:
                buckets.append(current)
            if len(buckets) >= weeks:
                return buckets[:weeks]
            current = []
            seen = set()
        seen.update((home, away))
        current.append(event)
    if current and len(buckets) < weeks:
        buckets.append(current)
    return buckets[:weeks]


def fixture_pack(
    ratings: dict[str, Any],
    team: str,
    opponent: str,
    *,
    home: bool,
) -> dict[str, Any]:
    pred = predict_lambdas(ratings, team, opponent, home=home)
    return {
        "opponent": opponent,
        "home": bool(home),
        **pred,
    }


def blend_goal_expectation(
    historical_pa: float,
    *,
    attack_mult: float,
    lambda_for: float | None,
    team_goal_rate: float | None,
    blend: float = MATCH_MODEL_BLEND,
) -> float:
    """Tarihsel maç başı gol ile takım λ × oyuncu payını karıştır."""
    hist = max(0.0, float(historical_pa or 0.0) * float(attack_mult or 1.0))
    if lambda_for is None or team_goal_rate is None or team_goal_rate <= 0.05:
        return hist
    share = _clip(float(historical_pa or 0.0) / float(team_goal_rate), 0.0, 0.55)
    model = share * float(lambda_for)
    w = _clip(blend, 0.0, 1.0)
    return (1.0 - w) * hist + w * model


def blend_cs_probability(
    historical_cs: float,
    *,
    cs_mult: float,
    p_cs: float | None,
    blend: float = MATCH_MODEL_BLEND,
) -> float:
    hist = _clip(float(historical_cs or 0.0) * float(cs_mult or 1.0), 0.0, 1.0)
    if p_cs is None:
        return hist
    w = _clip(blend, 0.0, 1.0)
    return _clip((1.0 - w) * hist + w * float(p_cs), 0.0, 1.0)
