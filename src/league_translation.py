"""Geçmiş transferlerden öğrenilen lig → Süper Lig performans dönüşümü."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import DATA_DIR

MODEL_PATH = DATA_DIR / "league_translation.json"
ATTACK_IDENTITY_METRICS = {"goals_p90", "assists_p90", "xg_p90", "xa_p90"}
_TIER2_MARKERS = (
    "1. lig",
    "1.lig",
    "championship",
    "serie b",
    "2. bundesliga",
    "ligue 2",
    "segunda",
)
_TIER1_MARKERS = (
    "premier league",
    "laliga",
    "la liga",
    "serie a",
    "ligue 1",
    "eredivisie",
    "primeira liga",
)


def _league_band(name: str) -> str | None:
    """Kaynak ligin üst/alt uçuş bandı; canlı ağ çağrısı yok."""
    n = (name or "").lower()
    if any(marker in n for marker in _TIER2_MARKERS):
        return "2"
    if "bundesliga" in n:
        return "2" if "2." in n else "1"
    if any(marker in n for marker in _TIER1_MARKERS):
        return "1"
    return None


@lru_cache(maxsize=2)
def load_translation_model(path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Üretilmiş kalibrasyonu yükle; dosya yoksa güvenli biçimde kimlik dönüşümü kullan."""
    target = Path(path)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def clear_translation_cache() -> None:
    load_translation_model.cache_clear()


def _position_models(container: dict[str, Any], position: str) -> dict[str, Any]:
    positions = container.get("positions") or {}
    return positions.get(position) or positions.get("ALL") or {}


def _weighted_metric_mean(parts: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [
        part
        for part in parts
        if float(part.get("n_players") or 0.0) > 0
    ]
    if not usable:
        return None
    total = sum(float(part["n_players"]) for part in usable)
    return {
        "intercept": sum(
            float(part.get("intercept") or 0.0) * float(part["n_players"]) for part in usable
        )
        / total,
        "slope": sum(
            float(part.get("slope") or 0.0) * float(part["n_players"]) for part in usable
        )
        / total,
        "cap": sum(
            float(part.get("cap") or 0.0) * float(part["n_players"]) for part in usable
        )
        / total,
        "n_players": int(round(total)),
        "local_weight": 0.0,
    }


def _peer_metric_model(
    model: dict[str, Any],
    tournament_id: int,
    position: str,
    metric: str,
    league: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    band = _league_band(str(league.get("name") or ""))
    if not band:
        return None, None
    peers: list[dict[str, Any]] = []
    for other_id, other in (model.get("leagues") or {}).items():
        if str(other_id) == str(int(tournament_id or 0)):
            continue
        if _league_band(str(other.get("name") or "")) != band:
            continue
        fitted = _position_models(other, position).get(metric)
        if not isinstance(fitted, dict):
            continue
        if float(fitted.get("local_weight") or 0.0) <= 0:
            continue
        peers.append(fitted)
    averaged = _weighted_metric_mean(peers)
    if not averaged:
        return None, None
    return averaged, {
        "level": "tier",
        "league": league.get("name") or "",
        "n_players": int(averaged.get("n_players") or 0),
        "local_weight": 0.0,
    }


def _identity_mix_weight(
    model: dict[str, Any],
    tournament_id: int,
    metric: str,
    source: float,
) -> float:
    """Ortalama transfer eğimi yıldız G/A oranını fazla küçültmesin."""
    if metric not in ATTACK_IDENTITY_METRICS or source <= 0:
        return 0.0
    league = (model.get("leagues") or {}).get(str(int(tournament_id or 0))) or {}
    typical_ga = float(league.get("observed_source_ga_p90") or 0.0)
    if typical_ga <= 0:
        return 0.0
    baseline = typical_ga / 2.0
    if baseline <= 0:
        return 0.0
    excess = source / baseline
    return max(0.0, min(0.70, (excess - 1.25) / 2.5))


def _metric_model(
    model: dict[str, Any],
    tournament_id: int,
    position: str,
    metric: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    leagues = model.get("leagues") or {}
    league = leagues.get(str(int(tournament_id or 0))) or {}
    exact = _position_models(league, position).get(metric)
    if isinstance(exact, dict) and float(exact.get("local_weight") or 0.0) > 0:
        return exact, {
            "level": "league",
            "league": league.get("name") or "",
            "n_players": int(exact.get("n_players") or league.get("n_players") or 0),
            "local_weight": float(exact.get("local_weight") or 0.0),
        }

    peer, peer_meta = _peer_metric_model(model, tournament_id, position, metric, league)
    if peer and peer_meta:
        return peer, peer_meta

    global_block = model.get("global") or {}
    fallback = _position_models(global_block, position).get(metric)
    if isinstance(fallback, dict):
        return fallback, {
            "level": "global",
            "league": league.get("name") or "",
            "n_players": int(fallback.get("n_players") or 0),
            "local_weight": 0.0,
        }
    if isinstance(exact, dict):
        return exact, {
            "level": "global",
            "league": league.get("name") or "",
            "n_players": int(exact.get("n_players") or league.get("n_players") or 0),
            "local_weight": 0.0,
        }
    return None, {"level": "identity", "league": league.get("name") or "", "n_players": 0}


def translate_metric(
    model: dict[str, Any],
    tournament_id: int,
    position: str,
    metric: str,
    source_value: float,
) -> tuple[float, dict[str, Any]]:
    """
    Kaynak lig oranını doğrusal ve küçültülmüş tarihsel modelle Süper Lig oranına çevir.

    Katsayı bulunamazsa veri uydurmaz; kaynak değeri aynen döndürür.
    """
    source = max(0.0, float(source_value or 0.0))
    fitted, meta = _metric_model(model, tournament_id, position, metric)
    if not fitted:
        return source, meta

    intercept = float(fitted.get("intercept") or 0.0)
    slope = float(fitted.get("slope") or 0.0)
    predicted = max(0.0, intercept + slope * source)
    cap = float(fitted.get("cap") or 0.0)
    if cap > 0:
        predicted = min(predicted, cap)
    mix = _identity_mix_weight(model, tournament_id, metric, source)
    if mix > 0:
        predicted = (1.0 - mix) * predicted + mix * source
    return predicted, {
        **meta,
        "metric": metric,
        "source": source,
        "predicted": predicted,
        "identity_mix": mix,
    }


def translate_metric_mixture(
    model: dict[str, Any],
    league_mix: list[dict[str, Any]],
    position: str,
    metric: str,
    source_value: float,
) -> tuple[float, dict[str, Any]]:
    """Birden fazla kaynak lig sezonu karıştıysa modelleri aynı ağırlıklarla birleştir."""
    valid = [
        part
        for part in league_mix
        if int(part.get("tournament_id") or 0) > 0 and float(part.get("weight") or 0.0) > 0
    ]
    if not valid:
        return max(0.0, float(source_value or 0.0)), {
            "level": "identity",
            "n_players": 0,
        }
    total = sum(float(part["weight"]) for part in valid)
    predicted = 0.0
    details = []
    for part in valid:
        weight = float(part["weight"]) / total
        value, detail = translate_metric(
            model,
            int(part["tournament_id"]),
            position,
            metric,
            source_value,
        )
        predicted += weight * value
        details.append((weight, detail))
    exact = [(weight, detail) for weight, detail in details if detail.get("level") == "league"]
    tier_fallback = [
        (weight, detail) for weight, detail in details if detail.get("level") == "tier"
    ]
    global_fallback = [
        (weight, detail) for weight, detail in details if detail.get("level") == "global"
    ]
    selected = exact or tier_fallback or global_fallback or details
    if exact:
        selected_level = "league"
    elif tier_fallback:
        selected_level = "tier"
    elif global_fallback:
        selected_level = "global"
    else:
        selected_level = "identity"
    return predicted, {
        "level": selected_level,
        "league": " + ".join(
            str(part.get("tournament") or "") for part in valid if part.get("tournament")
        ),
        "n_players": int(
            round(sum(weight * float(detail.get("n_players") or 0) for weight, detail in selected))
        ),
        "local_weight": sum(
            weight * float(detail.get("local_weight") or 0.0) for weight, detail in selected
        ),
        "metric": metric,
        "source": max(0.0, float(source_value or 0.0)),
        "predicted": predicted,
        "identity_mix": sum(
            weight * float(detail.get("identity_mix") or 0.0) for weight, detail in details
        ),
    }


def translate_external_rates(
    rates: dict[str, float],
    row: dict[str, Any],
    position: str,
    target_min_per_app: float,
    *,
    model: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Maç başı dış-lig oranlarını, dakika-temelli tarihsel Süper Lig oranlarına çevir."""
    calibration = model if model is not None else load_translation_model()
    if not calibration:
        unchanged = rates.copy()
        unchanged["min_per_app"] = float(target_min_per_app or rates.get("min_per_app") or 0.0)
        return unchanged, {"level": "identity", "n_players": 0}

    out = rates.copy()
    tid = int(row.get("tournament_id") or 0)
    raw_mix = row.get("league_mix")
    league_mix = (
        list(raw_mix)
        if isinstance(raw_mix, list) and raw_mix
        else [
            {
                "tournament_id": tid,
                "tournament": row.get("tournament") or "",
                "weight": 1.0,
            }
        ]
    )
    source_min = float(rates.get("min_per_app") or 0.0)
    target_min = float(target_min_per_app or source_min or 0.0)
    if source_min <= 0 or target_min <= 0:
        out["min_per_app"] = target_min
        return out, {"level": "identity", "n_players": 0}

    details: list[dict[str, Any]] = []
    translated_attack_source = 0.0
    translated_attack_target = 0.0

    per90_fields = {
        "gls_pa": "goals_p90",
        "ast_pa": "assists_p90",
        "xg_pa": "xg_p90",
        "xa_pa": "xa_p90",
        "sot_pa": "shots_on_target_p90",
        "key_passes_pa": "key_passes_p90",
        "yc_pa": "yellow_cards_p90",
        "rc_pa": "red_cards_p90",
    }
    if position == "GK":
        per90_fields.update({"saves_pa": "saves_p90", "ga_pa": "goals_conceded_p90"})

    for field, metric in per90_fields.items():
        source_pa = float(rates.get(field) or 0.0)
        # xG/xA ve vekil alanlarında 0 çoğu eski sezonda "veri yok" demektir.
        optional = metric in {
            "xg_p90",
            "xa_p90",
            "shots_on_target_p90",
            "key_passes_p90",
            "yellow_cards_p90",
            "red_cards_p90",
        }
        if optional and source_pa <= 0:
            continue
        source_p90 = source_pa * 90.0 / source_min
        translated_p90, detail = translate_metric_mixture(
            calibration, league_mix, position, metric, source_p90
        )
        out[field] = translated_p90 * target_min / 90.0
        details.append(detail)
        if metric in ("goals_p90", "assists_p90"):
            translated_attack_source += source_p90
            translated_attack_target += translated_p90

    if position in ("GK", "DF", "MF"):
        translated_cs, detail = translate_metric_mixture(
            calibration,
            league_mix,
            position,
            "clean_sheet_rate",
            float(rates.get("cs_rate") or 0.0),
        )
        out["cs_rate"] = min(1.0, translated_cs)
        details.append(detail)

    rating = float(rates.get("rating") or 0.0)
    if rating > 0:
        translated_rating, detail = translate_metric_mixture(
            calibration, league_mix, position, "rating", rating
        )
        out["rating"] = translated_rating
        details.append(detail)

    out["min_per_app"] = target_min
    exact = [d for d in details if d.get("level") == "league"]
    tier_fallback = [d for d in details if d.get("level") == "tier"]
    global_fallback = [d for d in details if d.get("level") == "global"]
    chosen = (
        exact[0]
        if exact
        else (
            tier_fallback[0]
            if tier_fallback
            else (
                global_fallback[0]
                if global_fallback
                else (details[0] if details else {"level": "identity"})
            )
        )
    )
    attack_factor = (
        translated_attack_target / translated_attack_source
        if translated_attack_source > 0
        else 1.0
    )
    attack_mixes = [
        float(detail.get("identity_mix") or 0.0)
        for detail in details
        if detail.get("metric") in ATTACK_IDENTITY_METRICS
    ]
    return out, {
        "level": chosen.get("level") or "identity",
        "league": chosen.get("league") or row.get("tournament") or "",
        "n_players": int(chosen.get("n_players") or 0),
        "local_weight": float(chosen.get("local_weight") or 0.0),
        "attack_factor": attack_factor,
        "identity_mix": max(attack_mixes) if attack_mixes else 0.0,
        "metrics": details,
    }
