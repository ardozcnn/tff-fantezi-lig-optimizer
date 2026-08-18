"""FotMob ikinci kaynak doğrulaması: ilk 11, derin sezon verisi ve güncel kulüp maçları."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import pandas as pd
import requests
import urllib3

from .config import CACHE_DIR
from .names import best_match, normalize_name

FOTMOB_BASE = "https://www.fotmob.com/api/data"
ProgressCb = Callable[[str], None] | None
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"fotmob_{key}.json"


def _get_json(path: str, cache_key: str, max_age_hours: float) -> Any:
    target = _cache_path(cache_key)
    if target.exists():
        age = (time.time() - target.stat().st_mtime) / 3600
        if age <= max_age_hours:
            try:
                return json.loads(target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
    response = requests.get(
        f"{FOTMOB_BASE}/{path.lstrip('/')}",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
        verify=os.environ.get("FBREF_SSL_VERIFY", "0") not in ("0", "false", "False"),
    )
    response.raise_for_status()
    payload = response.json()
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def _same_team(left: str, right: str) -> bool:
    a, b = normalize_name(left), normalize_name(right)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    generic = {"fk", "jk", "spor", "sportif", "faaliyetler", "istanbul"}
    at, bt = set(a.split()) - generic, set(b.split()) - generic
    return bool(at and bt and len(at & bt) / len(at | bt) >= 0.5)


def readiness_multiplier(recent_played: float, recent_minutes: float) -> float:
    evidence = min(1.0, max(recent_played / 4.0, recent_minutes / 280.0))
    return 0.65 + 0.35 * evidence


def _is_super_lig(match: dict[str, Any]) -> bool:
    name = str(match.get("leagueName") or "").casefold()
    return "super lig" in name or "süper lig" in name or "superlig" in name.replace(" ", "")


def _match_played_at(match: dict[str, Any]) -> datetime | None:
    raw = str(((match.get("matchDate") or {}).get("utcTime")) or "")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _soften_rate(total: float, apps: float, prior_rate: float, prior_apps: float = 6.0) -> float:
    """Tek maçlık 2 gol gibi gürültüyü sezon prior'una doğru küçültür."""
    apps = max(0.0, float(apps))
    return (float(total) + prior_rate * prior_apps) / (apps + prior_apps)


def hot_form_blend_weight(
    sl_apps: float,
    sl_minutes: float | None = None,
    *,
    early_season: bool = False,
) -> float:
    """Güncel sonucun ağırlığı; tek 90 dakika yaklaşık %8, dört maç en çok %25."""
    apps = max(0.0, float(sl_apps))
    minutes = max(0.0, float(sl_minutes if sl_minutes is not None else apps * 75.0))
    if apps <= 0 or minutes <= 0:
        return 0.0
    prior_minutes = 990.0 if early_season else 810.0
    return min(0.25, minutes / (minutes + prior_minutes))


_PRIOR_GLS = {"FW": 0.40, "MF": 0.15, "DF": 0.06, "GK": 0.0}
_PRIOR_AST = {"FW": 0.15, "MF": 0.18, "DF": 0.08, "GK": 0.0}


def hot_form_expected_points(
    validation: dict[str, Any],
    position: str,
    *,
    prior_rates: dict[str, float] | None = None,
    attack_mult: float = 1.0,
    cs_mult: float = 1.0,
    team_cs_rate: float | None = None,
) -> float | None:
    """Son Süper Lig maçlarından (Bayes yumuşatmalı) bir maçlık beklenen puan."""
    raw_apps = float(validation.get("fotmob_sl_apps") or 0.0)
    apps = float(
        validation.get("fotmob_sl_effective_apps")
        or raw_apps
        or 0.0
    )
    if raw_apps < 1 or apps <= 0:
        return None
    minutes = float(
        validation.get("fotmob_sl_effective_minutes")
        or validation.get("fotmob_sl_minutes")
        or 0.0
    )
    goals = float(
        validation.get("fotmob_sl_effective_goals")
        if validation.get("fotmob_sl_effective_goals") is not None
        else validation.get("fotmob_sl_goals")
        or 0.0
    )
    assists = float(
        validation.get("fotmob_sl_effective_assists")
        if validation.get("fotmob_sl_effective_assists") is not None
        else validation.get("fotmob_sl_assists")
        or 0.0
    )
    pos = position if position in _PRIOR_GLS else "MF"
    prior = prior_rates or {}
    prior_gls = max(0.0, float(prior.get("gls_pa") or _PRIOR_GLS[pos]))
    prior_ast = max(0.0, float(prior.get("ast_pa") or _PRIOR_AST[pos]))
    min_per_app = minutes / apps if apps else 0.0
    share_60 = 1.0 if min_per_app >= 60 else max(0.0, min_per_app / 60.0)
    rates = {
        "apps": apps,
        "apps_60": apps * share_60,
        "share_60": share_60,
        "min_per_app": min_per_app,
        "gls_pa": _soften_rate(goals, apps, prior_gls),
        "ast_pa": _soften_rate(assists, apps, prior_ast),
        "xg_pa": max(0.0, float(prior.get("xg_pa") or 0.0)),
        "xa_pa": max(0.0, float(prior.get("xa_pa") or 0.0)),
        "sot_pa": max(0.0, float(prior.get("sot_pa") or 0.0)),
        "key_passes_pa": max(0.0, float(prior.get("key_passes_pa") or 0.0)),
        "bcc_pa": max(0.0, float(prior.get("bcc_pa") or 0.0)),
        "yc_pa": 0.0,
        "rc_pa": 0.0,
        "rating": 6.8
        + (float(validation.get("fotmob_sl_rating") or 6.8) - 6.8)
        * apps
        / (apps + 8.0),
    }
    from .scoring import expected_points_from_rates

    return expected_points_from_rates(
        rates,
        pos,
        team_cs_rate,
        attack_mult=attack_mult,
        cs_mult=cs_mult,
    )


def search_player(name: str, team: str) -> dict[str, Any] | None:
    payload = _get_json(
        f"search/suggest?term={quote(name)}",
        f"search_{normalize_name(name).replace(' ', '_')}",
        72,
    )
    hits: dict[str, dict[str, Any]] = {}
    for group in payload if isinstance(payload, list) else []:
        for hit in group.get("suggestions") or []:
            if hit.get("type") == "player" and hit.get("id"):
                hits[str(hit["id"])] = hit
    if not hits:
        return None
    same_team = [h for h in hits.values() if _same_team(team, str(h.get("teamName") or ""))]
    pool = same_team or list(hits.values())
    matched, score = best_match(name, [str(h.get("name") or "") for h in pool], score_cutoff=80)
    if not matched or score < 80:
        return None
    return next(h for h in pool if str(h.get("name") or "") == matched)


def _deep_stats(payload: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    section = (payload.get("firstSeasonStats") or {}).get("statsSection") or {}
    for group in section.get("items") or []:
        for item in group.get("items") or []:
            title = str(item.get("title") or "").strip().lower()
            raw = str(item.get("statValue") or "").replace(",", "")
            try:
                out[title] = float(raw)
            except ValueError:
                continue
    return out


def _main_league_deep_stats(
    payload: dict[str, Any],
    player_id: str,
) -> dict[str, float]:
    main = payload.get("mainLeague") or {}
    season_name = str(main.get("season") or "")
    league_id = int(main.get("leagueId") or 0)
    entry_id = ""
    for season in payload.get("statSeasons") or []:
        if str(season.get("seasonName") or "") != season_name:
            continue
        for tournament in season.get("tournaments") or []:
            if int(tournament.get("tournamentId") or 0) == league_id:
                entry_id = str(tournament.get("entryId") or "")
                break
    if not entry_id:
        return {}
    stats = _get_json(
        f"playerStats?playerId={player_id}&seasonId={entry_id}&isFirstSeason=false",
        f"player_stats_{player_id}_{entry_id.replace('-', '_')}",
        24,
    )
    return _deep_stats({"firstSeasonStats": stats})


def fetch_player_validation(name: str, team: str) -> dict[str, Any] | None:
    hit = search_player(name, team)
    if not hit:
        return None
    player_id = str(hit["id"])
    payload = _get_json(f"playerData?id={player_id}", f"player_{player_id}", 12)
    primary = payload.get("primaryTeam") or {}
    primary_name = str(primary.get("teamName") or primary.get("name") or hit.get("teamName") or "")
    if primary_name and not _same_team(team, primary_name):
        return None

    main = payload.get("mainLeague") or {}
    totals: dict[str, float] = {}
    for item in main.get("stats") or []:
        key = str(item.get("localizedTitleId") or item.get("title") or "").lower()
        try:
            totals[key] = float(item.get("value") or 0)
        except (TypeError, ValueError):
            continue
    deep = _main_league_deep_stats(payload, player_id)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=120)
    sl_cutoff = now - timedelta(days=28)

    recent = []
    for match in payload.get("recentMatches") or []:
        if not _same_team(team, str(match.get("teamName") or "")):
            continue
        played_at = _match_played_at(match)
        if played_at is None or played_at < cutoff:
            continue
        recent.append(match)
        if len(recent) >= 8:
            break
    played = [m for m in recent if m.get("playedInMatch")]
    starts = [m for m in played if not m.get("onBench")]
    recent_minutes = sum(float(m.get("minutesPlayed") or 0) for m in played)
    recent_goals = sum(float(m.get("goals") or 0) for m in played)
    recent_assists = sum(float(m.get("assists") or 0) for m in played)

    sl_played = []
    for match in payload.get("recentMatches") or []:
        if not _same_team(team, str(match.get("teamName") or "")):
            continue
        if not _is_super_lig(match) or not match.get("playedInMatch"):
            continue
        played_at = _match_played_at(match)
        if played_at is None or played_at < sl_cutoff:
            continue
        sl_played.append(match)
        if len(sl_played) >= 4:
            break
    sl_minutes = sum(float(m.get("minutesPlayed") or 0) for m in sl_played)
    sl_goals = sum(float(m.get("goals") or 0) for m in sl_played)
    sl_assists = sum(float(m.get("assists") or 0) for m in sl_played)
    weighted = []
    for match in sl_played:
        played_at = _match_played_at(match)
        age_days = max(0.0, (now - played_at).total_seconds() / 86400.0) if played_at else 0.0
        weight = 0.5 ** (age_days / 42.0)
        weighted.append((match, weight))
    sl_effective_apps = sum(weight for _, weight in weighted)
    sl_effective_minutes = sum(
        float(match.get("minutesPlayed") or 0) * weight for match, weight in weighted
    )
    sl_effective_goals = sum(
        float(match.get("goals") or 0) * weight for match, weight in weighted
    )
    sl_effective_assists = sum(
        float(match.get("assists") or 0) * weight for match, weight in weighted
    )
    sl_ratings = []
    for m in sl_played:
        try:
            sl_ratings.append(float((m.get("ratingProps") or {}).get("rating") or 0))
        except (TypeError, ValueError):
            continue
    sl_rating = sum(sl_ratings) / len(sl_ratings) if sl_ratings else 0.0

    return {
        "fotmob_id": player_id,
        "fotmob_name": payload.get("name") or hit.get("name") or name,
        "fotmob_team": primary_name or team,
        "fotmob_league": main.get("leagueName") or "",
        "fotmob_season": main.get("season") or "",
        "fotmob_matches": totals.get("matches_uppercase", totals.get("matches", 0.0)),
        "fotmob_starts": totals.get("started", 0.0),
        "fotmob_minutes": totals.get("minutes_played", 0.0),
        "fotmob_goals": totals.get("goals", 0.0),
        "fotmob_assists": totals.get("assists", 0.0),
        "fotmob_rating": totals.get("rating", 0.0),
        "fotmob_yellow": totals.get("yellow_cards", 0.0),
        "fotmob_red": totals.get("red_cards", 0.0),
        "fotmob_xg": deep.get("xg", 0.0),
        "fotmob_xa": deep.get("expected assists", deep.get("xa", 0.0)),
        "fotmob_shots": deep.get("shots", 0.0),
        "fotmob_sot": deep.get("shots on target", 0.0),
        "fotmob_recent_matches": float(len(recent)),
        "fotmob_recent_played": float(len(played)),
        "fotmob_recent_starts": float(len(starts)),
        "fotmob_recent_minutes": recent_minutes,
        "fotmob_recent_goals": recent_goals,
        "fotmob_recent_assists": recent_assists,
        "fotmob_sl_apps": float(len(sl_played)),
        "fotmob_sl_minutes": sl_minutes,
        "fotmob_sl_goals": sl_goals,
        "fotmob_sl_assists": sl_assists,
        "fotmob_sl_rating": sl_rating,
        "fotmob_sl_effective_apps": sl_effective_apps,
        "fotmob_sl_effective_minutes": sl_effective_minutes,
        "fotmob_sl_effective_goals": sl_effective_goals,
        "fotmob_sl_effective_assists": sl_effective_assists,
        "fotmob_injury": bool(payload.get("injuryInformation")),
    }


def apply_fotmob_validation(
    players: pd.DataFrame,
    *,
    progress: ProgressCb = None,
    max_fetch: int = 55,
    early_season: bool = False,
) -> pd.DataFrame:
    """Önemli/az-formlu oyuncuları ikinci kaynaktan doğrular ve güncel kullanım ekler."""
    out = players.copy()
    price = pd.to_numeric(out.get("price_m"), errors="coerce").fillna(0.0)
    form_apps = pd.to_numeric(out.get("form_apps"), errors="coerce").fillna(0.0)
    candidate = (price >= 7.0) | ((price >= 5.5) & (form_apps < 4))
    todo = out.loc[candidate].sort_values(
        ["price_m", "projected_pts"], ascending=False
    ).head(max_fetch)
    if todo.empty:
        return out

    if progress:
        progress(f"FotMob ikinci kaynak: {len(todo)} oyuncu doğrulanıyor...")

    def one(index: Any, row: pd.Series) -> tuple[Any, dict[str, Any] | None]:
        try:
            return index, fetch_player_validation(
                str(row.get("display_name") or row.get("player") or ""),
                str(row.get("team") or ""),
            )
        except Exception:
            return index, None

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(one, idx, row) for idx, row in todo.iterrows()]
        for future in as_completed(futures):
            idx, validation = future.result()
            if not validation:
                continue
            for key, value in validation.items():
                if key not in out.columns:
                    out[key] = None
                out.at[idx, key] = value

            old_recency = float(out.at[idx, "recency_mult"] or 1.0)
            recent_played = float(validation.get("fotmob_recent_played") or 0.0)
            recent_minutes = float(validation.get("fotmob_recent_minutes") or 0.0)
            readiness = readiness_multiplier(recent_played, recent_minutes)
            combined = max(old_recency, readiness)
            if old_recency > 0 and combined > old_recency:
                out.at[idx, "projected_pts"] = (
                    float(out.at[idx, "projected_pts"]) * combined / old_recency
                )
                out.at[idx, "recency_mult"] = combined

            sl_apps = float(validation.get("fotmob_sl_apps") or 0.0)
            form_n = float(out.at[idx, "form_apps"] or 0.0) if "form_apps" in out.columns else 0.0
            tff_minutes = (
                float(out.at[idx, "tff_minutes"] or 0.0)
                if "tff_minutes" in out.columns and pd.notna(out.at[idx, "tff_minutes"])
                else 0.0
            )
            tff_points = (
                float(out.at[idx, "tff_points"] or 0.0)
                if "tff_points" in out.columns and pd.notna(out.at[idx, "tff_points"])
                else 0.0
            )
            has_fresh_tff = 0 < tff_minutes < 400 and tff_points != 0
            use_hot = sl_apps >= 1 and not has_fresh_tff and (
                early_season or form_n < 1.5
            )
            if use_hot:
                hot = hot_form_expected_points(
                    validation,
                    str(out.at[idx, "position"] or "MF"),
                    prior_rates={
                        key: float(out.at[idx, key] or 0.0)
                        if key in out.columns and pd.notna(out.at[idx, key])
                        else 0.0
                        for key in (
                            "gls_pa",
                            "ast_pa",
                            "xg_pa",
                            "xa_pa",
                            "sot_pa",
                            "key_passes_pa",
                            "bcc_pa",
                        )
                    },
                    attack_mult=float(out.at[idx, "fixture_attack_mult"] or 1.0)
                    if "fixture_attack_mult" in out.columns
                    else 1.0,
                    cs_mult=float(out.at[idx, "fixture_cs_mult"] or 1.0)
                    if "fixture_cs_mult" in out.columns
                    else 1.0,
                    team_cs_rate=(
                        float(out.at[idx, "team_cs_base"])
                        if "team_cs_base" in out.columns
                        and pd.notna(out.at[idx, "team_cs_base"])
                        else None
                    ),
                )
                effective_minutes = float(
                    validation.get("fotmob_sl_effective_minutes")
                    or validation.get("fotmob_sl_minutes")
                    or 0.0
                )
                weight = hot_form_blend_weight(
                    sl_apps,
                    effective_minutes,
                    early_season=early_season,
                )
                if hot is not None and weight > 0:
                    base_pts = float(out.at[idx, "projected_pts"] or 0.0)
                    blended = (1.0 - weight) * base_pts + weight * float(hot)
                    out.at[idx, "projected_pts"] = round(blended, 3)
                    out.at[idx, "fotmob_hot_weight"] = weight
                    out.at[idx, "fotmob_hot_pts"] = round(float(hot), 3)

    if progress:
        count = int(pd.to_numeric(out.get("fotmob_id"), errors="coerce").notna().sum())
        hot_n = int(pd.to_numeric(out.get("fotmob_hot_weight"), errors="coerce").fillna(0).gt(0).sum()) if "fotmob_hot_weight" in out.columns else 0
        progress(f"FotMob doğrulaması: {count} oyuncu" + (f", {hot_n} hot-form blend." if hot_n else "."))
    return out


def _fotmob_side_name(side: Any) -> str:
    if isinstance(side, str):
        return side
    if not isinstance(side, dict):
        return ""
    return str(side.get("name") or side.get("longName") or side.get("shortName") or "")


def fotmob_rows_to_sofa_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in rows or []:
        status = row.get("status") if isinstance(row.get("status"), dict) else {}
        if status.get("finished"):
            continue
        home = _fotmob_side_name(row.get("home") or row.get("homeTeam"))
        away = _fotmob_side_name(row.get("away") or row.get("awayTeam"))
        if not home or not away:
            continue
        raw = str(
            status.get("utcTime")
            or (row.get("matchDate") or {}).get("utcTime")
            or row.get("utcTime")
            or ""
        )
        stamp = 0
        if raw:
            try:
                stamp = int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp())
            except ValueError:
                stamp = 0
        events.append(
            {
                "startTimestamp": stamp,
                "homeTeam": {"name": home},
                "awayTeam": {"name": away},
            }
        )
    events.sort(key=lambda e: int(e.get("startTimestamp") or 0))
    return events


def fetch_upcoming_super_lig_events(year_start: int | None = None) -> list[dict[str, Any]]:
    year = int(year_start or datetime.now(timezone.utc).year)
    label = f"{year}/{year + 1}"
    data = _get_json(
        f"leagues?id=71&season={label}",
        f"fixtures_{year % 100:02d}",
        6.0,
    )
    rows = ((data.get("fixtures") or {}).get("allMatches")) or []
    return fotmob_rows_to_sofa_events(rows)
