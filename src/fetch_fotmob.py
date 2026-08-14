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

    cutoff = datetime.now(timezone.utc) - timedelta(days=120)

    def current_enough(match: dict[str, Any]) -> bool:
        raw = str(((match.get("matchDate") or {}).get("utcTime")) or "")
        if not raw:
            return False
        try:
            played_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return False
        return played_at >= cutoff

    recent = [
        match
        for match in payload.get("recentMatches") or []
        if _same_team(team, str(match.get("teamName") or "")) and current_enough(match)
    ][:8]
    played = [m for m in recent if m.get("playedInMatch")]
    starts = [m for m in played if not m.get("onBench")]
    recent_minutes = sum(float(m.get("minutesPlayed") or 0) for m in played)
    recent_goals = sum(float(m.get("goals") or 0) for m in played)
    recent_assists = sum(float(m.get("assists") or 0) for m in played)

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
        "fotmob_injury": bool(payload.get("injuryInformation")),
    }


def apply_fotmob_validation(
    players: pd.DataFrame,
    *,
    progress: ProgressCb = None,
    max_fetch: int = 55,
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

            # Lig L6 eksik görünse bile güncel hazırlık/Avrupa maçlarında oynuyorsa
            # eski recency cezasını geri al. Dört güncel kulüp maçı tam teyittir.
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

    if progress:
        count = int(pd.to_numeric(out.get("fotmob_id"), errors="coerce").notna().sum())
        progress(f"FotMob doğrulaması: {count} oyuncu.")
    return out
