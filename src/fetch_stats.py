"""Süper Lig istatistikleri — Sofascore API (FBref 403 engelli ortamlarda birincil)."""

from __future__ import annotations

import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from curl_cffi import requests as curl_requests

from .config import (
    CACHE_DIR,
    FIXTURE_CS_CEILING,
    FIXTURE_CS_FLOOR,
    FORM_MATCHES,
    REQUEST_DELAY_S,
    TEAM_CS_PRIOR_MATCHES,
    is_quiet,
)
from .names import normalize_name


def _log(msg: str) -> None:
    if not is_quiet():
        print(msg)

SOFA_BASE = "https://www.sofascore.com/api/v1"
UNIQUE_TOURNAMENT_ID = 52


class SofaNotFound(RuntimeError):
    """Sofascore 404 — sezon/istatistik henüz yok."""

_session = None
_session_profile = ""
_last_request = 0.0
_sofa_blocked = False
_stale_cache_warned = False
_IMPERSONATE_PROFILES = (
    "chrome146",
    "chrome131",
    "chrome124",
    "safari184",
)
_SOFA_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
    "Origin": "https://www.sofascore.com",
    "Referer": "https://www.sofascore.com/",
}


def _ssl_verify() -> bool:
    flag = os.environ.get("FBREF_SSL_VERIFY", "0").strip().lower()
    if flag in ("1", "true", "yes"):
        return True
    return False


def _get_session(profile: str | None = None):
    global _session, _session_profile
    wanted = profile or _session_profile or _IMPERSONATE_PROFILES[0]
    if _session is None or wanted != _session_profile:
        try:
            _session = curl_requests.Session(impersonate=wanted)
            _session_profile = wanted
        except Exception:
            _session = curl_requests.Session(impersonate="chrome")
            _session_profile = "chrome"
    return _session


def _throttle(delay: float | None = None) -> None:
    global _last_request
    d = REQUEST_DELAY_S if delay is None else delay
    elapsed = time.time() - _last_request
    if elapsed < d:
        time.sleep(d - elapsed)
    _last_request = time.time()


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)
    return CACHE_DIR / f"{safe}.json"


def _load_cache(key: str, max_age_hours: float | None = 18.0) -> Any | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    if max_age_hours is not None:
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if age_h > max_age_hours:
            return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(key: str, payload: Any) -> None:
    path = _cache_path(key)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _sofa_fetch(url: str, delay: float) -> Any:
    last_error: Exception | None = None
    for profile in _IMPERSONATE_PROFILES:
        _throttle(delay)
        try:
            resp = _get_session(profile).get(
                url,
                headers=_SOFA_HEADERS,
                timeout=45,
                verify=_ssl_verify(),
            )
        except Exception as exc:
            last_error = exc
            continue
        if resp.status_code == 404:
            raise SofaNotFound(url)
        if resp.status_code in (403, 429, 503):
            last_error = RuntimeError(f"Sofascore {resp.status_code}: {url}")
            continue
        if resp.status_code != 200:
            last_error = RuntimeError(f"Sofascore {resp.status_code}: {url}")
            continue
        return resp.json()
    if last_error:
        raise last_error
    raise RuntimeError(f"Sofascore istek başarısız: {url}")


def sofa_get(path: str, *, cache_key: str | None = None, max_age_hours: float = 18.0, delay: float = 0.35) -> Any:
    global _sofa_blocked, _stale_cache_warned
    if cache_key:
        cached = _load_cache(cache_key, max_age_hours=max_age_hours)
        if cached is not None:
            return cached
    url = path if path.startswith("http") else f"{SOFA_BASE}{path}"
    if not _sofa_blocked:
        try:
            data = _sofa_fetch(url, delay)
        except SofaNotFound:
            raise
        except Exception as exc:
            if "403" in str(exc) or "429" in str(exc):
                _sofa_blocked = True
            data = None
            fetch_error = exc
        else:
            if cache_key:
                _save_cache(cache_key, data)
            return data
    else:
        data = None
        fetch_error = RuntimeError(f"Sofascore engelli: {url}")
    if cache_key:
        stale = _load_cache(cache_key, max_age_hours=None)
        if stale is not None:
            if not _stale_cache_warned:
                warnings.warn(
                    "Sofascore geçici olarak engelli; kayıtlı önbellek kullanılıyor.",
                    stacklevel=2,
                )
                _stale_cache_warned = True
            return stale
    if data is None:
        raise fetch_error
    return data


def discover_current_season_start() -> int:
    import datetime as dt

    today = dt.date.today()
    if today.month >= 7:
        return today.year
    return today.year - 1


def season_label(year_start: int) -> str:
    return f"{year_start}-{year_start + 1}"


def list_seasons() -> list[dict[str, Any]]:
    data = sofa_get(
        f"/unique-tournament/{UNIQUE_TOURNAMENT_ID}/seasons",
        cache_key="sofa_seasons",
        max_age_hours=24 * 14,
        delay=0.2,
    )
    return list(data.get("seasons") or [])


def resolve_season_id(year_start: int) -> int:
    """
    year_start=2024 → '24/25' sezonu.
    """
    yy = f"{year_start % 100:02d}/{(year_start + 1) % 100:02d}"
    seasons = list_seasons()
    for s in seasons:
        if s.get("year") == yy:
            return int(s["id"])
    for s in seasons:
        if str(year_start) in str(s.get("name", "")):
            return int(s["id"])
    raise RuntimeError(f"Sofascore sezon bulunamadı: {yy}. Mevcut: {[s.get('year') for s in seasons[:8]]}")


def fetch_standings_teams(season_id: int) -> list[dict[str, Any]]:
    try:
        data = sofa_get(
            f"/unique-tournament/{UNIQUE_TOURNAMENT_ID}/season/{season_id}/standings/total",
            cache_key=f"sofa_standings_{season_id}",
            max_age_hours=12,
        )
    except SofaNotFound:
        return []
    teams: list[dict[str, Any]] = []
    for block in data.get("standings") or []:
        for row in block.get("rows") or []:
            t = row.get("team") or {}
            if t.get("id"):
                teams.append({"id": t["id"], "name": t.get("name", "")})
    seen = set()
    out = []
    for t in teams:
        if t["id"] not in seen:
            seen.add(t["id"])
            out.append(t)
    return out


def fetch_standings_rows(season_id: int) -> list[dict[str, Any]]:
    try:
        data = sofa_get(
            f"/unique-tournament/{UNIQUE_TOURNAMENT_ID}/season/{season_id}/standings/total",
            cache_key=f"sofa_standings_{season_id}",
            max_age_hours=12,
        )
    except SofaNotFound:
        return []
    rows: list[dict[str, Any]] = []
    for block in data.get("standings") or []:
        rows.extend(r for r in block.get("rows") or [] if isinstance(r, dict))
    return rows


def fetch_upcoming_events(season_id: int, max_pages: int = 2) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for page in range(max_pages):
        try:
            data = sofa_get(
                f"/unique-tournament/{UNIQUE_TOURNAMENT_ID}/season/{season_id}/events/next/{page}",
                cache_key=f"sofa_events_next_{season_id}_{page}",
                max_age_hours=3,
                delay=0.2,
            )
        except SofaNotFound:
            break
        batch = data.get("events") or []
        events.extend(batch)
        if not data.get("hasNextPage"):
            break
    events.sort(key=lambda e: e.get("startTimestamp") or 0)
    return events


def next_matchweek_fixtures(season_id: int) -> list[dict[str, str]]:
    """Bir kulübün ikinci maçı gelene kadar sıradaki maç haftasını sun."""
    events = fetch_upcoming_events(season_id)
    if not events:
        return []
    fixtures: list[dict[str, str]] = []
    seen_teams: set[str] = set()
    for event in events:
        stamp = int(event.get("startTimestamp") or 0)
        if not stamp:
            continue
        kickoff = datetime.fromtimestamp(stamp, tz=timezone.utc)
        home = str((event.get("homeTeam") or {}).get("name") or "")
        away = str((event.get("awayTeam") or {}).get("name") or "")
        if not home or not away:
            continue
        if normalize_name(home) in seen_teams or normalize_name(away) in seen_teams:
            break
        seen_teams.update((normalize_name(home), normalize_name(away)))
        fixtures.append(
            {
                "home": home,
                "away": away,
                "kickoff": kickoff.astimezone().strftime("%d.%m %H:%M"),
            }
        )
    return fixtures


def build_fixture_context(
    upcoming_season_id: int,
    strength_season_id: int,
    *,
    fallback_strength_season_id: int | None = None,
    prefer_fallback: bool = False,
) -> dict[str, dict[str, Any]]:
    """İlk gelecek maç + rakibin sezon hücum/savunma gücü."""
    primary_id = (
        fallback_strength_season_id
        if prefer_fallback and fallback_strength_season_id
        else strength_season_id
    )
    rows = fetch_standings_rows(primary_id)
    if (
        not prefer_fallback
        and fallback_strength_season_id
        and fallback_strength_season_id != primary_id
    ):
        played = sum(float(row.get("matches") or 0) for row in rows)
        teams = len(rows)
        if teams == 0 or (teams > 0 and played / teams < 4.0):
            rows = fetch_standings_rows(fallback_strength_season_id)
    metrics: dict[str, dict[str, float]] = {}
    total_goals = total_matches = 0.0
    for row in rows:
        team = row.get("team") or {}
        key = normalize_name(str(team.get("name") or ""))
        matches = float(row.get("matches") or 0)
        if not key or matches <= 0:
            continue
        gf = float(row.get("scoresFor") or 0)
        ga = float(row.get("scoresAgainst") or 0)
        metrics[key] = {"gf": gf / matches, "ga": ga / matches}
        total_goals += gf
        total_matches += matches

    league_goal_rate = total_goals / total_matches if total_matches else 1.35
    events = fetch_upcoming_events(upcoming_season_id)
    context: dict[str, dict[str, Any]] = {}

    def add(team: str, opponent: str, home: bool) -> None:
        key = normalize_name(team)
        if not key or key in context:
            return
        opp = metrics.get(normalize_name(opponent))
        if opp:
            opponent_defence = opp["ga"] / league_goal_rate
            opponent_attack = opp["gf"] / league_goal_rate
        else:
            opponent_defence = opponent_attack = 1.0
        attack_mult = opponent_defence * (1.06 if home else 0.94)
        cs_mult = (1.0 / max(opponent_attack, 0.55)) * (1.08 if home else 0.92)
        context[key] = {
            "opponent": opponent,
            "home": home,
            "attack_mult": max(0.78, min(1.22, attack_mult)),
            "cs_mult": max(FIXTURE_CS_FLOOR, min(FIXTURE_CS_CEILING, cs_mult)),
        }

    for event in events:
        home = str((event.get("homeTeam") or {}).get("name") or "")
        away = str((event.get("awayTeam") or {}).get("name") or "")
        if home and away:
            add(home, away, True)
            add(away, home, False)
        if len(context) >= 18:
            break
    return context


def _season_median_apps(df: pd.DataFrame) -> float:
    if df is None or df.empty or "mp" not in df.columns:
        return 0.0
    mp = pd.to_numeric(df["mp"], errors="coerce").fillna(0.0)
    active = mp[mp > 0]
    return float(active.median()) if not active.empty else 0.0


def _merge_top_players(top: dict[str, list]) -> dict[int, dict[str, Any]]:
    """Sofascore topPlayers kategorilerini player_id bazında birleştir."""
    by_id: dict[int, dict[str, Any]] = {}

    def ensure(item: dict[str, Any]) -> dict[str, Any]:
        p = item.get("player") or {}
        pid = int(p["id"])
        if pid not in by_id:
            team = item.get("team") or {}
            by_id[pid] = {
                "player_id": pid,
                "player": p.get("name", ""),
                "squad": team.get("name", ""),
                "pos": p.get("position") or item.get("position") or "",
                "mp": 0,
                "gls": 0,
                "ast": 0,
                "crdy": 0,
                "crdr": 0,
                "saves": 0,
                "cs": 0,
                "ga": 0,
                "int": 0,
                "tkl": 0,
                "xg": 0.0,
                "xa": 0.0,
                "minutes": 0.0,
                "rating": 0.0,
            }
        return by_id[pid]

    mapping = {
        "goals": "gls",
        "assists": "ast",
        "yellowCards": "crdy",
        "redCards": "crdr",
        "saves": "saves",
        "cleanSheet": "cs",
        "interceptions": "int",
        "tackles": "tkl",
        "mostConceded": "ga",
    }
    for cat, items in (top or {}).items():
        if not isinstance(items, list):
            continue
        for item in items:
            row = ensure(item)
            st = item.get("statistics") or {}
            apps = st.get("appearances")
            if apps is not None:
                row["mp"] = max(int(row["mp"] or 0), int(apps))
            if cat == "rating" and st.get("rating") is not None:
                row["rating"] = float(st["rating"])
            field = mapping.get(cat)
            if field and st.get(cat if cat not in mapping else cat) is not None:
                val = st.get(cat)
                if val is None and field in st:
                    val = st[field]
                if val is not None:
                    row[field] = val
            for sk, dk in (
                ("goals", "gls"),
                ("assists", "ast"),
                ("yellowCards", "crdy"),
                ("redCards", "crdr"),
                ("saves", "saves"),
                ("cleanSheet", "cs"),
                ("interceptions", "int"),
                ("tackles", "tkl"),
            ):
                if sk in st and st[sk] is not None:
                    row[dk] = st[sk]
    return by_id


def fetch_team_season_top(team_id: int, season_id: int) -> dict[int, dict[str, Any]]:
    data = sofa_get(
        f"/team/{team_id}/unique-tournament/{UNIQUE_TOURNAMENT_ID}/season/{season_id}/top-players/overall",
        cache_key=f"sofa_team_top_{team_id}_{season_id}",
        max_age_hours=18,
        delay=0.25,
    )
    return _merge_top_players(data.get("topPlayers") or {})


def fetch_league_top(season_id: int) -> dict[int, dict[str, Any]]:
    data = sofa_get(
        f"/unique-tournament/{UNIQUE_TOURNAMENT_ID}/season/{season_id}/top-players/overall",
        cache_key=f"sofa_league_top_{season_id}",
        max_age_hours=18,
        delay=0.4,
    )
    return _merge_top_players(data.get("topPlayers") or {})


def _num(st: dict[str, Any], *keys: str) -> float | None:
    for k in keys:
        v = st.get(k)
        if v is not None and v != "":
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def apply_overall_fields(row: dict[str, Any], st: dict[str, Any]) -> dict[str, Any]:
    """Sofascore overall → ortak kolonlar (gol, xG, şut, kilit pas, penaltı...)."""
    if not st:
        return row

    def set_if(dst: str, *keys: str) -> None:
        v = _num(st, *keys)
        if v is not None:
            row[dst] = v

    set_if("mp", "appearances")
    set_if("minutes", "minutesPlayed")
    set_if("gls", "goals")
    set_if("ast", "assists")
    set_if("crdy", "yellowCards")
    set_if("crdr", "redCards")
    set_if("saves", "saves")
    set_if("cs", "cleanSheet")
    set_if("ga", "goalsConceded")
    set_if("int", "interceptions")
    set_if("tkl", "tackles")
    set_if("xg", "expectedGoals")
    set_if("xa", "expectedAssists")
    set_if("rating", "rating")
    set_if("pen_save", "penaltySave")
    set_if("pen_miss", "attemptPenaltyMiss", "penaltyMiss")
    set_if("og", "ownGoals")
    set_if("key_passes", "keyPasses")
    set_if("bcc", "bigChancesCreated")
    set_if("bcm", "bigChancesMissed")
    set_if("sot", "onTargetScoringAttempt", "shotsOnTarget")
    set_if("shots", "totalShots", "shots")
    set_if("dribbles", "successfulDribbles")
    set_if("pen_won", "penaltyWon")
    return row


def fetch_player_overall(player_id: int, season_id: int) -> dict[str, Any]:
    data = sofa_get(
        f"/player/{player_id}/unique-tournament/{UNIQUE_TOURNAMENT_ID}/season/{season_id}/statistics/overall",
        cache_key=f"sofa_pstat_{player_id}_{season_id}",
        max_age_hours=24,
        delay=0.15,
    )
    return data.get("statistics") or {}


def enrich_with_overall(
    players: dict[int, dict[str, Any]],
    season_id: int,
    *,
    max_workers: int = 6,
) -> dict[int, dict[str, Any]]:
    """Dakika / kart / clean sheet vb. için overall endpoint (cache'li)."""
    need = [
        pid
        for pid, r in players.items()
        if not r.get("minutes") or (r.get("mp") or 0) > 0
    ]
    need = [pid for pid in players if not players[pid].get("minutes")]
    if not need:
        need = list(players.keys())

    def one(pid: int) -> tuple[int, dict[str, Any]]:
        try:
            return pid, fetch_player_overall(pid, season_id)
        except Exception:
            return pid, {}

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(one, pid) for pid in need]
        for fut in as_completed(futs):
            pid, st = fut.result()
            done += 1
            if done % 50 == 0:
                _log(f"    overall {done}/{len(need)}...")
            if not st or pid not in players:
                continue
            apply_overall_fields(players[pid], st)
    return players


def fetch_season_player_stats(year_start: int | None, *, enrich: bool = True) -> pd.DataFrame:
    if year_start is None:
        year_start = discover_current_season_start()
    try:
        season_id = resolve_season_id(year_start)
    except RuntimeError as exc:
        warnings.warn(str(exc), stacklevel=2)
        return pd.DataFrame()

    teams = fetch_standings_teams(season_id)
    merged: dict[int, dict[str, Any]] = {}
    for t in teams:
        try:
            part = fetch_team_season_top(int(t["id"]), season_id)
        except SofaNotFound:
            continue
        except Exception as exc:
            if "404" in str(exc) or _sofa_blocked:
                continue
            warnings.warn(f"Takım {t.get('name')}: {exc}", stacklevel=2)
            continue
        for pid, row in part.items():
            if pid not in merged:
                merged[pid] = row
            else:
                for k, v in row.items():
                    if k in ("player", "squad", "pos") and v:
                        merged[pid][k] = v
                    elif isinstance(v, (int, float)) and v:
                        if not merged[pid].get(k):
                            merged[pid][k] = v

    try:
        league = fetch_league_top(season_id)
        for pid, row in league.items():
            if pid not in merged:
                merged[pid] = row
            else:
                for k in ("saves", "cs", "ga", "gls", "ast", "int", "tkl", "crdy", "crdr"):
                    if row.get(k) and not merged[pid].get(k):
                        merged[pid][k] = row[k]
    except SofaNotFound:
        pass
    except Exception as exc:
        if "404" not in str(exc):
            warnings.warn(f"Lig top: {exc}", stacklevel=2)

    if enrich and merged:
        _log(f"  Overall istatistik zenginleştirme ({len(merged)} oyuncu, cache'li)...")
        merged = enrich_with_overall(merged, season_id)

    rows = []
    for row in merged.values():
        mins = float(row.get("minutes") or 0)
        mp = float(row.get("mp") or 0)
        if mins <= 0 and mp > 0:
            mins = mp * 75.0
        n90 = mins / 90.0 if mins else mp
        rows.append(
            {
                "player": row.get("player", ""),
                "squad": row.get("squad", ""),
                "pos": _map_sofa_pos(row.get("pos")),
                "mp": mp,
                "minutes": mins,
                "nineties": n90,
                "gls": float(row.get("gls") or 0),
                "ast": float(row.get("ast") or 0),
                "crdy": float(row.get("crdy") or 0),
                "crdr": float(row.get("crdr") or 0),
                "saves": float(row.get("saves") or 0),
                "ga": float(row.get("ga") or 0),
                "cs": float(row.get("cs") or 0),
                "int": float(row.get("int") or 0),
                "tkl": float(row.get("tkl") or 0),
                "xg": float(row.get("xg") or 0),
                "xa": float(row.get("xa") or 0),
                "rating": float(row.get("rating") or 0),
                "pen_save": float(row.get("pen_save") or 0),
                "pen_miss": float(row.get("pen_miss") or 0),
                "og": float(row.get("og") or 0),
                "key_passes": float(row.get("key_passes") or 0),
                "bcc": float(row.get("bcc") or 0),
                "bcm": float(row.get("bcm") or 0),
                "sot": float(row.get("sot") or 0),
                "shots": float(row.get("shots") or 0),
                "dribbles": float(row.get("dribbles") or 0),
                "pen_won": float(row.get("pen_won") or 0),
                "player_id": row.get("player_id"),
                "season": season_label(year_start),
                "season_start": year_start,
            }
        )
    return pd.DataFrame(rows)


def _map_sofa_pos(pos: Any) -> str:
    if not pos:
        return ""
    p = str(pos).upper().strip()
    if p in ("G", "GK"):
        return "GK"
    if p in ("D", "DF"):
        return "DF"
    if p in ("M", "MF"):
        return "MF"
    if p in ("F", "FW", "A"):
        return "FW"
    return p


def fetch_recent_events(season_id: int, max_pages: int = 4) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for page in range(max_pages):
        try:
            data = sofa_get(
                f"/unique-tournament/{UNIQUE_TOURNAMENT_ID}/season/{season_id}/events/last/{page}",
                cache_key=f"sofa_events_{season_id}_{page}",
                max_age_hours=6,
                delay=0.3,
            )
        except SofaNotFound:
            break
        batch = data.get("events") or []
        events.extend(batch)
        if not data.get("hasNextPage"):
            break
    out = []
    for e in events:
        st = (e.get("status") or {}).get("type")
        if st in ("finished", "ended") or e.get("homeScore"):
            out.append(e)
    out.sort(key=lambda x: x.get("startTimestamp") or 0, reverse=True)
    return out


def _empty_agg() -> dict[str, float]:
    return {
        "mp": 0.0,
        "minutes": 0.0,
        "apps_60": 0.0,
        "gls": 0.0,
        "ast": 0.0,
        "crdy": 0.0,
        "crdr": 0.0,
        "saves": 0.0,
        "ga": 0.0,
        "cs": 0.0,
        "int": 0.0,
        "tkl": 0.0,
        "xg": 0.0,
        "xa": 0.0,
        "key_passes": 0.0,
        "bcc": 0.0,
        "sot": 0.0,
        "shots": 0.0,
        "rating_sum": 0.0,
        "rating_n": 0.0,
    }


def aggregate_form_from_lineups(
    season_id: int,
    form_matches: int = FORM_MATCHES,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Son maçlardan (takım başına ~form_matches) lineups toplayıp oyuncu form tablosu üretir.
    Ayrıca takım clean sheet oranı döner.
    """
    events = fetch_recent_events(season_id, max_pages=5)
    team_apps: dict[int, int] = {}
    team_cs: dict[str, list[float]] = {}
    player_meta: dict[int, dict[str, Any]] = {}
    player_agg: dict[int, dict[str, float]] = {}
    player_team_apps: dict[int, int] = {}

    used_events = 0
    for e in events:
        hid = int((e.get("homeTeam") or {}).get("id") or 0)
        aid = int((e.get("awayTeam") or {}).get("id") or 0)
        hname = (e.get("homeTeam") or {}).get("name", "")
        aname = (e.get("awayTeam") or {}).get("name", "")
        if not hid or not aid:
            continue
        if team_apps.get(hid, 0) >= form_matches and team_apps.get(aid, 0) >= form_matches:
            continue
        if team_apps.get(hid, 0) >= form_matches + 2 and team_apps.get(aid, 0) >= form_matches + 2:
            continue

        eid = e["id"]
        try:
            lineups = sofa_get(
                f"/event/{eid}/lineups",
                cache_key=f"sofa_lineup_{eid}",
                max_age_hours=48,
                delay=0.2,
            )
        except Exception:
            continue

        hs = (e.get("homeScore") or {}).get("current")
        aws = (e.get("awayScore") or {}).get("current")
        if hs is None:
            hs = (e.get("homeScore") or {}).get("display")
        if aws is None:
            aws = (e.get("awayScore") or {}).get("display")
        try:
            hs_i, aw_i = int(hs), int(aws)
        except (TypeError, ValueError):
            continue

        for side, team_id, team_name, scored, conceded in (
            ("home", hid, hname, hs_i, aw_i),
            ("away", aid, aname, aw_i, hs_i),
        ):
            if team_apps.get(team_id, 0) >= form_matches:
                continue
            team_apps[team_id] = team_apps.get(team_id, 0) + 1
            team_cs.setdefault(team_name, []).append(1.0 if conceded == 0 else 0.0)

            for pl in (lineups.get(side) or {}).get("players") or []:
                p = pl.get("player") or {}
                pid = p.get("id")
                if not pid:
                    continue
                pid = int(pid)
                st = pl.get("statistics") or {}
                if not st:
                    continue
                if pid not in player_meta:
                    player_meta[pid] = {
                        "player": p.get("name", ""),
                        "squad": team_name,
                        "pos": _map_sofa_pos(p.get("position") or pl.get("position")),
                        "player_id": pid,
                    }
                player_team_apps[pid] = player_team_apps.get(pid, 0) + 1
                if player_team_apps[pid] > form_matches:
                    continue
                agg = player_agg.setdefault(pid, _empty_agg())
                mins = float(st.get("minutesPlayed") or 0)
                if mins <= 0:
                    continue
                agg["mp"] += 1
                agg["minutes"] += mins
                if mins >= 60:
                    agg["apps_60"] += 1
                agg["gls"] += float(st.get("goals") or 0)
                agg["ast"] += float(st.get("goalAssist") or st.get("assists") or 0)
                agg["crdy"] += float(st.get("yellowCard") or st.get("yellowCards") or 0)
                agg["crdr"] += float(st.get("redCard") or st.get("redCards") or 0)
                agg["saves"] += float(st.get("saves") or 0)
                agg["int"] += float(st.get("interceptionWon") or st.get("interceptions") or 0)
                agg["tkl"] += float(st.get("totalTackle") or st.get("tackles") or 0)
                agg["xg"] += float(st.get("expectedGoals") or 0)
                agg["xa"] += float(st.get("expectedAssists") or 0)
                agg["key_passes"] += float(st.get("keyPass") or st.get("keyPasses") or 0)
                agg["bcc"] += float(st.get("bigChanceCreated") or st.get("bigChancesCreated") or 0)
                agg["sot"] += float(
                    st.get("onTargetScoringAttempt") or st.get("shotsOnTarget") or 0
                )
                agg["shots"] += float(st.get("totalShots") or st.get("shots") or 0)
                rat = st.get("rating")
                if rat:
                    agg["rating_sum"] += float(rat)
                    agg["rating_n"] += 1
                if mins >= 60 and conceded == 0:
                    agg["cs"] += 1
                if mins >= 60:
                    agg["ga"] += conceded
                used_events += 1

        if teams_enough := sum(1 for v in team_apps.values() if v >= form_matches):
            if teams_enough >= 16 and used_events > 50:
                pass

    form_cs = {t: (sum(v) / len(v) if v else 0.0) for t, v in team_cs.items()}

    rows = []
    for pid, agg in player_agg.items():
        meta = player_meta.get(pid, {})
        mins = agg["minutes"]
        mp = agg["mp"]
        if mp <= 0:
            continue
        rows.append(
            {
                "player": meta.get("player", ""),
                "squad": meta.get("squad", ""),
                "pos": meta.get("pos", ""),
                "mp": mp,
                "minutes": mins,
                "nineties": mins / 90.0 if mins else 0,
                "gls": agg["gls"],
                "ast": agg["ast"],
                "crdy": agg["crdy"],
                "crdr": agg["crdr"],
                "saves": agg["saves"],
                "ga": agg["ga"],
                "cs": agg["cs"],
                "int": agg["int"],
                "tkl": agg["tkl"],
                "xg": agg.get("xg", 0.0),
                "xa": agg.get("xa", 0.0),
                "key_passes": agg.get("key_passes", 0.0),
                "bcc": agg.get("bcc", 0.0),
                "sot": agg.get("sot", 0.0),
                "shots": agg.get("shots", 0.0),
                "apps_60": agg.get("apps_60", 0.0),
                "rating": (
                    agg["rating_sum"] / agg["rating_n"]
                    if agg.get("rating_n")
                    else 0.0
                ),
                "player_id": pid,
                "season": f"form_L{form_matches}",
                "season_start": -2,
            }
        )
    return pd.DataFrame(rows), form_cs


def team_season_clean_sheet_rate_from_df(df: pd.DataFrame) -> dict[str, float]:
    """Sezon oyuncu CS toplamlarından kabaca takım oranı üretme (yedek)."""
    if df.empty or "cs" not in df.columns:
        return {}
    gks = df[df["pos"].isin(["GK", "G"])] if "pos" in df.columns else df
    rates: dict[str, float] = {}
    for squad, grp in gks.groupby("squad"):
        mp = float(pd.to_numeric(grp["mp"], errors="coerce").fillna(0).max() or 0)
        if mp < 1:
            continue
        cs = float(pd.to_numeric(grp["cs"], errors="coerce").fillna(0).max() or 0)
        rates[str(squad)] = cs / mp
    return rates


def team_season_apps_from_df(df: pd.DataFrame) -> dict[str, float]:
    """Takım başına mevcut sezon örneklem büyüklüğü (kaleci mp)."""
    if df.empty or "mp" not in df.columns:
        return {}
    gks = df[df["pos"].isin(["GK", "G"])] if "pos" in df.columns else df
    apps: dict[str, float] = {}
    for squad, grp in gks.groupby("squad"):
        mp = float(pd.to_numeric(grp["mp"], errors="coerce").fillna(0).max() or 0)
        if mp > 0:
            apps[str(squad)] = mp
    return apps


def blend_team_cs_rates(
    current: dict[str, float],
    previous: dict[str, float],
    current_apps: dict[str, float] | None = None,
    *,
    prior_matches: float = 8.0,
    league_default: float = 0.28,
) -> dict[str, float]:
    """
    Erken sezon tek maç CS=%100 / oynamayan takım CS=%0 gürültüsünü
    önceki sezonla sürekli Bayes ağırlığıyla küçültür (sert 4 maç cliff yok).
    """
    current_apps = current_apps or {}
    teams = set(current) | set(previous) | set(current_apps)
    out: dict[str, float] = {}
    for team in teams:
        n = float(current_apps.get(team) or 0.0)
        curr = current.get(team)
        prev = previous.get(team, league_default)
        if curr is None or n <= 0:
            out[team] = float(prev)
            continue
        weight = n / (n + float(prior_matches))
        out[team] = weight * float(curr) + (1.0 - weight) * float(prev)
    return out


def load_dual_season_stats(
    current_start: int | None = None,
    form_matches: int = FORM_MATCHES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, float], dict[str, Any]]:
    """
    form kaynağı: son L{form_matches} maç lineups (gerçek form).
    current_df: form tablosu (scoring form olarak kullanır).
    prev_df / base: baz için mevcut sezon ≥8 maç veya önceki sezon — scoring tarafında.
    Burada prev = önceki sezon toplam; current_season ayrı da verilir meta içine.
    """
    if current_start is None:
        current_start = discover_current_season_start()
    requested_start = current_start
    prev_start = current_start - 1
    meta: dict[str, Any] = {
        "current_start": current_start,
        "prev_start": prev_start,
        "requested_start": requested_start,
        "form_matches": form_matches,
        "source": "sofascore",
        "notes": [],
        "preseason": False,
    }

    _log(f"  Sezon baz verisi: {season_label(current_start)} ...")
    current_season = fetch_season_player_stats(current_start, enrich=True)
    mp_sum = float(pd.to_numeric(current_season.get("mp"), errors="coerce").fillna(0).sum()) if not current_season.empty else 0.0
    if current_season.empty or mp_sum < 80:
        why = "boş" if current_season.empty else "henüz maç yok"
        meta["notes"].append(
            f"{season_label(current_start)} {why}; {season_label(prev_start)} deneniyor."
        )
        meta["preseason"] = True
        current_season = fetch_season_player_stats(prev_start, enrich=True)
        current_start = prev_start
        prev_start = current_start - 1
        meta["current_start"] = current_start
        meta["prev_start"] = prev_start

    _log(f"  Önceki sezon baz: {season_label(prev_start)} ...")
    prev_season = fetch_season_player_stats(prev_start, enrich=True)

    form_cs: dict[str, float] = {}
    form_df = pd.DataFrame()
    form_by_season: dict[int, tuple[pd.DataFrame, dict[str, float]]] = {}
    form_season_starts: list[int] = []
    for start in (requested_start, current_start, requested_start - 1, prev_start):
        if start not in form_season_starts:
            form_season_starts.append(start)
    for form_start in form_season_starts:
        try:
            sid = resolve_season_id(form_start)
            _log(
                f"  Form L{form_matches}: son maç lineups "
                f"({season_label(form_start)}, season_id={sid})..."
            )
            candidate_df, candidate_cs = aggregate_form_from_lineups(
                sid, form_matches=form_matches
            )
            if candidate_df.empty:
                meta["notes"].append(
                    f"{season_label(form_start)} form lineups boş."
                )
                continue
            form_by_season[form_start] = (candidate_df, candidate_cs)
        except SofaNotFound:
            meta["notes"].append(
                f"{season_label(form_start)} form maçları henüz yok."
            )
        except Exception as exc:
            meta["notes"].append(
                f"{season_label(form_start)} form lineups alınamadı ({exc})."
            )

    primary_start = next((s for s in form_season_starts if s in form_by_season), None)
    if primary_start is not None:
        form_df, form_cs = form_by_season[primary_start]
        meta["form_season_start"] = primary_start
        meta["notes"].append(
            f"Oyuncu formu: {season_label(primary_start)} son ~{form_matches} maç "
            f"lineups (Sofascore); {len(form_df)} oyuncu, {len(form_cs)} takım CS."
        )
        fallback_start = next(
            (s for s in form_season_starts if s != primary_start and s in form_by_season),
            None,
        )
        if fallback_start is not None:
            fallback_df, fallback_cs = form_by_season[fallback_start]
            primary_teams = {
                normalize_name(str(t))
                for t in form_df.get("squad", pd.Series(dtype=str)).dropna().tolist()
            }
            missing_mask = ~fallback_df.get("squad", pd.Series(dtype=str)).map(
                lambda t: normalize_name(str(t)) in primary_teams
            )
            extra = fallback_df.loc[missing_mask]
            if not extra.empty:
                form_df = pd.concat([form_df, extra], ignore_index=True)
                for team, rate in fallback_cs.items():
                    key = normalize_name(team)
                    if key and key not in {normalize_name(t) for t in form_cs}:
                        form_cs[team] = rate
                meta["notes"].append(
                    f"Yeni sezonda maçı olmayan takımlar için "
                    f"{season_label(fallback_start)} form fallback: {len(extra)} oyuncu."
                )
    else:
        meta["notes"].append("Form proxy: mevcut sezon oranları.")
        form_df = current_season.copy()
        form_cs = team_season_clean_sheet_rate_from_df(current_season)

    if form_df.empty and not current_season.empty:
        form_df = current_season.copy()
        meta["notes"].append("Form boştu → mevcut sezon proxy.")

    base_cs_current = team_season_clean_sheet_rate_from_df(current_season)
    base_cs_prev = team_season_clean_sheet_rate_from_df(prev_season)
    base_cs = blend_team_cs_rates(
        base_cs_current,
        base_cs_prev,
        team_season_apps_from_df(current_season),
        prior_matches=TEAM_CS_PRIOR_MATCHES,
    )
    if not base_cs:
        base_cs = base_cs_prev or base_cs_current
    if meta.get("early_season") or meta.get("preseason"):
        meta["notes"].append(
            "Takım CS oranı erken sezonda önceki sezonla küçültüldü "
            "(tek maçlık %0/%100 gürültüsü engellendi)."
        )

    meta["current_season_rows"] = len(current_season)
    meta["prev_season_rows"] = len(prev_season)
    meta["form_rows"] = len(form_df)
    maturity_apps = _season_median_apps(
        current_season if not meta["preseason"] else pd.DataFrame()
    )
    meta["season_maturity_apps"] = maturity_apps
    meta["early_season"] = bool(meta["preseason"] or maturity_apps < 4.0)
    if meta["early_season"] and not meta["preseason"]:
        meta["notes"].append(
            f"Erken sezon: mevcut sezon medyan maç ~{maturity_apps:.1f}; "
            "rakip gücü için önceki sezon standings tercih edilir."
        )

    fixture_context: dict[str, dict[str, Any]] = {}
    try:
        upcoming_sid = resolve_season_id(requested_start)
        strength_sid = resolve_season_id(current_start)
        fallback_sid = resolve_season_id(requested_start - 1)
        fixture_context = build_fixture_context(
            upcoming_sid,
            strength_sid,
            fallback_strength_season_id=fallback_sid,
            prefer_fallback=bool(meta["early_season"]),
        )
        if fixture_context:
            strength_note = (
                f"önceki sezon güç ({season_label(requested_start - 1)})"
                if meta["early_season"]
                else f"mevcut sezon güç ({season_label(current_start)})"
            )
            meta["notes"].append(
                f"Haftalık rakip/iç-dış saha: {len(fixture_context)} takım; {strength_note}."
            )
    except Exception as exc:
        meta["notes"].append(f"Fikstür etkisi alınamadı ({exc}); nötr rakip varsayıldı.")

    return form_df, prev_season, form_cs, base_cs, {
        **meta,
        "current_season_df": current_season,
        "fixture_context": fixture_context,
    }
