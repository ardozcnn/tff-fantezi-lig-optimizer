"""Süper Lig dışı Sofascore önceliği — yeni transferler (Salah, Trossard, Asensio...)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable
from urllib.parse import quote

import pandas as pd

from .fetch_stats import apply_overall_fields, sofa_get
from .names import best_match, name_variants, normalize_name
from .scoring import expected_points_from_rates, map_position, rates_from_totals

# Sofascore uniqueTournament id → lig adı
DOMESTIC_LEAGUES: dict[int, str] = {
    17: "Premier League",
    8: "LaLiga",
    23: "Serie A",
    35: "Bundesliga",
    34: "Ligue 1",
    37: "Eredivisie",
    238: "Belgian Pro League",
    215: "Liga Portugal",
    44: "Liga Portugal",
    18: "Championship",
    172: "Championship",
    955: "Saudi Pro League",
    196: "MLS",
    242: "Liga MX",
    45: "Superliga",
    170: "Bundesliga 2",
    36: "2. Bundesliga",
    182: "Ligue 2",
    53: "Süper Lig",
    52: "Süper Lig",
    373: "UEFA Europa League",
    679: "UEFA Europa League",
}

# Dış lig hücum çarpanı (üst lig → SL biraz daha kolay)
ATTACK_MULT = {
    17: 1.10,
    8: 1.07,
    23: 1.07,
    35: 1.07,
    34: 1.07,
    7: 1.02,  # UCL
    679: 1.05,
    373: 1.05,
}
CS_MULT = {
    17: 0.92,
    8: 0.94,
    23: 0.94,
    35: 0.93,
    34: 0.94,
}

EURO_IDS = {7, 679, 373}  # UCL, UEL, UECL

SKIP_TOURNAMENTS = {
    16,  # World Cup
    11,  # WC Qual UEFA
    13,  # WC Qual CAF
    270,  # AFCON
    1848,
    346,  # Community Shield
    21,  # EFL Cup
    19,  # FA Cup
    10783,  # Nations League
    133,  # Copa America
}

ProgressCb = Callable[[str], None] | None


def search_players(query: str) -> list[dict[str, Any]]:
    if not query or len(query.strip()) < 2:
        return []
    q = query.strip()
    try:
        data = sofa_get(
            f"/search/all?q={quote(q)}",
            cache_key=f"sofa_search_{normalize_name(q).replace(' ', '_')[:80]}",
            max_age_hours=72,
            delay=0.18,
        )
    except Exception:
        return []
    out = []
    for item in data.get("results") or []:
        if item.get("type") != "player":
            continue
        e = item.get("entity") or {}
        if not e.get("id"):
            continue
        team = e.get("team") or {}
        out.append(
            {
                "player_id": int(e["id"]),
                "player": e.get("name") or "",
                "short_name": e.get("shortName") or "",
                "pos": e.get("position") or "",
                "squad": team.get("name") or "",
            }
        )
    return out


def pick_search_hit(
    query: str,
    team_hint: str,
    hits: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not hits:
        return None
    team_n = normalize_name(team_hint)
    # 1) Aynı kulüp
    if team_n:
        same = [h for h in hits if team_n and normalize_name(h.get("squad") or "") == team_n]
        if not same:
            same = [
                h
                for h in hits
                if team_n
                and (
                    team_n in normalize_name(h.get("squad") or "")
                    or normalize_name(h.get("squad") or "") in team_n
                )
            ]
        if same:
            names = [h["player"] for h in same]
            matched, _ = best_match(query, names, score_cutoff=55)
            if matched:
                return next(h for h in same if h["player"] == matched)
            return same[0]

    names = [h["player"] for h in hits]
    matched, score = best_match(query, names, score_cutoff=82)
    if matched and score >= 82:
        return next(h for h in hits if h["player"] == matched)
    # İlk isabet zayıfsa alma (yanlış Mohamed Salah)
    return None


def player_tournament_seasons(player_id: int) -> list[dict[str, Any]]:
    data = sofa_get(
        f"/player/{player_id}/statistics/seasons",
        cache_key=f"sofa_pseasons_{player_id}",
        max_age_hours=48,
        delay=0.15,
    )
    rows = []
    for ut in data.get("uniqueTournamentSeasons") or []:
        u = ut.get("uniqueTournament") or {}
        tid = u.get("id")
        if not tid or int(tid) in SKIP_TOURNAMENTS:
            continue
        tname = u.get("name") or DOMESTIC_LEAGUES.get(int(tid), "")
        for season in ut.get("seasons") or []:
            sid = season.get("id")
            if not sid:
                continue
            rows.append(
                {
                    "tournament_id": int(tid),
                    "tournament": tname,
                    "season_id": int(sid),
                    "year": season.get("year") or "",
                    "season_name": season.get("name") or "",
                    "domestic": int(tid) in DOMESTIC_LEAGUES,
                }
            )
    return rows


def fetch_overall_any(player_id: int, tournament_id: int, season_id: int) -> dict[str, Any]:
    data = sofa_get(
        f"/player/{player_id}/unique-tournament/{tournament_id}/season/{season_id}/statistics/overall",
        cache_key=f"sofa_pstat_{player_id}_{tournament_id}_{season_id}",
        max_age_hours=48,
        delay=0.12,
    )
    return data.get("statistics") or {}


def _apps(st: dict[str, Any]) -> float:
    return float(st.get("appearances") or 0)


def _gi_rate(st: dict[str, Any]) -> float:
    apps = max(_apps(st), 1.0)
    return (float(st.get("goals") or 0) + float(st.get("assists") or 0)) / apps


def _xg_rate(st: dict[str, Any]) -> float:
    apps = max(_apps(st), 1.0)
    return (float(st.get("expectedGoals") or 0) + float(st.get("expectedAssists") or 0)) / apps


_BLEND_KEYS = (
    "goals",
    "assists",
    "expectedGoals",
    "expectedAssists",
    "minutesPlayed",
    "yellowCards",
    "redCards",
    "saves",
    "cleanSheet",
    "goalsConceded",
    "interceptions",
    "tackles",
    "penaltySave",
    "attemptPenaltyMiss",
    "ownGoals",
    "keyPasses",
    "bigChancesCreated",
    "bigChancesMissed",
    "onTargetScoringAttempt",
    "totalShots",
    "successfulDribbles",
    "penaltyWon",
)


def mix_overall(parts: list[tuple[dict[str, Any], float]], scale_apps: float) -> dict[str, Any]:
    """Oranları ağırlıklı karıştır, sonra scale_apps maça çevir."""
    wsum = sum(w for _, w in parts) or 1.0
    out: dict[str, Any] = {}
    for key in _BLEND_KEYS:
        acc = 0.0
        for st, w in parts:
            apps = max(_apps(st), 1.0)
            acc += w * (float(st.get(key) or 0) / apps)
        out[key] = acc / wsum * scale_apps
    rating_acc = 0.0
    rw = 0.0
    for st, w in parts:
        r = float(st.get("rating") or 0)
        if r:
            rating_acc += w * r
            rw += w
    if rw:
        out["rating"] = rating_acc / rw
    out["appearances"] = scale_apps
    return out


def choose_prior_season(player_id: int) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Son 1–2 lig sezonunu (düşük yıl varsa öncekiyle) karıştır; Avrupa kupası ek kanıt."""
    seasons = player_tournament_seasons(player_id)
    if not seasons:
        return None

    domestic = [r for r in seasons if r["domestic"] and r["tournament_id"] not in (52, 53)]
    other = [
        r
        for r in seasons
        if not r["domestic"] and r["tournament_id"] not in (52, 53, *SKIP_TOURNAMENTS)
    ]
    domestic.sort(key=lambda r: r["year"], reverse=True)
    other.sort(key=lambda r: r["year"], reverse=True)

    collected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for cand in domestic[:6]:
        try:
            st = fetch_overall_any(player_id, cand["tournament_id"], cand["season_id"])
        except Exception:
            continue
        apps = _apps(st)
        mins = float(st.get("minutesPlayed") or 0)
        if apps < 6 and mins < 400:
            continue
        collected.append((cand, st))
        if len(collected) >= 2:
            break

    if not collected:
        for cand in other[:4]:
            try:
                st = fetch_overall_any(player_id, cand["tournament_id"], cand["season_id"])
            except Exception:
                continue
            apps = _apps(st)
            mins = float(st.get("minutesPlayed") or 0)
            if apps < 6 and mins < 400:
                continue
            collected.append((cand, st))
            if len(collected) >= 1:
                break

    if not collected:
        return None

    latest_meta, latest_st = collected[0]
    scale = max(_apps(latest_st), 10.0)
    parts: list[tuple[dict[str, Any], float]] = [(latest_st, 0.70)]
    note = latest_meta.get("tournament") or ""

    if len(collected) >= 2:
        prev_meta, prev_st = collected[1]
        latest_gi = _gi_rate(latest_st)
        prev_gi = _gi_rate(prev_st)
        latest_xg = _xg_rate(latest_st)
        prev_xg = _xg_rate(prev_st)
        # Düşük sezon (sakatlık / form) → önceki kapasiteyi atma (Salah vb.)
        down = (latest_gi < prev_gi * 0.62 and _apps(prev_st) >= 12) or (
            latest_xg > 0 and prev_xg > 0 and latest_xg < prev_xg * 0.62 and _apps(prev_st) >= 12
        )
        w_latest, w_prev = (0.48, 0.52) if down else (0.68, 0.32)
        parts = [(latest_st, w_latest), (prev_st, w_prev)]
        note = (
            f"{latest_meta.get('tournament')} {latest_meta.get('year')} "
            f"+ {prev_meta.get('tournament')} {prev_meta.get('year')} karışım"
        )
        latest_meta = {
            **latest_meta,
            "blend_note": note,
            "season_name": note,
        }

    # Aynı yıl Avrupa kupası: ek hücum kanıtı (düşük ağırlık)
    euro = [
        r
        for r in seasons
        if r["tournament_id"] in EURO_IDS and r.get("year") == latest_meta.get("year")
    ]
    for cand in euro[:1]:
        try:
            st = fetch_overall_any(player_id, cand["tournament_id"], cand["season_id"])
        except Exception:
            continue
        if _apps(st) >= 4:
            parts.append((st, 0.12))
            note = f"{note}; +{cand.get('tournament') or 'Avrupa'}"
            latest_meta["blend_note"] = note
            latest_meta["season_name"] = note

    blended = mix_overall(parts, scale)
    return latest_meta, blended


def recent_club_minutes(player_id: int, team_hint: str, max_pages: int = 5) -> float:
    """Yeni kulüp maçlarında (hazırlık / resmi) dakika."""
    team_n = normalize_name(team_hint)
    if not team_n:
        return 0.0
    total = 0.0
    seen = 0
    for page in range(max_pages):
        try:
            data = sofa_get(
                f"/player/{player_id}/events/last/{page}",
                cache_key=f"sofa_pevents_{player_id}_{page}",
                max_age_hours=12,
                delay=0.12,
            )
        except Exception:
            break
        events = data.get("events") or []
        if not events:
            break
        for e in events:
            home = normalize_name((e.get("homeTeam") or {}).get("name") or "")
            away = normalize_name((e.get("awayTeam") or {}).get("name") or "")
            club_match = team_n in home or team_n in away or home in team_n or away in team_n
            if not club_match:
                continue
            eid = e.get("id")
            if not eid:
                continue
            try:
                lineups = sofa_get(
                    f"/event/{eid}/lineups",
                    cache_key=f"sofa_lineup_{eid}",
                    max_age_hours=48,
                    delay=0.12,
                )
            except Exception:
                continue
            for side in ("home", "away"):
                for pl in (lineups.get(side) or {}).get("players") or []:
                    p = pl.get("player") or {}
                    if int(p.get("id") or 0) != int(player_id):
                        continue
                    st = pl.get("statistics") or {}
                    mins = float(st.get("minutesPlayed") or 0)
                    if mins <= 0 and pl.get("substitute") is False:
                        mins = 45.0
                    if mins > 0:
                        total += mins
                        seen += 1
            if seen >= 8:
                return total
    return total


def stats_row_from_overall(
    st: dict[str, Any],
    *,
    player: str,
    squad: str,
    pos: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    mins = float(st.get("minutesPlayed") or 0)
    mp = float(st.get("appearances") or 0)
    if mins <= 0 and mp > 0:
        mins = mp * 75.0
    row: dict[str, Any] = {
        "player": player,
        "squad": squad,
        "pos": pos,
        "mp": mp,
        "minutes": mins,
        "nineties": mins / 90.0 if mins else mp,
        "player_id": meta.get("player_id"),
        "season": meta.get("season_name") or meta.get("year") or "",
        "tournament": meta.get("tournament") or "",
        "tournament_id": meta.get("tournament_id"),
        "blend_note": meta.get("blend_note") or "",
    }
    apply_overall_fields(row, st)
    row["mp"] = float(row.get("mp") or mp)
    row["minutes"] = float(row.get("minutes") or mins)
    return row


def minutes_prior(price_m: float, last_min_per_app: float, friendly_min: float = 0.0) -> float:
    if price_m >= 10:
        prior = 85.0
    elif price_m >= 8:
        prior = 78.0
    elif price_m >= 6.5:
        prior = 68.0
    else:
        prior = 55.0
    if friendly_min >= 60:
        prior = max(prior, 80.0)
    if last_min_per_app > 0:
        return 0.45 * last_min_per_app + 0.55 * prior
    return prior


def appearance_prior(price_m: float, last_apps: float, friendly_min: float = 0.0) -> float:
    """Pahalı yıldız büyük ihtimalle XI; hazırlık dakikası varsa daha da net."""
    if friendly_min >= 45:
        return 0.94 if price_m >= 8 else 0.88
    if price_m >= 10:
        return 0.93
    if price_m >= 8:
        return 0.88
    if last_apps >= 20:
        return min(0.90, 0.58 + last_apps / 70.0)
    if price_m >= 6.5:
        return 0.78
    return 0.62


def project_external_player(
    row: dict[str, Any],
    *,
    tff_position: str,
    price_m: float,
    fixture_attack_mult: float = 1.0,
    fixture_cs_mult: float = 1.0,
) -> dict[str, Any]:
    rates = rates_from_totals(row)
    friendly_min = float(row.get("friendly_minutes") or 0)
    min_pa = minutes_prior(price_m, rates.get("min_per_app") or 0.0, friendly_min)
    rates["min_per_app"] = min_pa
    if min_pa >= 60:
        rates["share_60"] = min(1.0, 0.55 + (min_pa - 60) / 60.0)
    elif min_pa >= 1:
        rates["share_60"] = min_pa / 90.0 * 0.5
    else:
        rates["share_60"] = 0.0

    tid = int(row.get("tournament_id") or 0)
    attack_mult = (
        ATTACK_MULT.get(tid, 1.04 if row.get("tournament") else 1.0)
        * fixture_attack_mult
    )
    cs_mult = CS_MULT.get(tid, 1.0)
    pts = expected_points_from_rates(
        rates,
        tff_position,
        team_cs_rate=(rates.get("cs_rate") or 0.0) * cs_mult,
        appearance=appearance_prior(price_m, rates.get("apps") or 0.0, friendly_min),
        attack_mult=attack_mult,
        cs_mult=fixture_cs_mult,
    )
    season = str(row.get("season") or "").strip()
    tourn = str(row.get("tournament") or "").strip()
    blend = str(row.get("blend_note") or "").strip()
    if blend:
        src = blend
    elif tourn and tourn in season:
        src = season
    else:
        src = f"{tourn} {season}".strip() or "dış lig"
    exp_g = rates.get("gls_pa") or 0
    exp_a = rates.get("ast_pa") or 0
    reason = (
        f"Yeni transfer: {src}; "
        f"maç başı G/A kapasitesi ~{rates.get('xg_pa') or exp_g:.2f}/{rates.get('xa_pa') or exp_a:.2f} "
        f"(ham {row.get('gls', 0):.0f}G/{row.get('ast', 0):.0f}A, "
        f"xG/xA {row.get('xg', 0):.1f}/{row.get('xa', 0):.1f}, "
        f"{rates.get('apps', 0):.0f} maç ölçeği); "
        f"TFF {tff_position}"
    )
    return {
        "projected_pts": round(float(pts), 3),
        "base_pts": round(float(pts), 3),
        "form_pts": 0.0,
        "form_apps": 0.0,
        "base_apps": rates.get("apps") or 0.0,
        "min_per_app": round(min_pa, 1),
        "gls_pa": round(rates.get("gls_pa") or 0.0, 3),
        "ast_pa": round(rates.get("ast_pa") or 0.0, 3),
        "xg_pa": round(rates.get("xg_pa") or 0.0, 3),
        "xa_pa": round(rates.get("xa_pa") or 0.0, 3),
        "int_p90": round(rates.get("int_p90") or 0.0, 2),
        "tkl_p90": round(rates.get("tkl_p90") or 0.0, 2),
        "data_src": "external_prior",
        "base_src": src,
        "reason": reason,
        "stats_player": row.get("player"),
        "match_score": 100.0,
        "ext_league": row.get("tournament") or "",
        "ext_season": row.get("season") or "",
        "ext_gls": row.get("gls") or 0,
        "ext_ast": row.get("ast") or 0,
        "ext_xg": row.get("xg") or 0,
        "ext_xa": row.get("xa") or 0,
        "ext_apps": rates.get("apps") or 0,
        "ext_minutes": row.get("minutes") or 0,
        "rating": row.get("rating") or 0,
        "friendly_minutes": friendly_min,
    }


def _search_queries(player_name: str, extra_queries: list[str]) -> list[str]:
    queries: list[str] = []

    def add(q: str) -> None:
        q = (q or "").strip()
        if q and q not in queries:
            queries.append(q)

    for q in extra_queries:
        if q.strip() != player_name.strip():
            add(q)
    parts = [p for p in player_name.split() if p]
    if len(parts) >= 4:
        add(f"{parts[0]} {parts[-1]}")
        add(" ".join(parts[:2]))
    elif len(parts) >= 2:
        add(" ".join(parts[:2]))
        add(f"{parts[0]} {parts[-1]}")
    add(player_name)
    for v in name_variants(player_name):
        add(v)
    return queries


def resolve_one(
    player_name: str,
    team: str,
    extra_queries: list[str],
) -> dict[str, Any] | None:
    queries = _search_queries(player_name, extra_queries)

    hit = None
    for q in queries:
        hits = search_players(q)
        hit = pick_search_hit(player_name, team, hits)
        if hit:
            break
        if extra_queries:
            hit = pick_search_hit(extra_queries[0], team, hits)
            if hit:
                break
    if not hit:
        return None

    chosen = choose_prior_season(int(hit["player_id"]))
    if not chosen:
        return None
    meta, st = chosen
    meta = {**meta, "player_id": hit["player_id"]}
    pos = map_position(hit.get("pos")) or str(hit.get("pos") or "")
    row = stats_row_from_overall(
        st,
        player=hit["player"],
        squad=hit.get("squad") or team,
        pos=pos,
        meta=meta,
    )
    try:
        row["friendly_minutes"] = recent_club_minutes(int(hit["player_id"]), team)
    except Exception:
        row["friendly_minutes"] = 0.0
    return row


def apply_external_priors(
    merged: pd.DataFrame,
    *,
    progress: ProgressCb = None,
    min_price: float = 5.5,
    max_fetch: int = 90,
) -> pd.DataFrame:
    """SL örneği zayıf / eşleşmeyen pahalı oyunculara son lig sezonunu bağla."""
    df = merged.copy()
    for col in (
        "data_src",
        "xg_pa",
        "xa_pa",
        "ext_league",
        "ext_season",
        "ext_gls",
        "ext_ast",
        "ext_xg",
        "ext_xa",
        "ext_apps",
        "ext_minutes",
        "rating",
        "form_apps",
        "base_apps",
        "friendly_minutes",
    ):
        if col not in df.columns:
            df[col] = 0.0 if col not in ("data_src", "ext_league", "ext_season") else ""

    def needs_ext(r: pd.Series) -> bool:
        price = float(r.get("price_m") or 0)
        if price < min_price:
            return False
        unmatched = pd.isna(r.get("stats_player")) or not str(r.get("stats_player") or "").strip()
        sl_apps = float(r.get("form_apps") or 0) + float(r.get("base_apps") or 0)
        pts = float(r.get("projected_pts") or 0)
        if unmatched:
            return True
        # Eşleşmiş ve en az 4 SL maçı olan oyuncuyu eski dış lig verisiyle ezme.
        # Örn. 25/26'da 7 maçı olan Hajradinović'e 2018/19 HNL bağlanıyordu.
        if sl_apps >= 4:
            return False
        if sl_apps < 4:
            return True
        return False

    mask = df.apply(needs_ext, axis=1)
    todo = df.loc[mask].sort_values("price_m", ascending=False).head(max_fetch)
    if todo.empty:
        if progress:
            progress("Dış lig önceliği: eklenecek yıldız yok.")
        return df

    if progress:
        progress(f"Dış lig istatistikleri: {len(todo)} oyuncu (Salah, Trossard, Asensio...).")

    jobs: list[tuple[Any, dict[str, Any]]] = []
    for idx, r in todo.iterrows():
        extras = []
        for k in ("display_name", "match_name", "search_name"):
            v = str(r.get(k) or "").strip()
            if v:
                extras.append(v)
        jobs.append(
            (
                idx,
                {
                    "player_name": str(r["player"]),
                    "team": str(r.get("team") or ""),
                    "position": str(r.get("position") or "MF"),
                    "price_m": float(r["price_m"]),
                    "fixture_attack_mult": float(r.get("fixture_attack_mult") or 1.0),
                    "fixture_cs_mult": float(r.get("fixture_cs_mult") or 1.0),
                    "extras": extras,
                },
            )
        )

    def one(job: tuple[Any, dict[str, Any]]) -> tuple[Any, dict[str, Any] | None]:
        idx, spec = job
        try:
            raw = resolve_one(spec["player_name"], spec["team"], spec["extras"])
        except Exception:
            return idx, None
        if not raw:
            return idx, None
        proj = project_external_player(
            raw,
            tff_position=spec["position"],
            price_m=spec["price_m"],
            fixture_attack_mult=spec["fixture_attack_mult"],
            fixture_cs_mult=spec["fixture_cs_mult"],
        )
        return idx, proj

    done = 0
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(one, job) for job in jobs]
        for fut in as_completed(futs):
            idx, proj = fut.result()
            done += 1
            if progress and done % 8 == 0:
                progress(f"Dış lig: {done}/{len(jobs)}...")
            if not proj:
                continue
            sl_pts = float(df.at[idx, "projected_pts"] or 0)
            sl_apps = float(df.at[idx, "form_apps"] or 0) + float(df.at[idx, "base_apps"] or 0)
            if sl_apps >= 8 and sl_pts >= 2.0:
                continue
            if sl_apps >= 4 and sl_pts > 0:
                blended = 0.55 * sl_pts + 0.45 * proj["projected_pts"]
                proj["projected_pts"] = round(blended, 3)
                proj["reason"] = (
                    f"SL örnek sınırlı ({sl_apps:.0f} maç) + {proj['reason']}"
                )
            for k, v in proj.items():
                if k not in df.columns:
                    df[k] = None
                df.at[idx, k] = v

    df["projected_pts"] = pd.to_numeric(df["projected_pts"], errors="coerce").fillna(0.0)
    if progress:
        n = int((df["data_src"] == "external_prior").sum())
        progress(f"Dış lig önceliği bağlandı: {n} oyuncu.")
    return df
