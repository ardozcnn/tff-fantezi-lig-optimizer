"""TFF Fantezi puan tahmini (form + baz blend)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import (
    ASSIST_POINTS,
    AVAIL_MULT,
    BCC_TO_ASSIST,
    CONCEDED_PENALTY_PER_2,
    CS_POINTS,
    FORM_WEAK_APPS,
    GOAL_POINTS,
    KEYPASS_TO_ASSIST,
    LOW_SAMPLE_MULT,
    MIN_60_POINTS,
    MIN_APPS_FOR_CURRENT_BASE,
    MIN_FULL_POINTS,
    NEW_SIGNING_MULT,
    OWN_GOAL_PENALTY,
    PENALTY_MISS_PENALTY,
    PENALTY_SAVE_POINTS,
    POS_MAP,
    RED_PENALTY,
    SAVE_POINTS_PER_3,
    SOT_TO_GOAL,
    W_BASE_DEFAULT,
    W_BASE_WEAK,
    W_FORM_DEFAULT,
    W_FORM_WEAK,
    YELLOW_PENALTY,
)
from .names import normalize_name


def map_position(pos: str | float | None) -> str | None:
    if pos is None or (isinstance(pos, float) and pd.isna(pos)):
        return None
    raw = str(pos).upper().replace(",", " ").split()
    if not raw:
        return None
    primary = raw[0].strip()
    # Sofascore tek harf
    if primary in ("G", "GK"):
        return "GK"
    if primary in ("D", "DF"):
        return "DF"
    if primary in ("M", "MF"):
        return "MF"
    if primary in ("F", "FW", "A", "ST"):
        return "FW"
    if primary in POS_MAP:
        return POS_MAP[primary]
    for token in raw:
        t = token.strip()
        if t in POS_MAP:
            return POS_MAP[t]
    return None


def _safe_div(n: float, d: float) -> float:
    if d is None or d == 0 or pd.isna(d):
        return 0.0
    return float(n) / float(d)


def _empty_rates() -> dict[str, float]:
    return {
        "apps": 0.0,
        "min_per_app": 0.0,
        "share_60": 0.0,
        "gls_pa": 0.0,
        "ast_pa": 0.0,
        "xg_pa": 0.0,
        "xa_pa": 0.0,
        "yc_pa": 0.0,
        "rc_pa": 0.0,
        "saves_pa": 0.0,
        "ga_pa": 0.0,
        "cs_rate": 0.0,
        "int_p90": 0.0,
        "tkl_p90": 0.0,
        "rating": 0.0,
        "pen_save_pa": 0.0,
        "pen_miss_pa": 0.0,
        "og_pa": 0.0,
        "key_passes_pa": 0.0,
        "bcc_pa": 0.0,
        "bcm_pa": 0.0,
        "sot_pa": 0.0,
        "shots_pa": 0.0,
        "dribbles_pa": 0.0,
        "pen_won_pa": 0.0,
        "apps_60": 0.0,
    }


def _row_rates(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    if row is None:
        return _empty_rates()
    get = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    mp = float(get("mp") or 0) or 0.0
    minutes = float(get("minutes") or 0) or 0.0
    if mp <= 0 and minutes > 0:
        mp = max(minutes / 90.0, 1.0)
    n90 = float(get("nineties") or 0) or (minutes / 90.0 if minutes else mp)

    gls = float(get("gls") or 0) or 0.0
    ast = float(get("ast") or 0) or 0.0
    xg = float(get("xg") or 0) or 0.0
    xa = float(get("xa") or 0) or 0.0
    crdy = float(get("crdy") or 0) or 0.0
    crdr = float(get("crdr") or 0) or 0.0
    saves = float(get("saves") or 0) or 0.0
    ga = float(get("ga") or 0) or 0.0
    cs = float(get("cs") or 0) or 0.0
    inter = float(get("int") or 0) or 0.0
    tkl = float(get("tkl") or 0) or 0.0
    rating = float(get("rating") or 0) or 0.0
    pen_save = float(get("pen_save") or 0) or 0.0
    pen_miss = float(get("pen_miss") or 0) or 0.0
    og = float(get("og") or 0) or 0.0
    key_passes = float(get("key_passes") or 0) or 0.0
    bcc = float(get("bcc") or 0) or 0.0
    bcm = float(get("bcm") or 0) or 0.0
    sot = float(get("sot") or 0) or 0.0
    shots = float(get("shots") or 0) or 0.0
    dribbles = float(get("dribbles") or 0) or 0.0
    pen_won = float(get("pen_won") or 0) or 0.0
    apps_60 = float(get("apps_60") or 0) or 0.0

    apps = max(mp, 1.0) if (gls or ast or minutes or mp or xg or key_passes or sot) else 0.0
    if apps <= 0:
        return _empty_rates()

    min_per_app = minutes / apps if apps else 0.0
    if apps_60 > 0 and apps > 0:
        share_60 = min(1.0, apps_60 / apps)
    elif min_per_app >= 60:
        share_60 = min(1.0, 0.55 + (min_per_app - 60) / 60.0)
    elif min_per_app >= 1:
        share_60 = min_per_app / 90.0 * 0.5
    else:
        share_60 = 0.0

    return {
        "apps": apps,
        "min_per_app": min_per_app,
        "share_60": share_60,
        "gls_pa": gls / apps,
        "ast_pa": ast / apps,
        "xg_pa": xg / apps if xg else 0.0,
        "xa_pa": xa / apps if xa else 0.0,
        "yc_pa": crdy / apps,
        "rc_pa": crdr / apps,
        "saves_pa": saves / apps,
        "ga_pa": ga / apps if ga else _safe_div(ga, apps),
        "cs_rate": cs / apps if cs else 0.0,
        "int_p90": _safe_div(inter, n90) if n90 else 0.0,
        "tkl_p90": _safe_div(tkl, n90) if n90 else 0.0,
        "rating": rating,
        "pen_save_pa": pen_save / apps,
        "pen_miss_pa": pen_miss / apps,
        "og_pa": og / apps,
        "key_passes_pa": key_passes / apps,
        "bcc_pa": bcc / apps,
        "bcm_pa": bcm / apps,
        "sot_pa": sot / apps,
        "shots_pa": shots / apps,
        "dribbles_pa": dribbles / apps,
        "pen_won_pa": pen_won / apps,
        "apps_60": apps_60,
    }


def rates_from_totals(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    return _row_rates(row)


def _blend_attack(actual_pa: float, expected_pa: float, proxy_pa: float = 0.0) -> float:
    """xG/xA daha öngörülebilir; şut/kilit pas vekil; ham G/A şans gürültüsü."""
    parts: list[tuple[float, float]] = []
    if expected_pa and expected_pa > 0:
        parts.append((0.55, expected_pa))
        parts.append((0.30, actual_pa))
        if proxy_pa > 0:
            parts.append((0.15, proxy_pa))
    elif proxy_pa > 0:
        parts.append((0.45, actual_pa))
        parts.append((0.55, proxy_pa))
    else:
        return actual_pa
    wsum = sum(w for w, _ in parts)
    return sum(w * v for w, v in parts) / wsum if wsum else actual_pa


def _bonus_from_rating(rating: float) -> float:
    """Maçın en iyi 3'ü: 3/2/1. Rating yüksekse bonus beklenir."""
    if rating >= 7.8:
        return 1.15
    if rating >= 7.3:
        return 0.70
    if rating >= 6.9:
        return 0.35
    if rating >= 6.5:
        return 0.12
    return 0.04


def expected_points_from_rates(
    rates: dict[str, float],
    position: str,
    team_cs_rate: float | None = None,
    *,
    appearance: float | None = None,
    attack_mult: float = 1.0,
) -> float:
    """Bir maçlık beklenen TFF fantezi puanı (dakika, xG/xA, CS, kart, bonus, penaltı)."""
    if not rates or rates.get("apps", 0) <= 0:
        return 0.0
    pos = position if position in GOAL_POINTS else "MF"
    share_60 = rates["share_60"]
    if appearance is None:
        min_pa = rates.get("min_per_app") or 0.0
        appearance = 0.94 if min_pa >= 70 else (0.88 if min_pa >= 50 else 0.75)
    # 60 dk: 1p, 60+'dan fazla: 2p (resmi tablo)
    min_pts = appearance * (
        MIN_60_POINTS + share_60 * (MIN_FULL_POINTS - MIN_60_POINTS)
    )

    exp_gls = _blend_attack(
        rates.get("gls_pa") or 0.0,
        rates.get("xg_pa") or 0.0,
        (rates.get("sot_pa") or 0.0) * SOT_TO_GOAL,
    )
    kp_proxy = (rates.get("key_passes_pa") or 0.0) * KEYPASS_TO_ASSIST
    bcc_proxy = (rates.get("bcc_pa") or 0.0) * BCC_TO_ASSIST
    exp_ast = _blend_attack(
        rates.get("ast_pa") or 0.0,
        rates.get("xa_pa") or 0.0,
        max(kp_proxy, bcc_proxy),
    )
    goal_pts = exp_gls * GOAL_POINTS[pos] * attack_mult
    assist_pts = exp_ast * ASSIST_POINTS * attack_mult

    cs_rate = team_cs_rate if team_cs_rate is not None else rates.get("cs_rate", 0.0)
    if pos in ("GK", "DF"):
        cs_pts = cs_rate * share_60 * CS_POINTS[pos]
        ga_pa = rates.get("ga_pa") or (1.0 - cs_rate) * 1.2
        concede_pen = (ga_pa / 2.0) * CONCEDED_PENALTY_PER_2 * share_60
    elif pos == "MF":
        cs_pts = cs_rate * share_60 * CS_POINTS["MF"]
        concede_pen = 0.0
    else:
        cs_pts = 0.0
        concede_pen = 0.0

    save_pts = 0.0
    if pos == "GK":
        save_pts = (rates.get("saves_pa", 0.0) / 3.0) * SAVE_POINTS_PER_3
        save_pts += (rates.get("pen_save_pa") or 0.0) * PENALTY_SAVE_POINTS

    card_pen = rates.get("yc_pa", 0.0) * YELLOW_PENALTY + rates.get("rc_pa", 0.0) * RED_PENALTY
    pen_miss = (rates.get("pen_miss_pa") or 0.0) * PENALTY_MISS_PENALTY
    og_pen = (rates.get("og_pa") or 0.0) * OWN_GOAL_PENALTY
    bonus = appearance * _bonus_from_rating(rates.get("rating") or 0.0)

    return float(
        max(
            0.0,
            min_pts
            + goal_pts
            + assist_pts
            + cs_pts
            + save_pts
            + bonus
            - concede_pen
            - card_pen
            - pen_miss
            - og_pen,
        )
    )


def _squad_key(name: str) -> str:
    return normalize_name(name)


def blend_weights(form_apps: float) -> tuple[float, float]:
    if form_apps < FORM_WEAK_APPS:
        return W_FORM_WEAK, W_BASE_WEAK
    return W_FORM_DEFAULT, W_BASE_DEFAULT


def build_player_table(
    current: pd.DataFrame,
    prev: pd.DataFrame,
    form_cs: dict[str, float],
    base_cs: dict[str, float],
    meta: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Form = `current` (L6–L8 lineups veya sezon proxy).
    Baz = meta['current_season_df'] içinde ≥8 maç varsa o satır; yoksa `prev`.
    projected = w_form * form_pts + w_base * base_pts
    """
    meta = meta or {}
    current_season: pd.DataFrame = meta.get("current_season_df")
    if current_season is None or not isinstance(current_season, pd.DataFrame):
        current_season = pd.DataFrame()

    rows: list[dict[str, Any]] = []

    def index_by_player(df: pd.DataFrame) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        if df is None or df.empty:
            return out
        for _, r in df.iterrows():
            key = normalize_name(str(r.get("player", "")))
            if key:
                out[key] = r
        return out

    form_idx = index_by_player(current)
    cur_idx = index_by_player(current_season)
    prev_idx = index_by_player(prev)

    all_keys = set(form_idx) | set(cur_idx) | set(prev_idx)

    for key in all_keys:
        form_row = form_idx.get(key)
        cur = cur_idx.get(key)
        pr = prev_idx.get(key)
        sample = form_row if form_row is not None else (cur if cur is not None else pr)
        if sample is None:
            continue
        player = str(sample.get("player", ""))
        squad = str(
            (form_row if form_row is not None else sample).get("squad", "")
            or sample.get("squad", "")
        )
        pos_raw = (form_row if form_row is not None else sample).get("pos")
        if (pos_raw is None or str(pos_raw) in ("", "nan")) and cur is not None:
            pos_raw = cur.get("pos")
        if (pos_raw is None or str(pos_raw) in ("", "nan")) and pr is not None:
            pos_raw = pr.get("pos")
        position = map_position(pos_raw)
        # Sofascore pos already GK/DF/MF/FW
        if not position and pos_raw in ("GK", "DF", "MF", "FW"):
            position = str(pos_raw)
        if not position:
            continue

        form_rates = _row_rates(form_row) if form_row is not None else _empty_rates()
        form_apps = form_rates["apps"]

        # Baz seçimi: mevcut sezon ≥8 maç → current season; değilse prev; yoksa form
        cur_apps = float(cur.get("mp") or 0) if cur is not None else 0.0
        if cur is not None and cur_apps >= MIN_APPS_FOR_CURRENT_BASE:
            base_rates = _row_rates(cur)
            base_src = "current_season"
        elif pr is not None:
            base_rates = _row_rates(pr)
            base_src = "prev_season"
        elif cur is not None:
            base_rates = _row_rates(cur)
            base_src = "current_small"
        elif form_row is not None:
            base_rates = form_rates
            base_src = "form_only"
        else:
            continue

        if (form_rates.get("xg_pa") or 0) <= 0 and (base_rates.get("xg_pa") or 0) > 0:
            form_rates["xg_pa"] = base_rates["xg_pa"]
            form_rates["xa_pa"] = base_rates.get("xa_pa") or 0.0
        for k in ("key_passes_pa", "sot_pa", "bcc_pa", "rating"):
            if (form_rates.get(k) or 0) <= 0 and (base_rates.get(k) or 0) > 0:
                form_rates[k] = base_rates[k]

        team_form_cs = _lookup_cs(squad, form_cs)
        team_base_cs = _lookup_cs(squad, base_cs)

        form_pts = expected_points_from_rates(form_rates, position, team_form_cs)
        base_pts = expected_points_from_rates(base_rates, position, team_base_cs)

        if form_apps <= 0:
            w_f, w_b = 0.0, 1.0
        else:
            w_f, w_b = blend_weights(form_apps)

        projected = w_f * form_pts + w_b * base_pts

        if base_rates.get("min_per_app", 0) < 20 and form_rates.get("min_per_app", 0) < 20:
            projected *= 0.35

        reason_bits = []
        if position in ("DF", "GK"):
            reason_bits.append(f"CS~{(team_form_cs if team_form_cs is not None else form_rates.get('cs_rate', 0)):.0%}")
            if form_rates.get("int_p90", 0) > 0:
                reason_bits.append(f"Int/90={form_rates['int_p90']:.1f}")
        gls_show = form_rates.get("gls_pa") or base_rates.get("gls_pa") or 0
        ast_show = form_rates.get("ast_pa") or base_rates.get("ast_pa") or 0
        xg_show = form_rates.get("xg_pa") or base_rates.get("xg_pa") or 0
        xa_show = form_rates.get("xa_pa") or base_rates.get("xa_pa") or 0
        exp_g = _blend_attack(gls_show, xg_show, (form_rates.get("sot_pa") or base_rates.get("sot_pa") or 0) * SOT_TO_GOAL)
        exp_a = _blend_attack(
            ast_show,
            xa_show,
            max(
                (form_rates.get("key_passes_pa") or base_rates.get("key_passes_pa") or 0) * KEYPASS_TO_ASSIST,
                (form_rates.get("bcc_pa") or base_rates.get("bcc_pa") or 0) * BCC_TO_ASSIST,
            ),
        )
        if exp_g + exp_a + gls_show + ast_show > 0:
            reason_bits.append(
                f"beklenen G/A={exp_g:.2f}/{exp_a:.2f} (ham {gls_show:.2f}/{ast_show:.2f}, xG/xA {xg_show:.2f}/{xa_show:.2f})"
            )
        share = form_rates.get("share_60") or base_rates.get("share_60") or 0
        reason_bits.append(f"60+ dk ~{share:.0%} → {1 + share:.1f}p dakika")
        reason_bits.append(f"blend {w_f:.0%}/{w_b:.0%} ({base_src})")

        rows.append(
            {
                "player": player,
                "team": squad,
                "position": position,
                "pos_raw": pos_raw,
                "form_apps": form_apps,
                "base_apps": base_rates.get("apps", 0.0),
                "form_pts": round(form_pts, 3),
                "base_pts": round(base_pts, 3),
                "projected_pts": round(projected, 3),
                "w_form": w_f,
                "w_base": w_b,
                "base_src": base_src,
                "data_src": "super_lig",
                "int_p90": round(
                    form_rates.get("int_p90", 0.0) or base_rates.get("int_p90", 0.0), 2
                ),
                "tkl_p90": round(
                    form_rates.get("tkl_p90", 0.0) or base_rates.get("tkl_p90", 0.0), 2
                ),
                "gls_pa": round(form_rates.get("gls_pa", 0.0) or base_rates.get("gls_pa", 0.0), 3),
                "ast_pa": round(form_rates.get("ast_pa", 0.0) or base_rates.get("ast_pa", 0.0), 3),
                "xg_pa": round(form_rates.get("xg_pa", 0.0) or base_rates.get("xg_pa", 0.0), 3),
                "xa_pa": round(form_rates.get("xa_pa", 0.0) or base_rates.get("xa_pa", 0.0), 3),
                "share_60": round(form_rates.get("share_60") or base_rates.get("share_60") or 0.0, 3),
                "min_per_app": round(
                    form_rates.get("min_per_app") or base_rates.get("min_per_app") or 0.0,
                    1,
                ),
                "team_cs_form": team_form_cs,
                "reason": "; ".join(reason_bits),
                "player_key": key,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("projected_pts", ascending=False).reset_index(drop=True)


def _lookup_cs(squad: str, cs_map: dict[str, float]) -> float | None:
    if not cs_map or not squad:
        return None
    n = normalize_name(squad)
    for k, v in cs_map.items():
        if normalize_name(k) == n:
            return float(v)
    # kısmi
    for k, v in cs_map.items():
        nk = normalize_name(k)
        if n in nk or nk in n:
            return float(v)
    return None


def availability_multiplier(status: str | None, percent: float | None = None) -> float:
    raw = str(status or "").strip().upper()
    if not raw:
        return 1.0
    base = AVAIL_MULT.get(raw, 1.0)
    if percent is not None and percent > 0 and raw in ("DOUBTFUL", "INJURED"):
        return max(0.08, min(1.0, percent / 100.0))
    return base


def apply_context_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Yeni imza: son lig G/A + xG/xA (gol/asist kapasitesi korunur).
    Sakat/cezalı TFF availability ile kırpılır.
    """
    out = df.copy()
    if "projected_pts" not in out.columns:
        return out
    out["raw_pts"] = pd.to_numeric(out["projected_pts"], errors="coerce").fillna(0.0)
    src = out["data_src"].astype(str) if "data_src" in out.columns else ""
    form_apps = pd.to_numeric(out.get("form_apps", 0), errors="coerce").fillna(0.0)
    price = pd.to_numeric(out["price_m"], errors="coerce").fillna(0.0)

    ext = src == "external_prior" if isinstance(src, pd.Series) else False
    low = (~ext) & (form_apps < 4) & (price >= 7.0) if isinstance(src, pd.Series) else False

    pts = out["raw_pts"].copy()
    if isinstance(ext, pd.Series):
        pts = pts.where(~ext, pts * NEW_SIGNING_MULT)
        pts = pts.where(~low, pts * LOW_SAMPLE_MULT)

    avail = out["availability"].astype(str) if "availability" in out.columns else None
    pct = pd.to_numeric(out.get("avail_pct", None), errors="coerce") if "avail_pct" in out.columns else None
    if avail is not None:
        mults = []
        for i, st in avail.items():
            p = None
            if pct is not None:
                v = pct.loc[i] if i in pct.index else None
                p = float(v) if v is not None and pd.notna(v) else None
            mults.append(availability_multiplier(st, p))
        pts = pts * pd.Series(mults, index=pts.index)

    out["projected_pts"] = pts.round(3)

    def _note(row: pd.Series) -> str:
        reason = str(row.get("reason") or "")
        bits = []
        if str(row.get("data_src") or "") == "external_prior":
            bits.append("son lig G/A+xG (gol/asist kapasitesi korunur)")
            fm = float(row.get("friendly_minutes") or 0)
            if fm > 0:
                bits.append(f"yeni kulüp/hazırlık {fm:.0f} dk")
        st = str(row.get("availability") or "").upper()
        if st and st not in ("AVAILABLE", "", "NAN"):
            bits.append(f"TFF durum {st}")
        if bits:
            extra = "; ".join(bits)
            return f"{reason} | {extra}" if reason else extra
        return reason

    out["reason"] = out.apply(_note, axis=1)
    return out


def position_leaders(players: pd.DataFrame, n: int = 8) -> dict[str, pd.DataFrame]:
    leaders: dict[str, pd.DataFrame] = {}
    for pos in ("GK", "DF", "MF", "FW"):
        sub = players[players["position"] == pos].sort_values(
            "projected_pts", ascending=False
        ).head(n)
        leaders[pos] = sub
    return leaders
