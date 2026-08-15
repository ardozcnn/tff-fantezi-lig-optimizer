"""TFF Fantezi puan tahmini (form + baz blend)."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .config import (
    ASSIST_POINTS,
    AVAIL_MULT,
    BCC_TO_ASSIST,
    CONCEDED_PENALTY_PER_2,
    CS_POINTS,
    GOAL_POINTS,
    KEYPASS_TO_ASSIST,
    MIN_60_POINTS,
    MIN_FORM_MINUTES_FOR_RATES,
    MIN_FULL_POINTS,
    OWN_GOAL_PENALTY,
    PENALTY_MISS_PENALTY,
    PENALTY_SAVE_POINTS,
    POS_MAP,
    RED_PENALTY,
    SAVE_POINTS_PER_3,
    SOT_TO_GOAL,
    W_FORM_DEFAULT,
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
    """Sofa rating yalnızca BPS vekilidir; etkisi sınırlı ve yumuşak tutulur."""
    if rating <= 6.4:
        return 0.03
    return min(0.80, 0.45 * (rating - 6.4))


def _expected_concede_penalty(goals_conceded: float) -> float:
    """Poisson yaklaşımıyla E[floor(yenen gol / 2)]."""
    lam = max(0.0, float(goals_conceded or 0.0))
    if lam <= 0:
        return 0.0
    probability = math.exp(-lam)
    expected = 0.0
    for goals in range(1, 13):
        if goals > 1:
            probability *= lam / goals
        else:
            probability = math.exp(-lam) * lam
        expected += (goals // 2) * probability
    return expected


def expected_points_from_rates(
    rates: dict[str, float],
    position: str,
    team_cs_rate: float | None = None,
    *,
    appearance: float | None = None,
    attack_mult: float = 1.0,
    cs_mult: float = 1.0,
) -> float:
    """Bir maçlık beklenen TFF fantezi puanı (dakika, xG/xA, CS, kart, bonus, penaltı)."""
    if not rates or rates.get("apps", 0) <= 0:
        return 0.0
    pos = position if position in GOAL_POINTS else "MF"
    share_60 = rates["share_60"]
    if appearance is None:
        min_pa = rates.get("min_per_app") or 0.0
        appearance = 0.94 if min_pa >= 70 else (0.88 if min_pa >= 50 else 0.75)
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
    goal_pts = appearance * exp_gls * GOAL_POINTS[pos] * attack_mult
    assist_pts = appearance * exp_ast * ASSIST_POINTS * attack_mult

    cs_rate = team_cs_rate if team_cs_rate is not None else rates.get("cs_rate", 0.0)
    cs_rate = max(0.0, min(1.0, cs_rate * cs_mult))
    if pos in ("GK", "DF"):
        cs_pts = appearance * cs_rate * share_60 * CS_POINTS[pos]
        if team_cs_rate is not None:
            ga_pa = -math.log(max(0.03, min(0.97, cs_rate)))
        else:
            ga_pa = rates.get("ga_pa") or (1.0 - cs_rate) * 1.2
        concede_pen = (
            appearance
            * _expected_concede_penalty(ga_pa)
            * CONCEDED_PENALTY_PER_2
            * share_60
        )
    elif pos == "MF":
        cs_pts = appearance * cs_rate * share_60 * CS_POINTS["MF"]
        concede_pen = 0.0
    else:
        cs_pts = 0.0
        concede_pen = 0.0

    save_pts = 0.0
    if pos == "GK":
        save_pts = appearance * (rates.get("saves_pa", 0.0) / 3.0) * SAVE_POINTS_PER_3
        save_pts += appearance * (rates.get("pen_save_pa") or 0.0) * PENALTY_SAVE_POINTS

    card_pen = appearance * (
        rates.get("yc_pa", 0.0) * YELLOW_PENALTY
        + rates.get("rc_pa", 0.0) * RED_PENALTY
    )
    pen_miss = appearance * (rates.get("pen_miss_pa") or 0.0) * PENALTY_MISS_PENALTY
    og_pen = appearance * (rates.get("og_pa") or 0.0) * OWN_GOAL_PENALTY
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


def blend_weights(form_apps: float) -> tuple[float, float]:
    """L6 formunu örneklemle büyüt: 1 maç %5, 4 maç %20, 6 maç %30."""
    apps = max(0.0, min(float(form_apps or 0.0), 6.0))
    form_weight = W_FORM_DEFAULT * apps / 6.0
    return form_weight, 1.0 - form_weight


def _recency_multiplier(form_apps: float, form_matches: float) -> float:
    recent_matches = max(1.0, float(form_matches or 0))
    recent_presence = min(1.0, max(0.0, form_apps) / recent_matches)
    return 0.65 + 0.35 * recent_presence


def recency_for_projection(
    form_apps: float,
    form_matches: float,
    *,
    preseason: bool = False,
) -> float:
    """Yeni sezon maçı yoksa geçen sezonun L6'sı bu haftanın XI sinyali değildir."""
    if preseason:
        return 1.0
    return _recency_multiplier(form_apps, form_matches)


def _blend_rate_sets(
    current: dict[str, float],
    previous: dict[str, float],
    current_apps: float,
    *,
    prior_matches: float = 10.0,
) -> tuple[dict[str, float], float]:
    """Az mevcut sezon örneğini silmek yerine önceki sezonla küçültür."""
    if current.get("apps", 0) <= 0:
        return previous.copy(), 0.0
    if previous.get("apps", 0) <= 0:
        return current.copy(), 1.0

    weight = max(0.0, min(1.0, current_apps / (current_apps + prior_matches)))
    out: dict[str, float] = {}
    for key in current:
        if key in ("apps", "apps_60"):
            continue
        out[key] = weight * float(current.get(key, 0.0)) + (1.0 - weight) * float(
            previous.get(key, 0.0)
        )
    out["apps"] = current_apps
    out["apps_60"] = weight * float(current.get("apps_60", 0.0))
    return out, weight


def _position_prior_rates(position: str) -> dict[str, float]:
    """Tek maçlık CS/gol gürültüsünü küçültmek için mevki nötr prior."""
    rates = _empty_rates()
    rates.update(
        {
            "apps": 8.0,
            "apps_60": 7.0,
            "share_60": 0.85,
            "min_per_app": 75.0,
            "cs_rate": 0.28 if position in ("GK", "DF") else (0.25 if position == "MF" else 0.0),
            "ga_pa": 1.25 if position in ("GK", "DF") else 0.0,
            "saves_pa": 3.0 if position == "GK" else 0.0,
            "gls_pa": {"GK": 0.0, "DF": 0.05, "MF": 0.12, "FW": 0.28}.get(position, 0.1),
            "ast_pa": {"GK": 0.0, "DF": 0.06, "MF": 0.12, "FW": 0.12}.get(position, 0.08),
            "rating": 6.8,
        }
    )
    return rates


def shrink_small_sample_rates(
    rates: dict[str, float],
    position: str,
    *,
    prior_matches: float = 8.0,
) -> tuple[dict[str, float], float]:
    """1-3 maçlık örneklemi mevki prior'una doğru küçültür."""
    apps = float(rates.get("apps") or 0.0)
    if apps <= 0:
        return rates, 0.0
    if apps >= 4:
        return rates, 1.0
    prior = _position_prior_rates(position)
    return _blend_rate_sets(rates, prior, apps, prior_matches=prior_matches)


def build_player_table(
    current: pd.DataFrame,
    prev: pd.DataFrame,
    form_cs: dict[str, float],
    base_cs: dict[str, float],
    meta: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Form = `current` (L6–L8 lineups veya sezon proxy).
    Baz = mevcut sezon + önceki sezon örnek-büyüklüğü kontrollü karışım.
    projected = w_form * form_pts + w_base * base_pts
    """
    meta = meta or {}
    current_season: pd.DataFrame = meta.get("current_season_df")
    if current_season is None or not isinstance(current_season, pd.DataFrame):
        current_season = pd.DataFrame()
    fixture_context: dict[str, dict[str, Any]] = meta.get("fixture_context") or {}

    rows: list[dict[str, Any]] = []

    def index_by_player(df: pd.DataFrame) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        if df is None or df.empty:
            return out
        for _, r in df.iterrows():
            key = normalize_name(str(r.get("player", "")))
            if not key:
                continue
            existing = out.get(key)
            sample = float(r.get("mp") or 0) * 10000 + float(r.get("minutes") or 0)
            existing_sample = (
                float(existing.get("mp") or 0) * 10000
                + float(existing.get("minutes") or 0)
                if existing is not None
                else -1.0
            )
            if sample > existing_sample:
                out[key] = r
        return out

    form_idx = index_by_player(current)
    cur_idx = index_by_player(current_season)
    prev_idx = index_by_player(prev)

    all_keys = sorted(set(form_idx) | set(cur_idx) | set(prev_idx))

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
        if not position and pos_raw in ("GK", "DF", "MF", "FW"):
            position = str(pos_raw)
        if not position:
            continue

        form_rates = _row_rates(form_row) if form_row is not None else _empty_rates()
        form_apps = form_rates["apps"]
        form_minutes = float(form_row.get("minutes") or 0) if form_row is not None else 0.0

        cur_apps = float(cur.get("mp") or 0) if cur is not None else 0.0
        cur_rates = _row_rates(cur) if cur is not None else _empty_rates()
        prev_rates = _row_rates(pr) if pr is not None else _empty_rates()
        if cur_rates.get("apps", 0) > 0 and prev_rates.get("apps", 0) > 0:
            base_rates, season_weight = _blend_rate_sets(
                cur_rates, prev_rates, cur_apps
            )
            base_src = f"current+prev ({season_weight:.0%} current)"
        elif cur_rates.get("apps", 0) > 0:
            base_rates, sample_w = shrink_small_sample_rates(cur_rates, position)
            base_src = (
                "current_season"
                if sample_w >= 0.999
                else f"current_shrunk ({sample_w:.0%} observed)"
            )
        elif prev_rates.get("apps", 0) > 0:
            base_rates, sample_w = shrink_small_sample_rates(prev_rates, position)
            base_src = (
                "prev_season"
                if sample_w >= 0.999
                else f"prev_shrunk ({sample_w:.0%} observed)"
            )
        elif form_row is not None:
            base_rates, sample_w = shrink_small_sample_rates(form_rates, position)
            base_src = (
                "form_only"
                if sample_w >= 0.999
                else f"form_shrunk ({sample_w:.0%} observed)"
            )
        else:
            continue

        if (form_rates.get("xg_pa") or 0) <= 0 and (base_rates.get("xg_pa") or 0) > 0:
            form_rates["xg_pa"] = base_rates["xg_pa"]
            form_rates["xa_pa"] = base_rates.get("xa_pa") or 0.0
        for k in ("key_passes_pa", "sot_pa", "bcc_pa", "rating"):
            if (form_rates.get(k) or 0) <= 0 and (base_rates.get(k) or 0) > 0:
                form_rates[k] = base_rates[k]

        form_for_points = form_rates
        form_mpa = float(form_rates.get("min_per_app") or 0.0)
        if (
            form_mpa < MIN_FORM_MINUTES_FOR_RATES
            and form_minutes < 150
            and base_rates.get("apps", 0) >= 6
            and base_rates.get("min_per_app", 0) >= 45
        ):
            form_for_points = {
                **base_rates,
                "apps": max(form_rates.get("apps") or 0.0, 1.0),
            }

        team_form_cs = _lookup_cs(squad, form_cs)
        team_base_cs = _lookup_cs(squad, base_cs)
        fixture = lookup_fixture_context(squad, fixture_context)
        fixture_attack = float(fixture.get("attack_mult") or 1.0)
        fixture_cs = float(fixture.get("cs_mult") or 1.0)

        form_pts = expected_points_from_rates(
            form_for_points,
            position,
            team_form_cs,
            attack_mult=fixture_attack,
            cs_mult=fixture_cs,
        )
        base_pts = expected_points_from_rates(
            base_rates,
            position,
            team_base_cs,
            attack_mult=fixture_attack,
            cs_mult=fixture_cs,
        )

        if form_apps <= 0:
            w_f, w_b = 0.0, 1.0
        else:
            w_f, w_b = blend_weights(form_apps)

        projected = w_f * form_pts + w_b * base_pts

        recent_matches = max(1.0, float(meta.get("form_matches") or 6))
        early = bool(meta.get("early_season", meta.get("preseason")))
        form_from_requested = meta.get("form_season_start") == meta.get("requested_start")
        if cur_rates.get("apps", 0) > 0 and not current.empty:
            recency_mult = recency_for_projection(
                form_apps,
                recent_matches,
                preseason=bool(meta.get("preseason")),
            )
            projected *= recency_mult
        elif (
            early
            and form_from_requested
            and form_apps <= 0
            and not bool(meta.get("preseason"))
        ):
            recency_mult = _recency_multiplier(0.0, recent_matches)
            projected *= recency_mult
        else:
            recency_mult = 1.0

        if base_rates.get("min_per_app", 0) < 20 and form_rates.get("min_per_app", 0) < 20:
            projected *= 0.35

        show_rates = form_for_points if form_for_points.get("min_per_app", 0) >= 45 else base_rates
        reason_bits = []
        if position in ("DF", "GK"):
            reason_bits.append(f"CS~{(team_form_cs if team_form_cs is not None else form_rates.get('cs_rate', 0)):.0%}")
            if form_rates.get("int_p90", 0) > 0:
                reason_bits.append(f"Int/90={form_rates['int_p90']:.1f}")
        gls_show = show_rates.get("gls_pa") or 0
        ast_show = show_rates.get("ast_pa") or 0
        xg_show = show_rates.get("xg_pa") or 0
        xa_show = show_rates.get("xa_pa") or 0
        exp_g = _blend_attack(gls_show, xg_show, (show_rates.get("sot_pa") or 0) * SOT_TO_GOAL)
        exp_a = _blend_attack(
            ast_show,
            xa_show,
            max(
                (show_rates.get("key_passes_pa") or 0) * KEYPASS_TO_ASSIST,
                (show_rates.get("bcc_pa") or 0) * BCC_TO_ASSIST,
            ),
        )
        if exp_g + exp_a + gls_show + ast_show > 0:
            reason_bits.append(
                f"beklenen G/A={exp_g:.2f}/{exp_a:.2f} (ham {gls_show:.2f}/{ast_show:.2f}, xG/xA {xg_show:.2f}/{xa_show:.2f})"
            )
        share = show_rates.get("share_60") or base_rates.get("share_60") or 0
        reason_bits.append(f"60+ dk ~{share:.0%} → {1 + share:.1f}p dakika")
        if fixture:
            venue = "iç saha" if fixture.get("home") else "deplasman"
            reason_bits.append(f"{fixture.get('opponent')} ({venue})")
        reason_bits.append(f"L6 katılım {form_apps:.0f}/{recent_matches:.0f}")
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
                "gls_pa": round(show_rates.get("gls_pa", 0.0) or base_rates.get("gls_pa", 0.0), 3),
                "ast_pa": round(show_rates.get("ast_pa", 0.0) or base_rates.get("ast_pa", 0.0), 3),
                "xg_pa": round(show_rates.get("xg_pa", 0.0) or base_rates.get("xg_pa", 0.0), 3),
                "xa_pa": round(show_rates.get("xa_pa", 0.0) or base_rates.get("xa_pa", 0.0), 3),
                "share_60": round(show_rates.get("share_60") or base_rates.get("share_60") or 0.0, 3),
                "min_per_app": round(
                    show_rates.get("min_per_app") or base_rates.get("min_per_app") or 0.0,
                    1,
                ),
                "team_cs_form": team_form_cs,
                "team_cs_base": team_base_cs,
                "fixture_opponent": fixture.get("opponent") or "",
                "fixture_home": fixture.get("home"),
                "fixture_attack_mult": fixture_attack,
                "fixture_cs_mult": fixture_cs,
                "recency_mult": recency_mult,
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
    for k, v in cs_map.items():
        nk = normalize_name(k)
        if n in nk or nk in n:
            return float(v)
    return None


def lookup_fixture_context(
    squad: str,
    context: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not squad or not context:
        return {}
    normalized = normalize_name(squad)
    if normalized in context:
        return context[normalized]
    for key, value in context.items():
        if normalized in key or key in normalized:
            return value
    generic = {"fk", "jk", "spor", "sportif", "faaliyetler", "istanbul"}
    tokens = set(normalized.split()) - generic
    best: tuple[float, dict[str, Any]] | None = None
    for key, value in context.items():
        other = set(key.split()) - generic
        if not tokens or not other:
            continue
        score = len(tokens & other) / len(tokens | other)
        if score >= 0.5 and (best is None or score > best[0]):
            best = (score, value)
    return best[1] if best else {}


def availability_multiplier(status: str | None, percent: float | None = None) -> float:
    raw = str(status or "").strip().upper()
    if not raw:
        return 1.0
    base = AVAIL_MULT.get(raw, 1.0)
    if percent is not None and percent > 0 and raw in ("DOUBTFUL", "INJURED"):
        return max(0.08, min(1.0, percent / 100.0))
    return base


def apply_goalkeeper_start_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aynı takımdaki kalecileri ilk-11 olasılığıyla ayır.

    Bir yedek kalecinin tek eski temiz sayfası, uzun dönem takım birincisini
    geçmemeli. Önce güncel TFF dakikası, sonra form lineups, son olarak aynı
    kulüpteki geçmiş dakika kullanılır. Yeterli kanıt yoksa nötr bırakılır.
    """
    out = df.copy()
    if out.empty or not {"team", "position", "projected_pts"}.issubset(out.columns):
        return out

    out["gk_start_probability"] = 1.0
    out["gk_start_source"] = ""
    is_gk = out["position"].astype(str).str.upper().eq("GK")
    for _, idx in out[is_gk].groupby(out.loc[is_gk, "team"].map(normalize_name)).groups.items():
        members = list(idx)
        if len(members) < 2:
            continue

        group = out.loc[members]
        tff_minutes = pd.to_numeric(
            group.get("tff_minutes", pd.Series(0.0, index=members)),
            errors="coerce",
        ).fillna(0.0)
        form_apps = pd.to_numeric(
            group.get("form_apps", pd.Series(0.0, index=members)),
            errors="coerce",
        ).fillna(0.0)
        base_apps = pd.to_numeric(
            group.get("base_apps", pd.Series(0.0, index=members)),
            errors="coerce",
        ).fillna(0.0)
        min_per_app = pd.to_numeric(
            group.get("min_per_app", pd.Series(0.0, index=members)),
            errors="coerce",
        ).fillna(0.0)

        if float(tff_minutes.max()) >= 45:
            evidence, source = tff_minutes, "güncel TFF dakika"
        elif float(form_apps.max()) >= 1:
            evidence, source = form_apps * min_per_app.clip(lower=45.0) / 90.0, "son form lineups"
        elif float(base_apps.max()) >= 8:
            evidence, source = base_apps * min_per_app.clip(lower=45.0) / 90.0, "geçen sezon dakika"
        else:
            continue

        leader = float(evidence.max())
        if leader <= 0:
            continue
        probability = (0.05 + 0.90 * evidence / leader).clip(upper=0.95)
        out.loc[members, "gk_start_probability"] = probability
        out.loc[members, "gk_start_source"] = source
        out.loc[members, "projected_pts"] = (
            pd.to_numeric(out.loc[members, "projected_pts"], errors="coerce").fillna(0.0)
            * probability
        ).round(3)

    return out


def apply_context_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Yeni imza: son lig G/A + xG/xA (gol/asist kapasitesi korunur).
    Sakat/cezalı TFF availability ile kırpılır.
    """
    out = df.copy()
    if "projected_pts" not in out.columns:
        return out
    out["raw_pts"] = pd.to_numeric(out["projected_pts"], errors="coerce").fillna(0.0)
    form_apps = pd.to_numeric(
        out.get("form_apps", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0)
    pts = out["raw_pts"].copy()

    zeros = pd.Series(0.0, index=out.index)
    tff_minutes = pd.to_numeric(out.get("tff_minutes", zeros), errors="coerce").fillna(0.0)
    tff_starts = pd.to_numeric(out.get("tff_starts", zeros), errors="coerce").fillna(0.0)
    tff_points = pd.to_numeric(out.get("tff_points", zeros), errors="coerce").fillna(0.0)
    tff_ppm = pd.to_numeric(out.get("tff_ppm", zeros), errors="coerce").fillna(0.0)
    official_apps = pd.concat(
        [tff_starts, tff_minutes / 75.0], axis=1
    ).max(axis=1)
    official_ppg = tff_ppm.where(
        tff_ppm > 0,
        tff_points / official_apps.where(official_apps > 0, 1.0),
    )
    leftover_full_season = (tff_minutes >= 900) & (form_apps < 3)
    official_weight = (official_apps / 30.0).clip(lower=0.0, upper=0.35)
    early_official = (tff_minutes > 0) & (tff_minutes < 900)
    official_weight = official_weight.where(
        ~early_official,
        (official_apps / (official_apps + 12.0)).clip(lower=0.0, upper=0.25),
    )
    has_official = (tff_minutes > 0) & ~leftover_full_season
    pts = pts.where(
        ~has_official,
        (1.0 - official_weight) * pts + official_weight * official_ppg,
    )
    out["tff_calibration_weight"] = official_weight.where(has_official, 0.0)

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
    out = apply_goalkeeper_start_probabilities(out)
    pts = pd.to_numeric(out["projected_pts"], errors="coerce").fillna(0.0)

    unavailable = {"INJURED", "SUSPENDED", "UNAVAILABLE", "OUT"}
    status = (
        out["availability"].astype(str).str.strip().str.upper()
        if "availability" in out.columns
        else pd.Series("", index=out.index)
    )
    out["selection_eligible"] = ~status.isin(unavailable)
    out["selection_status"] = status.where(status != "", "UNKNOWN")
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
        gk_prob = float(row.get("gk_start_probability") or 1.0)
        if str(row.get("position") or "").upper() == "GK" and gk_prob < 0.99:
            bits.append(
                f"ilk 11 olasılığı ~{gk_prob:.0%} "
                f"({row.get('gk_start_source') or 'kaleci rotasyonu'})"
            )
        if bits:
            extra = "; ".join(bits)
            return f"{reason} | {extra}" if reason else extra
        return reason

    out["reason"] = out.apply(_note, axis=1)
    return out
