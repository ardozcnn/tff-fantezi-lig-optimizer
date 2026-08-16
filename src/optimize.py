"""Tamsayı programlama: en iyi diziliş + ilk 11 + yedekler."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pulp

from .autosub import expected_squad_points, order_bench_for_autosub
from .config import AUTOSUB_MONTE_CARLO_DRAWS, BUDGET_M, FORMATIONS, MAX_PER_CLUB, SQUAD
from .names import normalize_name


def _bench_of(xi: dict[str, int]) -> dict[str, int]:
    return {pos: SQUAD[pos] - xi[pos] for pos in SQUAD}


def _selection_value(row: pd.Series) -> float:
    """ILP için lineer vekil: oynama × puan; yedek için daha düşük."""
    pts = float(row.get("pts_if_plays") or row.get("projected_pts") or 0.0)
    play = float(row.get("play_probability") or 0.85)
    return pts * max(0.05, min(1.0, play))


def _sort_xi(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    order = {"GK": 0, "DF": 1, "MF": 2, "FW": 3}
    out["_o"] = out["position"].map(order)
    out["_v"] = out.apply(_selection_value, axis=1)
    return out.sort_values(["_o", "_v"], ascending=[True, False]).drop(
        columns=["_o", "_v"], errors="ignore"
    )


def assign_formation(
    squad_df: pd.DataFrame,
    formation: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sabit 15'li kadroda formasyona göre greedy ilk 11 / yedek ayrımı."""
    xi_parts: list[pd.DataFrame] = []
    bench_parts: list[pd.DataFrame] = []
    for pos, need in formation.items():
        pool = squad_df[squad_df["position"] == pos].copy()
        if pool.empty:
            continue
        pool["_v"] = pool.apply(_selection_value, axis=1)
        pool = pool.sort_values("_v", ascending=False).drop(columns=["_v"])
        xi_parts.append(pool.head(int(need)))
        bench_parts.append(pool.iloc[int(need) :])
    xi = _sort_xi(pd.concat(xi_parts, ignore_index=True)) if xi_parts else pd.DataFrame()
    bench_raw = (
        pd.concat(bench_parts, ignore_index=True) if bench_parts else pd.DataFrame()
    )
    bench = order_bench_for_autosub(bench_raw)
    return xi, bench


def _formation_feasible(squad_df: pd.DataFrame, formation: dict[str, int]) -> bool:
    for pos, need in formation.items():
        if int((squad_df["position"] == pos).sum()) < int(need):
            return False
    return True


def _local_formation_candidates(
    xi: pd.DataFrame,
    bench: pd.DataFrame,
    formation: dict[str, int],
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Aynı formasyon içinde pahalı/yüksek EV yedeği XI ile yer değiştir."""
    candidates = [(xi.copy(), bench.copy())]
    if xi.empty or bench.empty:
        return candidates
    for b_idx, b_row in bench.iterrows():
        b_pos = str(b_row.get("position") or "")
        xi_same = xi[xi["position"] == b_pos]
        if xi_same.empty:
            continue
        weak_idx = xi_same.apply(_selection_value, axis=1).idxmin()
        if _selection_value(b_row) <= _selection_value(xi.loc[weak_idx]) + 1e-9:
            continue
        new_xi = xi.copy()
        new_bench = bench.copy()
        new_xi.loc[weak_idx] = b_row
        new_bench.loc[b_idx] = xi.loc[weak_idx]
        candidates.append(
            (_sort_xi(new_xi.reset_index(drop=True)), order_bench_for_autosub(new_bench.reset_index(drop=True)))
        )
    return candidates


def rescore_formations(
    squad_df: pd.DataFrame,
    formations: dict[str, dict[str, int]],
    *,
    autosub_draws: int = AUTOSUB_MONTE_CARLO_DRAWS,
) -> dict[str, Any]:
    """15'li kadroyu tüm formasyonlarda gerçek autosub EV ile yeniden puanla."""
    best: dict[str, Any] | None = None
    comparisons: list[dict[str, Any]] = []
    clean = squad_df.drop(columns=["team_key", "role"], errors="ignore").copy()

    for name, shape in formations.items():
        if not _formation_feasible(clean, shape):
            continue
        base_xi, base_bench = assign_formation(clean, shape)
        for xi_df, bn_df in _local_formation_candidates(base_xi, base_bench, shape):
            autosub = expected_squad_points(xi_df, bn_df, draws=autosub_draws)
            ev = float(autosub["expected_pts"])
            comparisons.append(
                {
                    "formation": name,
                    "expected_pts": round(ev, 3),
                    "xi_expected": round(float(autosub["xi_expected"]), 3),
                    "bench_expected": round(float(autosub["bench_expected"]), 3),
                }
            )
            payload = {
                "formation": name,
                "xi": xi_df.assign(role="XI").reset_index(drop=True),
                "bench": bn_df.assign(role="yedek").reset_index(drop=True),
                "autosub": autosub,
                "expected_pts": ev,
            }
            if best is None or ev > float(best["expected_pts"]) + 1e-9:
                best = payload
            elif (
                best is not None
                and abs(ev - float(best["expected_pts"])) <= 1e-9
                and name < str(best["formation"])
            ):
                best = payload

    if best is None:
        raise RuntimeError("Formasyon EV yeniden skorlaması başarısız.")

    comparisons.sort(key=lambda row: row["expected_pts"], reverse=True)
    best["formation_comparisons"] = comparisons[:8]
    return best


def optimize_squad(
    players: pd.DataFrame,
    *,
    budget: float = BUDGET_M,
    max_per_club: int = MAX_PER_CLUB,
    squad: dict[str, int] | None = None,
    formations: dict[str, dict[str, int]] | None = None,
    bench_weight: float | None = None,
    autosub_draws: int = AUTOSUB_MONTE_CARLO_DRAWS,
    **_ignored: Any,
) -> dict[str, Any]:
    """
    15'li kadro: resmi 2-5-5-3.
    ILP lineer vekille aday 15'i seçer; ardından her formasyonu gerçek
    otomatik-yedek EV ile yeniden puanlar.
    """
    squad = squad or SQUAD
    formations = formations or FORMATIONS
    df = players.copy()
    df = df.dropna(subset=["price_m", "position", "projected_pts"])
    df = df[df["position"].isin(squad.keys())]
    df = df[df["price_m"] > 0].reset_index(drop=True)

    if df.empty:
        raise ValueError("Optimize edilecek oyuncu yok.")

    if "pts_if_plays" not in df.columns:
        df["pts_if_plays"] = pd.to_numeric(df["projected_pts"], errors="coerce").fillna(0.0)
    if "play_probability" not in df.columns:
        df["play_probability"] = 0.85

    for pos, need in squad.items():
        have = int((df["position"] == pos).sum())
        if have < need:
            raise ValueError(
                f"{pos} için {need} oyuncu gerekiyor, listede {have} var."
            )

    linear_bench = 0.22 if bench_weight is None else float(bench_weight)

    prob = pulp.LpProblem("tff_fantasy_formation", pulp.LpMaximize)
    idxs = list(df.index)
    start = pulp.LpVariable.dicts("xi", idxs, cat="Binary")
    bench = pulp.LpVariable.dicts("bn", idxs, cat="Binary")
    form_vars = {
        name: pulp.LpVariable(f"f_{name.replace('-', '_')}", cat="Binary")
        for name in formations
    }

    vals = {i: _selection_value(df.loc[i]) for i in idxs}
    price = {i: float(df.loc[i, "price_m"]) for i in idxs}

    prob += pulp.lpSum(
        start[i] * vals[i] + linear_bench * bench[i] * vals[i] for i in idxs
    )
    for i in idxs:
        prob += start[i] + bench[i] <= 1, f"one_{i}"
    prob += pulp.lpSum((start[i] + bench[i]) * price[i] for i in idxs) <= budget
    prob += pulp.lpSum(form_vars[n] for n in formations) == 1, "one_form"

    pos_idx = {
        pos: [i for i in idxs if df.loc[i, "position"] == pos] for pos in squad
    }
    for pos in squad:
        xi_need = pulp.lpSum(form_vars[n] * formations[n][pos] for n in formations)
        bn_need = pulp.lpSum(
            form_vars[n] * (squad[pos] - formations[n][pos]) for n in formations
        )
        prob += pulp.lpSum(start[i] for i in pos_idx[pos]) == xi_need, f"xi_{pos}"
        prob += pulp.lpSum(bench[i] for i in pos_idx[pos]) == bn_need, f"bn_{pos}"

    df["team_key"] = df["team"].map(lambda t: normalize_name(str(t)))
    for team_key, grp in df.groupby("team_key"):
        if not team_key:
            continue
        t_idx = list(grp.index)
        if len(t_idx) > max_per_club:
            prob += (
                pulp.lpSum(start[i] + bench[i] for i in t_idx) <= max_per_club,
                f"club_{team_key[:40]}",
            )

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(
            f"Optimum kadro bulunamadı ({pulp.LpStatus[status]})."
        )

    picked_idx = [
        i
        for i in idxs
        if (start[i].value() and start[i].value() > 0.5)
        or (bench[i].value() and bench[i].value() > 0.5)
    ]
    squad_pool = df.loc[picked_idx].copy()
    refined = rescore_formations(
        squad_pool,
        formations,
        autosub_draws=autosub_draws,
    )
    chosen = str(refined["formation"])
    xi_df = refined["xi"]
    bn_df = refined["bench"]
    autosub = refined["autosub"]
    squad_df = pd.concat([xi_df, bn_df], ignore_index=True)
    total_cost = float(squad_df["price_m"].sum())
    captain_row = xi_df.loc[xi_df.apply(_selection_value, axis=1).idxmax()]
    bench_shape = _bench_of(formations[chosen])

    return {
        "squad": squad_df.reset_index(drop=True),
        "xi": xi_df.reset_index(drop=True),
        "bench": bn_df.reset_index(drop=True),
        "formation": chosen,
        "xi_shape": formations[chosen],
        "bench_shape": bench_shape,
        "total_cost": total_cost,
        "bank": budget - total_cost,
        "total_projected": float(autosub["expected_pts"]),
        "xi_projected": float(autosub["xi_expected"]),
        "bench_projected": float(autosub["bench_expected"]),
        "autosub": autosub,
        "formation_comparisons": refined.get("formation_comparisons") or [],
        "captain": {
            "player": captain_row["player"],
            "display_name": str(captain_row.get("display_name") or captain_row["player"]),
            "projected_pts": float(captain_row.get("projected_pts") or 0.0),
            "pts_if_plays": float(captain_row.get("pts_if_plays") or captain_row.get("projected_pts") or 0.0),
            "play_probability": float(captain_row.get("play_probability") or 0.85),
            "team": captain_row["team"],
            "position": captain_row["position"],
            "price_m": float(captain_row["price_m"]),
            "reason": str(captain_row.get("reason") or ""),
        },
        "budget": budget,
    }
