"""Tamsayı programlama: en iyi diziliş + ilk 11 + yedekler."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pulp

from .config import BENCH_WEIGHT, BUDGET_M, FORMATIONS, MAX_PER_CLUB, SQUAD
from .names import normalize_name


def _bench_of(xi: dict[str, int]) -> dict[str, int]:
    return {pos: SQUAD[pos] - xi[pos] for pos in SQUAD}


def optimize_squad(
    players: pd.DataFrame,
    *,
    budget: float = BUDGET_M,
    max_per_club: int = MAX_PER_CLUB,
    squad: dict[str, int] | None = None,
    formations: dict[str, dict[str, int]] | None = None,
    bench_weight: float = BENCH_WEIGHT,
    **_ignored: Any,
) -> dict[str, Any]:
    """
    15'li kadro: resmi 2-5-5-3.
    İlk 11 dizilişi (4-4-2, 4-5-1, 3-5-2, …) içinden beklenen puanı max olanı seçer.
    Yedek = kadro − XI (4-5-1 → 2 forvet yedeği vb.).
    """
    squad = squad or SQUAD
    formations = formations or FORMATIONS
    df = players.copy()
    df = df.dropna(subset=["price_m", "position", "projected_pts"])
    df = df[df["position"].isin(squad.keys())]
    df = df[df["price_m"] > 0].reset_index(drop=True)

    if df.empty:
        raise ValueError("Optimize edilecek oyuncu yok.")

    for pos, need in squad.items():
        have = int((df["position"] == pos).sum())
        if have < need:
            raise ValueError(
                f"{pos} için {need} oyuncu gerekiyor, listede {have} var."
            )

    prob = pulp.LpProblem("tff_fantasy_formation", pulp.LpMaximize)
    idxs = list(df.index)
    start = pulp.LpVariable.dicts("xi", idxs, cat="Binary")
    bench = pulp.LpVariable.dicts("bn", idxs, cat="Binary")
    form_vars = {
        name: pulp.LpVariable(f"f_{name.replace('-', '_')}", cat="Binary")
        for name in formations
    }

    pts = {i: float(df.loc[i, "projected_pts"]) for i in idxs}
    price = {i: float(df.loc[i, "price_m"]) for i in idxs}

    prob += pulp.lpSum(
        start[i] * pts[i] + bench_weight * bench[i] * pts[i] for i in idxs
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

    chosen = next(
        (n for n, v in form_vars.items() if v.value() and v.value() > 0.5),
        "4-4-2",
    )
    xi_idx = [i for i in idxs if start[i].value() and start[i].value() > 0.5]
    bn_idx = [i for i in idxs if bench[i].value() and bench[i].value() > 0.5]
    order = {"GK": 0, "DF": 1, "MF": 2, "FW": 3}

    def _sort(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["_o"] = out["position"].map(order)
        return out.sort_values(["_o", "projected_pts"], ascending=[True, False]).drop(
            columns=["_o", "team_key"], errors="ignore"
        )

    xi_df = _sort(df.loc[xi_idx]).assign(role="XI")
    bn_df = _sort(df.loc[bn_idx]).assign(role="yedek")
    squad_df = pd.concat([xi_df, bn_df], ignore_index=True)

    total_cost = float(squad_df["price_m"].sum())
    xi_pts = float(xi_df["projected_pts"].sum())
    bn_pts = float(bn_df["projected_pts"].sum())
    captain_row = xi_df.loc[xi_df["projected_pts"].idxmax()]
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
        "total_projected": xi_pts + bn_pts,
        "xi_projected": xi_pts,
        "bench_projected": bn_pts,
        "captain": {
            "player": captain_row["player"],
            "display_name": str(captain_row.get("display_name") or captain_row["player"]),
            "projected_pts": float(captain_row["projected_pts"]),
            "team": captain_row["team"],
            "position": captain_row["position"],
            "price_m": float(captain_row["price_m"]),
            "reason": str(captain_row.get("reason") or ""),
        },
        "budget": budget,
    }
