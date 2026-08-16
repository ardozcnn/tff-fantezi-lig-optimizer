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
    İlk 11 dizilişi (4-4-2, 4-5-1, 3-5-2, …) içinden beklenen puanı max olanı seçer.
    Yedek değeri TFF otomatik değişim beklenen puanıyla hesaplanır.
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

    # Yedek lineer vekili: ortalama XI blank × yedek oynama (~0.18-0.28)
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

    chosen = next(
        (n for n, v in form_vars.items() if v.value() and v.value() > 0.5),
        "4-4-2",
    )
    xi_idx = [i for i in idxs if start[i].value() and start[i].value() > 0.5]
    bn_idx = [i for i in idxs if bench[i].value() and bench[i].value() > 0.5]
    order = {"GK": 0, "DF": 1, "MF": 2, "FW": 3}

    def _sort_xi(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["_o"] = out["position"].map(order)
        out["_v"] = out.apply(_selection_value, axis=1)
        return out.sort_values(["_o", "_v"], ascending=[True, False]).drop(
            columns=["_o", "_v", "team_key"], errors="ignore"
        )

    xi_df = _sort_xi(df.loc[xi_idx]).assign(role="XI")
    bn_df = order_bench_for_autosub(df.loc[bn_idx]).assign(role="yedek")
    if "team_key" in bn_df.columns:
        bn_df = bn_df.drop(columns=["team_key"], errors="ignore")
    squad_df = pd.concat([xi_df, bn_df], ignore_index=True)

    autosub = expected_squad_points(
        xi_df,
        bn_df,
        draws=autosub_draws,
    )
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
