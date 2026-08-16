"""Tamsayı programlama: en iyi diziliş + ilk 11 + yedekler."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pulp

from .autosub import expected_squad_points, order_bench_for_autosub
from .config import AUTOSUB_MONTE_CARLO_DRAWS, BUDGET_M, FORMATIONS, MAX_PER_CLUB, SQUAD
from .names import normalize_name


def _bench_of(
    xi: dict[str, int],
    squad: dict[str, int] | None = None,
) -> dict[str, int]:
    squad = squad or SQUAD
    return {pos: squad[pos] - xi[pos] for pos in squad}


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


def _solve_formation_candidate(
    df: pd.DataFrame,
    formation: dict[str, int],
    *,
    budget: float,
    max_per_club: int,
    squad: dict[str, int],
    bench_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    idxs = list(df.index)
    prob = pulp.LpProblem("tff_fantasy_fixed_formation", pulp.LpMaximize)
    start = pulp.LpVariable.dicts("xi", idxs, cat="Binary")
    bench = pulp.LpVariable.dicts("bn", idxs, cat="Binary")
    values = {i: _selection_value(df.loc[i]) for i in idxs}
    prices = {i: float(df.loc[i, "price_m"]) for i in idxs}

    prob += pulp.lpSum(
        start[i] * values[i] + bench_weight * bench[i] * values[i]
        for i in idxs
    )
    for i in idxs:
        prob += start[i] + bench[i] <= 1
    prob += pulp.lpSum(
        (start[i] + bench[i]) * prices[i] for i in idxs
    ) <= budget

    for pos, squad_need in squad.items():
        pos_idxs = [i for i in idxs if df.loc[i, "position"] == pos]
        prob += pulp.lpSum(start[i] for i in pos_idxs) == formation[pos]
        prob += (
            pulp.lpSum(bench[i] for i in pos_idxs)
            == squad_need - formation[pos]
        )

    for team_key, group in df.groupby("team_key"):
        if not team_key:
            continue
        team_idxs = list(group.index)
        if len(team_idxs) > max_per_club:
            prob += (
                pulp.lpSum(start[i] + bench[i] for i in team_idxs)
                <= max_per_club
            )

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return None

    xi_idxs = [
        i for i in idxs if start[i].value() and start[i].value() > 0.5
    ]
    bench_idxs = [
        i for i in idxs if bench[i].value() and bench[i].value() > 0.5
    ]
    xi = _sort_xi(df.loc[xi_idxs].drop(columns=["team_key"], errors="ignore"))
    ordered_bench = order_bench_for_autosub(
        df.loc[bench_idxs].drop(columns=["team_key"], errors="ignore")
    )
    return xi.reset_index(drop=True), ordered_bench.reset_index(drop=True)


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
        formation_best: dict[str, Any] | None = None
        for xi_df, bn_df in _local_formation_candidates(base_xi, base_bench, shape):
            autosub = expected_squad_points(xi_df, bn_df, draws=autosub_draws)
            ev = float(autosub["expected_pts"])
            payload = {
                "formation": name,
                "xi": xi_df.assign(role="XI").reset_index(drop=True),
                "bench": bn_df.assign(role="yedek").reset_index(drop=True),
                "autosub": autosub,
                "expected_pts": ev,
            }
            if (
                formation_best is None
                or ev > float(formation_best["expected_pts"]) + 1e-9
            ):
                formation_best = payload
            if best is None or ev > float(best["expected_pts"]) + 1e-9:
                best = payload
            elif (
                best is not None
                and abs(ev - float(best["expected_pts"])) <= 1e-9
                and name < str(best["formation"])
            ):
                best = payload
        if formation_best is not None:
            formation_autosub = formation_best["autosub"]
            comparisons.append(
                {
                    "formation": name,
                    "expected_pts": round(
                        float(formation_best["expected_pts"]), 3
                    ),
                    "xi_expected": round(
                        float(formation_autosub["xi_expected"]), 3
                    ),
                    "bench_expected": round(
                        float(formation_autosub["bench_expected"]), 3
                    ),
                }
            )

    if best is None:
        raise RuntimeError("Formasyon EV yeniden skorlaması başarısız.")

    comparisons.sort(key=lambda row: row["expected_pts"], reverse=True)
    best["formation_comparisons"] = comparisons
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
    Her formasyon için ayrı aday 15 seçer; farklı yedek ağırlıklı adayları
    gerçek otomatik-yedek EV ile karşılaştırır.
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

    df["team_key"] = df["team"].map(lambda t: normalize_name(str(t)))
    proxy_weights = (
        [float(bench_weight)]
        if bench_weight is not None
        else [0.12, 0.22, 0.40]
    )
    best: dict[str, Any] | None = None
    comparisons: list[dict[str, Any]] = []

    for formation_name, formation_shape in formations.items():
        formation_best: dict[str, Any] | None = None
        seen_candidates: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        for proxy_weight in proxy_weights:
            candidate = _solve_formation_candidate(
                df,
                formation_shape,
                budget=budget,
                max_per_club=max_per_club,
                squad=squad,
                bench_weight=proxy_weight,
            )
            if candidate is None:
                continue
            base_xi, base_bench = candidate
            for xi_candidate, bench_candidate in _local_formation_candidates(
                base_xi, base_bench, formation_shape
            ):
                candidate_key = (
                    tuple(sorted(xi_candidate["player"].astype(str))),
                    tuple(sorted(bench_candidate["player"].astype(str))),
                )
                if candidate_key in seen_candidates:
                    continue
                seen_candidates.add(candidate_key)
                autosub_candidate = expected_squad_points(
                    xi_candidate,
                    bench_candidate,
                    draws=autosub_draws,
                )
                expected = float(autosub_candidate["expected_pts"])
                payload = {
                    "formation": formation_name,
                    "xi": xi_candidate.assign(role="XI").reset_index(drop=True),
                    "bench": bench_candidate.assign(
                        role="yedek"
                    ).reset_index(drop=True),
                    "autosub": autosub_candidate,
                    "expected_pts": expected,
                    "proxy_bench_weight": proxy_weight,
                }
                if (
                    formation_best is None
                    or expected
                    > float(formation_best["expected_pts"]) + 1e-9
                ):
                    formation_best = payload
        if formation_best is None:
            continue
        formation_autosub = formation_best["autosub"]
        formation_xi = formation_best["xi"]
        formation_bench = formation_best["bench"]
        comparisons.append(
            {
                "formation": formation_name,
                "expected_pts": round(
                    float(formation_best["expected_pts"]), 3
                ),
                "xi_expected": round(
                    float(formation_autosub["xi_expected"]), 3
                ),
                "bench_expected": round(
                    float(formation_autosub["bench_expected"]), 3
                ),
                "xi_players": formation_xi["player"].astype(str).tolist(),
                "bench_players": formation_bench["player"].astype(str).tolist(),
            }
        )
        if (
            best is None
            or float(formation_best["expected_pts"])
            > float(best["expected_pts"]) + 1e-9
        ):
            best = formation_best

    if best is None:
        raise RuntimeError("Hiçbir formasyon için optimum kadro bulunamadı.")

    comparisons.sort(key=lambda row: row["expected_pts"], reverse=True)
    chosen = str(best["formation"])
    xi_df = best["xi"]
    bn_df = best["bench"]
    autosub = best["autosub"]
    squad_df = pd.concat([xi_df, bn_df], ignore_index=True)
    total_cost = float(squad_df["price_m"].sum())
    captain_row = xi_df.loc[xi_df.apply(_selection_value, axis=1).idxmax()]
    bench_shape = _bench_of(formations[chosen], squad)

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
        "formation_comparisons": comparisons,
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
