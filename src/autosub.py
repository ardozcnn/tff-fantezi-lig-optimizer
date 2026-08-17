"""TFF Fantezi Lig otomatik yedek girişi ve beklenen puan hesabı."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import AUTOSUB_MONTE_CARLO_DRAWS


def is_legal_xi(positions: list[str]) -> bool:
    """Resmi minimum: 1 kaleci, 3 defans, 1 forvet; toplam 11."""
    if len(positions) != 11:
        return False
    counts = {"GK": 0, "DF": 0, "MF": 0, "FW": 0}
    for pos in positions:
        key = str(pos or "").upper()
        if key not in counts:
            return False
        counts[key] += 1
    return counts["GK"] == 1 and counts["DF"] >= 3 and counts["FW"] >= 1


def _row_points(row: pd.Series) -> float:
    if "pts_if_plays" in row.index and pd.notna(row.get("pts_if_plays")):
        return float(row["pts_if_plays"])
    return float(row.get("projected_pts") or 0.0)


def _row_play_prob(row: pd.Series) -> float:
    if "play_probability" in row.index and pd.notna(row.get("play_probability")):
        return float(max(0.0, min(1.0, row["play_probability"])))
    return 0.85


def _bench_value(row: pd.Series) -> float:
    return _row_points(row) * _row_play_prob(row)


def _pick_bench_replacement(
    xi_work: pd.DataFrame,
    bench_work: pd.DataFrame,
    xi_pos: int,
    xi_row: pd.Series,
    used_bench: set[int],
) -> tuple[int, pd.Series] | None:
    """Yasal yedekler arasından aynı mevki öncelikli en yüksek EV'yi seç."""
    out_pos = str(xi_row.get("position") or "").upper()
    candidates: list[tuple[float, float, int, pd.Series]] = []

    for bn_pos, bn_row in bench_work.iterrows():
        if bn_pos in used_bench:
            continue
        in_pos = str(bn_row.get("position") or "").upper()
        if out_pos == "GK" and in_pos != "GK":
            continue
        if out_pos != "GK" and in_pos == "GK":
            continue
        trial_positions = [
            in_pos if idx == xi_pos else str(row.get("position") or "").upper()
            for idx, row in xi_work.iterrows()
        ]
        if not is_legal_xi(trial_positions):
            continue
        same_pos = 1.0 if in_pos == out_pos else 0.0
        candidates.append((same_pos, _bench_value(bn_row), bn_pos, bn_row))

    if not candidates:
        return None
    _, _, bn_pos, bn_row = max(candidates, key=lambda item: (item[0], item[1]))
    return bn_pos, bn_row


def apply_autosub(
    xi: pd.DataFrame,
    bench: pd.DataFrame,
    played: dict[Any, bool] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Oynamayan ilk-11 oyuncularının yerine yedekleri sırayla dene.

    Kaleci yalnız kaleciyle değişir. Sahada aynı mevki yedek varsa önce o
    tercih edilir; sonra yasal kalan en yüksek oynama×puan adayı seçilir.
    """
    if xi is None or xi.empty:
        return pd.DataFrame(), []

    xi_work = xi.reset_index(drop=True).copy()
    bench_work = (
        bench.reset_index(drop=True).copy()
        if isinstance(bench, pd.DataFrame)
        else pd.DataFrame()
    )
    played = played or {}

    def did_play(row: pd.Series) -> bool:
        key = row.get("player")
        if key in played:
            return bool(played[key])
        return True

    events: list[dict[str, Any]] = []
    used_bench: set[int] = set()

    for xi_pos, xi_row in xi_work.iterrows():
        if did_play(xi_row):
            continue
        picked = _pick_bench_replacement(
            xi_work,
            bench_work,
            xi_pos,
            xi_row,
            used_bench,
        )
        if picked is None:
            continue
        bn_pos, bn_row = picked
        out_pos = str(xi_row.get("position") or "").upper()
        in_pos = str(bn_row.get("position") or "").upper()
        xi_work.loc[xi_pos] = bn_row
        used_bench.add(bn_pos)
        events.append(
            {
                "out": str(xi_row.get("display_name") or xi_row.get("player") or ""),
                "in": str(bn_row.get("display_name") or bn_row.get("player") or ""),
                "out_pos": out_pos,
                "in_pos": in_pos,
            }
        )

    active = xi_work[
        [did_play(row) for _, row in xi_work.iterrows()]
    ].copy()
    return active, events


def score_final_xi(final_xi: pd.DataFrame) -> float:
    if final_xi is None or final_xi.empty:
        return 0.0
    return float(sum(_row_points(row) for _, row in final_xi.iterrows()))


def expected_squad_points(
    xi: pd.DataFrame,
    bench: pd.DataFrame,
    *,
    draws: int = AUTOSUB_MONTE_CARLO_DRAWS,
    seed: int = 20260817,
    full_bench: bool = False,
) -> dict[str, Any]:
    """Oynama olasılıklarıyla otomatik yedek beklenen puanı."""
    if xi is None or xi.empty:
        return {
            "expected_pts": 0.0,
            "xi_expected": 0.0,
            "bench_expected": 0.0,
            "captain_player": "",
            "captain_pts": 0.0,
        }

    xi_df = xi.reset_index(drop=True).copy()
    bench_df = (
        bench.reset_index(drop=True).copy()
        if isinstance(bench, pd.DataFrame)
        else pd.DataFrame()
    )
    rng = np.random.default_rng(seed)
    players = []
    for source, frame in (("xi", xi_df), ("bench", bench_df)):
        for idx, row in frame.iterrows():
            players.append(
                {
                    "source": source,
                    "idx": idx,
                    "player": row.get("player"),
                    "prob": _row_play_prob(row),
                    "pts": _row_points(row),
                }
            )

    totals = []
    xi_only = []
    bench_contrib = []
    for _ in range(max(1, int(draws))):
        played = {
            item["player"]: bool(rng.random() < item["prob"])
            for item in players
            if item["player"] is not None
        }
        final_xi, _events = apply_autosub(xi_df, bench_df, played=played)
        score = score_final_xi(final_xi)
        if full_bench and not bench_df.empty:
            used = set(final_xi["player"].tolist()) if "player" in final_xi.columns else set()
            for _, row in bench_df.iterrows():
                name = row.get("player")
                if name in used:
                    continue
                if played.get(name, False):
                    score += _row_points(row)
        base_xi = score_final_xi(
            xi_df[[played.get(row.get("player"), False) for _, row in xi_df.iterrows()]]
        )
        totals.append(score)
        xi_only.append(base_xi)
        bench_contrib.append(score - base_xi)

    captain_row = xi_df.loc[xi_df.apply(_row_points, axis=1).idxmax()]
    expected = float(np.mean(totals)) if totals else 0.0
    return {
        "expected_pts": round(expected, 3),
        "xi_expected": round(float(np.mean(xi_only)), 3) if xi_only else 0.0,
        "bench_expected": round(float(np.mean(bench_contrib)), 3) if bench_contrib else 0.0,
        "captain_player": str(captain_row.get("player") or ""),
        "captain_pts": round(_row_points(captain_row) * _row_play_prob(captain_row), 3),
        "draws": int(draws),
    }


def order_bench_for_autosub(bench: pd.DataFrame) -> pd.DataFrame:
    """
    Otomatik giriş sırası: kaleci yedeği GK için 1., saha yedekleri
    oynama×puan ile 2-4. Raporda bu sıra gösterilir.
    """
    if bench is None or bench.empty:
        return pd.DataFrame() if bench is None else bench.copy()
    out = bench.copy()
    out["_gk"] = out["position"].astype(str).str.upper().eq("GK").astype(int)
    out["_score"] = out.apply(_bench_value, axis=1)
    out = out.sort_values(["_gk", "_score"], ascending=[False, False]).drop(
        columns=["_gk", "_score"]
    )
    out = out.reset_index(drop=True)
    out["bench_rank"] = range(1, len(out) + 1)
    return out
