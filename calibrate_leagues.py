"""Geçmiş dış lig → Süper Lig geçişlerinden kalibrasyon modeli üret.

Çalıştırma:
    python calibrate_leagues.py

Ham örneklem yalnızca git-dışı ``data/cache`` altında tutulur. Çalışma zamanında
kullanılan küçük ve denetlenebilir özet ``data/league_translation.json`` olur.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, DATA_DIR
from src.fetch_external import (
    fetch_overall_any,
    is_domestic_league as is_sofa_domestic_league,
    player_tournament_seasons,
)
from src.fetch_stats import SofaNotFound, list_seasons, sofa_get
from src.names import normalize_name

TARGET_TOURNAMENT_IDS = {52}
POSITION_FILTERS = {"GK": "G", "DF": "D", "MF": "M", "FW": "F"}
MIN_MINUTES = 450.0

STAT_FIELDS = (
    "goals",
    "assists",
    "yellowCards",
    "redCards",
    "minutesPlayed",
    "appearances",
    "started",
    "saves",
    "cleanSheet",
    "goalsConcededInsideTheBox",
    "goalsConcededOutsideTheBox",
    "expectedGoals",
    "expectedAssists",
    "keyPasses",
    "shotsOnTarget",
    "rating",
)

def season_start(value: Any) -> int | None:
    raw = str(value or "").strip()
    match = re.search(r"(20\d{2}|19\d{2})", raw)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(\d{2})\s*/\s*\d{2}\b", raw)
    if match:
        yy = int(match.group(1))
        return 2000 + yy if yy < 80 else 1900 + yy
    return None


def is_domestic_league(meta: dict[str, Any]) -> bool:
    tid = int(meta.get("tournament_id") or 0)
    if tid in TARGET_TOURNAMENT_IDS:
        return False
    name = str(meta.get("tournament") or meta.get("season_name") or "")
    return is_sofa_domestic_league(tid, name)


def _canonical_stats(raw: dict[str, Any]) -> dict[str, Any]:
    conceded = raw.get("goalsConceded")
    if conceded is None:
        inside = raw.get("goalsConcededInsideTheBox")
        outside = raw.get("goalsConcededOutsideTheBox")
        if inside is not None or outside is not None:
            conceded = float(inside or 0.0) + float(outside or 0.0)
    return {
        "minutes": float(raw.get("minutesPlayed") or raw.get("minutes") or 0.0),
        "apps": float(raw.get("appearances") or raw.get("mp") or 0.0),
        "starts": float(raw.get("started") or 0.0),
        "goals": float(raw.get("goals") or raw.get("gls") or 0.0),
        "assists": float(raw.get("assists") or raw.get("ast") or 0.0),
        "yellow_cards": float(raw.get("yellowCards") or raw.get("crdy") or 0.0),
        "red_cards": float(raw.get("redCards") or raw.get("crdr") or 0.0),
        "saves": float(raw.get("saves") or 0.0),
        "clean_sheets": float(
            raw.get("cleanSheets")
            if raw.get("cleanSheets") is not None
            else raw.get("cleanSheet") or raw.get("cs") or 0.0
        ),
        "goals_conceded": float(conceded or raw.get("ga") or 0.0),
        "xg": float(raw.get("expectedGoals") or raw.get("xg") or 0.0),
        "xa": float(raw.get("expectedAssists") or raw.get("xa") or 0.0),
        "key_passes": float(raw.get("keyPasses") or raw.get("key_passes") or 0.0),
        "shots_on_target": float(
            raw.get("shotsOnTarget")
            or raw.get("onTargetScoringAttempt")
            or raw.get("sot")
            or 0.0
        ),
        "rating": float(raw.get("rating") or 0.0),
        "has_xg": raw.get("expectedGoals") is not None or raw.get("xg") is not None,
        "has_xa": raw.get("expectedAssists") is not None or raw.get("xa") is not None,
        "has_key_passes": raw.get("keyPasses") is not None
        or raw.get("key_passes") is not None,
        "has_shots_on_target": raw.get("shotsOnTarget") is not None
        or raw.get("onTargetScoringAttempt") is not None
        or raw.get("sot") is not None,
    }


def fetch_league_players(
    tournament_id: int,
    season_id: int,
    year_start: int,
) -> dict[int, dict[str, Any]]:
    fields = quote(",".join(STAT_FIELDS), safe="")
    players: dict[int, dict[str, Any]] = {}
    for position, code in POSITION_FILTERS.items():
        offset = 0
        for _ in range(10):
            path = (
                f"/unique-tournament/{tournament_id}/season/{season_id}/statistics"
                f"?limit=100&order=-rating&offset={offset}&accumulation=total"
                f"&fields={fields}&filters=position.in.{code}"
            )
            data = sofa_get(
                path,
                cache_key=f"cal2_players_{tournament_id}_{season_id}_{code}_{offset}",
                max_age_hours=24 * 180,
                delay=0.12,
            )
            batch = data.get("results") or []
            for item in batch:
                player = item.get("player") or {}
                team = item.get("team") or {}
                pid = int(player.get("id") or 0)
                if not pid:
                    continue
                row = {
                    **_canonical_stats(item),
                    "player_id": pid,
                    "player": player.get("name") or "",
                    "team": team.get("name") or "",
                    "team_id": int(team.get("id") or 0),
                    "position": position,
                    "season_start": year_start,
                    "season_id": season_id,
                }
                old = players.get(pid)
                if old is None or row["minutes"] > old["minutes"]:
                    players[pid] = row
            if not batch or int(data.get("page") or 1) >= int(data.get("pages") or 1):
                break
            offset += 100
    return players


def _same_team(left: str, right: str) -> bool:
    a, b = normalize_name(left), normalize_name(right)
    if not a or not b:
        return False
    return a == b or a in b or b in a


def fetch_transfer_history(player_id: int) -> list[dict[str, Any]]:
    try:
        data = sofa_get(
            f"/player/{player_id}/transfer-history",
            cache_key=f"cal_transfers_{player_id}",
            max_age_hours=24 * 90,
            delay=0.10,
        )
    except (SofaNotFound, RuntimeError):
        return []
    return list(data.get("transferHistory") or [])


def arrival_transfer(
    player_id: int,
    target_team: str,
    target_start: int,
) -> dict[str, Any] | None:
    lower = int(datetime(target_start, 5, 1, tzinfo=timezone.utc).timestamp())
    upper = int(datetime(target_start + 1, 3, 15, tzinfo=timezone.utc).timestamp())
    candidates = []
    for transfer in fetch_transfer_history(player_id):
        timestamp = int(transfer.get("transferDateTimestamp") or 0)
        to_name = str(
            (transfer.get("transferTo") or {}).get("name")
            or transfer.get("toTeamName")
            or ""
        )
        if lower <= timestamp <= upper and _same_team(to_name, target_team):
            candidates.append(transfer)
    if not candidates:
        return None
    return min(candidates, key=lambda t: int(t.get("transferDateTimestamp") or 0))


def _source_for_player(target: dict[str, Any]) -> dict[str, Any] | None:
    pid = int(target["player_id"])
    target_start = int(target["season_start"])
    try:
        history = player_tournament_seasons(pid)
    except Exception:
        return None
    if not history:
        return None

    prior_sl = [
        meta
        for meta in history
        if int(meta.get("tournament_id") or 0) in TARGET_TOURNAMENT_IDS
        and (season_start(meta.get("year")) or 9999) < target_start
    ]
    if prior_sl:
        return None

    arrival = arrival_transfer(pid, str(target.get("team") or ""), target_start)
    eligible: list[dict[str, Any]] = []
    for meta in history:
        source_start = season_start(meta.get("year") or meta.get("season_name"))
        if source_start not in (target_start, target_start - 1):
            continue
        if not is_domestic_league(meta):
            continue
        if source_start == target_start and arrival is None:
            continue
        eligible.append(meta)

    best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    for meta in eligible:
        try:
            stats = fetch_overall_any(
                pid,
                int(meta["tournament_id"]),
                int(meta["season_id"]),
            )
        except Exception:
            continue
        canonical = _canonical_stats(stats)
        minutes = canonical["minutes"]
        if minutes <= 0 and canonical["apps"] > 0:
            minutes = canonical["apps"] * 75.0
            canonical["minutes"] = minutes
        if minutes < MIN_MINUTES:
            continue
        recency_bonus = 120.0 if season_start(meta.get("year")) == target_start else 0.0
        score = minutes + recency_bonus
        if best is None or score > best[0]:
            best = (score, meta, canonical)
    if best is None:
        return None

    _, meta, source = best
    transfer_from = ""
    if arrival:
        transfer_from = str(
            (arrival.get("transferFrom") or {}).get("name")
            or arrival.get("fromTeamName")
            or ""
        )
    return {
        **source,
        "tournament_id": int(meta["tournament_id"]),
        "tournament": str(meta.get("tournament") or ""),
        "season_id": int(meta["season_id"]),
        "season": str(meta.get("year") or meta.get("season_name") or ""),
        "transfer_from": transfer_from,
        "arrival_verified": bool(arrival),
    }


def collect_samples(
    start_year: int,
    end_year: int,
    *,
    workers: int,
) -> list[dict[str, Any]]:
    seasons_by_start = {
        season_start(season.get("year")): season
        for season in list_seasons()
        if season_start(season.get("year")) is not None
    }
    target_by_year: dict[int, dict[int, dict[str, Any]]] = {}
    for year in range(start_year - 1, end_year + 1):
        season = seasons_by_start.get(year)
        if not season:
            continue
        print(f"Süper Lig oyuncuları: {year}/{str(year + 1)[-2:]} ...")
        target_by_year[year] = fetch_league_players(52, int(season["id"]), year)

    jobs: list[dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        current = target_by_year.get(year) or {}
        previous_ids = set(target_by_year.get(year - 1) or {})
        for pid, target in current.items():
            if pid in previous_ids or float(target.get("minutes") or 0.0) < MIN_MINUTES:
                continue
            jobs.append(target)

    print(f"Yeni giriş adayı: {len(jobs)} oyuncu; geçmiş ligleri eşleştiriliyor...")
    samples: list[dict[str, Any]] = []

    def one(target: dict[str, Any]) -> dict[str, Any] | None:
        source = _source_for_player(target)
        if not source:
            return None
        sample: dict[str, Any] = {
            "player_id": int(target["player_id"]),
            "player": target.get("player") or "",
            "position": target.get("position") or "",
            "target_season_start": int(target["season_start"]),
            "target_team": target.get("team") or "",
            "source_tournament_id": int(source["tournament_id"]),
            "source_league": source.get("tournament") or "",
            "source_season": source.get("season") or "",
            "source_team": source.get("transfer_from") or "",
            "arrival_verified": bool(source.get("arrival_verified")),
        }
        for prefix, values in (("source", source), ("target", target)):
            for key in (
                "minutes",
                "apps",
                "starts",
                "goals",
                "assists",
                "yellow_cards",
                "red_cards",
                "saves",
                "clean_sheets",
                "goals_conceded",
                "xg",
                "xa",
                "key_passes",
                "shots_on_target",
                "rating",
                "has_xg",
                "has_xa",
                "has_key_passes",
                "has_shots_on_target",
            ):
                sample[f"{prefix}_{key}"] = values.get(key, 0.0)
        return sample

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(one, target) for target in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            sample = future.result()
            if sample:
                samples.append(sample)
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)} işlendi, {len(samples)} eşleşme")

    samples.sort(key=lambda r: (r["target_season_start"], r["source_league"], r["player"]))
    return samples


@dataclass(frozen=True)
class MetricSpec:
    source: str
    target: str
    denominator: str
    positions: tuple[str, ...]
    availability: str | None = None
    hard_cap: float | None = None


METRICS: dict[str, MetricSpec] = {
    "goals_p90": MetricSpec("goals", "goals", "minutes", ("DF", "MF", "FW")),
    "assists_p90": MetricSpec("assists", "assists", "minutes", ("DF", "MF", "FW")),
    "xg_p90": MetricSpec("xg", "xg", "minutes", ("DF", "MF", "FW"), "has_xg"),
    "xa_p90": MetricSpec("xa", "xa", "minutes", ("DF", "MF", "FW"), "has_xa"),
    "shots_on_target_p90": MetricSpec(
        "shots_on_target",
        "shots_on_target",
        "minutes",
        ("DF", "MF", "FW"),
        "has_shots_on_target",
    ),
    "key_passes_p90": MetricSpec(
        "key_passes",
        "key_passes",
        "minutes",
        ("DF", "MF", "FW"),
        "has_key_passes",
    ),
    "yellow_cards_p90": MetricSpec(
        "yellow_cards", "yellow_cards", "minutes", ("GK", "DF", "MF", "FW")
    ),
    "red_cards_p90": MetricSpec(
        "red_cards", "red_cards", "minutes", ("GK", "DF", "MF", "FW")
    ),
    "saves_p90": MetricSpec("saves", "saves", "minutes", ("GK",)),
    "goals_conceded_p90": MetricSpec(
        "goals_conceded", "goals_conceded", "minutes", ("GK",)
    ),
    "clean_sheet_rate": MetricSpec(
        "clean_sheets", "clean_sheets", "apps", ("GK", "DF", "MF"), hard_cap=1.0
    ),
    "minutes_per_app": MetricSpec(
        "minutes", "minutes", "per_app", ("GK", "DF", "MF", "FW"), hard_cap=90.0
    ),
    "rating": MetricSpec(
        "rating", "rating", "identity", ("GK", "DF", "MF", "FW"), hard_cap=10.0
    ),
}


def _metric_frame(samples: pd.DataFrame, position: str, spec: MetricSpec) -> pd.DataFrame:
    frame = samples[samples["position"] == position].copy()
    if frame.empty:
        return frame
    if spec.availability:
        frame = frame[
            frame[f"source_{spec.availability}"].astype(bool)
            & frame[f"target_{spec.availability}"].astype(bool)
        ]
    if spec.denominator == "minutes":
        src_den = frame["source_minutes"] / 90.0
        tgt_den = frame["target_minutes"] / 90.0
        x = frame[f"source_{spec.source}"] / src_den
        y = frame[f"target_{spec.target}"] / tgt_den
        exposure = 2.0 / (1.0 / src_den.clip(lower=0.1) + 1.0 / tgt_den.clip(lower=0.1))
    elif spec.denominator == "apps":
        src_den = frame["source_apps"].clip(lower=1.0)
        tgt_den = frame["target_apps"].clip(lower=1.0)
        x = frame[f"source_{spec.source}"] / src_den
        y = frame[f"target_{spec.target}"] / tgt_den
        exposure = 2.0 / (1.0 / src_den + 1.0 / tgt_den)
    elif spec.denominator == "per_app":
        x = frame["source_minutes"] / frame["source_apps"].clip(lower=1.0)
        y = frame["target_minutes"] / frame["target_apps"].clip(lower=1.0)
        exposure = 2.0 / (
            1.0 / frame["source_apps"].clip(lower=1.0)
            + 1.0 / frame["target_apps"].clip(lower=1.0)
        )
    else:
        x = frame[f"source_{spec.source}"]
        y = frame[f"target_{spec.target}"]
        exposure = 2.0 / (
            1.0 / frame["source_apps"].clip(lower=1.0)
            + 1.0 / frame["target_apps"].clip(lower=1.0)
        )
        frame = frame[(x > 0) & (y > 0)]
        x, y, exposure = x.loc[frame.index], y.loc[frame.index], exposure.loc[frame.index]

    frame = frame.assign(x=x, y=y, weight=exposure.clip(lower=1.0, upper=20.0))
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y", "weight"])
    if frame.empty:
        return frame
    x_cap = float(frame["x"].quantile(0.995))
    y_cap = float(frame["y"].quantile(0.995))
    frame["x"] = frame["x"].clip(lower=0.0, upper=max(x_cap, 1e-6))
    frame["y"] = frame["y"].clip(lower=0.0, upper=max(y_cap, 1e-6))
    return frame


def _weighted_mae(actual: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> float:
    return float(np.average(np.abs(actual - predicted), weights=weight))


def _fit_global(frame: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    x = frame["x"].to_numpy(float)
    y = frame["y"].to_numpy(float)
    w = frame["weight"].to_numpy(float)
    mu = float(np.average(x, weights=w))
    variance = float(np.average((x - mu) ** 2, weights=w))
    scale = max(math.sqrt(variance), 1e-4)
    z = (x - mu) / scale
    design = np.column_stack([np.ones(len(z)), z])
    beta = np.linalg.pinv(design.T @ (w[:, None] * design)) @ (design.T @ (w * y))
    beta[1] = max(0.0, float(beta[1]))
    return beta, mu, scale


def _raw_coefficients(beta: np.ndarray, mu: float, scale: float) -> tuple[float, float]:
    slope = max(0.0, float(beta[1]) / scale)
    intercept = float(beta[0]) - slope * mu
    return intercept, min(slope, 2.5)


def _fit_local(
    frame: pd.DataFrame,
    global_beta: np.ndarray,
    mu: float,
    scale: float,
    prior_players: float,
) -> tuple[np.ndarray, float]:
    if len(frame) < 3 or prior_players >= 1e8:
        return global_beta.copy(), 0.0
    x = frame["x"].to_numpy(float)
    y = frame["y"].to_numpy(float)
    w = frame["weight"].to_numpy(float)
    w = w * len(w) / max(w.sum(), 1e-9)
    z = (x - mu) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = float(prior_players) * np.eye(2)
    beta = np.linalg.pinv(design.T @ (w[:, None] * design) + penalty) @ (
        design.T @ (w * y) + penalty @ global_beta
    )
    beta[1] = max(0.0, float(beta[1]))
    n_eff = float((w.sum() ** 2) / max(np.square(w).sum(), 1e-9))
    local_weight = n_eff / (n_eff + float(prior_players))
    return beta, local_weight


def _apply_strength(
    intercept: float,
    slope: float,
    strength: float,
) -> tuple[float, float]:
    return strength * intercept, (1.0 - strength) + strength * slope


def _fit_family(
    frame: pd.DataFrame,
    prior_players: float,
    strength: float,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    global_beta, mu, scale = _fit_global(frame)
    intercept, slope = _raw_coefficients(global_beta, mu, scale)
    intercept, slope = _apply_strength(intercept, slope, strength)
    cap = float(frame["y"].quantile(0.995) * 1.20)
    global_model = {
        "intercept": intercept,
        "slope": slope,
        "cap": cap,
        "n_players": int(len(frame)),
        "local_weight": 0.0,
    }
    locals_: dict[int, dict[str, Any]] = {}
    for tid, local in frame.groupby("source_tournament_id"):
        beta, local_weight = _fit_local(
            local, global_beta, mu, scale, prior_players
        )
        local_intercept, local_slope = _raw_coefficients(beta, mu, scale)
        local_intercept, local_slope = _apply_strength(
            local_intercept, local_slope, strength
        )
        locals_[int(tid)] = {
            "intercept": local_intercept,
            "slope": local_slope,
            "cap": cap,
            "n_players": int(len(local)),
            "local_weight": local_weight,
        }
    return global_model, locals_


def _cross_validate(frame: pd.DataFrame) -> dict[str, float]:
    years = sorted(int(y) for y in frame["target_season_start"].unique())
    validation_years = years[-3:] if len(years) >= 5 else years[-1:]
    priors = (5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 1e9)
    strengths = (0.0, 0.25, 0.50, 0.75, 1.0)
    best: tuple[float, float, float] | None = None
    identity_errors: list[tuple[float, float]] = []

    for prior in priors:
        for strength in strengths:
            errors: list[tuple[float, float]] = []
            for year in validation_years:
                train = frame[frame["target_season_start"] < year]
                valid = frame[frame["target_season_start"] == year]
                if len(train) < 12 or valid.empty:
                    continue
                global_model, locals_ = _fit_family(train, prior, strength)
                predicted = []
                for _, row in valid.iterrows():
                    fitted = locals_.get(int(row["source_tournament_id"]), global_model)
                    predicted.append(
                        max(
                            0.0,
                            min(
                                float(fitted["cap"]),
                                float(fitted["intercept"])
                                + float(fitted["slope"]) * float(row["x"]),
                            ),
                        )
                    )
                weight = valid["weight"].to_numpy(float)
                error = _weighted_mae(
                    valid["y"].to_numpy(float), np.asarray(predicted), weight
                )
                errors.append((error, float(weight.sum())))
                if prior == priors[0] and strength == strengths[0]:
                    identity_errors.append(
                        (
                            _weighted_mae(
                                valid["y"].to_numpy(float),
                                valid["x"].to_numpy(float),
                                weight,
                            ),
                            float(weight.sum()),
                        )
                    )
            if not errors:
                continue
            score = float(np.average([e for e, _ in errors], weights=[w for _, w in errors]))
            candidate = (score, prior, strength)
            if best is None or candidate < best:
                best = candidate

    identity = (
        float(
            np.average(
                [error for error, _ in identity_errors],
                weights=[weight for _, weight in identity_errors],
            )
        )
        if identity_errors
        else 0.0
    )
    if best is None:
        return {"prior_players": 1e9, "strength": 0.0, "cv_mae": identity, "identity_mae": identity}
    return {
        "prior_players": float(best[1]),
        "strength": float(best[2]),
        "cv_mae": float(best[0]),
        "identity_mae": identity,
    }


def fit_calibration(samples: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(samples)
    global_positions: dict[str, dict[str, Any]] = {}
    leagues: dict[str, dict[str, Any]] = {}
    validation: dict[str, Any] = {}

    league_names = (
        frame.groupby("source_tournament_id")["source_league"]
        .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0])
        .to_dict()
    )
    league_counts = frame.groupby("source_tournament_id")["player_id"].nunique().to_dict()
    for tid, name in league_names.items():
        leagues[str(int(tid))] = {
            "name": str(name),
            "n_players": int(league_counts.get(tid, 0)),
            "positions": {},
        }

    for metric, spec in METRICS.items():
        for position in spec.positions:
            data = _metric_frame(frame, position, spec)
            if len(data) < 10:
                continue
            cv = _cross_validate(data)
            global_model, local_models = _fit_family(
                data,
                cv["prior_players"],
                cv["strength"],
            )
            if spec.hard_cap is not None:
                global_model["cap"] = min(float(global_model["cap"]), spec.hard_cap)
            global_positions.setdefault(position, {})[metric] = global_model
            validation[f"{position}:{metric}"] = cv
            for tid, local_model in local_models.items():
                if float(local_model.get("local_weight") or 0.0) <= 0:
                    continue
                if spec.hard_cap is not None:
                    local_model["cap"] = min(float(local_model["cap"]), spec.hard_cap)
                leagues.setdefault(
                    str(tid),
                    {"name": str(league_names.get(tid, tid)), "n_players": 0, "positions": {}},
                )
                leagues[str(tid)]["positions"].setdefault(position, {})[metric] = local_model

    for tid, league in leagues.items():
        subset = frame[
            (frame["source_tournament_id"] == int(tid))
            & frame["position"].isin(["DF", "MF", "FW"])
        ].copy()
        if subset.empty:
            continue
        source_n90 = subset["source_minutes"].clip(lower=1.0) / 90.0
        target_n90 = subset["target_minutes"].clip(lower=1.0) / 90.0
        source_rate = (subset["source_goals"] + subset["source_assists"]) / source_n90
        target_rate = (subset["target_goals"] + subset["target_assists"]) / target_n90
        weight = (
            2.0 / (1.0 / source_n90.clip(lower=0.1) + 1.0 / target_n90.clip(lower=0.1))
        ).clip(upper=20.0)
        source_mean = float(np.average(source_rate, weights=weight))
        target_mean = float(np.average(target_rate, weights=weight))
        league["observed_attack_retention"] = (
            target_mean / source_mean if source_mean > 0 else 1.0
        )
        league["observed_source_ga_p90"] = source_mean
        league["observed_target_ga_p90"] = target_mean
        if len(subset) >= 4 and source_mean > 0:
            rng = np.random.default_rng(20260814 + int(tid))
            ratios = []
            source_values = source_rate.to_numpy(float)
            target_values = target_rate.to_numpy(float)
            weights = weight.to_numpy(float)
            for _ in range(400):
                picked = rng.integers(0, len(subset), len(subset))
                src = float(np.average(source_values[picked], weights=weights[picked]))
                tgt = float(np.average(target_values[picked], weights=weights[picked]))
                if src > 0:
                    ratios.append(tgt / src)
            if ratios:
                league["observed_attack_retention_p10"] = float(
                    np.quantile(ratios, 0.10)
                )
                league["observed_attack_retention_p90"] = float(
                    np.quantile(ratios, 0.90)
                )

    weighted_improvements = []
    for result in validation.values():
        identity = float(result.get("identity_mae") or 0.0)
        fitted = float(result.get("cv_mae") or 0.0)
        if identity > 0:
            weighted_improvements.append(1.0 - fitted / identity)

    return {
        "version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_tournament_id": 52,
        "method": {
            "description": (
                "Aynı oyuncunun dış lig sezonu ile ilk sonraki Süper Lig sezonu; "
                "450+ dakika iki tarafta; ilk kez gelenler; dakika ağırlıklı doğrusal "
                "kalibrasyon; lig katsayıları global mevki modeline küçültülür."
            ),
            "min_minutes_each_side": MIN_MINUTES,
            "cross_validation": (
                "Son üç sezon ileri-zaman doğrulaması; küçültme gücü ve kimlik-model "
                "karışımı MAE ile seçilir."
            ),
            "tournament_identity": (
                "Turnuva adı ve tier bilgisi canlı Sofascore metadata ile doğrulanır; "
                "xG/xA bulunmayan eski sezonlar sıfır sayılmaz."
            ),
            "mean_cv_improvement_vs_identity": float(np.mean(weighted_improvements))
            if weighted_improvements
            else 0.0,
        },
        "sample": {
            "n_players": int(frame["player_id"].nunique()),
            "n_pairs": int(len(frame)),
            "target_seasons": sorted(int(y) for y in frame["target_season_start"].unique()),
            "n_leagues": int(frame["source_tournament_id"].nunique()),
        },
        "global": {"positions": global_positions},
        "leagues": leagues,
        "validation": validation,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value).__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2018)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "league_translation.json",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=CACHE_DIR / "league_translation_samples.json",
    )
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()

    if args.fit_only:
        samples = json.loads(args.samples.read_text(encoding="utf-8"))
    else:
        samples = collect_samples(args.start, args.end, workers=args.workers)
        args.samples.parent.mkdir(parents=True, exist_ok=True)
        args.samples.write_text(
            json.dumps(samples, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
    if not samples:
        raise RuntimeError("Kalibrasyon örneklemi boş.")

    model = fit_calibration(samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(model, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(
        f"Tamam: {model['sample']['n_pairs']} eşleşme, "
        f"{model['sample']['n_leagues']} lig → {args.output}"
    )


if __name__ == "__main__":
    main()
