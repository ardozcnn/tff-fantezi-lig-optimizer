"""Ortak analiz hattı: TFF fiyat + SL form + dış lig önceliği + ILP kadro."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .config import BUDGET_M, CACHE_DIR, COOKIE_FILE, DATA_DIR, FORM_MATCHES, PRICES_FILE, is_quiet
from .fetch_external import apply_external_priors
from .fetch_fotmob import apply_fotmob_validation
from .fetch_stats import load_dual_season_stats, next_matchweek_fixtures, resolve_season_id
from .load_prices import load_prices, merge_prices
from .manager_cards import choose_manager_card, manager_card_advice
from .names import normalize_name
from .optimize import optimize_squad
from .scoring import (
    apply_context_adjustments,
    build_player_table,
    lookup_fixture_context,
)
from .tff_client import fetch_tff_prices, load_saved_login, save_prices_csv
from .weekly_report import write_weekly_png

ProgressCb = Callable[[str], None]


def _say(progress: ProgressCb | None, msg: str) -> None:
    if progress:
        progress(msg)
    elif not is_quiet():
        print(msg)


def _df_records(df: pd.DataFrame, n: int | None = None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    show = df.head(n) if n else df
    out = []
    for rec in show.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if pd.isna(v):
                clean[k] = None
            elif hasattr(v, "item"):
                clean[k] = v.item()
            else:
                clean[k] = v
        out.append(clean)
    return out


def run_pipeline(
    *,
    season: int | None = None,
    prices_path: str | Path = PRICES_FILE,
    budget: float = BUDGET_M,
    form_matches: int = FORM_MATCHES,
    cookie_file: str | Path = COOKIE_FILE,
    fetch_prices: bool = True,
    refresh_cache: bool = False,
    report_png: str | Path | None = None,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    os.environ.setdefault("FBREF_SSL_VERIFY", "0")
    prices_path = Path(prices_path)
    cookie_path = Path(cookie_file)

    if refresh_cache and CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        _say(progress, "Cache temizlendi.")

    cookie_text = cookie_path.read_text(encoding="utf-8") if cookie_path.exists() else None
    saved_email, saved_pass = load_saved_login()
    should_fetch = fetch_prices and bool(
        cookie_text or saved_email or os.environ.get("TFF_EMAIL")
    )

    if should_fetch:
        _say(
            progress,
            "TFF Fantezi fiyatları çekiliyor...",
        )
        try:
            pdf = fetch_tff_prices(
                cookie=cookie_text,
                email=saved_email,
                password=saved_pass,
                raw_dump=str(DATA_DIR / "tff_raw.json"),
            )
            save_prices_csv(pdf, prices_path)
            _say(progress, f"TFF fiyatları: {len(pdf)} oyuncu")
        except Exception as exc:  # noqa: BLE001
            _say(progress, f"TFF fiyat hatası: {exc} — kayıtlı CSV deneniyor.")
            if not prices_path.exists():
                raise

    _say(progress, "Süper Lig form + sezon istatistikleri (Sofascore)...")
    form_df, prev, form_cs, base_cs, meta = load_dual_season_stats(
        current_start=season,
        form_matches=form_matches,
    )
    _say(
        progress,
        f"Form {len(form_df)} / önceki sezon {len(prev)} / mevcut {meta.get('current_season_rows', 0)}",
    )

    players = build_player_table(form_df, prev, form_cs, base_cs, meta)
    if players.empty:
        raise RuntimeError("Oyuncu istatistiği boş.")

    if not prices_path.exists():
        raise FileNotFoundError(f"Fiyat dosyası yok: {prices_path}")

    prices = load_prices(prices_path)
    merged = merge_prices(players, prices)
    fixture_context = meta.get("fixture_context") or {}
    cs_context = {
        normalize_name(str(team)): {"cs_rate": float(rate)}
        for team, rate in (base_cs or {}).items()
    }
    cs_values = [
        float(rate)
        for rate in (base_cs or {}).values()
        if rate is not None and pd.notna(rate)
    ]
    league_cs = float(pd.Series(cs_values).median()) if cs_values else 0.28
    for idx, row in merged.iterrows():
        team_name = str(row.get("team") or "")
        cs = lookup_fixture_context(team_name, cs_context)
        if cs and cs.get("cs_rate") is not None:
            merged.at[idx, "team_cs_base"] = float(cs["cs_rate"])
        elif pd.isna(row.get("team_cs_base")):
            merged.at[idx, "team_cs_base"] = league_cs
        fixture = lookup_fixture_context(team_name, fixture_context)
        if fixture:
            merged.at[idx, "fixture_opponent"] = fixture.get("opponent") or ""
            merged.at[idx, "fixture_home"] = fixture.get("home")
            merged.at[idx, "fixture_attack_mult"] = fixture.get("attack_mult") or 1.0
            merged.at[idx, "fixture_cs_mult"] = fixture.get("cs_mult") or 1.0
    matched = int(merged["stats_player"].notna().sum())
    _say(progress, f"SL eşleşmesi: {matched}/{len(merged)}")

    merged = apply_external_priors(merged, progress=progress)
    _say(progress, "FotMob ikinci kaynak (ilk 11 + güncel kulüp maçları)...")
    merged = apply_fotmob_validation(
        merged,
        progress=progress,
        early_season=bool(meta.get("early_season", meta.get("preseason"))),
    )
    merged = apply_context_adjustments(merged)
    prices_numeric = pd.to_numeric(merged["price_m"], errors="coerce").replace(0, pd.NA)
    merged["ppm"] = (
        pd.to_numeric(merged["projected_pts"], errors="coerce").fillna(0.0)
        / prices_numeric
    ).fillna(0.0)
    ext_n = int((merged.get("data_src") == "external_prior").sum()) if "data_src" in merged.columns else 0
    eligible = merged[
        merged.get("selection_eligible", pd.Series(True, index=merged.index))
    ].copy()
    blocked = int(len(merged) - len(eligible))
    if eligible.empty:
        raise RuntimeError("Seçilebilir oyuncu kalmadı; TFF sakat/cezalı durumlarını kontrol et.")
    _say(progress, f"En iyi diziliş + XI + yedek optimize ediliyor ({ext_n} yeni imza)...")

    result = optimize_squad(eligible, budget=budget)
    squad: pd.DataFrame = result["squad"].copy()
    if "display_name" not in squad.columns:
        squad["display_name"] = squad["player"]
    result["squad"] = squad
    for key in ("xi", "bench"):
        part = result.get(key)
        if isinstance(part, pd.DataFrame) and "display_name" not in part.columns:
            part["display_name"] = part["player"]
            result[key] = part

    cap = result["captain"]
    xi = result.get("xi", squad)
    cap_row = xi.loc[xi["projected_pts"].idxmax()]
    cap["display_name"] = str(cap_row.get("display_name") or cap["player"])
    cap["reason"] = str(cap_row.get("reason") or "")
    cap["data_src"] = str(cap_row.get("data_src") or "")
    result["captain"] = cap

    fixtures: list[dict[str, str]] = []
    try:
        fixtures = next_matchweek_fixtures(resolve_season_id(int(meta["requested_start"])))
    except Exception as exc:  # noqa: BLE001
        meta.setdefault("notes", []).append(f"Haftalık fikstür alınamadı ({exc}).")
    cards = manager_card_advice(result, eligible, budget=budget)
    card_decision = choose_manager_card(cards)
    report_path = None
    if report_png:
        report_path = str(write_weekly_png(report_png, result, card_decision))

    leaders = {}
    for pos in ("GK", "DF", "MF", "FW"):
        sub = merged[merged["position"] == pos].sort_values("projected_pts", ascending=False)
        leaders[pos] = _df_records(sub, 8)

    src = merged["data_src"] if "data_src" in merged.columns else ""
    new_signings = merged[
        (src == "external_prior")
        | (
            (merged["price_m"] >= 8)
            & (merged["projected_pts"] > 0)
        )
    ].sort_values("projected_pts", ascending=False)
    unmatched = merged[
        (merged["projected_pts"] <= 0) & (merged["price_m"] >= 6)
    ].sort_values("price_m", ascending=False)

    meta_clean = {
        k: v
        for k, v in meta.items()
        if not isinstance(v, pd.DataFrame)
    }
    meta_clean.update(
        {
            "matched_sl": matched,
            "n_prices": len(merged),
            "external_priors": ext_n,
            "blocked_unavailable": blocked,
            "fixtures": fixtures,
            "manager_card": card_decision,
            "report_png": report_path,
            "formation": result.get("formation", "4-4-2"),
            "bench_shape": result.get("bench_shape") or {},
            "method": (
                "dakika (60+=2p) + xG/xA + şut/kilit pas + CS/kart/bonus/penaltı; "
                "mevcut/önceki sezon örnek küçültme; haftalık rakip ve iç/dış saha; "
                "Sofascore + FotMob ilk 11/xG/güncel maç doğrulaması; "
                "resmî TFF puan kalibrasyonu; yeni imza son 1–2 lig sezonu ve "
                "geçmiş lig→SL transferlerinden ileri-zaman doğrulanmış dönüşüm; "
                "diziliş otomatik (4-4-2/4-5-1/3-5-2…); yedek otomatik giriş."
            ),
        }
    )

    return {
        "ok": True,
        "meta": meta_clean,
        "result": {
            "squad": _df_records(squad),
            "xi": _df_records(result.get("xi")),
            "bench": _df_records(result.get("bench")),
            "formation": result.get("formation", "4-4-2"),
            "total_cost": result["total_cost"],
            "bank": result["bank"],
            "total_projected": result["total_projected"],
            "xi_projected": result.get("xi_projected"),
            "bench_projected": result.get("bench_projected"),
            "captain": result["captain"],
            "budget": result["budget"],
        },
        "leaders": leaders,
        "new_signings": _df_records(new_signings, 25),
        "unmatched": _df_records(unmatched, 20),
        "fixtures": fixtures,
        "manager_card": card_decision,
        "report_png": report_path,
        "merged": merged,
        "raw_result": result,
    }


def run_cli_pipeline(args: Any) -> int:
    from .report import print_leaders, print_meta, print_squad

    try:
        payload = run_pipeline(
            season=args.season,
            prices_path=args.prices,
            budget=args.budget,
            form_matches=args.form_matches,
            fetch_prices=not args.no_fetch_prices,
            refresh_cache=args.refresh_cache,
            report_png=args.report_png,
        )
    except FileNotFoundError:
        print("Fiyat dosyası yok.", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    verbose = getattr(args, "verbose", False)
    if verbose:
        print_meta(payload["meta"])
    if args.export_stats:
        payload["merged"].to_csv(args.export_stats, index=False)
        if verbose:
            print(f"İstatistik tablosu: {args.export_stats}")
    if payload.get("report_png"):
        print(f"PNG raporu: {payload['report_png']}")
    if args.stats_only:
        print_leaders(payload["merged"], n=args.leaders or 8)
        return 0
    if args.leaders:
        print_leaders(payload["merged"], n=args.leaders)
    print_squad(payload["raw_result"])
    return 0
