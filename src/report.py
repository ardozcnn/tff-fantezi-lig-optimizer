"""Türkçe CLI rapor — sade kadro çıktısı."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import SQUAD

POS_SHORT = {"GK": "KL", "DF": "DF", "MF": "OS", "FW": "FV"}


def _name(row: pd.Series) -> str:
    full = str(row.get("player") or "").strip()
    disp = str(row.get("display_name") or "").strip()
    match = str(row.get("match_name") or "").strip()
    if disp and len(disp.split()) >= 2 and disp.lower() != match.lower():
        return disp[:32]
    if match and full:
        first = full.split()[0]
        if first.lower() not in match.lower():
            return f"{first} {match}"[:32]
        if len(match.split()) >= 2:
            return match[:32]
    parts = full.split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[-1]}"[:32]
    return (disp or full)[:32]


def _line(row: pd.Series) -> str:
    pos = POS_SHORT.get(str(row["position"]), str(row["position"]))
    team = str(row.get("team") or "")[:20]
    price = float(row.get("price_m") or 0)
    return f"  {pos}  {_name(row):<28}  {team:<20}  {price:4.1f}M"


def _bench_label(bshape: dict[str, int]) -> str:
    if not bshape:
        return ""
    parts = [
        f"{int(bshape[p])} {POS_SHORT[p]}"
        for p in ("GK", "DF", "MF", "FW")
        if bshape.get(p)
    ]
    return ", ".join(parts)


def print_squad(result: dict[str, Any]) -> None:
    xi = result.get("xi")
    bench = result.get("bench")
    if xi is None:
        xi = result["squad"]
        bench = pd.DataFrame()

    formation = result.get("formation", "4-4-2")
    bshape = result.get("bench_shape") or {}
    btxt = _bench_label(bshape)

    print()
    print("=" * 46)
    print("  TFF FANTEZİ LİG — ÖNERİLEN KADRO")
    print("=" * 46)
    print()
    print(f"Diziliş: {formation}")
    print()
    print("İLK 11")
    if xi is None or xi.empty:
        print("  (yok)")
    else:
        for _, r in xi.iterrows():
            print(_line(r))
    print()
    if btxt:
        print(f"YEDEKLER ({btxt})")
    else:
        print("YEDEKLER")
    if bench is None or bench.empty:
        print("  (yok)")
    else:
        for _, r in bench.iterrows():
            print(_line(r))
    print()
    cap = result["captain"]
    cap_name = cap.get("display_name") or cap["player"]
    print(f"Kaptan : {cap_name} ({cap.get('team', '')})")
    print(f"Maliyet: {result['total_cost']:.1f} / {result['budget']:.1f} M TL")
    if result.get("bank", 0) > 0:
        print(f"Kasa   : {result['bank']:.1f} M TL")
    sq = result["squad"]
    counts = sq["position"].value_counts().to_dict()
    chk = "  ".join(f"{POS_SHORT[p]} {counts.get(p, 0)}/{need}" for p, need in SQUAD.items())
    print(f"Kadro  : {chk}")
    print()


def print_meta(meta: dict[str, Any]) -> None:
    """--verbose: analiz özeti."""
    print("=== Analiz ===")
    form = meta.get("formation", "4-4-2")
    bench = meta.get("bench_shape") or {}
    if bench:
        btxt = " / ".join(
            f"{bench.get(p, 0)} {POS_SHORT[p]}" for p in ("GK", "DF", "MF", "FW") if bench.get(p, 0)
        )
        print(f"  Diziliş: {form}  |  yedek {btxt}")
    print(f"  SL eşleşme: {meta.get('matched_sl', '?')}/{meta.get('n_prices', '?')}")
    print(f"  Dış lig önceliği: {meta.get('external_priors', 0)} oyuncu")
    if meta.get("blocked_unavailable"):
        print(f"  Sakat/cezalı nedeniyle dışarıda: {meta['blocked_unavailable']}")
    for n in meta.get("notes") or []:
        print(f"  • {n}")
    fixtures = meta.get("fixtures") or []
    if fixtures:
        print("  Yaklaşan fikstür:")
        for fixture in fixtures:
            print(
                f"    {fixture.get('kickoff', '')}  "
                f"{fixture.get('home', '')} - {fixture.get('away', '')}"
            )
    card = meta.get("manager_card") or {}
    if card:
        action = "KULLAN" if card.get("use") else "SAKLA"
        print(f"  Menajer kartı: {action} — {card.get('card')}")
        print(f"    {card.get('why', '')}")
    print()


def print_leaders(players: pd.DataFrame, n: int = 8) -> None:
    """--stats-only / --verbose."""
    pos_tr = {"GK": "Kaleci", "DF": "Defans", "MF": "Orta saha", "FW": "Forvet"}
    print("=== Mevki liderleri ===")
    for pos in ("GK", "DF", "MF", "FW"):
        sub = players[players["position"] == pos].sort_values("projected_pts", ascending=False).head(n)
        print(f"-- {pos_tr[pos]} --")
        name_col = "display_name" if "display_name" in sub.columns else "player"
        for _, r in sub.iterrows():
            print(
                f"  {_name(r):<28}  {str(r['team']):<18}  "
                f"{float(r['projected_pts']):5.2f}p  {float(r['price_m']):4.1f}M"
            )
        print()
