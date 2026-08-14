"""TFF Fantezi Lig menajer kartları için şeffaf haftalık fırsat hesabı."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import BENCH_WEIGHT, BUDGET_M
from .optimize import optimize_squad

CARD_USE_THRESHOLDS = {
    "Dört Dörtlük Kaptan": 12.0,
    "Tripleks Kaptan": 6.0,
    "Tüm Takım Sahaya": 8.0,
    "Hücum": 2.0,
}


def manager_card_advice(
    result: dict[str, Any],
    available_players: pd.DataFrame,
    *,
    budget: float = BUDGET_M,
) -> list[dict[str, Any]]:
    """Kartları mevcut haftanın beklenen puan farkına göre sırala.

    Normal kaptan 2x kabul edilir. Bu nedenle Tripleks'in getirisi +1x,
    Dört Dörtlük'ün getirisi +2x kaptan puanıdır. Kart envanteri kullanıcı
    hesabından okunmadığı için bu fonksiyon kullanım adedi değil, fırsat
    sıralaması üretir.
    """
    xi = result.get("xi")
    bench = result.get("bench")
    captain = result.get("captain") or {}
    if not isinstance(xi, pd.DataFrame) or xi.empty:
        return []
    if not isinstance(bench, pd.DataFrame):
        bench = pd.DataFrame()

    captain_pts = float(captain.get("projected_pts") or 0.0)
    bench_pts = float(
        pd.to_numeric(bench.get("projected_pts", pd.Series(dtype=float)), errors="coerce")
        .fillna(0.0)
        .sum()
    )
    advice = [
        {
            "card": "Dört Dörtlük Kaptan",
            "extra_pts": round(2.0 * captain_pts, 2),
            "why": f"{captain.get('display_name') or captain.get('player') or 'Kaptan'} için 4x",
        },
        {
            "card": "Tripleks Kaptan",
            "extra_pts": round(captain_pts, 2),
            "why": f"{captain.get('display_name') or captain.get('player') or 'Kaptan'} için 3x",
        },
        {
            "card": "Tüm Takım Sahaya",
            "extra_pts": round((1.0 - BENCH_WEIGHT) * bench_pts, 2),
            "why": f"4 yedeğin projeksiyonu {bench_pts:.2f}p",
        },
    ]

    # Kartın resmi diziliş istisnaları uygulama içinde görünmediği için burada
    # yalnızca doğrulanabilir 5M bütçe etkisini hesapla.
    try:
        attack_result = optimize_squad(available_players, budget=budget + 5.0)
        normal_total = float(result.get("total_projected") or 0.0)
        attack_total = float(attack_result.get("total_projected") or 0.0)
        advice.append(
            {
                "card": "Hücum",
                "extra_pts": round(max(0.0, attack_total - normal_total), 2),
                "why": "105M ile yeniden optimize edildi; uygulamadaki özel diziliş ayrıca doğrulanmalı",
            }
        )
    except Exception:
        pass

    # Limitsiz Bütçe transfer anındaki kadro kurma kartıdır; haftalık sabit
    # puan bonusu olmadığı için mevcut kadroda sayı uydurulmaz.
    advice.append(
        {
            "card": "Limitsiz Bütçe",
            "extra_pts": None,
            "why": "Transfer haftasında pahalı hedef için kullan; sabit haftalık puan bonusu yok",
        }
    )
    return sorted(
        advice,
        key=lambda row: float(row["extra_pts"]) if row["extra_pts"] is not None else -1.0,
        reverse=True,
    )


def choose_manager_card(advice: list[dict[str, Any]]) -> dict[str, Any]:
    """Haftada en fazla bir kart öner; fırsat zayıfsa tüm kartları sakla."""
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in advice:
        threshold = CARD_USE_THRESHOLDS.get(str(item.get("card") or ""))
        extra = item.get("extra_pts")
        if threshold and extra is not None:
            scored.append((float(extra) / threshold, item))
    if not scored:
        return {
            "use": False,
            "card": "Kart kullanma",
            "extra_pts": 0.0,
            "why": "Bu hafta ölçülebilir kart fırsatı yok; hakları sonraki haftaya sakla.",
        }

    ratio, best = max(scored, key=lambda pair: pair[0])
    threshold = CARD_USE_THRESHOLDS[str(best["card"])]
    if ratio < 1.0:
        return {
            "use": False,
            "card": "Kart kullanma",
            "extra_pts": round(float(best["extra_pts"]), 2),
            "candidate": best["card"],
            "why": (
                f"En iyi aday {best['card']} (+{float(best['extra_pts']):.2f}p), "
                f"ama kullanım eşiği +{threshold:.0f}p. Haftada tek kart ve kart başına "
                "iki hak olduğu için bu hafta sakla; sonraki hafta yeniden bak."
            ),
        }
    return {
        "use": True,
        "card": best["card"],
        "extra_pts": round(float(best["extra_pts"]), 2),
        "why": (
            f"{best['why']}; +{float(best['extra_pts']):.2f}p fırsat "
            f"+{threshold:.0f}p kullanım eşiğini geçti. Bu hafta yalnız bu kartı kullan."
        ),
    }
