"""Haftalık ilk 11, yedekler ve tek kart kararını saha görseline dönüştür."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/segoeui.ttf")
        if bold
        else ("C:/Windows/Fonts/segoeui.ttf",)
    )
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def _player_name(row: pd.Series) -> str:
    name = str(row.get("display_name") or row.get("player") or "")
    return name if len(name) <= 18 else f"{name[:16]}…"


def _gradient_background(image: Image.Image, top: str, bottom: str) -> None:
    top_rgb = tuple(int(top[i : i + 2], 16) for i in (1, 3, 5))
    bottom_rgb = tuple(int(bottom[i : i + 2], 16) for i in (1, 3, 5))
    draw = ImageDraw.Draw(image)
    for y in range(image.height):
        t = y / max(1, image.height - 1)
        color = tuple(int(a + (b - a) * t) for a, b in zip(top_rgb, bottom_rgb))
        draw.line((0, y, image.width, y), fill=color)


def _card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str = "#111c2f",
    outline: str | None = "#243550",
    radius: int = 18,
    shadow: bool = True,
    width: int = 1,
) -> None:
    x1, y1, x2, y2 = box
    if shadow:
        draw.rounded_rectangle(
            (x1 + 5, y1 + 7, x2 + 5, y2 + 7),
            radius=radius,
            fill="#07101e",
        )
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=fill,
        outline=outline,
        width=width,
    )


def _pill(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    fill: str,
    text_fill: str = "#f8fafc",
    font: ImageFont.ImageFont,
) -> None:
    draw.rounded_rectangle(box, radius=(box[3] - box[1]) // 2, fill=fill)
    _centered(draw, box, text, fill=text_fill, font=font)


def _centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    fill: str,
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    width, height = right - left, bottom - top
    x1, y1, x2, y2 = box
    draw.text(
        (x1 + (x2 - x1 - width) / 2, y1 + (y2 - y1 - height) / 2),
        text,
        fill=fill,
        font=font,
    )


def _player_box(
    draw: ImageDraw.ImageDraw,
    row: pd.Series,
    x: int,
    y: int,
    *,
    captain_name: str,
) -> None:
    width, height = 138, 78
    name = _player_name(row)
    is_captain = str(row.get("player") or "") == captain_name
    status = str(row.get("availability") or "").upper()
    border = "#fbbf24" if is_captain else "#2c4664"
    if status in {"DOUBTFUL", "INJURED", "SUSPENDED", "OUT", "UNAVAILABLE"}:
        border = "#fb7185"
    _card(
        draw,
        (x, y, x + width, y + height),
        fill="#101d30",
        outline=border,
        radius=14,
        width=2,
    )
    _centered(
        draw,
        (x + 5, y + 5, x + width - 5, y + 29),
        name,
        fill="#f8fafc",
        font=_font(12, bold=True),
    )
    opponent = str(row.get("fixture_opponent") or "rakip bekleniyor")
    home = row.get("fixture_home")
    venue = "İ" if home is True else ("D" if home is False else "")
    _centered(
        draw,
        (x + 5, y + 30, x + width - 5, y + 51),
        f"{opponent[:14]} · {venue}".strip(" ·"),
        fill="#9fb0c6",
        font=_font(10),
    )
    _pill(
        draw,
        (x + 33, y + 54, x + width - 33, y + 72),
        f"{float(row.get('projected_pts') or 0):.2f} XP",
        fill="#123d36",
        text_fill="#70e1b2",
        font=_font(10, bold=True),
    )
    if is_captain:
        _pill(
            draw,
            (x + width - 28, y - 8, x + width + 5, y + 17),
            "C",
            fill="#fbbf24",
            text_fill="#172033",
            font=_font(11, bold=True),
        )


def write_weekly_png(
    path: str | Path,
    result: dict[str, Any],
    card_decision: dict[str, Any],
) -> Path:
    """11 oyuncuyu sahaya, 4 yedeği kulübeye çiz."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    xi = result.get("xi")
    bench = result.get("bench")
    if not isinstance(xi, pd.DataFrame):
        xi = pd.DataFrame()
    if not isinstance(bench, pd.DataFrame):
        bench = pd.DataFrame()

    width, height = 1440, 960
    image = Image.new("RGB", (width, height), "#091322")
    _gradient_background(image, "#14233a", "#07111f")
    draw = ImageDraw.Draw(image)
    title = _font(31, bold=True)
    subtitle = _font(18, bold=True)
    normal = _font(14)
    muted = "#9fb0c6"
    white = "#f8fafc"
    green = "#70e1b2"

    draw.rounded_rectangle((36, 22, 1404, 134), radius=24, fill="#101c2e", outline="#263a57")
    draw.rounded_rectangle((36, 22, 48, 134), radius=6, fill="#38d39f")
    draw.text((70, 42), "TFF FANTEZİ LİG", fill=white, font=title)
    draw.text((72, 84), "HAFTALIK KADRO PLANI", fill=green, font=_font(13, bold=True))
    cap = result.get("captain") or {}
    metrics = [
        ("DİZİLİŞ", str(result.get("formation") or "-")),
        ("KAPTAN", str(cap.get("display_name") or cap.get("player") or "-")[:16]),
        (
            "BÜTÇE",
            f"{float(result.get('total_cost') or 0):.1f} / "
            f"{float(result.get('budget') or 100):.0f}M",
        ),
        ("HAFTA XP", f"{float(result.get('total_projected') or 0):.1f}"),
    ]
    metric_x = [642, 812, 1030, 1228]
    metric_w = [146, 194, 174, 142]
    for (label, value), x, box_w in zip(metrics, metric_x, metric_w):
        draw.text((x, 44), label, fill=muted, font=_font(10, bold=True))
        draw.text((x, 66), value, fill=white, font=_font(18, bold=True))
        draw.rounded_rectangle((x, 105, x + box_w, 109), radius=2, fill="#263a57")
        draw.rounded_rectangle(
            (x, 105, x + min(box_w, max(25, int(box_w * 0.72))), 109),
            radius=2,
            fill="#38d39f",
        )

    # Saha: koyu, çizgili ve daha sakin bir yayın grafiği.
    pitch = (38, 160, 946, 822)
    _card(draw, pitch, fill="#0d543f", outline="#39896b", radius=26, width=2)
    px1, py1, px2, py2 = 56, 178, 928, 804
    stripe_h = (py2 - py1) / 8
    for i in range(8):
        if i % 2 == 0:
            draw.rectangle(
                (px1, int(py1 + i * stripe_h), px2, int(py1 + (i + 1) * stripe_h)),
                fill="#105b44",
            )
    line = "#80c9a9"
    draw.rounded_rectangle((px1, py1, px2, py2), radius=12, outline=line, width=2)
    mid_y = (py1 + py2) // 2
    draw.line((px1, mid_y, px2, mid_y), fill=line, width=2)
    draw.ellipse((431, mid_y - 58, 553, mid_y + 58), outline=line, width=2)
    draw.ellipse((487, mid_y - 4, 495, mid_y + 4), fill=line)
    draw.rectangle((332, py1, 652, py1 + 88), outline=line, width=2)
    draw.rectangle((332, py2 - 88, 652, py2), outline=line, width=2)
    _pill(
        draw,
        (68, 190, 154, 220),
        "İLK 11",
        fill="#0b3f31",
        text_fill="#baf7dd",
        font=_font(12, bold=True),
    )

    captain_name = str(cap.get("player") or "")
    row_y = {"FW": 252, "MF": 397, "DF": 542, "GK": 687}
    for pos in ("FW", "MF", "DF", "GK"):
        group = xi[xi["position"] == pos]
        count = len(group)
        if count == 0:
            continue
        usable_left, usable_right = 72, 912
        centers = [
            usable_left + (i + 1) * (usable_right - usable_left) / (count + 1)
            for i in range(count)
        ]
        for center, (_, row) in zip(centers, group.iterrows()):
            _player_box(
                draw,
                row,
                int(center - 69),
                row_y[pos],
                captain_name=captain_name,
            )

    # Sağ panel: tek net karar ve yedek kulübesi.
    panel_x = 980
    draw.text((panel_x, 164), "HAFTA STRATEJİSİ", fill=white, font=subtitle)
    draw.text((panel_x, 190), "Tek kart kararı · otomatik hesaplanır", fill=muted, font=_font(11))
    use = bool(card_decision.get("use"))
    decision_fill = "#70e1b2" if use else "#fbbf24"
    _card(
        draw,
        (panel_x, 220, 1400, 382),
        fill="#111e31",
        outline=decision_fill,
        radius=20,
        width=2,
    )
    _pill(
        draw,
        (panel_x + 18, 240, panel_x + 104, 270),
        "KULLAN" if use else "SAKLA",
        fill="#16483b" if use else "#4a381a",
        text_fill=decision_fill,
        font=_font(11, bold=True),
    )
    draw.text(
        (panel_x + 18, 282),
        str(card_decision.get("card") or "Kart kullanma"),
        fill=decision_fill,
        font=_font(22, bold=True),
    )
    why = str(card_decision.get("why") or "")
    words, lines, current = why.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if len(trial) > 48:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    for line_no, line in enumerate(lines[:3]):
        draw.text(
            (panel_x + 18, 322 + line_no * 17),
            line,
            fill=muted,
            font=_font(11),
        )

    draw.text((panel_x, 414), "YEDEK KULÜBESİ", fill=white, font=subtitle)
    draw.text((panel_x + 182, 419), "4 OYUNCU", fill=green, font=_font(10, bold=True))
    y = 450
    for _, row in bench.iterrows():
        _card(
            draw,
            (panel_x, y, 1400, y + 70),
            fill="#101c2e",
            outline="#263a57",
            radius=14,
        )
        _pill(
            draw,
            (panel_x + 14, y + 14, panel_x + 58, y + 55),
            str(row.get("position") or "-"),
            fill="#1b314a",
            text_fill=green,
            font=_font(11, bold=True),
        )
        draw.text(
            (panel_x + 72, y + 10),
            _player_name(row),
            fill=white,
            font=_font(14, bold=True),
        )
        opponent = str(row.get("fixture_opponent") or "rakip bekleniyor")
        draw.text(
            (panel_x + 72, y + 38),
            f"{opponent[:19]} · {'İ' if row.get('fixture_home') is True else 'D'}",
            fill=muted,
            font=_font(11),
        )
        _pill(
            draw,
            (1332, y + 21, 1386, y + 49),
            f"{float(row.get('projected_pts') or 0):.1f}",
            fill="#123d36",
            text_fill=green,
            font=_font(11, bold=True),
        )
        y += 82

    _card(
        draw,
        (panel_x, 792, 1400, 854),
        fill="#0e1a2a",
        outline="#22344e",
        radius=14,
        shadow=False,
    )
    draw.text(
        (panel_x + 16, 806),
        "✓  Sakat / cezalı oyuncular kadro havuzuna alınmaz",
        fill=white,
        font=_font(11, bold=True),
    )
    draw.text(
        (panel_x + 16, 830),
        "↗  Rakip bilgisi takımın sıradaki maçını gösterir",
        fill=muted,
        font=_font(10),
    )

    draw.line((38, 895, 1402, 895), fill="#243550", width=1)
    draw.text(
        (40, 914),
        "MODEL NOTU  ·  İlk hafta verisi düşük ağırlıklıdır  ·  Tahmin olasılıksaldır",
        fill=muted,
        font=_font(11, bold=True),
    )
    draw.text(
        (1190, 914),
        "HER ÇALIŞTIRMADA GÜNCELLENİR",
        fill="#526982",
        font=_font(10, bold=True),
    )
    image.save(target, "PNG", optimize=True)
    return target
