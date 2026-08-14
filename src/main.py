"""CLI giriş noktası."""

from __future__ import annotations

import argparse
import os

from .config import BUDGET_M, FORM_MATCHES, PRICES_FILE


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tff-optimizer",
        description="TFF Fantezi Lig: haftanın 15'li kadro önerisi.",
    )
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--prices", type=str, default=str(PRICES_FILE))
    p.add_argument("--budget", type=float, default=BUDGET_M)
    p.add_argument("--form-matches", type=int, default=FORM_MATCHES)
    p.add_argument("--leaders", type=int, default=0)
    p.add_argument("--stats-only", action="store_true")
    p.add_argument("--no-fetch-prices", action="store_true")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--export-stats", type=str, default=None)
    p.add_argument(
        "--report-png",
        type=str,
        default="data/weekly_report.png",
        help="Haftalık kadro/fikstür/kart özet PNG dosyası",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Ayrıntılı analiz logu (varsayılan: sade kadro çıktısı)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("FBREF_SSL_VERIFY", "0")
    os.environ.setdefault("TFF_QUIET", "0" if args.verbose else "1")

    from .pipeline import run_cli_pipeline

    return run_cli_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
