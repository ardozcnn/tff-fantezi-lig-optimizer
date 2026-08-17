# AGENTS.md

## Cursor Cloud specific instructions

This repo is a single **Python 3.10+ CLI** app (no web server, DB, or Docker): a TFF Fantezi Lig
(Turkish Fantasy Football) squad recommender. Dependencies come from `requirements.txt` and are
installed by the startup update script (`pip install -r requirements.txt`). Standard run/test
commands live in `README.md` and `calistir.bat`.

### Services / commands
- Tests (offline, no network): `python3 -m unittest discover -s tests -v` (no live APIs required for the suite).
- Run the app: `python3 -m src.main` (flags documented in `README.md`).
- No linter/formatter and no CI are configured — there is nothing to run for lint.

### Non-obvious gotchas
- **Sofascore API is blocked from cloud VMs.** The stats fetch hits `www.sofascore.com/api/v1/...`
  which returns `403 {"reason":"challenge"}` from datacenter/cloud IPs even with `curl_cffi` browser
  impersonation (the site homepage returns 200, but the API is challenged). Because the pipeline
  requires Süper Lig stats, a full live squad run (`python3 -m src.main`) fails with
  `HATA: Oyuncu istatistiği boş.` here. This is an external network block, not a setup problem.
  A residential IP / proxy is needed for true end-to-end live runs.
- **Offline prices for local runs:** `cp data/prices.example.csv data/prices.csv` then use
  `python3 -m src.main --no-fetch-prices`. This still needs Sofascore for stats, so it only gets
  past price-fetching, not the stats step.
- To exercise the core ILP optimizer (`src/optimize.py`) + report (`src/report.py`) without network,
  feed `optimize_squad` a DataFrame with columns `player, display_name, team, position, price_m,
  projected_pts`. It enforces TFF rules (100M budget, 2 GK / 5 DF / 5 MF / 3 FW, max 3 per club) and
  auto-picks formation + XI + bench + captain via PuLP/CBC (CBC ships with PuLP).
- `calibrate_leagues.py` (offline maintenance) imports `numpy`, which is **not** pinned in
  `requirements.txt` (usually present transitively via pandas). Not needed for normal squad runs —
  the calibrated model `data/league_translation.json` is already committed.
- TFF live prices need credentials (`data/tff_login.txt` or `TFF_EMAIL`/`TFF_PASSWORD`); the TFF/
  Keycloak endpoints may also be geo/IP-restricted from cloud VMs.
