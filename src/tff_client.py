"""TFF Fantezi Lig API istemcisi (tfffantezilig.com).

/istatistikler ve oyuncu listesi giriş ister.
Backend proxy: /api/backend/{path}  (credentials: cookie)

Bilinçli path'ler (401/403 yanıtlarıyla doğrulandı):
  players, stats, teams, users/me, leagues, leagues, elements, ...
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import urllib3

from .config import is_quiet

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

TFF_BASE = os.environ.get("TFF_BASE", "https://tfffantezilig.com").rstrip("/")
KEYCLOAK_TOKEN_URL = os.environ.get(
    "TFF_KEYCLOAK_TOKEN_URL",
    "https://auth.tfffantezilig.com/realms/tff/protocol/openid-connect/token",
)
KEYCLOAK_CLIENT_ID = os.environ.get("TFF_KEYCLOAK_CLIENT_ID", "web-client")
VERIFY_SSL = os.environ.get("FBREF_SSL_VERIFY", "0").strip().lower() not in (
    "0",
    "false",
    "no",
)

AUTH_COOKIE_NAMES = (
    "kc_access_token",
    "kc_refresh_token",
    "auth_session_user",
    "AUTH_SESSION_ID",
    "KEYCLOAK_IDENTITY",
    "KEYCLOAK_SESSION",
    "KEYCLOAK_REMEMBER_ME",
)
TRACKING_COOKIES = {
    "_ga",
    "_gid",
    "_gat",
    "_fbp",
    "_ttp",
    "_clck",
    "_clsk",
    "AMP_TOKEN",
}

POS_NUM = {1: "GK", 2: "DF", 3: "MF", 4: "FW"}
POS_ALIASES = {
    "G": "GK",
    "GK": "GK",
    "KALECİ": "GK",
    "KALECI": "GK",
    "D": "DF",
    "DF": "DF",
    "DEF": "DF",
    "DEF": "DF",
    "DEFANS": "DF",
    "M": "MF",
    "MF": "MF",
    "MID": "MF",
    "MID": "MF",
    "OS": "MF",
    "ORTA": "MF",
    "F": "FW",
    "FW": "FW",
    "FWD": "FW",
    "ST": "FW",
    "FWD": "FW",
    "FORVET": "FW",
    "A": "FW",
}


class TFFAuthError(RuntimeError):
    """401/403 — oturum yok veya cookie eksik."""


class TFFHttpError(RuntimeError):
    def __init__(self, path: str, status: int, body: str):
        self.path = path
        self.status = status
        super().__init__(f"HTTP {status} {path}: {body[:180]}")


def _cookie_pairs_from_dump(raw: str) -> dict[str, str]:
    """Chrome Cookie başlığı VEYA Application → Cookies tablo yapıştırmasını parse et."""
    text = raw.strip().strip('"').strip("'")
    if text.lower().startswith("cookie:"):
        text = text.split(":", 1)[1].strip()

    pairs: dict[str, str] = {}

    # 1) Klasik başlık: a=b; c=d
    if "\n" not in text and "\t" not in text and ";" in text:
        for part in text.split(";"):
            part = part.strip()
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            pairs[k.strip()] = v.strip()
        return pairs

    # 2) Application paneli: satır = name <tab> value <tab> domain ...
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("name") and "value" in line.lower():
            continue
        cols = re.split(r"\t+", line)
        if len(cols) >= 2 and re.match(r"^[\w.-]+$", cols[0]):
            name, value = cols[0], cols[1]
            if name.lower() in {"medium", "session", "lax", "strict"}:
                continue
            pairs[name] = value
            continue
        # name=value satırı
        if "=" in line and "://" not in line.split("=", 1)[0]:
            k, v = line.split("=", 1)
            k = k.strip()
            if re.match(r"^[\w.-]+$", k):
                pairs[k] = v.strip()

    # 3) JWT'ler satır içinde gömülüyse
    for name in ("kc_access_token", "kc_refresh_token"):
        if name in pairs:
            continue
        m = re.search(rf"{re.escape(name)}\s+([A-Za-z0-9._\-]+)", text)
        if m:
            pairs[name] = m.group(1)

    return pairs


def _clean_cookie(raw: str | None) -> str | None:
    if not raw:
        return None
    pairs = _cookie_pairs_from_dump(raw)
    if not pairs:
        s = " ".join(raw.strip().split())
        return s or None
    # Analitik çerezleri at, oturum olanları tut
    keep = []
    for k, v in pairs.items():
        if k in TRACKING_COOKIES or k.startswith("_ga"):
            continue
        keep.append(f"{k}={v}")
    if not keep:
        # hiç oturum yoksa ham bırak (analytics_only yakalasın)
        keep = [f"{k}={v}" for k, v in pairs.items()]
    return "; ".join(keep)


def _analytics_only(cookie: str) -> bool:
    parts = [p.split("=", 1)[0].strip() for p in cookie.split(";") if p.strip()]
    if not parts:
        return True
    real = [
        p
        for p in parts
        if p not in TRACKING_COOKIES and not p.startswith("_ga_") and p != "_ga"
    ]
    return len(real) == 0


def _token_from_cookie_header(cookie: str, name: str) -> str | None:
    for part in cookie.split(";"):
        part = part.strip()
        if part.startswith(name + "="):
            return part.split("=", 1)[1].strip()
    return None


def refresh_access_token(refresh_token: str) -> dict[str, str]:
    """Keycloak public client refresh (client_id=web-client)."""
    r = requests.post(
        KEYCLOAK_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "client_id": KEYCLOAK_CLIENT_ID,
            "refresh_token": refresh_token,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
        verify=VERIFY_SSL,
    )
    if r.status_code >= 400:
        raise TFFAuthError(
            f"Keycloak token yenileme başarısız (HTTP {r.status_code}). "
            "Siteye tarayıcıda tekrar giriş yapıp kc_refresh_token'ı taze kopyala."
        )
    data = r.json()
    access = data.get("access_token")
    if not access:
        raise TFFAuthError("Keycloak yanıtında access_token yok.")
    return {
        "access_token": access,
        "refresh_token": data.get("refresh_token") or refresh_token,
    }


def persist_tokens(path: Path, access: str, refresh: str, cookie_header: str | None = None) -> None:
    user = _token_from_cookie_header(cookie_header, "auth_session_user") if cookie_header else None
    lines = [f"kc_access_token={access}", f"kc_refresh_token={refresh}"]
    if user:
        lines.append(f"auth_session_user={user}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_saved_login() -> tuple[str | None, str | None]:
    from .config import LOGIN_FILE

    path = Path(os.environ.get("TFF_LOGIN_FILE") or LOGIN_FILE)
    if not path.exists():
        return None, None
    email = password = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip().lower(), v.strip()
        if k in ("email", "user", "username"):
            email = v
        elif k in ("password", "pass", "sifre"):
            password = v
    return email, password


def login_with_password(email: str, password: str) -> dict[str, str]:
    """Keycloak password grant → kc_access_token + kc_refresh_token dosyaya yazılır."""
    r = requests.post(
        KEYCLOAK_TOKEN_URL,
        data={
            "grant_type": "password",
            "client_id": KEYCLOAK_CLIENT_ID,
            "username": email.strip(),
            "password": password,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
        verify=VERIFY_SSL,
    )
    if r.status_code >= 400:
        raise TFFAuthError(
            f"TFF e-posta/şifre girişi başarısız (HTTP {r.status_code})."
        )
    data = r.json()
    access = data.get("access_token")
    refresh = data.get("refresh_token")
    if not access or not refresh:
        raise TFFAuthError("Keycloak yanıtında token yok.")
    persist_tokens(_default_cookie_file(), access, refresh)
    return {"access_token": access, "refresh_token": refresh}


def _default_cookie_file() -> Path:
    from .config import COOKIE_FILE

    env = os.environ.get("TFF_COOKIE_FILE")
    return Path(env) if env else COOKIE_FILE


def _session(cookie: str | None = None) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Referer": f"{TFF_BASE}/istatistikler",
            "Origin": TFF_BASE,
        }
    )
    raw = cookie or os.environ.get("TFF_COOKIE")
    cookie_file = _default_cookie_file()
    if not raw and cookie_file.exists():
        raw = cookie_file.read_text(encoding="utf-8")

    cookie_header = _clean_cookie(raw)
    bearer = os.environ.get("TFF_ACCESS_TOKEN") or os.environ.get("TFF_BEARER")

    if cookie_header:
        if _analytics_only(cookie_header):
            raise RuntimeError(
                "TFF_COOKIE sadece _ga / _gid gibi analitik çerezleri içeriyor. "
                "Gerekli olan: kc_access_token ve kc_refresh_token."
            )
        s.headers["Cookie"] = cookie_header
        if not bearer:
            bearer = _token_from_cookie_header(cookie_header, "kc_access_token")
        refresh = _token_from_cookie_header(cookie_header, "kc_refresh_token")
        if refresh:
            try:
                tokens = refresh_access_token(refresh)
                bearer = tokens["access_token"]
                s.headers["Cookie"] = re.sub(
                    r"kc_access_token=[^;]*",
                    "kc_access_token=" + tokens["access_token"],
                    cookie_header,
                )
                if "kc_access_token=" not in s.headers["Cookie"]:
                    s.headers["Cookie"] += "; kc_access_token=" + tokens["access_token"]
                s.headers["Cookie"] = re.sub(
                    r"kc_refresh_token=[^;]*",
                    "kc_refresh_token=" + tokens["refresh_token"],
                    s.headers["Cookie"],
                )
                persist_tokens(
                    cookie_file,
                    tokens["access_token"],
                    tokens["refresh_token"],
                    s.headers["Cookie"],
                )
            except TFFAuthError:
                pass

    if bearer:
        s.headers["Authorization"] = f"Bearer {bearer}"
    return s


def login(email: str | None = None, password: str | None = None) -> requests.Session:
    """POST /api/auth/login → session cookie'li Session."""
    email = email or os.environ.get("TFF_EMAIL") or os.environ.get("TFF_USER")
    password = password or os.environ.get("TFF_PASSWORD") or os.environ.get("TFF_PASS")
    if not email or not password:
        raise RuntimeError(
            "TFF girişi için TFF_EMAIL + TFF_PASSWORD veya TFF_COOKIE gerekli. "
            "Site: https://tfffantezilig.com (istatistikler giriş ister)."
        )
    s = _session()
    r = s.post(
        f"{TFF_BASE}/api/auth/login",
        json={"email": email.strip(), "password": password},
        timeout=30,
        verify=VERIFY_SSL,
    )
    try:
        body = r.json()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"TFF login JSON okunamadı ({r.status_code}): {exc}") from exc
    if r.status_code != 200 or not body.get("ok"):
        raise RuntimeError(
            f"TFF login başarısız (HTTP {r.status_code}). "
            f"Yanıt: {str(body)[:200]}"
        )
    return s


def backend_get(
    path: str,
    *,
    session: requests.Session | None = None,
    cookie: str | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 45.0,
) -> Any:
    """GET /api/backend/{path}"""
    path = path.lstrip("/")
    s = session or _session(cookie)
    url = f"{TFF_BASE}/api/backend/{path}"
    r = s.get(url, params=params, timeout=timeout, verify=VERIFY_SSL)
    if r.status_code in (401, 403):
        raise TFFAuthError(
            f"TFF oturumu geçersiz (HTTP {r.status_code}) path={path}. "
            "_ga çerezi yetmez. Siteye giriş yap → F12 Network → "
            "`/api/backend/` içeren isteğin Cookie başlığının tamamını kopyala "
            "veya TFF_EMAIL + TFF_PASSWORD kullan."
        )
    if r.status_code >= 400:
        raise TFFHttpError(path, r.status_code, r.text)
    try:
        return r.json()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"JSON değil ({path}): {r.text[:200]}") from exc


def _candidate_paths(league_id: str) -> list[str]:
    q = f"league-id={league_id}"
    return [
        f"projection/stats/player-stats?{q}",
        f"players?{q}",
        f"projection/players?{q}",
        f"projection/stats/players?{q}",
        f"stats?{q}",
        "players",
        "stats",
    ]


def parse_tff_official_players(data: Any, clubs: dict[int, str] | None = None) -> pd.DataFrame:
    """projection/stats/player-stats yanıtı."""
    clubs = clubs or {}
    items = _unwrap_list(data)
    if not items and isinstance(data, dict):
        items = data.get("data") if isinstance(data.get("data"), list) else []
    rows = []
    for p in items:
        if not isinstance(p, dict):
            continue
        first = str(p.get("name") or "").strip()
        last = str(p.get("surname") or "").strip()
        name = f"{first} {last}".strip() or str(p.get("matchName") or p.get("shortName") or "").strip()
        pos = _norm_pos(p.get("position"))
        price = _norm_price(p.get("cost") if p.get("cost") is not None else p.get("price"))
        club_id = p.get("clubId")
        team = ""
        if club_id is not None:
            try:
                team = clubs.get(int(club_id), "") or str(club_id)
            except (TypeError, ValueError):
                team = str(club_id)
        if not name or price is None or not pos:
            continue
        if p.get("pickable") is False or p.get("active") is False:
            continue
        match_name = str(p.get("matchName") or "").strip()
        short_name = str(p.get("shortName") or first).strip()
        if match_name and short_name and short_name.lower() != match_name.lower():
            if short_name.lower() in match_name.lower():
                display = match_name
            else:
                display = f"{short_name} {match_name}".strip()
        elif match_name and first and first.lower() not in match_name.lower():
            display = f"{first} {match_name}".strip()
        else:
            display = name
        if len(display.split()) < 2:
            display = name
        rows.append(
            {
                "player_name": name,
                "display_name": display,
                "match_name": match_name,
                "search_name": display,
                "team": team,
                "position": pos,
                "price_m": float(price),
                "availability": str(p.get("availabilityStatus") or ""),
                "avail_pct": p.get("availabilityPercent"),
                "avail_news": str(p.get("availabilityNews") or ""),
                "tff_form": p.get("form") or 0,
                "selected_by": p.get("selectedByPct") or 0,
                "tff_xg": p.get("xGTotal") or 0,
                "tff_xa": p.get("xATotal") or 0,
                "tff_points": p.get("totalPoints") or 0,
                "tff_ppm": p.get("pointsPerMatch") or 0,
                "tff_minutes": p.get("minutes") or 0,
                "tff_starts": p.get("starts") or 0,
                "tff_goals": p.get("goals") or 0,
                "tff_assists": p.get("assists") or 0,
                "tff_bonus": p.get("bonus") or 0,
                "tff_bps": p.get("bps") or 0,
            }
        )
    if not rows:
        raise ValueError("TFF player-stats listesi bos veya parse edilemedi.")
    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["player_name", "team"], keep="first")
        .reset_index(drop=True)
    )


def fetch_club_map(session: requests.Session, league_id: str) -> dict[int, str]:
    try:
        data = backend_get(
            f"projection/stats/club-stats?league-id={league_id}",
            session=session,
        )
    except Exception:
        return {}
    items = _unwrap_list(data)
    if not items and isinstance(data, dict) and isinstance(data.get("data"), list):
        items = data["data"]
    out: dict[int, str] = {}
    for c in items:
        if not isinstance(c, dict) or c.get("id") is None:
            continue
        try:
            out[int(c["id"])] = str(c.get("name") or c.get("shortName") or "")
        except (TypeError, ValueError):
            continue
    return out


def _unwrap_list(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    # { status: true, data: [...] } veya nested
    for key in (
        "players",
        "data",
        "items",
        "elements",
        "results",
        "list",
        "rows",
        "content",
    ):
        val = data.get(key)
        if isinstance(val, list):
            return val
        if isinstance(val, dict):
            nested = _unwrap_list(val)
            if nested:
                return nested
    return []


def _deep_find_list_of_dicts(data: Any, min_len: int = 20) -> list[dict]:
    """JSON içinde oyuncu listesi gibi görünen ilk uzun dict listesini bul."""
    if isinstance(data, list) and len(data) >= min_len and all(
        isinstance(x, dict) for x in data[:5]
    ):
        return data
    if isinstance(data, dict):
        for v in data.values():
            found = _deep_find_list_of_dicts(v, min_len=min_len)
            if found:
                return found
    if isinstance(data, list):
        for v in data:
            found = _deep_find_list_of_dicts(v, min_len=min_len)
            if found:
                return found
    return []


def _pick(d: dict, *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] is not None and d[k] != "":
            return d[k]
    # case-insensitive
    lower = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        if k.lower() in lower and lower[k.lower()] not in (None, ""):
            return lower[k.lower()]
    return None


def _norm_pos(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return POS_NUM.get(int(raw))
    s = str(raw).strip().upper()
    if s.isdigit():
        return POS_NUM.get(int(s))
    if s in POS_ALIASES:
        return POS_ALIASES[s]
    # "Goalkeeper" vb.
    if "KEEP" in s or "KALE" in s:
        return "GK"
    if "DEF" in s or "BACK" in s:
        return "DF"
    if "MID" in s or "ORTA" in s:
        return "MF"
    if "FORW" in s or "STRIK" in s or "ATT" in s or "FORV" in s:
        return "FW"
    return POS_ALIASES.get(s[:2]) if len(s) >= 2 else None


def _norm_price(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        v = float(raw)
        # FPL gibi 55 (=5.5) ölçeği
        if v > 30:
            v = v / 10.0
        return v
    s = str(raw).strip().replace(",", ".")
    s = re.sub(r"[^\d.]", "", s)
    if not s:
        return None
    v = float(s)
    if v > 30:
        v = v / 10.0
    return v


def _player_name(p: dict) -> str | None:
    name = _pick(
        p,
        "name",
        "player_name",
        "playerName",
        "fullName",
        "full_name",
        "web_name",
        "webName",
        "displayName",
        "display_name",
    )
    if name:
        return str(name).strip()
    first = _pick(p, "firstName", "first_name", "name")
    last = _pick(p, "lastName", "last_name", "surname")
    if first or last:
        return f"{first or ''} {last or ''}".strip()
    nested = p.get("player")
    if isinstance(nested, dict):
        return _player_name(nested)
    return None


def _player_team(p: dict) -> str:
    team = _pick(p, "team", "teamName", "team_name", "squad", "club", "clubName")
    if isinstance(team, dict):
        team = _pick(team, "name", "shortName", "short_name") or ""
    if team:
        return str(team).strip()
    nested = p.get("player")
    if isinstance(nested, dict):
        return _player_team(nested)
    return ""


def parse_players_payload(data: Any) -> pd.DataFrame:
    """Oyuncu/fiyat JSON → DataFrame."""
    items = _unwrap_list(data)
    if len(items) < 5:
        items = _deep_find_list_of_dicts(data, min_len=5) or items

    rows = []
    for p in items:
        if not isinstance(p, dict):
            continue
        # nested player object
        src = p
        if "player" in p and isinstance(p["player"], dict):
            # merge top-level price onto nested if needed
            merged = dict(p["player"])
            for k, v in p.items():
                if k != "player" and k not in merged:
                    merged[k] = v
            src = merged

        name = _player_name(src)
        price = _norm_price(
            _pick(
                src,
                "price",
                "price_m",
                "now_cost",
                "nowCost",
                "value",
                "cost",
                "currentPrice",
                "current_price",
                "sellingPrice",
                "purchasePrice",
            )
        )
        pos = _norm_pos(
            _pick(
                src,
                "position",
                "pos",
                "element_type",
                "elementType",
                "positionId",
                "position_id",
                "type",
            )
        )
        team = _player_team(src)
        if not name or price is None:
            continue
        if not pos:
            continue
        rows.append(
            {
                "player_name": name,
                "team": team,
                "position": pos,
                "price_m": float(price),
            }
        )

    if not rows:
        # debug yardım
        sample = str(data)[:400]
        raise ValueError(
            "TFF yanıtından oyuncu/fiyat okunamadı. "
            f"Örnek gövde: {sample}"
        )
    df = pd.DataFrame(rows)
    # dedupe by name+team keep max price
    df = (
        df.sort_values("price_m", ascending=False)
        .drop_duplicates(subset=["player_name", "team"], keep="first")
        .reset_index(drop=True)
    )
    return df


def fetch_tff_prices(
    url: str | None = None,
    cookie: str | None = None,
    email: str | None = None,
    password: str | None = None,
    timeout: float = 45.0,
    raw_dump: str | Path | None = None,
) -> pd.DataFrame:
    """
    Fiyat listesini çeker.

    Sıra:
      1) TFF_PLAYERS_URL veya url parametresi (tam URL)
      2) Login (email/pass) veya Cookie ile /api/backend/players
      3) Fallback: /api/backend/stats, /api/backend/elements
    """
    session: requests.Session | None = None
    if not email or not password:
        saved_e, saved_p = load_saved_login()
        email = email or saved_e or os.environ.get("TFF_EMAIL")
        password = password or saved_p or os.environ.get("TFF_PASSWORD")
    if email and password:
        try:
            login_with_password(email, password)
        except TFFAuthError:
            session = login(email, password)

    endpoint = url or os.environ.get("TFF_PLAYERS_URL") or ""
    attempts: list[str] = []
    payloads: list[tuple[str, Any]] = []
    last_err: Exception | None = None
    league_id = os.environ.get("TFF_LEAGUE_ID", "1").strip() or "1"

    if endpoint:
        s = session or _session(cookie)
        r = s.get(endpoint, timeout=timeout, verify=VERIFY_SSL)
        attempts.append(f"{endpoint} → {r.status_code}")
        if r.status_code in (401, 403):
            raise TFFAuthError(f"TFF URL yetkisiz ({r.status_code}): {endpoint}")
        r.raise_for_status()
        payloads.append(("custom", r.json()))
    else:
        s = session or _session(cookie)
        for path in _candidate_paths(league_id):
            try:
                data = backend_get(path, session=s, timeout=timeout)
                attempts.append(f"{path} → 200")
                payloads.append((path, data))
                break  # ilk başarılı JSON yeterli
            except TFFAuthError:
                raise
            except TFFHttpError as exc:
                attempts.append(f"{path} → {exc.status}")
                last_err = exc
                continue
            except Exception as exc:  # noqa: BLE001
                attempts.append(f"{path} → {exc}")
                last_err = exc
                continue

    if not payloads:
        detail = "\n  ".join(attempts) or "(istek yok)"
        raise RuntimeError(
            "TFF oyuncu listesi alınamadı.\n"
            f"  Denenen path'ler:\n  {detail}\n"
            "Cookie dosyasını yenile (data/tff_cookies.txt) veya siteye tekrar giriş yap."
        ) from last_err

    if raw_dump:
        dump_path = Path(raw_dump)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(
            json.dumps({k: v for k, v in payloads}, ensure_ascii=False, indent=2)[:2_000_000],
            encoding="utf-8",
        )

    s = session or _session(cookie)
    clubs = fetch_club_map(s, league_id)
    for name, data in payloads:
        try:
            df = parse_tff_official_players(data, clubs)
            if not is_quiet():
                print(f"  TFF kaynak: {name} ({len(df)} oyuncu)")
            return df
        except ValueError:
            pass
        try:
            df = parse_players_payload(data)
            if not is_quiet():
                print(f"  TFF kaynak (genel parse): {name} ({len(df)} oyuncu)")
            return df
        except ValueError:
            continue
    raise ValueError(
        "TFF API yanıtı alındı ama fiyat tablosuna çevrilemedi. "
        f"Path: {[n for n, _ in payloads]}."
    )


def save_prices_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.rename(
        columns={
            "player": "player_name",
            "name": "player_name",
        }
    )
    cols = [
        c
        for c in (
            "player_name",
            "display_name",
            "match_name",
            "search_name",
            "team",
            "position",
            "price_m",
            "availability",
            "avail_pct",
            "avail_news",
            "tff_form",
            "selected_by",
            "tff_xg",
            "tff_xa",
            "tff_points",
            "tff_ppm",
            "tff_minutes",
            "tff_starts",
            "tff_goals",
            "tff_assists",
            "tff_bonus",
            "tff_bps",
        )
        if c in out.columns
    ]
    out[cols].to_csv(path, index=False)
