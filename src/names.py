"""Türkçe isim normalizasyonu ve fuzzy eşleştirme."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz, process

_TR_MAP = str.maketrans(
    {
        "ç": "c",
        "Ç": "c",
        "ğ": "g",
        "Ğ": "g",
        "ı": "i",
        "İ": "i",
        "I": "i",
        "ö": "o",
        "Ö": "o",
        "ş": "s",
        "Ş": "s",
        "ü": "u",
        "Ü": "u",
    }
)

COMMON_FIRST = {
    "mohamed",
    "mohammed",
    "muhammad",
    "muhammed",
    "ahmed",
    "ahmad",
    "ali",
    "mustafa",
    "yusuf",
    "emir",
    "john",
    "david",
    "lucas",
    "gabriel",
    "jose",
    "antonio",
    "marco",
    "daniel",
    "anderson",
    "junior",
}


def _tokens(name: str) -> list[str]:
    return [t for t in normalize_name(name).split() if t]


def _plausible_match(query: str, candidate: str, score: float) -> bool:
    """Uzun resmi adın yanlış 'Mohamed' eşleşmesini ele."""
    if score >= 96:
        return True
    qt = _tokens(query)
    ct = _tokens(candidate)
    if not qt or not ct:
        return False
    q_set, c_set = set(qt), set(ct)
    if q_set & c_set and (qt[-1] == ct[-1] or qt[-1] in c_set or ct[-1] in q_set):
        return True
    distinctive = (q_set - COMMON_FIRST) & (c_set - COMMON_FIRST)
    if distinctive:
        return True
    if fuzz.ratio(qt[-1], ct[-1]) >= 86:
        return True
    return False
KNOWN_ALIASES = {
    "mohamed salah hamed mahrous ghaly": "mohamed salah",
    "leandro trossard": "leandro trossard",
    "marco asensio willemsen": "marco asensio",
    "mason will john greenwood": "mason greenwood",
    "leroy aziz sane": "leroy sane",
    "victor james osimhen": "victor osimhen",
    "anderson souza conceicao": "talisca",
    "muhammed kerem akturkoglu": "kerem akturkoglu",
    "ederson santana de moraes": "ederson",
    "n golo kante": "ngolo kante",
    "ngolo kante": "ngolo kante",
    "frederico rodrigues de paula santos": "fred",
    "matteo elias kenzo guendouzi olie": "matteo guendouzi",
    "ebere paul onuachu": "paul onuachu",
    "andre onana onana": "andre onana",
    "nathan benjamin ake": "nathan ake",
    "nelson cabral semedo": "nelson semedo",
    "onyinye wilfred ndidi": "wilfred ndidi",
    "badobre emmanuel elysee djedje agbadou": "emmanuel agbadou",
    "lucas sebastian torreira di pascua": "lucas torreira",
    "gabriel davi gomes sara": "gabriel sara",
    "davinson sanchez mina": "davinson sanchez",
    "mario rene junior lemina": "mario lemina",
}


def normalize_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    s = name.translate(_TR_MAP)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def name_variants(name: str) -> list[str]:
    """Uzun resmi addan arama / eşleşme varyantları."""
    n = normalize_name(name)
    if not n:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def add(x: str) -> None:
        x = x.strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)

    add(n)
    alias = KNOWN_ALIASES.get(n)
    if alias:
        add(alias)

    tokens = n.split()
    if len(tokens) >= 2:
        add(f"{tokens[0]} {tokens[-1]}")
        if len(tokens) >= 3:
            add(f"{tokens[0]} {tokens[1]}")
            add(f"{tokens[0]} {tokens[1]} {tokens[-1]}")
            add(" ".join(tokens[:2]))
        if len(tokens) >= 4:
            add(" ".join(tokens[:3]))
    return out


def display_from_parts(
    full_name: str,
    *,
    match_name: str | None = None,
    short_name: str | None = None,
) -> str:
    """UI'da kısa okunur ad: 'Mohamed Salah'."""
    mn = (match_name or "").strip()
    sn = (short_name or "").strip()
    if sn and mn and sn.lower() not in mn.lower():
        return f"{sn} {mn}".strip()
    if mn and len(mn.split()) >= 1 and len(full_name.split()) > 3:
        if sn:
            return f"{sn} {mn}".strip()
        first = full_name.split()[0]
        return f"{first} {mn}".strip()
    if mn and len(full_name.split()) <= 3:
        return full_name
    return full_name.strip()


def _score_pair(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 100.0
    set_s = fuzz.token_set_ratio(a, b)
    sort_s = fuzz.token_sort_ratio(a, b)
    partial = fuzz.partial_ratio(a, b)
    return float(max(set_s, sort_s, partial * 0.92))


def best_match(
    query: str,
    candidates: list[str],
    *,
    score_cutoff: int = 78,
    extra_queries: list[str] | None = None,
) -> tuple[str | None, float]:
    if not query or not candidates:
        return None, 0.0

    queries = name_variants(query)
    for extra in extra_queries or []:
        for v in name_variants(extra):
            if v not in queries:
                queries.append(v)
    if not queries:
        return None, 0.0

    mapping = {c: normalize_name(c) for c in candidates}
    inv: dict[str, str] = {}
    for orig, norm in mapping.items():
        inv.setdefault(norm, orig)

    best_name: str | None = None
    best_score = 0.0

    norms = list(inv.keys())
    for q in queries:
        for scorer in (fuzz.token_set_ratio, fuzz.token_sort_ratio):
            result = process.extractOne(q, norms, scorer=scorer, score_cutoff=min(score_cutoff, 70))
            if result:
                matched_norm, score, _ = result
                if score > best_score:
                    best_score = float(score)
                    best_name = inv.get(matched_norm)

    if best_score < score_cutoff:
        return None, best_score
    if best_name and not _plausible_match(query, best_name, best_score):
        return None, best_score
    return best_name, best_score
