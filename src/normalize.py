"""Stage 0: multi-view normalization.

We keep several views of each field instead of one aggressively cleaned string,
because over-normalizing destroys evidence (e.g. a city token that separates two
branches). Seed dictionaries below are deliberately small and country-agnostic in
spirit; the plan is to extend them with substitutions MINED from ground-truth
positives (see analysis/forensics.py), not by hand-coding per country.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List

import pandas as pd

# canonical form <- variants. Applied token-wise after basic cleaning.
LEGAL_CANON: Dict[str, str] = {
    "pvt": "private", "pvt.": "private", "prv": "private", "private": "private",
    "ltd": "limited", "limited": "limited", "ltda": "limited",
    "inc": "incorporated", "incorporated": "incorporated",
    "corp": "corporation", "corporation": "corporation",
    "co": "company", "company": "company", "cos": "company",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "opc": "opc",
    "pllc": "pllc", "pc": "pc",
    # common EU forms -- present only so they are split off, not country logic
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl",
    "sci": "sci", "snc": "snc", "gmbh": "gmbh", "ag": "ag", "bv": "bv",
}
# multi-token legal phrases collapsed before tokenizing
LEGAL_PHRASES = [
    (re.compile(pat), rep) for pat, rep in [
        (r"\bl\s*l\s*c\b", "llc"), (r"\bl\s*l\s*p\b", "llp"),
        (r"\bs\s*a\s*r\s*l\b", "sarl"), (r"\bs\s*a\s*s\b", "sas"),
        (r"\bprivate\s+limited\b", "private limited"),
    ]
]
_LEGAL_VALUES = frozenset(LEGAL_CANON.values())
_STOP = frozenset({"the", "and", "of"})

ADDR_CANON: Dict[str, str] = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "ln": "lane", "dr": "drive",
    "hwy": "highway", "pkwy": "parkway", "ct": "court", "pl": "place",
    "sq": "square", "ste": "suite", "apt": "apartment", "fl": "floor",
    "flr": "floor", "bldg": "building", "blk": "block", "sec": "sector",
    "ph": "phase", "nr": "near", "opp": "opposite", "mkt": "market",
    "ngr": "nagar", "clny": "colony", "no": "number", "num": "number",
    "n": "north", "s": "south", "e": "east", "w": "west",
}
LANDMARK_WORDS = {"near", "opposite", "behind", "beside", "next", "adjacent", "facing"}

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")
_POSTAL = re.compile(r"(?<!\d)\d{4,6}(?!\d)")


def fold(s: str) -> str:
    """Unicode NFKD, strip accents, lowercase, '&' -> 'and', punctuation -> space."""
    s = s or ""
    if s.isascii():  # NFKD + combining-strip are no-ops on ASCII; skip them
        s = s.lower()
    else:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = s.replace("&", " and ").replace("'", "")  # O'Brien -> obrien
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _canon_tokens(tokens: List[str], table: Dict[str, str]) -> List[str]:
    return [table.get(t, t) for t in tokens]


def name_views(raw: str) -> Dict[str, str]:
    base = fold(raw)
    for pat, rep in LEGAL_PHRASES:
        base = pat.sub(rep, base)
    toks = _canon_tokens(base.split(), LEGAL_CANON)
    legal = [t for t in toks if t in _LEGAL_VALUES]
    core = [t for t in toks if t not in _LEGAL_VALUES]
    if not core:  # name was only legal words -- keep something
        core = toks
    core_nostop = [t for t in core if t not in _STOP] or core
    return {
        "name_norm": " ".join(toks),
        "name_core": " ".join(core),
        "name_legal": " ".join(sorted(set(legal))),
        "name_sorted": " ".join(sorted(set(core_nostop))),
        "name_acronym": "".join(t[0] for t in core_nostop if t),
        "name_compact": "".join(core),  # 'mc donalds' == 'mcdonalds'
    }


def addr_views(raw: str) -> Dict[str, str]:
    base = fold(raw)
    toks = _canon_tokens(base.split(), ADDR_CANON)
    digits = _DIGITS.findall(base)
    postals = _POSTAL.findall(base)
    return {
        "addr_norm": " ".join(toks),
        "addr_sorted": " ".join(sorted(set(toks))),
        "addr_digits": " ".join(digits),
        "addr_postal": postals[-1] if postals else "",
        "addr_first_num": digits[0] if digits else "",
        "addr_landmark": int(any(t in LANDMARK_WORDS for t in toks)),
        "addr_ntok": len(toks),
    }


NAME_VIEW_COLS = ["name_norm", "name_core", "name_legal", "name_sorted", "name_acronym", "name_compact"]
ADDR_VIEW_COLS = ["addr_norm", "addr_sorted", "addr_digits", "addr_postal", "addr_first_num",
                  "addr_landmark", "addr_ntok"]


def _columnar(views: Iterable[Dict[str, object]], cols: List[str]) -> Dict[str, list]:
    """Collect per-row view dicts straight into per-column lists (never a list of dicts)."""
    out: Dict[str, list] = {c: [] for c in cols}
    appends = [(c, out[c].append) for c in cols]
    for v in views:
        for c, app in appends:
            app(v[c])
    return out


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add all views to a source frame (entity_id, business_name, business_address, country).

    Output is several times the size of the input: call it on ONE CHUNK at a time
    (see io_utils.iter_source), never on a whole multi-million-row source.
    """
    out = df.copy()
    for c, vals in _columnar((name_views(x) for x in df["business_name"]), NAME_VIEW_COLS).items():
        out[c] = vals
    for c, vals in _columnar((addr_views(x) for x in df["business_address"]), ADDR_VIEW_COLS).items():
        out[c] = vals
    folded = {c: fold(c) for c in out["country"].unique()}  # a handful of countries
    out["country_norm"] = out["country"].map(folded)
    return out
