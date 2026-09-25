"""Country-agnostic text normalization for names and addresses.

Every rule here is derived from noise patterns measured in the Phase-2 audit
and none depends on a specific country label.
"""
from __future__ import annotations

import re
import unicodedata
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .config import ADDRESS_CANON, FILLER_TOKENS, LEGAL_TOKENS, N_JOBS

_DOTTED_ACRONYM = re.compile(r"\b(?:\w\.){2,}\w?\.?")  # l.l.c. / p.c. / s.a.r.l.
_APOS = re.compile(r"[\'’‘`´]")
_NON_WORD = re.compile(r"[^\w\u0900-\u0dff]+")  # keep Indic vowel signs/viramas
_INDIC = re.compile(r"[\u0900-\u0dff]")
_DIGITS = re.compile(r"\d+")
_WEB_SUFFIX = {"com", "net", "org", "www", "in", "co", "biz", "io"}


def _base(s: str) -> str:
    """NFKC, strip accents, lowercase, collapse acronyms/apostrophes, keep \\w."""
    s = unicodedata.normalize("NFKC", s)
    # Strip Latin diacritics only (U+0300-036F); Indic vowel signs/viramas stay.
    s = "".join(c for c in unicodedata.normalize("NFD", s) if not "\u0300" <= c <= "\u036f")
    s = unicodedata.normalize("NFC", s).lower()
    s = _DOTTED_ACRONYM.sub(lambda m: m.group(0).replace(".", ""), s)
    s = _APOS.sub("", s)
    s = s.replace("&", " and ")
    return _NON_WORD.sub(" ", s).replace("_", " ").strip()


def _strip_zeros(tok: str) -> str:
    return (tok.lstrip("0") or "0") if tok.isdigit() else tok


def normalize_name(s: str) -> tuple[str, str, str]:
    """Return (full, core, concat).

    full   : normalized tokens
    core   : full minus legal/filler tokens (falls back to full if empty)
    concat : core tokens joined without spaces (matches domains/handles like
             'willissfreshsigns.com' or '@womenshealth')
    """
    toks = _base(s).split()
    core = [t for t in toks if t not in LEGAL_TOKENS and t not in FILLER_TOKENS]
    if not core:
        core = toks
    web = [t for t in core if t not in _WEB_SUFFIX] or core
    return " ".join(toks), " ".join(core), "".join(web)


def normalize_address(s: str) -> tuple[str, str]:
    """Return (tokens, sorted unique numbers) with canonical abbreviations."""
    out = []
    for t in _base(s).split():
        t = ADDRESS_CANON.get(t, t)
        if not t:
            continue
        out.append(_strip_zeros(t))
    nums = sorted({_strip_zeros(n) for n in _DIGITS.findall(" ".join(out))})
    return " ".join(out), " ".join(nums)


def _norm_chunk(args):
    names, addrs = args
    n = [normalize_name(x) for x in names]
    a = [normalize_address(x) for x in addrs]
    return (
        [x[0] for x in n], [x[1] for x in n], [x[2] for x in n],
        [x[0] for x in a], [x[1] for x in a],
        [bool(_INDIC.search(x)) for x in names],
    )


def normalize_frame(df: pd.DataFrame, n_jobs: int = N_JOBS) -> pd.DataFrame:
    """Add normalized columns to a source frame (parallel over chunks)."""
    names = df["business_name"].tolist()
    addrs = df["business_address"].tolist()
    step = 50_000
    chunks = [(names[i:i + step], addrs[i:i + step]) for i in range(0, len(names), step)]
    with Pool(n_jobs) as pool:
        res = pool.map(_norm_chunk, chunks)
    cols = ["name_full", "name_core", "name_concat", "addr_norm", "addr_nums", "name_indic"]
    out = df.copy()
    for i, c in enumerate(cols):
        out[c] = [v for r in res for v in r[i]]
    out["name_indic"] = out["name_indic"].astype(bool)
    out["addr_empty"] = (out["addr_norm"] == "").to_numpy()
    # Blocking key for exact pass: sorted unique core tokens (order-insensitive).
    out["name_key"] = [" ".join(sorted(set(x.split()))) for x in out["name_core"]]
    return out
