"""Blocking-miss diagnostic: incremental recall of candidate retrieval methods vs C.

    PYTHONPATH=src python experiments/scripts/blocking_diag.py --method translit
    PYTHONPATH=src python experiments/scripts/blocking_diag.py --method char_name
    PYTHONPATH=src python experiments/scripts/blocking_diag.py --method char_addr
    PYTHONPATH=src python experiments/scripts/blocking_diag.py --analyze

Each method is an independent top-k retrieval pass over the full train S2+S3
pool (within country), queried with the EXP001 validation S1 (200k). Nothing
is fed to the matcher; the output is only candidate pairs + measurements.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import resource
import time
import unicodedata
from multiprocessing import get_context

import numpy as np
import pandas as pd

from business_entity_resolution.blocking import Blocker, TfidfIndex
from business_entity_resolution.config import N_JOBS, BlockingConfig, work_dir

log = logging.getLogger("diag")
TOPK = 10
_INDIC = re.compile(r"[ऀ-෿]")
_SCRIPTS = [((0x900, 0x97f), "devanagari"), ((0x980, 0x9ff), "bengali"), ((0xa00, 0xa7f), "gurmukhi"),
            ((0xa80, 0xaff), "gujarati"), ((0xb00, 0xb7f), "oriya"), ((0xb80, 0xbff), "tamil"),
            ((0xc00, 0xc7f), "telugu"), ((0xc80, 0xcff), "kannada"), ((0xd00, 0xd7f), "malayalam")]


def squash(s: str) -> str:
    """Symmetric phonetic folding applied to BOTH Latin S1 names and transliterations."""
    s = s.lower().replace("ch", "\x01").replace("sh", "s").replace("ph", "f").replace("th", "t")
    s = s.replace("dh", "d").replace("bh", "b").replace("kh", "k").replace("gh", "g").replace("ck", "k")
    s = s.replace("x", "ks").replace("q", "k").replace("w", "v").replace("c", "k").replace("z", "j")
    s = s.replace("y", "i").replace("ee", "i").replace("oo", "u").replace("\x01", "ch")
    s = re.sub(r"(.)\1+", r"\1", s)          # collapse doubled letters
    return re.sub(r"[^a-z0-9 ]", "", s)


def _tokscript(tok: str):
    for ch in tok:
        o = ord(ch)
        for (a, b), name in _SCRIPTS:
            if a <= o <= b:
                return name
    return None


def translit_name(s: str) -> str:
    from indic_transliteration import sanscript
    out = []
    for tok in s.split():
        sc = _tokscript(tok)
        if sc:
            t = sanscript.transliterate(tok, getattr(sanscript, sc.upper()), sanscript.ITRANS)
            t = unicodedata.normalize("NFKD", t.lower())
            t = re.sub(r"[^a-z0-9]", "", t)
            if len(t) > 3 and t.endswith("a") and t[-2] not in "aeiou":
                t = t[:-1]                          # drop inherent final vowel
            out.append(t)
        else:
            out.append(tok)
    return squash(" ".join(out))


def _translit_chunk(names):
    return [translit_name(x) for x in names]


def run_method(method: str) -> None:
    t0 = time.time()
    wd = work_dir()
    rows = np.load(wd / "audit" / "val_rows.npy")
    s1 = pd.read_parquet(wd / "train_s1_norm.parquet", columns=["country", "name_core", "addr_norm"])
    tg = pd.read_parquet(wd / "train_t_norm.parquet", columns=["country", "name_core", "name_indic", "addr_norm"])
    q = s1.iloc[rows]
    out = []
    for c, qidx in q.groupby("country").indices.items():
        tidx = np.flatnonzero(tg["country"].to_numpy() == c)
        if method == "translit":
            tidx = tidx[tg["name_indic"].to_numpy()[tidx]]
            if not len(tidx):
                continue
            names = tg["name_core"].to_numpy()[tidx].tolist()
            step = 20_000
            with get_context("spawn").Pool(N_JOBS) as pool:
                docs = sum(pool.map(_translit_chunk, [names[i:i + step] for i in range(0, len(names), step)]), [])
            qdocs = [squash(x) for x in q["name_core"].to_numpy()[qidx]]
            idx = TfidfIndex(docs, max_df=30_000, char_ngrams=3)
        elif method == "char_name":
            docs = tg["name_core"].to_numpy()[tidx].tolist()
            qdocs = q["name_core"].to_numpy()[qidx].tolist()
            idx = TfidfIndex(docs, max_df=30_000, char_ngrams=3)
        elif method == "char_addr":
            tidx = tidx[tg["addr_norm"].to_numpy()[tidx] != ""]
            docs = tg["addr_norm"].to_numpy()[tidx].tolist()
            qdocs = q["addr_norm"].to_numpy()[qidx].tolist()
            idx = TfidfIndex(docs, max_df=30_000, char_ngrams=3)
        log.info("%s country=%s targets=%d index built (%.0fs)", method, c, len(tidx), time.time() - t0)
        fr = Blocker._topk_frame(idx.TT, idx.transform(qdocs), TOPK, BlockingConfig(), 0)
        out.append(pd.DataFrame({"s1_row": rows[qidx][fr["qi"].to_numpy()],
                                 "t_row": tidx[fr["tj"].to_numpy()]}))
        del idx
    res = pd.concat(out, ignore_index=True)
    d = wd / "diag"
    d.mkdir(exist_ok=True)
    res.to_parquet(d / f"{method}.parquet", index=False)
    meta = {"method": method, "pairs": len(res), "runtime_s": time.time() - t0,
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    (d / f"{method}.json").write_text(json.dumps(meta, indent=1))
    log.info("done %s", meta)


def analyze() -> None:
    from rapidfuzz import fuzz
    wd = work_dir()
    d = wd / "diag"
    rows = np.load(wd / "audit" / "val_rows.npy")
    key = lambda df: df["s1_row"].to_numpy().astype(np.int64) << 32 | df["t_row"].to_numpy()
    C = set(key(pd.read_parquet(wd / "audit" / "C_val_cand.parquet", columns=["s1_row", "t_row"])))
    gt = pd.read_parquet(wd / "audit" / "C_val_gt.parquet")
    gk = key(gt)
    missed = gt[~np.isin(gk, list(C))].reset_index(drop=True)
    mk = key(missed)
    s1 = pd.read_parquet(wd / "train_s1_norm.parquet", columns=["country", "name_core"])
    tg = pd.read_parquet(wd / "train_t_norm.parquet", columns=["country", "name_core", "name_indic", "addr_empty"])
    kc = tg.groupby(["country", "name_core"]).size()
    a, b = s1.iloc[missed["s1_row"]], tg.iloc[missed["t_row"]]
    ratio = np.array([fuzz.ratio(x, y) for x, y in zip(a["name_core"], b["name_core"])])
    cat = {
        "Indic target": b["name_indic"].to_numpy(),
        "India": a["country"].to_numpy() == "India",
        "empty-address target": b["addr_empty"].to_numpy(),
        "common-name S1 (>50)": kc.reindex(pd.MultiIndex.from_arrays(
            [a["country"].to_numpy(), a["name_core"].to_numpy()])).fillna(0).to_numpy() > 50,
        "Latin name-typo (0.8<=ratio<1)": (~b["name_indic"].to_numpy()) & (ratio >= 80) & (ratio < 100),
        "Latin other": (~b["name_indic"].to_numpy()) & ~((ratio >= 80) & (ratio < 100)),
    }
    n_true = len(gt)
    report, hit = {"true_pairs": n_true, "missed_by_C": len(missed),
                   "missed_by_category": {k: int(v.sum()) for k, v in cat.items()}}, {}
    for m in ("translit", "char_name", "char_addr"):
        f = d / f"{m}.parquet"
        if not f.exists():
            continue
        r = pd.read_parquet(f)
        rk = key(r)
        is_true = np.isin(rk, gk)
        in_c = np.isin(rk, list(C))
        h = np.isin(mk, rk)
        hit[m] = h
        meta = json.loads((d / f"{m}.json").read_text())
        report[m] = {
            "pairs": len(r), "true_found": int(is_true.sum()),
            "recall_alone": float(is_true.sum() / n_true),
            "new_unique_true_beyond_C": int(h.sum()),
            "share_of_C_misses_recovered": float(h.mean()),
            "blocking_recall_C_plus_method": float((n_true - len(missed) + h.sum()) / n_true),
            "overlap_with_C_(share_of_method_true_already_in_C)": float((is_true & in_c).sum() / max(is_true.sum(), 1)),
            "new_candidates_beyond_C": int((~in_c).sum()),
            "new_candidates_per_S1": float((~in_c).sum() / len(rows)),
            "precision_of_new_candidates": float((is_true & ~in_c).sum() / max((~in_c).sum(), 1)),
            "recovered_by_category": {k: f"{int((h & v).sum())}/{int(v.sum())}" for k, v in cat.items()},
            "runtime_s": meta["runtime_s"], "peak_rss_gb": meta["peak_rss_gb"],
        }
    ms = list(hit)
    report["overlap_among_recovered"] = {f"{x}&{y}": int((hit[x] & hit[y]).sum())
                                         for i, x in enumerate(ms) for y in ms[i + 1:]}
    if ms:
        u = np.logical_or.reduce([hit[m] for m in ms])
        report["union_all_methods"] = {"recovered": int(u.sum()),
                                       "blocking_recall": float((n_true - len(missed) + u.sum()) / n_true)}
    (d / "analysis.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["translit", "char_name", "char_addr"])
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze() if a.analyze else run_method(a.method)


if __name__ == "__main__":
    main()
