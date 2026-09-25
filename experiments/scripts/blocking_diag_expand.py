"""Diagnostic method 4: neighbour expansion from M2-predicted targets.

For each validation S1, take its targets predicted by the frozen M2 stack
(inference-time information only). Retrieve the top-k most similar targets
to each one using the existing hybrid name+address index; those become
candidates for that S1. Reports recall potential only; nothing is fed to M2.

    PYTHONPATH=src python experiments/scripts/blocking_diag_expand.py [--k 5]
"""
from __future__ import annotations

import argparse
import json
import logging
import resource
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz

from business_entity_resolution.blocking import Blocker, addr_doc, name_doc
from business_entity_resolution.config import BlockingConfig, work_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()
    wd = work_dir()
    thr = json.loads((wd / "exp" / "M2.json").read_text())["val"]["threshold"]
    sc = pd.read_parquet(wd / "exp" / "M2_val_scores.parquet")
    pred = sc.loc[sc["score"] >= thr, ["s1_row", "t_row"]]
    key = lambda s, t: np.asarray(s, np.int64) << 32 | np.asarray(t)
    C = set(key(sc["s1_row"], sc["t_row"]))
    gt = pd.read_parquet(wd / "audit" / "C_val_gt.parquet")
    gk = key(gt["s1_row"], gt["t_row"])
    missed = gt[~np.isin(gk, list(C))].reset_index(drop=True)
    tg = pd.read_parquet(wd / "train_t_norm.parquet")

    # (a) upper bound: missed target ~ near-duplicate of an already-predicted target of the same S1
    pm = pred.groupby("s1_row")["t_row"].agg(list)
    best = []
    for s, t in zip(missed["s1_row"], missed["t_row"]):
        L = pm.get(s, [])
        v = 0.0
        for x in L:
            nm = fuzz.token_set_ratio(tg["name_core"].iat[t], tg["name_core"].iat[x])
            ad = fuzz.token_set_ratio(tg["addr_norm"].iat[t], tg["addr_norm"].iat[x]) \
                if tg["addr_norm"].iat[t] and tg["addr_norm"].iat[x] else nm
            v = max(v, (nm + ad) / 2)
        best.append(v if L else np.nan)
    best = np.array(best)
    ub = {"missed": len(missed), "S1_has_predictions": float(np.mean(~np.isnan(best))),
          "max_sim_to_predicted_sibling>=80": int(np.nansum(best >= 80)),
          "max_sim_to_predicted_sibling>=90": int(np.nansum(best >= 90))}
    logging.info("upper bound %s", ub)

    # (b) actual retrieval: hybrid (name + address) index, queried with predicted targets
    blocker = Blocker(tg, BlockingConfig(hybrid=True))
    out = []
    for c, m in blocker.by_country.items():
        sub = pred[tg["country"].to_numpy()[pred["t_row"].to_numpy()] == c]
        if sub.empty:
            continue
        loc = pd.Series(np.arange(len(m["idx"])), index=m["idx"])
        q = tg.iloc[sub["t_row"].to_numpy()]
        Qh = sp.hstack([m["name"].transform(name_doc(q)), m["addr"].transform(addr_doc(q))]).tocsr()
        fr = Blocker._topk_frame(m["hybrid_TT"], Qh, a.k + 1, blocker.cfg, 0)  # +1: self is retrieved
        out.append(pd.DataFrame({"s1_row": sub["s1_row"].to_numpy()[fr["qi"].to_numpy()],
                                 "t_row": m["idx"][fr["tj"].to_numpy()]}))
        del loc
    r = pd.concat(out).drop_duplicates()
    rk = key(r["s1_row"], r["t_row"])
    new = ~np.isin(rk, list(C))
    is_true = np.isin(rk, gk)
    mk = key(missed["s1_row"], missed["t_row"])
    hit = np.isin(mk, rk)
    s1 = pd.read_parquet(wd / "train_s1_norm.parquet", columns=["country", "name_core"])
    kc = tg.groupby(["country", "name_core"]).size()
    a1, b1 = s1.iloc[missed["s1_row"]], tg.iloc[missed["t_row"]]
    ratio = np.array([fuzz.ratio(x, y) for x, y in zip(a1["name_core"], b1["name_core"])])
    cat = {"Indic target": b1["name_indic"].to_numpy(), "India": a1["country"].to_numpy() == "India",
           "empty-address target": b1["addr_empty"].to_numpy(),
           "common-name S1 (>50)": kc.reindex(pd.MultiIndex.from_arrays(
               [a1["country"].to_numpy(), a1["name_core"].to_numpy()])).fillna(0).to_numpy() > 50,
           "Latin name-typo (0.8<=ratio<1)": (~b1["name_indic"].to_numpy()) & (ratio >= 80) & (ratio < 100),
           "Latin other": (~b1["name_indic"].to_numpy()) & ~((ratio >= 80) & (ratio < 100))}
    diag = wd / "diag"
    prev = {}
    for m_ in ("translit", "char_name", "char_addr"):
        p = pd.read_parquet(diag / f"{m_}.parquet")
        prev[m_] = int((hit & np.isin(mk, key(p["s1_row"], p["t_row"]))).sum())
    n_true = len(gt)
    rep = {"upper_bound": ub, "k_per_predicted_target": a.k, "pairs": len(r),
           "new_unique_true_beyond_C": int(hit.sum()), "share_of_C_misses_recovered": float(hit.mean()),
           "blocking_recall_C_plus_method": float((n_true - len(missed) + hit.sum()) / n_true),
           "overlap_with_C_(share_of_method_true_already_in_C)": float((is_true & ~new).sum() / max(is_true.sum(), 1)),
           "new_candidates_per_S1": float(new.sum() / 200_000),
           "precision_of_new_candidates": float((is_true & new).sum() / max(new.sum(), 1)),
           "recovered_by_category": {k: f"{int((hit & v).sum())}/{int(v.sum())}" for k, v in cat.items()},
           "overlap_with_other_methods": prev, "runtime_s": time.time() - t0,
           "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    (diag / "expand.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
