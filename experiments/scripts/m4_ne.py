"""M4-NE: neighbour expansion through the FROZEN M2 stack (validation fold).

    PYTHONPATH=src python experiments/scripts/m4_ne.py --phase expand [--k 5]
    PYTHONPATH=src python experiments/scripts/m4_ne.py --phase score  [--no-expand]

expand: M2-predicted targets of each val S1 -> top-k hybrid (name+address)
        neighbours -> new candidates (not already in C) with blocking cosines.
score : enlarged candidate set -> matcher features -> frozen M1b (p1) ->
        sibling features -> frozen M2 (cascade p1 >= 0.01) -> M2 threshold.
        --no-expand scores C alone through the same code (must equal M2).
No model is retrained; nothing is tuned on validation.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import resource
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from business_entity_resolution import evaluate as ev
from business_entity_resolution.blocking import EXTRA_PASSES, PASSES, Blocker, addr_doc, name_doc, pair_cosine
from business_entity_resolution.config import BlockingConfig, work_dir
from business_entity_resolution.features import build_features
from business_entity_resolution.pipeline import key_freqs, prepared
from business_entity_resolution.siblings import sibling_features

log = logging.getLogger("m4ne")
KEY = lambda s, t: np.asarray(s, np.int64) << 32 | np.asarray(t, np.int64)


def load_c_val(wd):
    files = sorted(glob.glob(str(wd / "matcher_data" / "C" / "val" / "cand_*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def phase_expand(k: int) -> None:
    t0 = time.time()
    wd = work_dir()
    thr = json.loads((wd / "exp" / "M2.json").read_text())["val"]["threshold"]
    sc = pd.read_parquet(wd / "exp" / "M2_val_scores.parquet")
    pred = sc.loc[sc["score"] >= thr, ["s1_row", "t_row"]]
    cset = KEY(sc["s1_row"], sc["t_row"])
    s1, tg = prepared("train")
    blocker = Blocker(tg, BlockingConfig(hybrid=True))
    t_fit = time.time() - t0
    out = []
    for c, m in blocker.by_country.items():
        sub = pred[tg["country"].to_numpy()[pred["t_row"].to_numpy()] == c]
        if sub.empty:
            continue
        q = tg.iloc[sub["t_row"].to_numpy()]
        Qh = sp.hstack([m["name"].transform(name_doc(q)), m["addr"].transform(addr_doc(q))]).tocsr()
        fr = Blocker._topk_frame(m["hybrid_TT"], Qh, k + 1, blocker.cfg, 0)  # +1: self is retrieved
        new = pd.DataFrame({"s1_row": sub["s1_row"].to_numpy()[fr["qi"].to_numpy()],
                            "t_row": m["idx"][fr["tj"].to_numpy()]}).drop_duplicates()
        new = new[~np.isin(KEY(new["s1_row"], new["t_row"]), cset)]
        # blocking cosines for the new pairs (same definition as C's candidates)
        loc = pd.Series(np.arange(len(m["idx"])), index=m["idx"])
        us = np.unique(new["s1_row"].to_numpy())
        qpos = pd.Series(np.arange(len(us)), index=us)
        qs = s1.iloc[us]
        qi = qpos.reindex(new["s1_row"]).to_numpy()
        tj = loc.reindex(new["t_row"]).to_numpy()
        new["cos_name"] = pair_cosine(m["name"].transform(name_doc(qs)), m["name"].T, qi, tj)
        new["cos_addr"] = pair_cosine(m["addr"].transform(addr_doc(qs)), m["addr"].T, qi, tj)
        out.append(new)
    new = pd.concat(out, ignore_index=True)
    new.to_parquet(wd / "exp" / "M4NE_new_pairs.parquet", index=False)
    meta = {"k": k, "new_pairs": len(new), "index_fit_s": t_fit, "runtime_s": time.time() - t0,
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    (wd / "exp" / "M4NE_expand.json").write_text(json.dumps(meta, indent=1))
    log.info("expand done %s", meta)


def phase_score(no_expand: bool) -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    wd = work_dir()
    name = "M4NE_noexpand" if no_expand else "M4NE"
    c = load_c_val(wd)
    c["x_expand"] = False
    if not no_expand:
        new = pd.read_parquet(wd / "exp" / "M4NE_new_pairs.parquet")
        for p in (*PASSES, *EXTRA_PASSES):
            new[p] = False
        for p in PASSES:
            new[f"rank_{p}"] = np.nan
        new["x_expand"] = True
        new["y"] = 0
        c = pd.concat([c, new[c.columns]], ignore_index=True)
    c = c.sort_values(["s1_row", "t_row"], kind="stable").reset_index(drop=True)
    gt = pd.read_parquet(wd / "audit" / "C_val_gt.parquet")
    gk = KEY(gt["s1_row"], gt["t_row"])
    c["y"] = np.isin(KEY(c["s1_row"], c["t_row"]), gk).astype(np.int8)

    s1, tg = prepared("train")
    s1f, tf = key_freqs(s1, tg)
    X = build_features(c, s1, tg, s1f, tf)
    m1 = lgb.Booster(model_file=str(wd / "models" / "M1b.txt"))
    rounds = int(json.loads((wd / "exp" / "M1b.json").read_text())["best_iteration"])
    c["p1"] = m1.predict(X[m1.feature_name()], num_iteration=rounds).astype(np.float32)
    S = sibling_features(c[["s1_row", "t_row", "p1"]], tg)
    m2 = lgb.Booster(model_file=str(wd / "models" / "M2.txt"))
    m2info = json.loads((wd / "exp" / "M2.json").read_text())
    thr, cut = m2info["val"]["threshold"], m2info["min_p1"]
    XS = pd.concat([X, S], axis=1)
    score = c["p1"].to_numpy().astype(np.float64)
    mask = score >= cut
    score[mask] = m2.predict(XS.loc[mask, m2.feature_name()], num_iteration=m2.best_iteration or None)
    c["score"] = score
    del X, S, XS

    ids = np.load(wd / "audit" / "val_rows.npy")
    truth = ev.as_sets(gt, "s1_row", "t_row")
    pred = ev.as_sets(c.loc[c["score"] >= thr, ["s1_row", "t_row"]], "s1_row", "t_row")
    per = np.array([ev.f05(pred.get(s, set()), truth.get(s, set())) for s in ids])
    y = c["y"].to_numpy() == 1
    p = c["score"].to_numpy() >= thr
    n_true = len(gt)
    res = {"macro_f05": float(per.mean()), "threshold": thr,
           "pair_precision": float((p & y).sum() / p.sum()), "pair_recall": float((p & y).sum() / n_true),
           "false_positives": int((p & ~y).sum()), "false_negatives": int(n_true - (p & y).sum()),
           "blocking_recall": float(y.sum() / n_true), "cands_per_s1": len(c) / len(ids),
           "cands_p95": float(c.groupby("s1_row").size().reindex(ids, fill_value=0).quantile(0.95)),
           "ceiling": ev.macro_f05(ev.as_sets(c[y], "s1_row", "t_row"), truth, ids),
           "singleton_f05": float(per[[s not in truth for s in ids]].mean())}
    ne = c["x_expand"].to_numpy()
    res["expansion"] = {"new_candidates": int(ne.sum()), "new_true": int((ne & y).sum()),
                        "new_predicted": int((ne & p).sum()), "new_predicted_true": int((ne & p & y).sum()),
                        "new_false_positives": int((ne & p & ~y).sum())}
    res["runtime_s"] = time.time() - t0
    res["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    (wd / "exp" / f"{name}.json").write_text(json.dumps(res, indent=1))
    c[["s1_row", "t_row", "y", "score", "x_expand"]].to_parquet(wd / "exp" / f"{name}_val_scores.parquet", index=False)
    log.info("RESULT %s %s", name, json.dumps(res))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["expand", "score"], required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-expand", action="store_true")
    a = ap.parse_args()
    phase_expand(a.k) if a.phase == "expand" else phase_score(a.no_expand)


if __name__ == "__main__":
    main()
