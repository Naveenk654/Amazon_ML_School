"""Phase 4A: blocking experiments evaluated with the UNCHANGED EXP001 matcher.

    python -m business_entity_resolution.blocking_experiments <experiment-name>

Each run reuses EXP001's validation fold (same deterministic sampling), the
saved LightGBM model and the saved threshold; only the blocking config changes.
Run each experiment in its own process so peak memory is measured cleanly.
"""
from __future__ import annotations

import argparse
import json
import logging
import resource
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import evaluate as ev
from .blocking import EXTRA_PASSES, PASSES, Blocker
from .config import BlockingConfig, ValidationConfig, work_dir
from .data import load_ground_truth
from .features import build_features
from .inference import decide
from .model import predict as model_predict
from .pipeline import generate, key_freqs, make_roles, prepared, with_ids, within_country_space

log = logging.getLogger("ber.exp")

EXPERIMENTS = {
    "E0_baseline": {},
    "A1_adaptive_k": dict(adaptive_k=((51, 30), (201, 60))),
    "A2_adaptive_k_deep": dict(adaptive_k=((11, 20), (51, 60), (201, 120))),
    "B_char_name": dict(char_name=True),
    "C_hybrid": dict(hybrid=True),
    "D_addr_expand": dict(addr_expand=True),
    "E1_empty_addr_name": dict(empty_addr_name=True),
}

BINS = [("rare_0-10", 0, 10), ("medium_11-50", 11, 50), ("common_51-200", 51, 200),
        ("common_>200", 201, np.inf), ("common_>50", 51, np.inf)]


def peak_mb() -> dict:
    return {"peak_rss_main_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "peak_rss_worker_mb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024}


def run(name: str, overrides: dict) -> dict:
    t_start = time.time()
    bcfg = BlockingConfig(**overrides)
    vcfg = ValidationConfig()
    wd = work_dir()
    model = lgb.Booster(model_file=str(wd / "baseline_model.txt"))
    thr = json.loads((wd / "baseline_threshold.json").read_text())["threshold"]

    s1, tg = prepared("train")
    s1f, tf = key_freqs(s1, tg)
    gt = load_ground_truth("train")
    gt_s1 = pd.Index(s1["entity_id"]).get_indexer(gt["s1_id"])
    rows = np.sort(make_roles(s1, np.bincount(gt_s1, minlength=len(s1)), vcfg)["val"])
    ids = s1["entity_id"].to_numpy()[rows]
    gt = gt[np.isin(gt_s1, rows)].reset_index(drop=True)
    truth = ev.as_sets(gt)

    t0 = time.time()
    blocker = Blocker(tg, bcfg)
    t_fit = time.time() - t0
    t0 = time.time()
    cand = generate(blocker, s1, tg, rows)
    t_query = time.time() - t0
    del blocker

    t0 = time.time()
    X = build_features(cand, s1, tg, s1f, tf)
    cand["score"] = model_predict(model, X)
    t_feat = time.time() - t0
    del X

    c = with_ids(cand, s1, tg)
    passes = [p for p in (*PASSES, *EXTRA_PASSES) if c[p].any()]
    within, full = within_country_space(s1.iloc[rows], tg)
    rep = ev.blocking_report(c, gt, ids, full, passes)
    rep["reduction_ratio_within_country"] = 1 - len(c) / within
    rep["oracle_macro_f05"] = ev.oracle_f05(c, truth, ids)

    # unique contribution of each pass (true pairs found by that pass only)
    key = c["s1_id"] + "|" + c["t_id"]
    is_true = key.isin(set(gt["s1_id"] + "|" + gt["t_id"])).to_numpy()
    flags = c[passes].to_numpy()
    only = flags.sum(1) == 1
    for i, p in enumerate(passes):
        rep[f"unique_true_{p}"] = int((is_true & only & flags[:, i]).sum())

    # recall by S1 name commonness (target count of S1 core name, as in EXP001)
    kc = tg.groupby(["country", "name_core"]).size()
    s1i = s1.set_index("entity_id")
    a = s1i.loc[gt["s1_id"].to_numpy()]
    f = kc.reindex(pd.MultiIndex.from_arrays([a["country"].to_numpy(), a["name_core"].to_numpy()])) \
          .fillna(0).to_numpy()
    blocked = (gt["s1_id"] + "|" + gt["t_id"]).isin(set(key)).to_numpy()
    rep["recall_by_name_freq"] = {b: {"share_of_true_pairs": float(((f >= lo) & (f <= hi)).mean()),
                                      "recall": float(blocked[(f >= lo) & (f <= hi)].mean())}
                                  for b, lo, hi in BINS}

    pred_pairs = decide(c, thr)
    pred = ev.as_sets(pred_pairs)
    res = {"threshold": thr, "macro_f05": ev.macro_f05(pred, truth, ids),
           **ev.pair_metrics(pred_pairs, gt),
           "pred_empty_rate": float(np.mean([s not in pred for s in ids])),
           "singleton_false_match": int(sum(1 for s in ids if s not in truth and s in pred))}
    bd = ev.breakdown(pred, truth, s1.iloc[rows])
    out = {"experiment": name, "overrides": {k: str(v) for k, v in overrides.items()},
           "n_val_s1": len(ids), "blocking": rep, "end_to_end": res,
           "breakdown": bd["macro_f05"].round(4).to_dict(),
           "runtime_s": {"index_fit": t_fit, "blocking_query": t_query,
                         "features_and_scoring": t_feat, "total": time.time() - t_start},
           **peak_mb()}
    d = wd / "exp"
    d.mkdir(exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
    log.info("RESULT %s", json.dumps(out, default=float))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=list(EXPERIMENTS))
    a = ap.parse_args()
    run(a.name, EXPERIMENTS[a.name])


if __name__ == "__main__":
    main()
