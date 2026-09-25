"""Candidate generation: several simple, individually measurable passes.

All passes operate *within* a country value taken from the data (open set, no
hard-coded labels); the audit found zero cross-country true pairs.

Passes
------
name_exact : identical order-insensitive core-name key (skips huge blocks)
name_tfidf : top-k cosine over IDF-weighted core-name tokens (+ concat token)
addr_tfidf : top-k cosine over IDF-weighted normalized address tokens

No pass compares all pairs: exact is a hash join, the TF-IDF passes use a
sparse top-k matrix product over a vocabulary pruned of very frequent tokens.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sparse_dot_topn import sp_matmul_topn

from .config import N_JOBS, BlockingConfig

log = logging.getLogger(__name__)

PASSES = ("name_exact", "name_tfidf", "addr_tfidf")


def name_doc(df: pd.DataFrame) -> list[str]:
    return [f"{c} #{k}" if k else c for c, k in zip(df["name_core"], df["name_concat"])]


def addr_doc(df: pd.DataFrame) -> list[str]:
    return df["addr_norm"].tolist()


class TfidfIndex:
    """IDF-weighted, L2-normalized binary token vectors fitted on the target pool.

    Tokens are hashed (2^24 buckets, deterministic murmurhash) so no Python
    vocabulary is held in memory; near-unique tokens such as concatenated names
    are therefore cheap.
    """

    N_FEATURES = 2 ** 24

    def __init__(self, target_docs: list[str], max_df: int):
        self.hv = HashingVectorizer(analyzer=str.split, n_features=self.N_FEATURES,
                                    alternate_sign=False, norm=None, binary=True,
                                    dtype=np.float32)
        X = self.hv.transform(target_docs).tocsr()
        df = np.bincount(X.indices, minlength=self.N_FEATURES)
        n = X.shape[0]
        idf = (np.log((n + 1) / (df + 1)) + 1).astype(np.float32)
        idf[df > max_df] = 0  # prune very frequent tokens from blocking vectors
        self.idf = idf
        self.T = self._weight(X)
        self.TT = self.T.T.tocsr()

    def _weight(self, X: sp.csr_matrix) -> sp.csr_matrix:
        X = X.tocsr(copy=True)
        X.data = self.idf[X.indices] * X.data
        X.eliminate_zeros()
        norm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
        norm[norm == 0] = 1
        X.data = X.data / np.repeat(norm, np.diff(X.indptr)).astype(np.float32)
        return X

    def transform(self, docs: list[str]) -> sp.csr_matrix:
        return self._weight(self.hv.transform(docs).tocsr())

    def topk(self, Q: sp.csr_matrix, k: int, min_score: float, chunk: int):
        out_i, out_j, out_s, out_r = [], [], [], []
        for a in range(0, Q.shape[0], chunk):
            C = sp_matmul_topn(Q[a:a + chunk], self.TT, top_n=k, threshold=min_score,
                               sort=True, n_threads=N_JOBS).tocsr()
            cnt = np.diff(C.indptr)
            out_i.append(np.repeat(np.arange(C.shape[0]) + a, cnt))
            out_j.append(C.indices.astype(np.int64))
            out_s.append(C.data)
            # rank within row (results are sorted, best first)
            out_r.append(np.arange(C.nnz) - np.repeat(C.indptr[:-1], cnt))
        return (np.concatenate(out_i), np.concatenate(out_j),
                np.concatenate(out_s), np.concatenate(out_r))


def pair_cosine(Q: sp.csr_matrix, T: sp.csr_matrix, qi: np.ndarray, tj: np.ndarray,
                chunk: int = 2_000_000) -> np.ndarray:
    """Row-wise cosine for explicit (query row, target row) pairs."""
    out = np.empty(len(qi), np.float32)
    for a in range(0, len(qi), chunk):
        b = a + chunk
        out[a:b] = np.asarray(Q[qi[a:b]].multiply(T[tj[a:b]]).sum(1)).ravel()
    return out


class Blocker:
    """Fits per-country indexes on the target pool; queries any S1 subset."""

    def __init__(self, targets: pd.DataFrame, cfg: BlockingConfig):
        self.cfg = cfg
        self.tg = targets
        self.by_country = {}
        for c, idx in targets.groupby("country").indices.items():
            sub = targets.iloc[idx]
            log.info("fit indexes country=%s targets=%d", c, len(idx))
            self.by_country[c] = dict(
                idx=idx,
                name=TfidfIndex(name_doc(sub), cfg.name_max_df),
                addr=TfidfIndex(addr_doc(sub), cfg.addr_max_df),
                key_counts=sub["name_key"].value_counts(),
            )

    def candidates(self, s1: pd.DataFrame) -> pd.DataFrame:
        """Union of all passes for the given S1 rows.

        Returns one row per (s1 row position, target row position) with a
        boolean flag and rank per pass plus both TF-IDF cosines for every pair.
        """
        cfg = self.cfg
        frames = []
        for c, qidx in s1.groupby("country").indices.items():
            if c not in self.by_country:
                continue  # no targets in this country -> no candidates
            m = self.by_country[c]
            q = s1.iloc[qidx]
            tkeys = self.tg["name_key"].to_numpy()[m["idx"]]
            parts = []
            # pass 1: exact core-name key, skipping oversized blocks
            ok = m["key_counts"][m["key_counts"] <= cfg.exact_max_block].index
            lk = pd.DataFrame({"qi": np.arange(len(q)), "name_key": q["name_key"].to_numpy()})
            lk = lk[lk["name_key"].isin(ok)]
            rk = pd.DataFrame({"tj": np.arange(len(tkeys)), "name_key": tkeys})
            ex = lk.merge(rk, on="name_key")[["qi", "tj"]].astype(np.int32)
            ex["pass"] = 0
            ex["rank"] = np.int16(0)
            parts.append(ex)
            # passes 2-3: TF-IDF top-k
            Qn = m["name"].transform(name_doc(q))
            Qa = m["addr"].transform(addr_doc(q))
            for code, idx_, Q, k in ((1, m["name"], Qn, cfg.name_topk),
                                      (2, m["addr"], Qa, cfg.addr_topk)):
                i, j, s, r = idx_.topk(Q, k, cfg.min_score, cfg.chunk_rows)
                parts.append(pd.DataFrame({"qi": i.astype(np.int32), "tj": j.astype(np.int32),
                                           "pass": np.int8(code), "rank": r.astype(np.int16)}))
            allp = pd.concat(parts, ignore_index=True)
            ranks = allp.groupby(["qi", "tj", "pass"])["rank"].min().unstack("pass")
            ranks = ranks.reindex(columns=range(len(PASSES)))
            ranks.columns = list(PASSES)
            del allp, parts
            u = ranks.add_prefix("rank_")
            for p in PASSES:
                u[p] = ranks[p].notna().to_numpy()
            u = u.reset_index()
            qi = u["qi"].to_numpy()
            tj = u["tj"].to_numpy()
            u["cos_name"] = pair_cosine(Qn, m["name"].T, qi, tj)
            u["cos_addr"] = pair_cosine(Qa, m["addr"].T, qi, tj)
            u["s1_row"] = qidx[qi]
            u["t_row"] = m["idx"][tj]
            frames.append(u.drop(columns=["qi", "tj"]))
            log.info("country=%s s1=%d candidates=%d", c, len(q), len(u))
        cols = ["s1_row", "t_row", *PASSES, *[f"rank_{p}" for p in PASSES], "cos_name", "cos_addr"]
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
        for p in PASSES:
            if f"rank_{p}" not in out:
                out[f"rank_{p}"] = np.nan
        return out[cols]
