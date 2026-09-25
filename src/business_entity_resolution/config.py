"""Paths and baseline hyper-parameters.

Paths come from environment variables so the pipeline never hard-codes where
the (non-committed) competition data lives:

    BER_DATA  directory containing train/ and test/ with the challenge TSVs
    BER_WORK  scratch directory for caches (normalized parquet, candidates, models)
    BER_OUT   directory for the submission TSVs (default: <repo>/output)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("BER_DATA", REPO_ROOT / "data"))


def work_dir() -> Path:
    p = Path(os.environ.get("BER_WORK", REPO_ROOT / "work"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def out_dir() -> Path:
    p = Path(os.environ.get("BER_OUT", REPO_ROOT / "output"))
    p.mkdir(parents=True, exist_ok=True)
    return p


N_JOBS = int(os.environ.get("BER_JOBS", os.cpu_count() or 1))
SEED = 13

# Tokens that the Phase-2 audit showed are routinely added/dropped between an S1
# name and its true S2/S3 records (legal forms, honorifics, filler, web noise).
# They are removed only from the *core* name used for blocking keys and some
# features; the full normalized name is always kept as well.
LEGAL_TOKENS = frozenset(
    """
    llc inc incorporated ltd limited pvt private corp corporation co company cie
    lp llp plc pllc pc
    sarl sas sasu sa sci eurl snc ei scop
    """.split()
)
FILLER_TOKENS = frozenset(
    """
    the and of dba formerly doing business as aka trading
    services service center group partners
    shri sri smt mr dr
    com www
    india france
    """.split()
)

# Canonical forms for address tokens (abbreviation -> full), from audit examples.
ADDRESS_CANON = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "dr": "drive", "ln": "lane", "ct": "court", "blvd": "boulevard", "bd": "boulevard",
    "hwy": "highway", "pkwy": "parkway", "pl": "place", "sq": "square", "rte": "route",
    "r": "rue", "all": "allee", "imp": "impasse", "ste": "suite", "apt": "apartment",
    "fl": "floor", "flr": "floor", "no": "", "nr": "near", "opp": "opposite",
    "null": "", "na": "", "n/a": "", "none": "",
}


@dataclass
class BlockingConfig:
    # Exact core-name key pass: skip keys whose target block is larger than this
    # (very common names such as "primary care" would otherwise explode).
    exact_max_block: int = 50
    # TF-IDF top-k passes.
    name_topk: int = 10
    addr_topk: int = 10
    # Tokens whose document frequency among targets exceeds this are dropped
    # from the blocking vectors (they carry little identity and dominate cost).
    name_max_df: int = 20000
    addr_max_df: int = 20000
    min_score: float = 0.05
    chunk_rows: int = 200_000


@dataclass
class ValidationConfig:
    n_folds: int = 5
    train_folds: tuple = (0, 1, 2)
    tune_fold: int = 3
    val_fold: int = 4
    # S1 entities sampled from each role (the blocking pool is always the full
    # train S2+S3 set, as in the test setting).
    n_train: int = 300_000
    n_tune: int = 150_000
    n_val: int = 200_000


@dataclass
class ModelConfig:
    params: dict = field(
        default_factory=lambda: dict(
            objective="binary",
            learning_rate=0.1,
            num_leaves=63,
            min_data_in_leaf=100,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=1,
            verbose=-1,
            seed=SEED,
        )
    )
    num_boost_round: int = 300
