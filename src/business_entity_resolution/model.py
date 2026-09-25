"""Pairwise matcher: gradient-boosted trees (LightGBM, MIT license)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import ModelConfig


def train(X: pd.DataFrame, y: np.ndarray, cfg: ModelConfig) -> lgb.Booster:
    ds = lgb.Dataset(X, label=y, free_raw_data=True)
    return lgb.train(cfg.params, ds, num_boost_round=cfg.num_boost_round)


def predict(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X[model.feature_name()])
