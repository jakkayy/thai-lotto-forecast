"""LightGBM model — ranking-based approach"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger

from models.base_model import BaseLotteryModel

def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    from features.engineer import FEATURE_COLS
    digit_cols = [c for c in df.columns if c.startswith("digit_")]
    return [c for c in FEATURE_COLS + digit_cols if c in df.columns]


class LGBMModel(BaseLotteryModel):
    """
    LightGBM ในโหมด binary classification
    Train ด้วย class_weight='balanced' เพราะ is_winner=1 หายากมาก
    """
    name = "lgbm"

    def __init__(self):
        self._model: lgb.Booster | None = None
        self._feature_cols: list[str] = []

    def fit(self, df_train: pd.DataFrame) -> None:
        self._feature_cols = _get_feature_cols(df_train)
        X = df_train[self._feature_cols].astype(float)
        y = df_train["is_winner"].astype(int)

        pos = y.sum()
        neg = len(y) - pos
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "scale_pos_weight": scale_pos_weight,
            "min_child_samples": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "random_state": 42,
        }

        dtrain = lgb.Dataset(X, label=y)
        self._model = lgb.train(
            params,
            dtrain,
            num_boost_round=params["n_estimators"],
            valid_sets=[dtrain],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
        )
        logger.info(f"[lgbm] trained on {len(X)} rows, {len(self._feature_cols)} features")

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        if self._model is None:
            return pd.Series(np.zeros(len(df)), index=df.index)
        X = df[self._feature_cols].astype(float)
        scores = self._model.predict(X)
        return pd.Series(scores, index=df.index)

    def feature_importance(self) -> pd.DataFrame:
        if self._model is None:
            return pd.DataFrame()
        imp = self._model.feature_importance(importance_type="gain")
        return pd.DataFrame({"feature": self._feature_cols, "importance": imp}).sort_values(
            "importance", ascending=False
        )

    def save(self, path: Path) -> None:
        if self._model:
            self._model.save_model(str(path))

    def load(self, path: Path) -> None:
        self._model = lgb.Booster(model_file=str(path))
