"""Ensemble model — weighted average ของหลายโมเดล"""
import numpy as np
import pandas as pd
from loguru import logger

from models.base_model import BaseLotteryModel


class EnsembleModel(BaseLotteryModel):
    """
    Weighted average ของ scores จากหลายโมเดล
    Weights ถูกปรับจาก walk-forward validation performance
    """
    name = "ensemble"

    def __init__(self, models: list[BaseLotteryModel], weights: list[float] | None = None):
        self.sub_models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        assert len(self.sub_models) == len(self.weights)

    def fit(self, df_train: pd.DataFrame) -> None:
        for model in self.sub_models:
            logger.info(f"[ensemble] training sub-model: {model.name}")
            model.fit(df_train)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        all_scores = []
        for model, w in zip(self.sub_models, self.weights):
            scores = model.predict_proba(df)
            # normalize ก่อน weight
            s_min, s_max = scores.min(), scores.max()
            if s_max > s_min:
                scores = (scores - s_min) / (s_max - s_min)
            all_scores.append(scores * w)

        return sum(all_scores)

    def update_weights_from_performance(self, hit_rates: dict[str, float]) -> None:
        """ปรับ weights ตาม hit_rate ของแต่ละ sub-model"""
        new_weights = []
        for model in self.sub_models:
            rate = hit_rates.get(model.name, 0.0)
            new_weights.append(max(rate, 1e-6))
        total = sum(new_weights)
        self.weights = [w / total for w in new_weights]
        logger.info(f"[ensemble] updated weights: {dict(zip([m.name for m in self.sub_models], self.weights))}")
