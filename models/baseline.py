"""Baseline: random + frequency-weighted random"""
import numpy as np
import pandas as pd

from models.base_model import BaseLotteryModel


class BaselineModel(BaseLotteryModel):
    """
    Uniform random — ใช้เป็น baseline เพื่อเปรียบเทียบกับโมเดลอื่น
    ทุก candidate ได้ score เท่ากัน (แบบ random)
    """
    name = "baseline_random"

    def fit(self, df_train: pd.DataFrame) -> None:
        pass

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        rng = np.random.default_rng()
        return pd.Series(rng.random(len(df)), index=df.index)


class FrequencyBaselineModel(BaseLotteryModel):
    """
    Frequency-weighted — candidate ที่ออกบ่อยได้ score สูงกว่า
    ยังเป็น baseline แต่ดีกว่า uniform
    """
    name = "baseline_frequency"

    def __init__(self):
        self._freq: dict[str, float] = {}

    def fit(self, df_train: pd.DataFrame) -> None:
        freq = df_train.groupby("candidate")["is_winner"].sum()
        total = freq.sum()
        self._freq = (freq / total).to_dict() if total > 0 else {}

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        return df["candidate"].map(lambda c: self._freq.get(c, 1e-6))
