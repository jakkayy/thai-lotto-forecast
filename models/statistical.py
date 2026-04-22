"""Statistical model: Gap + Frequency combined score"""
import numpy as np
import pandas as pd

from models.base_model import BaseLotteryModel


class StatisticalModel(BaseLotteryModel):
    """
    Score = w_gap * gap_score + w_freq * freq_score
    - gap_score: candidate ที่ไม่ออกนาน (overdue) ได้คะแนนสูง
    - freq_score: candidate ที่ออกบ่อยใน 30 งวดล่าสุดได้คะแนนสูง
    """
    name = "statistical"

    def __init__(self, w_gap: float = 0.5, w_freq: float = 0.5):
        self.w_gap = w_gap
        self.w_freq = w_freq

    def fit(self, df_train: pd.DataFrame) -> None:
        pass  # ไม่ต้อง train — ใช้ features ที่คำนวณไว้แล้วโดยตรง

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        scores = pd.Series(np.zeros(len(df)), index=df.index)

        if "gap" in df.columns and "avg_gap" in df.columns:
            # gap score: normalized (gap / avg_gap) — ยิ่งนานยิ่ง overdue
            avg_gap_safe = df["avg_gap"].replace(0, 1)
            gap_ratio = df["gap"] / avg_gap_safe
            gap_score = (gap_ratio - gap_ratio.min()) / (gap_ratio.max() - gap_ratio.min() + 1e-9)
            scores += self.w_gap * gap_score

        if "freq_rate_30" in df.columns:
            freq = df["freq_rate_30"]
            freq_score = (freq - freq.min()) / (freq.max() - freq.min() + 1e-9)
            scores += self.w_freq * freq_score

        return scores
