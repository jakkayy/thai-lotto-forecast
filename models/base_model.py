from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseLotteryModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, df_train: pd.DataFrame) -> None:
        """Train บน DataFrame ที่มี columns: candidate, is_winner, + features"""

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """คืน Series ของ score/probability สำหรับแต่ละ row (candidate)"""

    def rank_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """คืน DataFrame ของ candidate เรียงตาม score สูง→ต่ำ"""
        scores = self.predict_proba(df)
        result = df[["candidate"]].copy()
        result["score"] = scores.values
        return result.sort_values("score", ascending=False).reset_index(drop=True)

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass
