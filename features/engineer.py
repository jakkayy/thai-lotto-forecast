"""Feature Engineering สำหรับแต่ละ target type"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from db.connection import get_session
from db.models import LotteryDraw, LotteryFeatures

FEATURE_COLS = [
    "month", "day", "year", "quarter", "draw_index", "is_first_of_month",
    "freq_all", "freq_10", "freq_30", "freq_60",
    "gap", "avg_gap", "is_overdue",
    "freq_rate_all", "freq_rate_30",
]

# target_type → จำนวน candidates ที่เป็นไปได้
TARGET_CANDIDATES = {
    "back2": [str(i).zfill(2) for i in range(100)],         # 00-99
    "back3": [str(i).zfill(3) for i in range(1000)],        # 000-999
    "front3": [str(i).zfill(3) for i in range(1000)],       # 000-999
    "prize1_last2": [str(i).zfill(2) for i in range(100)],  # เฉพาะ 2 หลักท้ายของรางวัลที่ 1
}

TARGET_WINNER_FIELD = {
    "back2": "prize_back_2",
    "back3": "prize_back_3",
    "front3": "prize_front_3",
    "prize1_last2": "prize_1",
}


class FeatureEngineer:
    def __init__(self, draws: list[LotteryDraw]):
        self.draws = sorted(draws, key=lambda d: d.draw_date)
        self.df = self._to_dataframe()

    def _to_dataframe(self) -> pd.DataFrame:
        rows = []
        for d in self.draws:
            rows.append({
                "draw_date": d.draw_date,
                "prize_1": d.prize_1 or "",
                "prize_back_2": d.prize_back_2 or "",
                "prize_back_3": d.prize_back_3 or [],
                "prize_front_3": d.prize_front_3 or [],
                "prize1_last2": (d.prize_1 or "")[-2:] if d.prize_1 else "",
            })
        return pd.DataFrame(rows)

    def compute_all(self, target_type: str) -> list[dict[str, Any]]:
        """คำนวณ features สำหรับทุก draw ใน target_type หนึ่ง"""
        candidates = TARGET_CANDIDATES[target_type]
        winner_field = TARGET_WINNER_FIELD[target_type]
        records = []

        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            draw_date = row["draw_date"]
            history = self.df.iloc[:idx]  # ข้อมูลก่อนงวดนี้เท่านั้น

            # winners ของงวดนี้
            winner_val = row[winner_field if winner_field in row else winner_field]
            if target_type in ("back3", "front3"):
                winners = set(row[winner_field] if isinstance(row[winner_field], list) else [])
            elif target_type == "prize1_last2":
                winners = {row["prize1_last2"]} if row["prize1_last2"] else set()
            else:
                winners = {row[winner_field]} if row[winner_field] else set()

            context_feats = self._context_features(draw_date, history, target_type)

            for candidate in candidates:
                cand_feats = self._candidate_features(candidate, history, target_type)
                features = {**context_feats, **cand_feats}
                is_winner = candidate in winners

                records.append({
                    "draw_date": draw_date,
                    "target_type": target_type,
                    "candidate": candidate,
                    "features": features,
                    "is_winner": is_winner,
                })

        logger.info(f"[features] computed {len(records)} records for {target_type}")
        return records

    def _context_features(self, draw_date: date, history: pd.DataFrame, target_type: str) -> dict:
        """Features ที่ไม่ขึ้นกับ candidate — เหมือนกันทุก candidate ในงวดนั้น"""
        return {
            "month": draw_date.month,
            "day": draw_date.day,  # 1 หรือ 16
            "year": draw_date.year,
            "quarter": (draw_date.month - 1) // 3 + 1,
            "draw_index": len(history),
            "is_first_of_month": int(draw_date.day == 1),
        }

    def _candidate_features(self, candidate: str, history: pd.DataFrame, target_type: str) -> dict:
        """Features เฉพาะ candidate นี้"""
        if len(history) == 0:
            return self._zero_candidate_features(candidate, target_type)

        winner_field = TARGET_WINNER_FIELD[target_type]

        # สร้าง binary series: 1 ถ้างวดนั้น candidate นี้ออก
        if target_type in ("back3", "front3"):
            appeared = history[winner_field].apply(
                lambda x: int(candidate in (x if isinstance(x, list) else []))
            )
        elif target_type == "prize1_last2":
            appeared = (history["prize1_last2"] == candidate).astype(int)
        else:
            col = "prize_back_2" if target_type == "back2" else winner_field
            appeared = (history[col] == candidate).astype(int)

        n = len(appeared)
        freq_all = appeared.sum()
        freq_10 = appeared.iloc[-10:].sum() if n >= 10 else appeared.sum()
        freq_30 = appeared.iloc[-30:].sum() if n >= 30 else appeared.sum()
        freq_60 = appeared.iloc[-60:].sum() if n >= 60 else appeared.sum()

        # Gap: จำนวน draws นับจากครั้งล่าสุดที่ออก
        last_idx = appeared[::-1].values.tolist()
        gap = next((i for i, v in enumerate(last_idx) if v == 1), n)

        # Average gap ระหว่างครั้งที่ออก
        indices_appeared = appeared[appeared == 1].index.tolist()
        if len(indices_appeared) >= 2:
            gaps = [indices_appeared[i+1] - indices_appeared[i] for i in range(len(indices_appeared)-1)]
            avg_gap = float(np.mean(gaps))
        else:
            total_candidates = len(TARGET_CANDIDATES[target_type])
            avg_gap = float(total_candidates)  # expected gap ถ้า uniform

        # Digit-level features
        digit_feats = self._digit_features(candidate, history)

        return {
            "freq_all": int(freq_all),
            "freq_10": int(freq_10),
            "freq_30": int(freq_30),
            "freq_60": int(freq_60),
            "gap": gap,
            "avg_gap": round(avg_gap, 2),
            "is_overdue": int(gap > avg_gap),
            "freq_rate_all": round(freq_all / n, 4) if n > 0 else 0.0,
            "freq_rate_30": round(appeared.iloc[-30:].sum() / min(n, 30), 4) if n > 0 else 0.0,
            **digit_feats,
        }

    def _digit_features(self, candidate: str, history: pd.DataFrame) -> dict:
        """Frequency ของแต่ละ digit position"""
        feats = {}
        for pos, digit in enumerate(candidate):
            col = f"prize_back_2" if len(candidate) == 2 else None
            feats[f"digit_{pos}_val"] = int(digit)
        return feats

    def _zero_candidate_features(self, candidate: str, target_type: str) -> dict:
        digit_feats = self._digit_features(candidate, pd.DataFrame())
        return {
            "freq_all": 0, "freq_10": 0, "freq_30": 0, "freq_60": 0,
            "gap": 0, "avg_gap": 0.0, "is_overdue": 0,
            "freq_rate_all": 0.0, "freq_rate_30": 0.0,
            **digit_feats,
        }

    def save_to_db(self, records: list[dict[str, Any]]) -> int:
        """Bulk upsert features เข้า DB"""
        if not records:
            return 0
        try:
            rows = [
                {
                    "draw_date": r["draw_date"],
                    "target_type": r["target_type"],
                    "candidate": r["candidate"],
                    "features": r["features"],
                    "is_winner": r["is_winner"],
                }
                for r in records
            ]
            stmt = (
                insert(LotteryFeatures)
                .values(rows)
                .on_conflict_do_update(
                    constraint="uq_features_draw_target_candidate",
                    set_={"features": insert(LotteryFeatures).excluded.features,
                          "is_winner": insert(LotteryFeatures).excluded.is_winner},
                )
            )
            with get_session() as session:
                session.execute(stmt)
            logger.info(f"[features] saved {len(rows)} rows to DB")
            return len(rows)
        except Exception as e:
            logger.error(f"[features] save_to_db failed: {e}")
            return 0

    def to_dataframe(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """แปลง records → DataFrame พร้อม flatten features"""
        rows = []
        for r in records:
            row = {"draw_date": r["draw_date"], "candidate": r["candidate"], "is_winner": int(r["is_winner"])}
            row.update(r["features"])
            rows.append(row)
        return pd.DataFrame(rows)
