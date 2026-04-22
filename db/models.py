from datetime import date, datetime

from sqlalchemy import (
    BigInteger, Boolean, Date, DateTime, Float, Integer,
    String, Text, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class LotteryDraw(Base):
    """ผลการออกรางวัลสลากกินแบ่งรัฐบาลแต่ละงวด"""

    __tablename__ = "lottery_draws"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    draw_date: Mapped[date] = mapped_column(Date, unique=True, nullable=False, index=True)

    # รางวัลที่ 1 (1 รางวัล)
    prize_1: Mapped[str | None] = mapped_column(String(6))

    # รางวัลข้างเคียงรางวัลที่ 1 (2 รางวัล)
    prize_near_1: Mapped[list[str] | None] = mapped_column(ARRAY(String(6)))

    # รางวัลที่ 2 (5 รางวัล)
    prize_2: Mapped[list[str] | None] = mapped_column(ARRAY(String(6)))

    # รางวัลที่ 3 (10 รางวัล)
    prize_3: Mapped[list[str] | None] = mapped_column(ARRAY(String(6)))

    # รางวัลที่ 4 (50 รางวัล)
    prize_4: Mapped[list[str] | None] = mapped_column(ARRAY(String(6)))

    # รางวัลที่ 5 (100 รางวัล)
    prize_5: Mapped[list[str] | None] = mapped_column(ARRAY(String(6)))

    # เลขหน้า 3 ตัว (2 ชุด)
    prize_front_3: Mapped[list[str] | None] = mapped_column(ARRAY(String(3)))

    # เลขท้าย 3 ตัว (2 ชุด)
    prize_back_3: Mapped[list[str] | None] = mapped_column(ARRAY(String(3)))

    # เลขท้าย 2 ตัว (1 ชุด)
    prize_back_2: Mapped[str | None] = mapped_column(String(2))

    source: Mapped[str | None] = mapped_column(String(100))
    raw_data: Mapped[dict | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class LotteryFeatures(Base):
    """Feature store — คำนวณ features สำหรับแต่ละงวด"""

    __tablename__ = "lottery_features"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    draw_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    target_type: Mapped[str] = mapped_column(String(20), nullable=False)  # back2, back3, front3, prize1
    candidate: Mapped[str] = mapped_column(String(10), nullable=False)   # เลขที่เป็น candidate
    features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    is_winner: Mapped[bool] = mapped_column(Boolean, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("draw_date", "target_type", "candidate", name="uq_features_draw_target_candidate"),
    )


class Prediction(Base):
    """คำทำนายของแต่ละงวด"""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    target_draw_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    target_type: Mapped[str] = mapped_column(String(20), nullable=False)  # back2, back3, front3
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    predicted_numbers: Mapped[list] = mapped_column(JSONB, nullable=False)  # [{number, score}, ...]
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("target_draw_date", "target_type", "model_name", name="uq_prediction_draw_target_model"),
    )


class ModelPerformance(Base):
    """บันทึก performance ของแต่ละโมเดลในแต่ละงวด"""

    __tablename__ = "model_performance"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    target_type: Mapped[str] = mapped_column(String(20), nullable=False)
    eval_draw_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    hit_top1: Mapped[bool] = mapped_column(Boolean)   # ถูกในอันดับ 1
    hit_top5: Mapped[bool] = mapped_column(Boolean)   # ถูกใน top-5
    hit_top10: Mapped[bool] = mapped_column(Boolean)  # ถูกใน top-10
    correct_rank: Mapped[int | None] = mapped_column(Integer)  # อันดับที่ถูก (None = ไม่ถูก)
    metrics: Mapped[dict | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
