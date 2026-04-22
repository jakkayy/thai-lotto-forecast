"""Load normalized draw dict เข้า PostgreSQL"""
from datetime import date
from typing import Any

from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from db.connection import get_session
from db.models import LotteryDraw


def upsert_draw(draw: dict[str, Any]) -> bool:
    """
    Insert หรือ update draw ใน DB (upsert on draw_date)
    Returns True ถ้าสำเร็จ
    """
    try:
        row = {
            "draw_date": draw["draw_date"],
            "prize_1": draw.get("prize_1"),
            "prize_near_1": draw.get("prize_near_1") or [],
            "prize_2": draw.get("prize_2") or [],
            "prize_3": draw.get("prize_3") or [],
            "prize_4": draw.get("prize_4") or [],
            "prize_5": draw.get("prize_5") or [],
            "prize_front_3": draw.get("prize_front_3") or [],
            "prize_back_3": draw.get("prize_back_3") or [],
            "prize_back_2": draw.get("prize_back_2"),
            "source": draw.get("source"),
            "raw_data": draw.get("raw_data"),
        }

        stmt = (
            insert(LotteryDraw)
            .values(**row)
            .on_conflict_do_update(
                index_elements=["draw_date"],
                set_={
                    k: v for k, v in row.items()
                    if k != "draw_date" and v is not None
                },
            )
        )

        with get_session() as session:
            session.execute(stmt)

        logger.info(f"[etl.load] upserted draw {draw['draw_date']}")
        return True

    except Exception as e:
        logger.error(f"[etl.load] upsert failed for {draw.get('draw_date')}: {e}")
        return False


def get_all_draws(min_date: date | None = None) -> list[LotteryDraw]:
    """ดึง draw ทั้งหมดจาก DB เรียงตาม draw_date"""
    from sqlalchemy.orm import make_transient
    with get_session() as session:
        q = session.query(LotteryDraw).order_by(LotteryDraw.draw_date)
        if min_date:
            q = q.filter(LotteryDraw.draw_date >= min_date)
        draws = q.all()
        # expunge ออกจาก session เพื่อให้ใช้งานได้หลัง session ปิด
        for d in draws:
            session.expunge(d)
        return draws


def get_latest_draw_date() -> date | None:
    with get_session() as session:
        row = (
            session.query(LotteryDraw.draw_date)
            .order_by(LotteryDraw.draw_date.desc())
            .first()
        )
        return row[0] if row else None
