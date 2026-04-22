"""Data quality checks ก่อน load เข้า DB"""
import re
from datetime import date
from typing import Any

from loguru import logger

_DRAW_DAYS = {1, 16}
_RE_6 = re.compile(r"^\d{6}$")
_RE_3 = re.compile(r"^\d{3}$")
_RE_2 = re.compile(r"^\d{2}$")


def validate_draw(draw: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    ตรวจสอบ draw dict
    Returns (is_valid, list_of_errors)
    """
    errors: list[str] = []

    if draw is None:
        return False, ["draw is None"]

    # draw_date
    draw_date = draw.get("draw_date")
    if not isinstance(draw_date, date):
        errors.append(f"draw_date invalid: {draw_date!r}")
    elif draw_date.day not in _DRAW_DAYS:
        errors.append(f"draw_date day={draw_date.day} not in {_DRAW_DAYS}")
    elif draw_date > date.today():
        errors.append(f"draw_date {draw_date} is in the future")

    # prize_1 (รางวัลที่ 1 — 6 หลัก)
    p1 = draw.get("prize_1")
    if p1 is not None and not _RE_6.match(str(p1)):
        errors.append(f"prize_1 invalid: {p1!r}")

    # prize_back_2 (2 หลัก)
    pb2 = draw.get("prize_back_2")
    if pb2 is not None and not _RE_2.match(str(pb2)):
        errors.append(f"prize_back_2 invalid: {pb2!r}")

    # prize_back_3 (list ของ 3 หลัก)
    pb3 = draw.get("prize_back_3") or []
    for n in pb3:
        if not _RE_3.match(str(n)):
            errors.append(f"prize_back_3 entry invalid: {n!r}")

    # prize_front_3 (list ของ 3 หลัก)
    pf3 = draw.get("prize_front_3") or []
    for n in pf3:
        if not _RE_3.match(str(n)):
            errors.append(f"prize_front_3 entry invalid: {n!r}")

    # ต้องมี prize_1 หรือ prize_back_2 อย่างน้อยหนึ่งอย่าง
    if not draw.get("prize_1") and not draw.get("prize_back_2"):
        errors.append("both prize_1 and prize_back_2 are missing")

    is_valid = len(errors) == 0
    if not is_valid:
        logger.warning(f"[etl.validate] draw {draw.get('draw_date')} has errors: {errors}")

    return is_valid, errors
