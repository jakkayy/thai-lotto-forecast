"""Bootstrap pipeline — ดึงข้อมูลย้อนหลังทั้งหมดครั้งแรก"""
from datetime import date

from loguru import logger

from etl.transform import normalize_draw
from etl.validate import validate_draw
from etl.load import upsert_draw, get_latest_draw_date
from scraper.github_archive import GithubArchiveScraper


def run_bootstrap(from_date: date | None = None, to_date: date | None = None) -> int:
    """
    ดึงข้อมูลย้อนหลังจาก GitHub archive แล้ว load เข้า DB
    Returns จำนวน draws ที่ load สำเร็จ
    """
    latest_date = get_latest_draw_date()
    if from_date is None:
        if latest_date:
            from_date = latest_date
            logger.info(f"[bootstrap] resuming from {from_date}")
        else:
            from_date = date(2007, 1, 1)
            logger.info(f"[bootstrap] starting fresh from {from_date}")

    to_date = to_date or date.today()
    logger.info(f"[bootstrap] fetching {from_date} → {to_date}")

    scraper = GithubArchiveScraper()
    success = 0
    failed = 0

    for raw in scraper.fetch_all_historical(from_date, to_date):
        normalized = normalize_draw(raw)
        if normalized is None:
            failed += 1
            continue

        is_valid, errors = validate_draw(normalized)
        if not is_valid:
            logger.debug(f"[bootstrap] skipping invalid draw: {errors}")
            failed += 1
            continue

        if upsert_draw(normalized):
            success += 1
        else:
            failed += 1

    logger.info(f"[bootstrap] done: {success} success, {failed} failed/skipped")
    return success
