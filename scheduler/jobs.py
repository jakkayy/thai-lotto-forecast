"""APScheduler jobs — รันอัตโนมัติวันที่ 2 และ 17 ของทุกเดือน เวลา 09:00"""
from datetime import date

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from config import settings

_scheduler: BackgroundScheduler | None = None


def fetch_and_process_job() -> None:
    """
    Job หลัก: ดึงผลงวดล่าสุด → ETL → retrain → predict
    รันหลังงวดออก 1 วัน (วันที่ 2 และ 17) เวลา 09:00
    """
    logger.info("[scheduler] === fetch_and_process_job started ===")

    # 1. ดึงผลงวดล่าสุด (ออกวันที่ 1 หรือ 16)
    _fetch_latest_draw()

    # 2. Retrain models
    _retrain_models()

    # 3. สร้างคำทำนายงวดถัดไป
    _generate_predictions()

    logger.info("[scheduler] === fetch_and_process_job completed ===")


def _fetch_latest_draw() -> None:
    from scraper.glo_api import GloApiScraper
    from scraper.rayriffy_api import RayrifffyApiScraper
    from scraper.glo_website import GloWebsiteScraper
    from etl.transform import normalize_draw, cross_validate_draws
    from etl.validate import validate_draw
    from etl.load import upsert_draw

    scrapers_with_normalizers = [
        (GloApiScraper(), GloApiScraper.normalize),
        (RayrifffyApiScraper(), RayrifffyApiScraper.normalize),
        (GloWebsiteScraper(), GloWebsiteScraper.normalize),
    ]

    draws = []
    for scraper, _ in scrapers_with_normalizers:
        raw = scraper.fetch_latest()
        if raw:
            normalized = normalize_draw(raw)
            if normalized:
                draws.append(normalized)
        if len(draws) >= 2:
            break

    if not draws:
        logger.error("[scheduler] all sources failed — no draw fetched")
        return

    # cross-validate ถ้าได้ >= 2 sources
    if len(draws) >= 2:
        if not cross_validate_draws(draws[0], draws[1]):
            logger.warning("[scheduler] source mismatch — using primary source only")

    draw = draws[0]
    is_valid, errors = validate_draw(draw)
    if not is_valid:
        logger.error(f"[scheduler] invalid draw: {errors}")
        return

    upsert_draw(draw)
    logger.info(f"[scheduler] fetched draw {draw.get('draw_date')}")


def _retrain_models() -> None:
    try:
        from pipeline.train import run_all_targets
        run_all_targets()
    except Exception as e:
        logger.error(f"[scheduler] retrain failed: {e}")


def _generate_predictions() -> None:
    try:
        from pipeline.predict import run_prediction
        results = run_prediction()
        for target, preds in results.items():
            top5 = [p["number"] for p in preds[:5]]
            logger.info(f"[scheduler] {target} top-5: {top5}")
    except Exception as e:
        logger.error(f"[scheduler] prediction failed: {e}")


def start_scheduler() -> BackgroundScheduler:
    global _scheduler

    _scheduler = BackgroundScheduler(timezone="Asia/Bangkok")

    # รันวันที่ 2 และ 17 เวลา 09:00 (Bangkok time)
    _scheduler.add_job(
        fetch_and_process_job,
        trigger=CronTrigger(
            day="2,17",
            hour=settings.FETCH_HOUR,
            minute=settings.FETCH_MINUTE,
            timezone="Asia/Bangkok",
        ),
        id="fetch_and_process",
        name="Fetch lottery result, retrain, predict",
        replace_existing=True,
        misfire_grace_time=3600,  # ถ้า miss ไม่เกิน 1 ชั่วโมงยังรันได้
    )

    _scheduler.start()
    logger.info(
        f"[scheduler] started — job runs on day 2,17 at "
        f"{settings.FETCH_HOUR:02d}:{settings.FETCH_MINUTE:02d} Bangkok time"
    )
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("[scheduler] stopped")
