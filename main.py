"""
Thai Lottery Prediction System
Entry point — bootstrap → train → predict → schedule
"""
import signal
import sys
import time
import argparse

from loguru import logger

from config import settings
from db.connection import engine
from db.models import Base


def init_db() -> None:
    """สร้าง tables ถ้ายังไม่มี (ใช้ alembic สำหรับ migration จริง)"""
    from alembic.config import Config
    from alembic import command

    alembic_cfg = Config("alembic.ini")
    try:
        command.upgrade(alembic_cfg, "head")
        logger.info("[main] DB migration: up to date")
    except Exception as e:
        logger.warning(f"[main] alembic migration failed, fallback to create_all: {e}")
        Base.metadata.create_all(engine)


def cmd_bootstrap(args) -> None:
    from pipeline.bootstrap import run_bootstrap
    from datetime import date

    from_date = date.fromisoformat(args.from_date) if args.from_date else None
    to_date = date.fromisoformat(args.to_date) if args.to_date else None
    n = run_bootstrap(from_date=from_date, to_date=to_date)
    logger.info(f"[main] bootstrap loaded {n} draws")


def cmd_train(args) -> None:
    from pipeline.train import run_training, run_all_targets
    if args.target == "all":
        run_all_targets()
    else:
        run_training(args.target)


def cmd_predict(args) -> None:
    from pipeline.predict import run_prediction, get_latest_predictions
    from datetime import date

    target_date = date.fromisoformat(args.date) if args.date else None
    results = run_prediction(target_draw_date=target_date)

    print("\n" + "=" * 60)
    print("  LOTTERY PREDICTIONS")
    print("=" * 60)
    for target_type, predictions in results.items():
        print(f"\n[{target_type.upper()}]")
        print(f"  {'Rank':<6} {'Number':<10} {'Score'}")
        print(f"  {'-'*30}")
        for rank, pred in enumerate(predictions[:10], 1):
            print(f"  {rank:<6} {pred['number']:<10} {pred['score']:.4f}")
    print("=" * 60)


def cmd_serve(args) -> None:
    """รัน scheduler แบบ continuous"""
    from scheduler.jobs import start_scheduler, stop_scheduler

    scheduler = start_scheduler()

    def _shutdown(sig, frame):
        logger.info("[main] shutting down...")
        stop_scheduler()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("[main] scheduler running — Ctrl+C to stop")
    while True:
        time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Thai Lottery Prediction System")
    sub = parser.add_subparsers(dest="command")

    # bootstrap
    bp = sub.add_parser("bootstrap", help="ดึงข้อมูลย้อนหลังจาก GitHub archive")
    bp.add_argument("--from-date", dest="from_date", help="YYYY-MM-DD")
    bp.add_argument("--to-date", dest="to_date", help="YYYY-MM-DD")

    # train
    tp = sub.add_parser("train", help="Train โมเดล")
    tp.add_argument("--target", default="all", choices=["all", "back2", "back3", "front3", "prize1_last2"])

    # predict
    pp = sub.add_parser("predict", help="สร้างคำทำนายงวดถัดไป")
    pp.add_argument("--date", help="target draw date YYYY-MM-DD")

    # serve
    sub.add_parser("serve", help="รัน scheduler อัตโนมัติ")

    args = parser.parse_args()

    # init DB ก่อนทุก command
    init_db()

    if args.command == "bootstrap":
        cmd_bootstrap(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        # default: serve
        logger.info("[main] no command specified — running full pipeline then serve")
        from etl.load import get_latest_draw_date
        latest = get_latest_draw_date()
        if latest is None:
            logger.info("[main] no data found — running bootstrap first")
            from pipeline.bootstrap import run_bootstrap
            run_bootstrap()

        logger.info("[main] running initial training...")
        from pipeline.train import run_all_targets
        run_all_targets()

        logger.info("[main] generating initial predictions...")
        from pipeline.predict import run_prediction
        run_prediction()

        cmd_serve(args)


if __name__ == "__main__":
    main()
