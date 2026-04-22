"""Evaluation metrics สำหรับ lottery prediction"""
from datetime import date

from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from db.connection import get_session
from db.models import ModelPerformance


def evaluate_predictions(
    ranked_candidates: list[str],
    actual_winners: set[str],
    top_k: int = 10,
) -> dict[str, float]:
    """
    คำนวณ metrics
    - hit_top1: ถูกใน rank 1
    - hit_top5: ถูกใน top-5
    - hit_top10: ถูกใน top-10
    - correct_rank: rank ของ winner ที่ถูก (None = ไม่ถูก)
    - mrr: Mean Reciprocal Rank
    """
    correct_rank = None
    for rank, cand in enumerate(ranked_candidates[:top_k * 2], start=1):
        if cand in actual_winners:
            correct_rank = rank
            break

    return {
        "hit_top1": float(correct_rank == 1),
        "hit_top5": float(correct_rank is not None and correct_rank <= 5),
        "hit_top10": float(correct_rank is not None and correct_rank <= 10),
        "correct_rank": float(correct_rank) if correct_rank else 0.0,
        "mrr": float(1.0 / correct_rank) if correct_rank else 0.0,
    }


def save_performance(
    model_name: str,
    target_type: str,
    eval_date: date,
    ranked_candidates: list[str],
    actual_winners: set[str],
) -> None:
    metrics = evaluate_predictions(ranked_candidates, actual_winners)
    correct_rank = int(metrics["correct_rank"]) if metrics["correct_rank"] > 0 else None

    row = {
        "model_name": model_name,
        "target_type": target_type,
        "eval_draw_date": eval_date,
        "hit_top1": bool(metrics["hit_top1"]),
        "hit_top5": bool(metrics["hit_top5"]),
        "hit_top10": bool(metrics["hit_top10"]),
        "correct_rank": correct_rank,
        "metrics": metrics,
    }

    try:
        with get_session() as session:
            session.execute(insert(ModelPerformance).values(**row).on_conflict_do_nothing())
    except Exception as e:
        logger.error(f"[evaluate] save_performance failed: {e}")
