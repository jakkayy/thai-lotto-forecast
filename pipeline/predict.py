"""Prediction pipeline — สร้างคำทำนายสำหรับงวดถัดไป"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from config import settings
from db.connection import get_session
from db.models import LotteryDraw, Prediction
from etl.load import get_all_draws, get_latest_draw_date
from features.engineer import FeatureEngineer, TARGET_CANDIDATES
from models.baseline import BaselineModel, FrequencyBaselineModel
from models.statistical import StatisticalModel
from models.lgbm_model import LGBMModel
from models.lstm_model import LSTMModel
from models.ensemble import EnsembleModel

_TARGET_TYPES = ["back2", "back3", "front3", "prize1_last2"]
_TOP_N = 15


def _load_models(target_type: str) -> list:
    artifacts = settings.artifacts_path / target_type
    models = []

    lgbm = LGBMModel()
    lgbm_path = artifacts / "lgbm.bin"
    if lgbm_path.exists():
        lgbm.load(lgbm_path)
        models.append(lgbm)

    lstm = LSTMModel()
    lstm_path = artifacts / "lstm.bin"
    if lstm_path.exists():
        lstm.load(lstm_path)
        models.append(lstm)

    stat = StatisticalModel()
    freq = FrequencyBaselineModel()
    models.extend([stat, freq])

    ensemble = EnsembleModel(models=[lgbm, stat, freq] if lgbm_path.exists() else [stat, freq])
    models.append(ensemble)

    return models


def _next_draw_date(after: date | None = None) -> date:
    """หาวันงวดถัดไป (วันที่ 1 หรือ 16)"""
    today = after or date.today()
    if today.day < 1:
        return date(today.year, today.month, 1)
    elif today.day < 16:
        return date(today.year, today.month, 16)
    else:
        if today.month == 12:
            return date(today.year + 1, 1, 1)
        return date(today.year, today.month + 1, 1)


def run_prediction(target_draw_date: date | None = None) -> dict[str, list[dict]]:
    """
    สร้างคำทำนายสำหรับงวดถัดไป
    Returns dict[target_type → list of {number, score}]
    """
    draws = get_all_draws()
    if len(draws) < 10:
        logger.warning("[predict] not enough historical data")
        return {}

    if target_draw_date is None:
        target_draw_date = _next_draw_date(get_latest_draw_date())

    logger.info(f"[predict] generating predictions for {target_draw_date}")
    results = {}

    for target_type in _TARGET_TYPES:
        try:
            engineer = FeatureEngineer(draws)
            records = engineer.compute_all(target_type)
            df_all = engineer.to_dataframe(records)
            df_all = df_all.sort_values("draw_date").reset_index(drop=True)

            # สร้าง feature row สำหรับงวดที่จะทำนาย (ใช้ข้อมูลล่าสุดทั้งหมด)
            predict_records = _build_predict_features(draws, target_draw_date, target_type)
            if not predict_records:
                continue

            df_predict = pd.DataFrame(predict_records)
            for col in df_all.columns:
                if col not in df_predict.columns:
                    df_predict[col] = 0

            models = _load_models(target_type)
            # train models บน data ทั้งหมดก่อน predict
            for model in models:
                if hasattr(model, "_model") and model._model is None:
                    model.fit(df_all)
                elif hasattr(model, "_net") and model._net is None:
                    model.fit(df_all)
                elif model.name in ("statistical", "baseline_frequency"):
                    model.fit(df_all)
                elif model.name == "ensemble":
                    model.fit(df_all)

            # ใช้ ensemble เป็น primary
            ensemble = next((m for m in models if m.name == "ensemble"), models[-1])
            ranked = ensemble.rank_candidates(df_predict)
            top = ranked.head(_TOP_N)

            predictions = [
                {"number": row["candidate"], "score": round(float(row["score"]), 6)}
                for _, row in top.iterrows()
            ]

            results[target_type] = predictions
            _save_predictions(target_draw_date, target_type, "ensemble", predictions)

            logger.info(f"[predict] {target_type} top-5: {[p['number'] for p in predictions[:5]]}")

        except Exception as e:
            logger.error(f"[predict] {target_type} failed: {e}")

    return results


def _build_predict_features(
    draws: list[LotteryDraw],
    target_draw_date: date,
    target_type: str,
) -> list[dict]:
    """สร้าง feature rows สำหรับทุก candidate ของงวดที่จะทำนาย"""
    from features.engineer import TARGET_CANDIDATES

    engineer = FeatureEngineer(draws)
    df_hist = engineer.df
    candidates = TARGET_CANDIDATES[target_type]
    rows = []

    context = engineer._context_features(target_draw_date, df_hist, target_type)
    for cand in candidates:
        cand_feats = engineer._candidate_features(cand, df_hist, target_type)
        row = {"draw_date": target_draw_date, "candidate": cand, "is_winner": 0}
        row.update(context)
        row.update(cand_feats)
        rows.append(row)
    return rows


def _save_predictions(
    target_draw_date: date,
    target_type: str,
    model_name: str,
    predictions: list[dict],
) -> None:
    try:
        row = {
            "target_draw_date": target_draw_date,
            "target_type": target_type,
            "model_name": model_name,
            "predicted_numbers": predictions,
        }
        with get_session() as session:
            session.execute(
                insert(Prediction)
                .values(**row)
                .on_conflict_do_update(
                    constraint="uq_prediction_draw_target_model",
                    set_={"predicted_numbers": predictions},
                )
            )
    except Exception as e:
        logger.error(f"[predict] save failed: {e}")


def get_latest_predictions(target_draw_date: date | None = None) -> dict[str, list[dict]]:
    """ดึงคำทำนายล่าสุดจาก DB"""
    with get_session() as session:
        q = session.query(Prediction).order_by(Prediction.target_draw_date.desc())
        if target_draw_date:
            q = q.filter(Prediction.target_draw_date == target_draw_date)
        rows = q.all()

    results = {}
    for row in rows:
        key = f"{row.target_draw_date}_{row.target_type}"
        if key not in results:
            results[row.target_type] = row.predicted_numbers
    return results
