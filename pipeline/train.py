"""Training pipeline พร้อม Walk-forward validation"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import socket

import mlflow
import pandas as pd
from loguru import logger

from config import settings
from db.models import LotteryDraw
from etl.load import get_all_draws
from features.engineer import FeatureEngineer, TARGET_CANDIDATES
from models.baseline import BaselineModel, FrequencyBaselineModel
from models.statistical import StatisticalModel
from models.lgbm_model import LGBMModel
from models.lstm_model import LSTMModel
from models.ensemble import EnsembleModel
from pipeline.evaluate import evaluate_predictions, save_performance

_TARGET_TYPES = ["back2", "back3", "front3", "prize1_last2"]
_MIN_TRAIN_DRAWS = 50
_WALK_FORWARD_START = 200  # train บน 200 งวดแรก test บน ~200 งวดหลัง


def _mlflow_server_available() -> bool:
    try:
        host, port = "localhost", 5001
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _setup_mlflow() -> None:
    if _mlflow_server_available():
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        logger.info("[train] MLflow: using server")
    else:
        mlflow.set_tracking_uri("file:./mlruns")
        logger.info("[train] MLflow: server unavailable, using local ./mlruns")
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)


def _build_walkforward_models() -> list:
    """Models สำหรับ walk-forward — ไม่มี LSTM เพราะช้าเกินไป"""
    lgbm = LGBMModel()
    stat = StatisticalModel()
    freq = FrequencyBaselineModel()
    ensemble = EnsembleModel(models=[lgbm, stat, freq], weights=[0.5, 0.3, 0.2])
    return [BaselineModel(), freq, stat, lgbm, ensemble]


def _build_final_models() -> list:
    """Models สำหรับ final train — รวม LSTM ด้วย"""
    lgbm = LGBMModel()
    stat = StatisticalModel()
    freq = FrequencyBaselineModel()
    lstm = LSTMModel(epochs=20)
    ensemble = EnsembleModel(models=[lgbm, stat, freq], weights=[0.5, 0.3, 0.2])
    return [BaselineModel(), freq, stat, lgbm, lstm, ensemble]


def run_training(target_type: str = "back2", save_models: bool = True) -> dict[str, Any]:
    """
    Walk-forward training + evaluation
    Returns dict ของ metrics สรุปทุก model
    """
    draws = get_all_draws()
    if len(draws) < _MIN_TRAIN_DRAWS:
        logger.warning(f"[train] only {len(draws)} draws — need at least {_MIN_TRAIN_DRAWS}")
        return {}

    logger.info(f"[train] {len(draws)} draws loaded, target={target_type}")

    engineer = FeatureEngineer(draws)
    records = engineer.compute_all(target_type)
    df_all = engineer.to_dataframe(records)
    df_all = df_all.sort_values("draw_date").reset_index(drop=True)

    draw_dates = sorted(df_all["draw_date"].unique())
    test_dates = draw_dates[_WALK_FORWARD_START:]

    if len(test_dates) == 0:
        logger.warning(f"[train] not enough draws for walk-forward (need >{_WALK_FORWARD_START})")
        return {}

    logger.info(f"[train] walk-forward: {len(test_dates)} test steps")

    _setup_mlflow()

    models = _build_walkforward_models()
    all_metrics: dict[str, list] = {m.name: [] for m in models}

    for step, test_date in enumerate(test_dates):
        df_train = df_all[df_all["draw_date"] < test_date]
        df_test = df_all[df_all["draw_date"] == test_date]

        winners = set(df_test[df_test["is_winner"] == 1]["candidate"].tolist())
        if not winners:
            continue

        for model in models:
            try:
                model.fit(df_train)
                ranked = model.rank_candidates(df_test)
                top_candidates = ranked["candidate"].tolist()
                metrics = evaluate_predictions(top_candidates, winners)
                all_metrics[model.name].append(metrics)
                save_performance(model.name, target_type, test_date, top_candidates, winners)
            except Exception as e:
                logger.error(f"[train] {model.name} step {step} failed: {e}")

        if (step + 1) % 20 == 0:
            logger.info(f"[train] walk-forward step {step+1}/{len(test_dates)}")

    # สรุป metrics รวม
    summary = {}
    for model_name, step_metrics in all_metrics.items():
        if not step_metrics:
            continue
        avg = {
            k: sum(m[k] for m in step_metrics) / len(step_metrics)
            for k in step_metrics[0]
        }
        summary[model_name] = avg
        logger.info(f"[train] {model_name}: {avg}")

    try:
        with mlflow.start_run(run_name=f"walk_forward_{target_type}"):
            mlflow.log_param("target_type", target_type)
            mlflow.log_param("n_test_steps", len(test_dates))
            mlflow.log_param("n_train_draws", len(draws))
            for model_name, avg in summary.items():
                for k, v in avg.items():
                    mlflow.log_metric(f"{model_name}_{k}", round(v, 4))
    except Exception as e:
        logger.warning(f"[train] MLflow logging skipped: {e}")

    # Final train บน data ทั้งหมด รวม LSTM แล้ว save
    if save_models:
        artifacts = settings.artifacts_path / target_type
        artifacts.mkdir(parents=True, exist_ok=True)
        logger.info(f"[train] final training (with LSTM) on all {len(draws)} draws...")
        for model in _build_final_models():
            try:
                model.fit(df_all)
                model.save(artifacts / f"{model.name}.bin")
                logger.info(f"[train] saved {model.name} → {artifacts}/{model.name}.bin")
            except Exception as e:
                logger.error(f"[train] save {model.name} failed: {e}")

    return summary


def run_all_targets() -> dict[str, dict]:
    results = {}
    for target in _TARGET_TYPES:
        logger.info(f"[train] === {target} ===")
        results[target] = run_training(target)
    return results
