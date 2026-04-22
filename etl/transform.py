"""Normalize raw scraper output ผ่าน scraper-specific normalizer"""
from datetime import date
from typing import Any

from loguru import logger

from scraper.glo_api import GloApiScraper
from scraper.rayriffy_api import RayrifffyApiScraper
from scraper.glo_website import GloWebsiteScraper
from scraper.github_archive import GithubArchiveScraper

_NORMALIZERS = {
    "glo_api": GloApiScraper.normalize,
    "rayriffy_api": RayrifffyApiScraper.normalize,
    "glo_website": GloWebsiteScraper.normalize,
    "github_archive": GithubArchiveScraper.normalize,
}


def normalize_draw(raw_response: dict) -> dict[str, Any] | None:
    source = raw_response.get("source", "")
    normalizer = _NORMALIZERS.get(source)
    if normalizer is None:
        logger.error(f"[etl.transform] unknown source: {source}")
        return None
    return normalizer(raw_response)


def cross_validate_draws(draw_a: dict, draw_b: dict) -> bool:
    """ตรวจว่า draw สองชุดจาก source ต่างกัน ตรงกันในส่วนสำคัญไหม"""
    if draw_a is None or draw_b is None:
        return False
    key_fields = ("draw_date", "prize_1", "prize_back_2")
    for field in key_fields:
        if draw_a.get(field) != draw_b.get(field):
            logger.warning(
                f"[etl.transform] cross-validate mismatch on '{field}': "
                f"{draw_a.get(field)} vs {draw_b.get(field)}"
            )
            return False
    return True


def merge_draws(primary: dict, fallback: dict) -> dict:
    """รวมข้อมูลจาก 2 sources — ใช้ primary เป็นหลัก เติม fallback เฉพาะ field ที่ None"""
    merged = dict(primary)
    for key, val in fallback.items():
        if merged.get(key) is None and val is not None:
            merged[key] = val
    return merged
