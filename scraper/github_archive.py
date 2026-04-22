"""Bootstrap scraper — ดึงข้อมูลย้อนหลังจาก vicha-w/thai-lotto-archive"""
from datetime import date
from typing import Any, Generator

from loguru import logger

from scraper.base import BaseScraper

_ARCHIVE_BASE = (
    "https://raw.githubusercontent.com/"
    "vicha-w/thai-lotto-archive/master/lottonumbers"
)

_ARCHIVE_START = date(2006, 12, 30)


def _all_draw_dates(from_date: date, to_date: date) -> list[date]:
    """สร้าง list ของ draw dates ทุกวันที่ 1 และ 16"""
    draws = []
    current = date(from_date.year, from_date.month, 1)
    while current <= to_date:
        for day in (1, 16):
            try:
                d = date(current.year, current.month, day)
                if from_date <= d <= to_date:
                    draws.append(d)
            except ValueError:
                pass
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return sorted(draws)


def _parse_txt(text: str) -> dict[str, list[str]]:
    """
    แปลง text format → dict
    format: LABEL num1 num2 ...  (space-separated, ไม่มี colon)
    บรรทัดแรกเป็น URL (source) ข้ามไป
    """
    prizes: dict[str, list[str]] = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("http"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        label = parts[0].upper()
        values = parts[1:]
        prizes[label] = values
    return prizes


class GithubArchiveScraper(BaseScraper):
    """ดึงข้อมูลย้อนหลังจาก GitHub archive — ใช้ตอน bootstrap เท่านั้น"""

    name = "github_archive"

    def fetch_latest(self) -> dict[str, Any] | None:
        return None

    def fetch_by_date(self, draw_date: date) -> dict[str, Any] | None:
        url = f"{_ARCHIVE_BASE}/{draw_date.isoformat()}.txt"
        try:
            resp = self._get(url)
            if resp.status_code == 200 and resp.text.strip():
                logger.debug(f"[{self.name}] {draw_date} OK")
                return {"source": self.name, "raw": resp.text, "draw_date": str(draw_date)}
        except Exception as e:
            logger.debug(f"[{self.name}] {draw_date} failed: {e}")
        return None

    def fetch_all_historical(
        self,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        from_date = from_date or _ARCHIVE_START
        to_date = to_date or date.today()
        draws = _all_draw_dates(from_date, to_date)
        total = len(draws)
        logger.info(f"[{self.name}] bootstrapping {total} draw dates ({from_date} → {to_date})")

        for i, d in enumerate(draws, 1):
            result = self.fetch_by_date(d)
            if result:
                yield result
            if i % 50 == 0:
                logger.info(f"[{self.name}] progress: {i}/{total}")

    @staticmethod
    def normalize(raw_response: dict) -> dict[str, Any] | None:
        try:
            raw_text = raw_response.get("raw", "")
            draw_date_str = raw_response.get("draw_date", "")

            draw_date = date.fromisoformat(draw_date_str) if draw_date_str else None
            if draw_date is None:
                return None

            prizes = _parse_txt(raw_text)

            def _pad(nums: list[str], pad: int) -> list[str]:
                return [str(n).zfill(pad) for n in nums if n]

            first = prizes.get("FIRST", [])

            # เลขท้าย 3 ตัว: post-2015 ใช้ THREE_FIRST / THREE_LAST
            # pre-2015 ใช้ THREE (4 ตัว: 2 หน้า + 2 หลัง)
            front3 = _pad(prizes.get("THREE_FIRST", []), 3)
            back3 = _pad(prizes.get("THREE_LAST", []), 3)
            if not front3 and not back3 and "THREE" in prizes:
                three = prizes["THREE"]
                mid = len(three) // 2
                front3 = _pad(three[:mid], 3)
                back3 = _pad(three[mid:], 3)

            two = prizes.get("TWO", [])
            back2 = str(two[0]).zfill(2) if two else None

            return {
                "draw_date": draw_date,
                "prize_1": first[0].zfill(6) if first else None,
                "prize_near_1": _pad(prizes.get("NEAR_FIRST", []), 6),
                "prize_2": _pad(prizes.get("SECOND", []), 6),
                "prize_3": _pad(prizes.get("THIRD", []), 6),
                "prize_4": _pad(prizes.get("FOURTH", []), 6),
                "prize_5": _pad(prizes.get("FIFTH", []), 6),
                "prize_front_3": front3,
                "prize_back_3": back3,
                "prize_back_2": back2,
                "source": "github_archive",
                "raw_data": {"text": raw_text},
            }
        except Exception as e:
            logger.error(f"[github_archive] normalize failed: {e}")
            return None
