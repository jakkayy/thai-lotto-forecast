from datetime import date
from typing import Any

from bs4 import BeautifulSoup
from loguru import logger

from config import settings
from scraper.base import BaseScraper


class GloWebsiteScraper(BaseScraper):
    """Last resort: scrape glo.or.th HTML เมื่อ API ทุก sources ล้มเหลว"""

    name = "glo_website"

    def fetch_latest(self) -> dict[str, Any] | None:
        try:
            resp = self._get(settings.GLO_WEBSITE_URL)
            soup = BeautifulSoup(resp.text, "lxml")
            return {"source": self.name, "raw": self._parse_html(soup), "type": "latest"}
        except Exception as e:
            logger.warning(f"[{self.name}] fetch_latest failed: {e}")
            return None

    def fetch_by_date(self, draw_date: date) -> dict[str, Any] | None:
        try:
            be_year = draw_date.year + 543
            params = {
                "date": str(draw_date.day).zfill(2),
                "month": str(draw_date.month).zfill(2),
                "year": str(be_year),
            }
            resp = self._get(settings.GLO_WEBSITE_URL, params=params)
            soup = BeautifulSoup(resp.text, "lxml")
            return {"source": self.name, "raw": self._parse_html(soup), "type": "by_date"}
        except Exception as e:
            logger.warning(f"[{self.name}] fetch_by_date {draw_date} failed: {e}")
            return None

    def _parse_html(self, soup: BeautifulSoup) -> dict:
        """Parse HTML ของ glo.or.th — flexible เพราะ structure อาจเปลี่ยน"""
        result = {}
        try:
            # รางวัลที่ 1
            prize1_el = soup.select_one(".prize1 .number, #prize1 .number, [data-prize='1'] .number")
            if prize1_el:
                result["prize_1"] = prize1_el.get_text(strip=True).zfill(6)

            # เลขท้าย 2 ตัว
            back2_el = soup.select_one(".last2 .number, #last2digit .number, [data-prize='last2'] .number")
            if back2_el:
                result["prize_back_2"] = back2_el.get_text(strip=True).zfill(2)

            # เลขท้าย 3 ตัว
            back3_els = soup.select(".last3 .number, #last3digit .number, [data-prize='last3'] .number")
            if back3_els:
                result["prize_back_3"] = [el.get_text(strip=True).zfill(3) for el in back3_els]

            # เลขหน้า 3 ตัว
            front3_els = soup.select(".front3 .number, #front3digit .number, [data-prize='front3'] .number")
            if front3_els:
                result["prize_front_3"] = [el.get_text(strip=True).zfill(3) for el in front3_els]

        except Exception as e:
            logger.error(f"[{self.name}] HTML parse error: {e}")

        return result

    @staticmethod
    def normalize(raw_response: dict) -> dict[str, Any] | None:
        raw = raw_response.get("raw", {})
        if not raw:
            return None
        return {
            "prize_1": raw.get("prize_1"),
            "prize_near_1": raw.get("prize_near_1", []),
            "prize_2": raw.get("prize_2", []),
            "prize_3": raw.get("prize_3", []),
            "prize_4": raw.get("prize_4", []),
            "prize_5": raw.get("prize_5", []),
            "prize_front_3": raw.get("prize_front_3", []),
            "prize_back_3": raw.get("prize_back_3", []),
            "prize_back_2": raw.get("prize_back_2"),
            "source": "glo_website",
            "raw_data": raw,
        }
