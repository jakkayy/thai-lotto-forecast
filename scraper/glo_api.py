from datetime import date
from typing import Any

from loguru import logger

from config import settings
from scraper.base import BaseScraper


def _be_to_ce_year(be_year: int) -> int:
    """แปลง พ.ศ. → ค.ศ."""
    return be_year - 543


def _parse_glo_date(date_str: str) -> date | None:
    """แปลง '16/04/2569' (พ.ศ.) → date object (ค.ศ.)"""
    try:
        parts = date_str.strip().split("/")
        day, month, be_year = int(parts[0]), int(parts[1]), int(parts[2])
        return date(_be_to_ce_year(be_year), month, day)
    except Exception:
        return None


class GloApiScraper(BaseScraper):
    """ดึงข้อมูลจาก Official GLO API (glo.or.th)"""

    name = "glo_api"

    def fetch_latest(self) -> dict[str, Any] | None:
        try:
            resp = self._post(settings.GLO_LATEST_URL, json={})
            data = resp.json()
            logger.info(f"[{self.name}] fetch_latest OK")
            return {"source": self.name, "raw": data, "type": "latest"}
        except Exception as e:
            logger.warning(f"[{self.name}] fetch_latest failed: {e}")
            return None

    def fetch_by_date(self, draw_date: date) -> dict[str, Any] | None:
        try:
            be_year = draw_date.year + 543
            payload = {
                "date": str(draw_date.day).zfill(2),
                "month": str(draw_date.month).zfill(2),
                "year": str(be_year),
            }
            resp = self._post(settings.GLO_RESULT_URL, json=payload)
            data = resp.json()
            logger.info(f"[{self.name}] fetch_by_date {draw_date} OK")
            return {"source": self.name, "raw": data, "type": "by_date", "requested_date": str(draw_date)}
        except Exception as e:
            logger.warning(f"[{self.name}] fetch_by_date {draw_date} failed: {e}")
            return None

    @staticmethod
    def normalize(raw_response: dict) -> dict[str, Any] | None:
        """แปลง GLO API response → normalized dict"""
        try:
            data = raw_response.get("raw", {})

            # หา prizes ใน response (GLO API มีหลาย format)
            lotto = (
                data.get("response", {}).get("lotto") or
                data.get("lotto") or
                data.get("data", {}).get("lotto") or
                data
            )

            prizes = lotto.get("prizes", {})
            date_str = lotto.get("date", "")
            draw_date = _parse_glo_date(date_str)

            if draw_date is None:
                logger.warning(f"[glo_api] cannot parse date: {date_str}")
                return None

            def _get_numbers(key: str) -> list[str]:
                p = prizes.get(key, {})
                if isinstance(p, dict):
                    nums = p.get("number", p.get("numbers", []))
                    return [str(n).zfill(6) for n in (nums if isinstance(nums, list) else [nums])]
                return []

            def _get_numbers_3(key: str) -> list[str]:
                p = prizes.get(key, {})
                if isinstance(p, dict):
                    front = p.get("front", [])
                    back = p.get("back", [])
                    nums = front + back if (front or back) else p.get("number", [])
                    return [str(n).zfill(3) for n in (nums if isinstance(nums, list) else [nums])]
                return []

            first_nums = _get_numbers("1") or _get_numbers("first")
            back2_raw = prizes.get("back2", prizes.get("lastTwo", {}))
            back2_nums = back2_raw.get("number", "") if isinstance(back2_raw, dict) else ""
            if isinstance(back2_nums, list):
                back2_nums = back2_nums[0] if back2_nums else ""
            back2 = str(back2_nums).zfill(2) if back2_nums else None

            return {
                "draw_date": draw_date,
                "prize_1": first_nums[0] if first_nums else None,
                "prize_near_1": _get_numbers("1near") or _get_numbers("nearFirst"),
                "prize_2": _get_numbers("2") or _get_numbers("second"),
                "prize_3": _get_numbers("3") or _get_numbers("third"),
                "prize_4": _get_numbers("4") or _get_numbers("fourth"),
                "prize_5": _get_numbers("5") or _get_numbers("fifth"),
                "prize_front_3": _get_numbers_3("front3") or _get_numbers_3("lastThree"),
                "prize_back_3": _get_numbers_3("back3") or _get_numbers_3("lastThree"),
                "prize_back_2": back2,
                "source": "glo_api",
                "raw_data": raw_response.get("raw"),
            }
        except Exception as e:
            logger.error(f"[glo_api] normalize failed: {e}")
            return None
