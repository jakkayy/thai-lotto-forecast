from datetime import date
from typing import Any

from loguru import logger

from config import settings
from scraper.base import BaseScraper


class RayrifffyApiScraper(BaseScraper):
    """Fallback: rayriffy Thai Lotto API (community, free)"""

    name = "rayriffy_api"

    def fetch_latest(self) -> dict[str, Any] | None:
        try:
            url = f"{settings.RAYRIFFY_API_URL}/latest"
            resp = self._get(url)
            data = resp.json()
            logger.info(f"[{self.name}] fetch_latest OK")
            return {"source": self.name, "raw": data}
        except Exception as e:
            logger.warning(f"[{self.name}] fetch_latest failed: {e}")
            return None

    def fetch_by_date(self, draw_date: date) -> dict[str, Any] | None:
        try:
            date_str = draw_date.strftime("%Y-%m-%d")
            url = f"{settings.RAYRIFFY_API_URL}/{date_str}"
            resp = self._get(url)
            data = resp.json()
            logger.info(f"[{self.name}] fetch_by_date {draw_date} OK")
            return {"source": self.name, "raw": data}
        except Exception as e:
            logger.warning(f"[{self.name}] fetch_by_date {draw_date} failed: {e}")
            return None

    @staticmethod
    def normalize(raw_response: dict) -> dict[str, Any] | None:
        try:
            data = raw_response.get("raw", {})
            if data.get("status") != "ok":
                return None

            resp_data = data.get("response", {}).get("data", data.get("data", {}))
            prizes = resp_data.get("prizes", {})

            date_str = resp_data.get("date", "")
            try:
                parts = date_str.split("/")
                draw_date = date(int(parts[2]) - 543, int(parts[1]), int(parts[0]))
            except Exception:
                return None

            def _nums(key: str, pad: int = 6) -> list[str]:
                p = prizes.get(key, {})
                nums = p.get("number", []) if isinstance(p, dict) else []
                return [str(n).zfill(pad) for n in (nums if isinstance(nums, list) else [nums])]

            last_three = prizes.get("lastThree", {})
            front3 = [str(n).zfill(3) for n in last_three.get("front", [])]
            back3 = [str(n).zfill(3) for n in last_three.get("back", [])]

            last_two = prizes.get("lastTwo", {})
            back2_raw = last_two.get("number", "") if isinstance(last_two, dict) else ""
            back2 = str(back2_raw).zfill(2) if back2_raw else None

            return {
                "draw_date": draw_date,
                "prize_1": _nums("first")[0] if _nums("first") else None,
                "prize_near_1": _nums("nearFirst"),
                "prize_2": _nums("second"),
                "prize_3": _nums("third"),
                "prize_4": _nums("fourth"),
                "prize_5": _nums("fifth"),
                "prize_front_3": front3,
                "prize_back_3": back3,
                "prize_back_2": back2,
                "source": "rayriffy_api",
                "raw_data": raw_response.get("raw"),
            }
        except Exception as e:
            logger.error(f"[rayriffy_api] normalize failed: {e}")
            return None
