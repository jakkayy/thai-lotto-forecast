from abc import ABC, abstractmethod
from datetime import date
from typing import Any

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import settings


class BaseScraper(ABC):
    name: str = "base"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; LottoBot/1.0)",
            "Accept": "application/json",
        })
        self.timeout = settings.REQUEST_TIMEOUT

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, ValueError)),
        reraise=True,
    )
    def _get(self, url: str, **kwargs) -> requests.Response:
        resp = self.session.get(url, timeout=self.timeout, **kwargs)
        resp.raise_for_status()
        return resp

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, ValueError)),
        reraise=True,
    )
    def _post(self, url: str, **kwargs) -> requests.Response:
        resp = self.session.post(url, timeout=self.timeout, **kwargs)
        resp.raise_for_status()
        return resp

    @abstractmethod
    def fetch_latest(self) -> dict[str, Any] | None:
        """ดึงผลงวดล่าสุด คืน raw dict หรือ None ถ้าล้มเหลว"""

    @abstractmethod
    def fetch_by_date(self, draw_date: date) -> dict[str, Any] | None:
        """ดึงผลตามวันที่ คืน raw dict หรือ None ถ้าล้มเหลว"""
