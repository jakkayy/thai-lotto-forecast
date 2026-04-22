from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    DATABASE_URL: str = "postgresql://lotto:lotto@localhost:5432/lotto"

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5001"
    MLFLOW_EXPERIMENT_NAME: str = "thai-lottery"

    # Scraper endpoints
    GLO_LATEST_URL: str = "https://www.glo.or.th/api/lottery/getLatestLottery"
    GLO_RESULT_URL: str = "https://www.glo.or.th/api/checking/getLotteryResult"
    RAYRIFFY_API_URL: str = "https://lotto.api.rayriffy.com"
    GLO_WEBSITE_URL: str = "https://www.glo.or.th/check_result/index.php"
    GITHUB_ARCHIVE_OWNER: str = "vicha-w"
    GITHUB_ARCHIVE_REPO: str = "thai-lotto-archive"

    # Scraper settings
    REQUEST_TIMEOUT: int = 30
    REQUEST_RETRIES: int = 3

    # Scheduler
    FETCH_HOUR: int = 9
    FETCH_MINUTE: int = 0

    # Paths
    ARTIFACTS_DIR: str = "artifacts"

    @property
    def artifacts_path(self) -> Path:
        p = Path(self.ARTIFACTS_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
