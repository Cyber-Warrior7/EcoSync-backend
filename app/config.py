import os
from functools import lru_cache
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Runtime configuration pulled from environment variables."""

    def __init__(self) -> None:
        self.project_name: str = os.getenv("PROJECT_NAME", "EcoSync API")
        self.environment: str = os.getenv("ENV", "development")
        self.offline_mode: bool = os.getenv("OFFLINE_MODE", "false").lower() == "true"
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        self.default_reward_per_post: int = int(os.getenv("DEFAULT_REWARD_PER_POST", "10"))
        self.default_reward_per_found_item: int = int(os.getenv("DEFAULT_REWARD_PER_FOUND_ITEM", "8"))
        cors_origins_raw = os.getenv("CORS_ORIGINS", "*")
        self.cors_origins: List[str] = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]
        self.jwt_secret: str = os.getenv("JWT_SECRET", "change-me")
        self.jwt_exp_hours: int = int(os.getenv("JWT_EXP_HOURS", "12"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
