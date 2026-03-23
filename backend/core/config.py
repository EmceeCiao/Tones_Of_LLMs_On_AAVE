from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import json

class Settings(BaseSettings):
    DATABASE_URL: Optional[str] = "sqlite:///./app.db"
    FIREBASE_SERVICE_ACCOUNT_PATH: Optional[str] = "/Users/kellychen/repos/Tones_Of_LLMs_On_AAVE/backend/firebase_service_acc.json"
    DEBUG: bool = False
    APP_ENV: str = "dev"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()