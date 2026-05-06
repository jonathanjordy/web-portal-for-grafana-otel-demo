from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ClickHouse
    clickhouse_host:     str = "10.184.0.14"
    clickhouse_port:     int = 8123
    clickhouse_user:     str = "default"
    clickhouse_password: str = ""
    clickhouse_database: str = "otel"

    # Gemini API
    gemini_api_key: str = ""

    # App
    debug: bool = False

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()