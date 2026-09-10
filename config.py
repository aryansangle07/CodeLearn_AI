from functools import lru_cache
from typing import List, Optional
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App & Server Settings
    APP_NAME: str = "CodeLearn AI"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    CORS_ORIGINS: List[str] = ["http://localhost:8501", "http://localhost:3000"]

    # PostgreSQL Database
    DATABASE_URL: Optional[str] = None
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "codelearn_ai"

    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30

    # Embeddings & Index Directory
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    INDEX_STORAGE_DIR: str = "./data/indices"
    MAX_CHUNKS_PER_REPO: int = 2000

    # Ingestion Boundary Limits
    MAX_REPOSITORY_SIZE_MB: int = 50
    MAX_FILE_SIZE_MB: int = 2
    MAX_FILES_PER_REPOSITORY: int = 500
    MAX_QUERY_LENGTH: int = 1000

    # Free-Tier LLM Limits
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "qwen/qwen3.8-27b"
    GROQ_RPM_LIMIT: int = 30
    GROQ_RPD_LIMIT: int = 14400

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_RPM_LIMIT: int = 15
    GEMINI_RPD_LIMIT: int = 1500

    # Auto-Seed Demo Repositories (Opt-In Flag)
    SEED_DEMO_REPOS: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @model_validator(mode="after")
    def assemble_database_url(self) -> "Settings":
        if not self.DATABASE_URL:
            # Assemble connection string from decomposed parts
            # URL-encode password in case of special characters
            import urllib.parse
            user = urllib.parse.quote_plus(self.POSTGRES_USER)
            password = urllib.parse.quote_plus(self.POSTGRES_PASSWORD)
            host = self.POSTGRES_HOST
            port = self.POSTGRES_PORT
            db = self.POSTGRES_DB
            self.DATABASE_URL = f"postgresql://{user}:{password}@{host}:{port}/{db}"
        return self


@lru_cache()
def get_settings() -> Settings:
    return Settings()
