"""Application configuration module.

Loads all settings from environment variables using pydantic-settings.
Provides type-safe, validated configuration for the entire application.

Example:
    >>> from app.core.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.app_name)
    AI Automation Platform
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All configuration is loaded from the .env file or actual environment
    variables. Sensitive values (API keys, secrets) must never be hard-coded.

    Attributes:
        app_name: Display name of the application.
        app_version: Current semantic version.
        app_env: Deployment environment (development/staging/production).
        debug: Enable debug mode and verbose output.
        secret_key: Secret key for JWT token signing (min 32 chars).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # Application
    # =========================================================================
    app_name: str = Field(default="AI Automation Platform")
    app_version: str = Field(default="1.0.0")
    app_env: Literal["development", "staging", "production"] = Field(
        default="development"
    )
    debug: bool = Field(default=False)
    secret_key: str = Field(..., min_length=32)

    # =========================================================================
    # API
    # =========================================================================
    api_v1_prefix: str = Field(default="/api/v1")
    allowed_hosts: list[str] = Field(default_factory=lambda: ["*"])

    # =========================================================================
    # Database
    # =========================================================================
    database_url: str = Field(...)
    database_pool_size: int = Field(default=10, ge=1, le=100)
    database_max_overflow: int = Field(default=20, ge=0, le=100)
    database_pool_timeout: int = Field(default=30, ge=5)
    database_echo: bool = Field(default=False)

    # =========================================================================
    # Redis
    # =========================================================================
    redis_url: str = Field(...)
    redis_ttl: int = Field(default=3600, ge=1)

    # =========================================================================
    # AI Providers
    # =========================================================================
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o")
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    google_api_key: str = Field(default="")
    google_model: str = Field(default="gemini-1.5-pro")
    google_embedding_model: str = Field(default="models/embedding-001")

    # =========================================================================
    # Provider Selection
    # =========================================================================
    llm_provider: Literal["openai", "google"] = Field(default="openai")
    embedding_provider: Literal["openai", "google"] = Field(default="openai")

    # =========================================================================
    # Vector Database
    # =========================================================================
    vector_db: Literal["faiss", "chroma"] = Field(default="chroma")
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    faiss_index_path: str = Field(default="./data/faiss")

    # =========================================================================
    # Facebook
    # =========================================================================
    facebook_app_id: str = Field(default="")
    facebook_app_secret: str = Field(default="")
    facebook_page_token: str = Field(default="")
    facebook_verify_token: str = Field(default="")
    facebook_webhook_secret: str = Field(default="")
    facebook_api_version: str = Field(default="v19.0")

    # =========================================================================
    # TikTok
    # =========================================================================
    tiktok_client_id: str = Field(default="")
    tiktok_client_secret: str = Field(default="")
    tiktok_redirect_uri: str = Field(default="")

    # =========================================================================
    # YouTube
    # =========================================================================
    youtube_client_id: str = Field(default="")
    youtube_client_secret: str = Field(default="")
    youtube_redirect_uri: str = Field(default="")

    # =========================================================================
    # Telegram
    # =========================================================================
    telegram_bot_token: str = Field(default="")
    telegram_webhook_url: str = Field(default="")

    # =========================================================================
    # Security
    # =========================================================================
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30, ge=1)
    refresh_token_expire_days: int = Field(default=7, ge=1)

    # =========================================================================
    # Logging
    # =========================================================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    log_format: Literal["json", "console"] = Field(default="json")
    log_file: str = Field(default="logs/app.log")

    # =========================================================================
    # RAG
    # =========================================================================
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)
    retriever_top_k: int = Field(default=5, ge=1, le=50)

    # =========================================================================
    # Rate Limiting
    # =========================================================================
    rate_limit_requests: int = Field(default=100, ge=1)
    rate_limit_window: int = Field(default=60, ge=1)

    # =========================================================================
    # HTTP Client
    # =========================================================================
    http_timeout: float = Field(default=30.0, ge=1.0)
    http_max_retries: int = Field(default=3, ge=0, le=10)

    @field_validator("database_url", mode="before")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL is not empty.

        Args:
            v: Database URL string.

        Returns:
            Validated URL string.

        Raises:
            ValueError: If the URL is empty.
        """
        if not v or not v.strip():
            raise ValueError("DATABASE_URL must not be empty")
        return v

    @field_validator("redis_url", mode="before")
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        """Validate Redis URL is not empty.

        Args:
            v: Redis URL string.

        Returns:
            Validated URL string.

        Raises:
            ValueError: If the URL is empty.
        """
        if not v or not v.strip():
            raise ValueError("REDIS_URL must not be empty")
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production environment.

        Returns:
            True if environment is production.
        """
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment.

        Returns:
            True if environment is development.
        """
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    Uses lru_cache to ensure settings are loaded only once per process.

    Returns:
        Settings: Singleton settings instance.
    """
    return Settings()
