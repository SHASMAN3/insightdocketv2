"""
Application configuration via Pydantic BaseSettings.

All settings are loaded from environment variables (or .env file).
Interview note: Pydantic v2 BaseSettings validates types at startup —
misconfigured environments fail fast before serving any traffic.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings(BaseSettings):
    """Central settings object. One instance per process via get_settings()."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Google AI ──────────────────────────────────────────────
    google_api_key: str = Field(...,validation_alias="GOOGLE_API_KEY",description="Gemini API key")

    # ── LangSmith ──────────────────────────────────────────────
    langchain_tracing_v2: bool = Field(default=True)
    langchain_api_key: str = Field(default="", description="LangSmith API key")
    langchain_project: str = Field(default="insightdocket")
    langchain_endpoint: str = Field(default="https://api.smith.langchain.com")

    # ── MongoDB ────────────────────────────────────────────────
    mongodb_uri: str = Field(
        default="mongodb://server:7ygLIIK6T76876@localhost:27017/?authSource=admin&directConnection=true"
    )
    mongodb_database: str = Field(default="insightdocket")
    mongodb_collection: str = Field(default="chunks")
    mongodb_vector_index: str = Field(default="vector_index")

    # ── MySQL ──────────────────────────────────────────────────
    mysql_host: str = Field(default="localhost")
    mysql_port: int = Field(default=3307)
    mysql_user: str = Field(default="insightdocket")
    mysql_password: str = Field(default="insightdocket_pass")
    mysql_database: str = Field(default="insightdocket")

    @property
    def mysql_dsn(self) -> str:
        """Async SQLAlchemy DSN constructed from individual fields."""
        return (
            f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    # ── Application ────────────────────────────────────────────
    app_env: str = Field(default="development")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    app_log_level: str = Field(default="INFO")

    # ── Storage ────────────────────────────────────────────────
    pdf_storage_path: Path = Field(default=Path("./storage/pdfs"))

    @field_validator("pdf_storage_path", mode="before")
    @classmethod
    def resolve_storage_path(cls, v: str | Path) -> Path:
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    # ── Rate Limiting ──────────────────────────────────────────
    gemini_rpm_limit: int = Field(default=5, ge=1, le=15)
    embedding_batch_size: int = Field(default=10, ge=1, le=50)

    # ── RAG Tuning ─────────────────────────────────────────────
    confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    vector_top_k: int = Field(default=20, ge=1, le=100)
    text_top_k: int = Field(default=20, ge=1, le=100)
    final_top_k: int = Field(default=5, ge=1, le=20)

    # ── Reranker ───────────────────────────────────────────────
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ── Security ───────────────────────────────────────────────
    secret_key: str = Field(default="change_this_in_production")
    api_key_header: str = Field(default="X-API-Key")

    # ── Audit Logs ─────────────────────────────────────────────
    audit_log_dir: Path = Field(default=Path("./logs"))

    @field_validator("audit_log_dir", mode="before")
    @classmethod
    def resolve_audit_dir(cls, v: str | Path) -> Path:
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Enforce stricter checks in production environment."""
        if self.app_env == "production":
            if self.secret_key == "change_this_in_production":
                raise ValueError("SECRET_KEY must be changed in production")
            if not self.langchain_api_key:
                raise ValueError("LANGCHAIN_API_KEY required in production")
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    # ── Gemini model names ─────────────────────────────────────
    gemini_generation_model: str = "gemini-3.1-flash-lite"
    gemini_vision_model: str = "gemini-3.1-flash-lite"
    embedding_model: str = "gemini-embedding-001"
    # text-embedding-004 produces 768-dim vectors (free tier)
    # For 3072-dim, use "models/embedding-001" — change MONGODB_VECTOR_DIMENSIONS too
    embedding_dimensions: int = 768


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return cached Settings singleton.
    lru_cache ensures .env is parsed exactly once at startup.
    """
    return Settings()
