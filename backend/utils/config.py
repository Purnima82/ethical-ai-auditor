"""
config.py — Centralised application configuration loaded from .env
All settings validated with Pydantic for safety.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # CORS
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        env="CORS_ORIGINS",
    )

    # Limits
    max_dataset_rows: int = Field(default=50_000, env="MAX_DATASET_ROWS")
    audit_cache_ttl: int = Field(default=3600, env="AUDIT_CACHE_TTL")

    # Model
    claude_model: str = Field(
        default="claude-sonnet-4-20250514", env="CLAUDE_MODEL"
    )
    claude_max_tokens: int = Field(default=1500, env="CLAUDE_MAX_TOKENS")

    # Fairness thresholds
    nist_dp_threshold: float = Field(default=0.10, env="NIST_DP_THRESHOLD")
    nist_eo_threshold: float = Field(default=0.10, env="NIST_EO_THRESHOLD")
    disparate_impact_threshold: float = Field(
        default=0.80, env="DISPARATE_IMPACT_THRESHOLD"
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def api_key_configured(self) -> bool:
        return bool(self.anthropic_api_key and self.anthropic_api_key.startswith("sk-"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
