"""
Central configuration for the gateway.

Loads environment variables and application settings using Pydantic,
including provider credentials, fallback order, retry behaviour,
rate-limiting, cache settings, and circuit breaker configuration.

Provides a cached Settings instance shared across the application.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    default_provider: str = os.getenv("DEFAULT_PROVIDER", "ollama")
    fallback_chain: list[str] = ["groq","openai", "anthropic", "gemini", "ollama"]

    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 10

    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_seconds: int = 30

    cache_ttl_seconds: int = 3600
    max_retries: int = 3
    base_backoff_seconds: float = 0.5

    class Config:
        env_file = ".env"

@lru_cache
def get_settings():
    return Settings()
