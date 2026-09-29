"""
Centralized Application Configuration.
Uses pydantic-settings to manage environment variables with validation,
type safety, and secure secret handling.
"""

from functools import lru_cache
from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Application settings
    APP_ENV: Literal["development", "staging", "production", "test"] = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    CORS_ORIGINS: str = "*"

    # Knowledge Base
    PDF_URL: str = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
    DATA_DIR: str = "./data"

    # Vector Database (Pinecone)
    PINECONE_API_KEY: Optional[str] = Field(default=None, description="Pinecone API Key")
    PINECONE_INDEX_NAME: str = "agentic-ai-rag"
    PINECONE_NAMESPACE: str = "agentic-ai-ebook"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    # Embedding Configuration
    EMBEDDING_PROVIDER: Literal["openai", "gemini", "local"] = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536

    # LLM Configuration
    LLM_PROVIDER: Literal["openai", "gemini"] = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1024

    # API Keys (Optional at startup; validated when corresponding service is called)
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API Key")
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Google Gemini API Key")

    # RAG & Guardrail Parameters
    RETRIEVAL_TOP_K: int = 4
    RELEVANCE_THRESHOLD: float = 0.65
    MAX_CONTEXT_CHUNKS: int = 4

    @field_validator("RETRIEVAL_TOP_K")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v <= 0 or v > 20:
            raise ValueError("RETRIEVAL_TOP_K must be between 1 and 20.")
        return v

    @field_validator("RELEVANCE_THRESHOLD")
    @classmethod
    def validate_relevance_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("RELEVANCE_THRESHOLD must be between 0.0 and 1.0.")
        return v


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
