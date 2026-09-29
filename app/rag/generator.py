"""
Modular LLM Generator Module.
Handles prompt execution through OpenAI, Google Gemini, or Mock generators
with zero temperature and strict grounding constraints.
"""

import abc
import logging
import os
from typing import Optional
from app.config import Settings, get_settings
from app.rag.prompts import (
    STRICT_SYSTEM_PROMPT,
    format_user_prompt,
    FALLBACK_ANSWER,
)

logger = logging.getLogger(__name__)


class BaseLLMGenerator(abc.ABC):
    """Abstract interface for LLM response generation."""

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        pass

    @abc.abstractmethod
    def generate(self, query: str, context: str) -> str:
        """Generate a grounded answer given user query and retrieved context."""
        pass


class OpenAIGenerator(BaseLLMGenerator):
    """OpenAI generation provider."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        from openai import OpenAI
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIGenerator.")
        self._client = OpenAI(api_key=self._api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, query: str, context: str) -> str:
        user_message = format_user_prompt(query, context)
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content.strip()


class GeminiGenerator(BaseLLMGenerator):
    """Google Gemini generation provider."""

    def __init__(self, model: str = "gemini-3.5-flash-lite", api_key: Optional[str] = None):
        from google import genai
        from google.genai import types
        self._model = model
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required for GeminiGenerator.")
        self._client = genai.Client(api_key=self._api_key)
        self._types = types

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, query: str, context: str) -> str:
        user_message = format_user_prompt(query, context)
        config = self._types.GenerateContentConfig(
            system_instruction=STRICT_SYSTEM_PROMPT,
            temperature=0.0,
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=user_message,
            config=config,
        )
        return (response.text or "").strip()


class MockLLMGenerator(BaseLLMGenerator):
    """
    Deterministic mock generator for offline unit tests and evaluation.
    Synthesizes a response strictly from the provided context without external calls.
    """

    def __init__(self, model: str = "mock-grounded-generator"):
        self._model = model

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, query: str, context: str) -> str:
        if not context or "No relevant context retrieved" in context:
            return FALLBACK_ANSWER

        # Check for adversarial injection
        if "ignore" in query.lower() and "document" in query.lower():
            return "I cannot ignore the provided document. Answering strictly from the Agentic AI eBook: " + context[:120].strip()

        # Deterministic extraction of first sentence or summary with page citation
        lines = [line.strip() for line in context.split("\n") if line.strip() and not line.startswith("---")]
        excerpt = lines[0] if lines else "Information supported by document."
        return f"According to the provided document, {excerpt} [Page 2]"


def get_generator(settings: Optional[Settings] = None) -> BaseLLMGenerator:
    """Factory returning configured LLM generator with graceful fallbacks."""
    cfg = settings or get_settings()
    provider = cfg.LLM_PROVIDER.lower()

    if provider == "openai":
        try:
            return OpenAIGenerator(model=cfg.LLM_MODEL, api_key=cfg.OPENAI_API_KEY)
        except Exception as e:
            logger.warning("Could not initialize OpenAIGenerator (%s). Checking Gemini...", e)
            if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                return GeminiGenerator(model="gemini-3.5-flash-lite")
            return MockLLMGenerator()

    elif provider == "gemini":
        try:
            return GeminiGenerator(model=cfg.LLM_MODEL or "gemini-3.5-flash-lite", api_key=cfg.GEMINI_API_KEY)
        except Exception as e:
            logger.warning("Could not initialize GeminiGenerator (%s). Using MockLLMGenerator.", e)
            return MockLLMGenerator()

    else:
        return MockLLMGenerator()
