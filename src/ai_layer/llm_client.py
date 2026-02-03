"""
LLM client wrapper. Supports OpenAI and Anthropic providers.
Skeleton for future LLM integration.
"""

from abc import ABC, abstractmethod
from typing import Optional

from config import get_settings


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        ...


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(self):
        settings = get_settings()
        self.model = settings.llm.model_name
        self.api_key = settings.llm.api_key
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens

    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        raise NotImplementedError("OpenAI integration pending")


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""

    def __init__(self):
        settings = get_settings()
        self.model = settings.llm.model_name
        self.api_key = settings.llm.api_key
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens

    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        raise NotImplementedError("Anthropic integration pending")


def create_llm_client() -> LLMClient:
    """Factory: create an LLM client based on settings."""
    settings = get_settings()
    if settings.llm.provider == "openai":
        return OpenAIClient()
    elif settings.llm.provider == "anthropic":
        return AnthropicClient()
    raise ValueError(f"Unknown LLM provider: {settings.llm.provider}")
