"""
LLM client wrapper. Supports Groq (free), OpenAI, and Anthropic providers.
"""

import json
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from config import get_settings


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        ...

    @abstractmethod
    def generate_sync(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """Synchronous generation for batch processing."""
        ...


class GroqClient(LLMClient):
    """
    Groq API client - FREE and FAST.
    Models: llama-3.1-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768
    """

    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self):
        settings = get_settings()
        self.model = settings.llm.model_name
        self.api_key = settings.llm.api_key
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens

        if not self.api_key or self.api_key == "your-groq-api-key-here":
            raise ValueError(
                "Groq API key not set. Get free key at https://console.groq.com"
            )

    def _build_messages(self, prompt: str, system_prompt: Optional[str] = None) -> list:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": self._build_messages(prompt, system_prompt),
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    def generate_sync(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": self._build_messages(prompt, system_prompt),
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    BASE_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self):
        settings = get_settings()
        self.model = settings.llm.model_name
        self.api_key = settings.llm.api_key
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens

    def _build_messages(self, prompt: str, system_prompt: Optional[str] = None) -> list:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": self._build_messages(prompt, system_prompt),
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    def generate_sync(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": self._build_messages(prompt, system_prompt),
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""

    BASE_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self):
        settings = get_settings()
        self.model = settings.llm.model_name
        self.api_key = settings.llm.api_key
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens

    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            body = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                body["system"] = system_prompt

            response = await client.post(
                self.BASE_URL,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]

    def generate_sync(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        with httpx.Client(timeout=60.0) as client:
            body = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                body["system"] = system_prompt

            response = client.post(
                self.BASE_URL,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]


def create_llm_client() -> LLMClient:
    """Factory: create an LLM client based on settings."""
    settings = get_settings()
    provider = settings.llm.provider.lower()

    if provider == "groq":
        return GroqClient()
    elif provider == "openai":
        return OpenAIClient()
    elif provider == "anthropic":
        return AnthropicClient()
    raise ValueError(f"Unknown LLM provider: {provider}. Use 'groq', 'openai', or 'anthropic'")
