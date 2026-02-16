"""
LLM client for OpenAI-compatible APIs (OpenAI, Gemini, etc.).
"""

from typing import Optional
import httpx
from config import get_settings

# Provider별 OpenAI-compatible 엔드포인트
PROVIDER_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
}


class LLMClient:
    """OpenAI-compatible API client."""

    def __init__(self):
        settings = get_settings()
        self.model = settings.llm.model_name
        self.api_key = settings.llm.api_key
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens
        self.provider = settings.llm.provider
        self.base_url = PROVIDER_URLS.get(
            self.provider, PROVIDER_URLS["openai"]
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
                self.base_url,
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
                self.base_url,
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


def create_llm_client() -> LLMClient:
    """Create LLM client."""
    return LLMClient()
