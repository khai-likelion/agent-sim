"""
LLM client wrapper. Supports SambaNova, Groq, OpenAI, and Anthropic providers.
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


class SambaNovaClient(LLMClient):
    """
    SambaNova Cloud API client - FREE with generous limits.
    Models: Meta-Llama-3.1-70B-Instruct, Meta-Llama-3.1-8B-Instruct
    Rate limits: Much more generous than Groq free tier.
    """

    BASE_URL = "https://api.sambanova.ai/v1/chat/completions"

    def __init__(self):
        settings = get_settings()
        self.model = settings.llm.model_name
        self.api_key = settings.llm.api_key
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens

        if not self.api_key:
            raise ValueError(
                "SambaNova API key not set. Get free key at https://cloud.sambanova.ai"
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


class HuggingFaceClient(LLMClient):
    """
    Hugging Face Inference API client using huggingface_hub.
    Models: Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3.1-8B-Instruct, etc.
    """

    def __init__(self):
        from huggingface_hub import InferenceClient

        settings = get_settings()
        self.model = settings.llm.model_name
        self.api_key = settings.llm.api_key
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens

        if not self.api_key:
            raise ValueError(
                "Hugging Face API key not set. Get free key at https://huggingface.co/settings/tokens"
            )

        self._client = InferenceClient(token=self.api_key)

    def _build_messages(self, prompt: str, system_prompt: Optional[str] = None) -> list:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        # huggingface_hub InferenceClient is sync, run in thread for async
        import asyncio
        return await asyncio.to_thread(self.generate_sync, prompt, system_prompt)

    def generate_sync(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt, system_prompt),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


class DeepSeekClient(LLMClient):
    """
    DeepSeek API client - OpenAI compatible, generous free tier.
    Models: deepseek-chat (DeepSeek-V3), deepseek-reasoner (DeepSeek-R1)
    Free tier: $5 credit for new users, very affordable pricing after.
    Rate limits: Much more generous than other providers.
    """

    BASE_URL = "https://api.deepseek.com/chat/completions"

    def __init__(self):
        settings = get_settings()
        self.model = settings.llm.model_name
        self.api_key = settings.llm.api_key
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens

        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not set. Get key at https://platform.deepseek.com/api_keys"
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
        async with httpx.AsyncClient(timeout=120.0) as client:
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
        with httpx.Client(timeout=120.0) as client:
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


class OllamaClient(LLMClient):
    """
    Ollama local LLM client - runs models locally, no API key needed.
    Models: qwen2.5:7b, llama3.1:8b, mistral:7b, etc.
    Requires Ollama running locally at http://localhost:11434
    """

    def __init__(self):
        settings = get_settings()
        self.model = settings.llm.model_name
        self.base_url = settings.llm.api_key or "http://localhost:11434"
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
        import asyncio
        return await asyncio.to_thread(self.generate_sync, prompt, system_prompt)

    def generate_sync(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": self._build_messages(prompt, system_prompt),
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except:
            return False


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

    if provider == "ollama":
        return OllamaClient()
    elif provider == "sambanova":
        return SambaNovaClient()
    elif provider == "groq":
        return GroqClient()
    elif provider == "openai":
        return OpenAIClient()
    elif provider == "anthropic":
        return AnthropicClient()
    elif provider == "huggingface":
        return HuggingFaceClient()
    elif provider == "deepseek":
        return DeepSeekClient()
    raise ValueError(f"Unknown LLM provider: {provider}. Use 'ollama', 'sambanova', 'groq', 'openai', 'anthropic', 'huggingface', or 'deepseek'")
