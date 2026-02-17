"""
Centralized configuration using pydantic-settings.
Loads from .env file and provides typed access to all constants.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class SimulationSettings(BaseSettings):
    """Simulation engine parameters."""

    h3_resolution: int = Field(default=10, description="H3 격자 해상도")
    k_ring: int = Field(default=1, description="인접 탐색 범위 (graph hops)")

    model_config = {"env_prefix": "SIM_", "env_file": ".env", "extra": "ignore"}


class PathSettings(BaseSettings):
    """File path configuration."""

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    store_data_dir: Optional[Path] = Field(
        default=None,
        description="매장 JSON 데이터 디렉토리 경로 (설정 시 기본 경로 대신 사용)"
    )

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def split_store_dir(self) -> Path:
        """split_by_store_id 디렉토리 (STORE_DATA_DIR 환경변수로 오버라이드 가능)"""
        if self.store_data_dir is not None:
            return self.store_data_dir
        return self.data_dir / "split_by_store_id"

    model_config = {"env_prefix": "PATH_", "env_file": ".env", "extra": "ignore"}

    @property
    def output_dir(self) -> Path:
        return self.project_root / "data" / "output"

    @property
    def stores_csv(self) -> Path:
        return self.data_dir / "stores.csv"


class LLMSettings(BaseSettings):
    """LLM API configuration. Default: Ollama with Qwen2.5-7B."""

    provider: str = Field(
        default="ollama", description="LLM provider: ollama | deepseek | sambanova | huggingface | groq | openai | anthropic"
    )
    model_name: str = Field(
        default="qwen2.5:7b",
        description="Model name. Ollama: qwen2.5:7b, llama3.1:8b. DeepSeek: deepseek-chat. SambaNova: Meta-Llama-3.3-70B-Instruct"
    )
    api_key: str = Field(default="http://localhost:11434", description="API key or Ollama base URL")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1024)

    model_config = {"env_prefix": "LLM_", "env_file": ".env", "extra": "ignore"}


class Settings(BaseSettings):
    """Root settings aggregator."""

    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
