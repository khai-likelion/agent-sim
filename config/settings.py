"""
Centralized configuration using pydantic-settings.
Loads from .env file and provides typed access to all constants.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class PathSettings(BaseSettings):
    """File path configuration."""

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def split_store_dir(self) -> Path:
        return self.data_dir / "split_by_store_id_ver5"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "data" / "output"

    @property
    def stores_csv(self) -> Path:
        return self.data_dir / "stores.csv"

    model_config = {"env_prefix": "PATH_", "env_file": ".env", "extra": "ignore"}


class LLMSettings(BaseSettings):
    """LLM API configuration."""

    provider: str = Field(default="openai", description="LLM provider: openai | gemini")
    model_name: str = Field(default="gpt-4o-mini", description="Model name")
    api_key: str = Field(default="", description="API key")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=512)

    model_config = {"env_prefix": "LLM_", "env_file": ".env", "extra": "ignore"}

# PathSettings, LLMSettings를 하나로 묶는 루트 컨테이너
class Settings(BaseSettings):
    """Root settings aggregator."""

    paths: PathSettings = Field(default_factory=PathSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

# .env 파싱은 최초 1회만 하고, 이후엔 메모리에 캐시된 값을 바로 씁니다.
# 위에서 최초 1회 생성 -> run_before_after_sim.py 에서 import 하며 재사용
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
