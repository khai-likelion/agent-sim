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

    model_config = {
        "env_prefix": "PATH_",
        "env_file": str(Path(__file__).resolve().parent.parent / ".env"),
        "extra": "ignore"
    }


class LLMSettings(BaseSettings):
    """LLM API configuration for Gemini."""

    model_name: str = Field(default="gemini-2.0-flash")
    lite_model_name: str = Field(default="gemini-2.5-flash-lite")
    eval_model_name: str = Field(default="gemini-3-flash")
    api_key: str = Field(default="")

    model_config = {
        "env_prefix": "GEMINI_",
        "env_file": str(Path(__file__).resolve().parent.parent / ".env"),
        "extra": "ignore"
    }


class AreaSettings(BaseSettings):
    """Area configuration for simulation."""

    area_code: str = Field(default="11440690")
    quarter: str = Field(default="20244")
    lat_min: float = 37.550
    lat_max: float = 37.560
    lng_min: float = 126.900
    lng_max: float = 126.915

    model_config = {
        "env_prefix": "AREA_",
        "env_file": str(Path(__file__).resolve().parent.parent / ".env"),
        "extra": "ignore"
    }


class SimSettings(BaseSettings):
    """Simulation behavior configuration."""

    agent_count: int = Field(default=160)
    simulation_days: int = Field(default=7)

    model_config = {
        "env_prefix": "SIM_",
        "env_file": str(Path(__file__).resolve().parent.parent / ".env"),
        "extra": "ignore"
    }


class Settings(BaseSettings):
    """Root settings aggregator."""

    paths: PathSettings = Field(default_factory=PathSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    area: AreaSettings = Field(default_factory=AreaSettings)
    sim: SimSettings = Field(default_factory=SimSettings)

    model_config = {
        "env_file": str(Path(__file__).resolve().parent.parent / ".env"),
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
