"""
Centralized configuration using pydantic-settings.
Loads from .env file and provides typed access to all constants.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class AreaSettings(BaseSettings):
    """Geographic area configuration for Mangwon-dong."""

    area_code: str = Field(default="11440690", description="행정동 코드 (망원1동)")
    quarter: str = Field(default="20244", description="분기 코드 (2024년 4분기)")
    lat_min: float = Field(default=37.550, description="위도 하한")
    lat_max: float = Field(default=37.560, description="위도 상한")
    lng_min: float = Field(default=126.900, description="경도 하한")
    lng_max: float = Field(default=126.915, description="경도 상한")

    model_config = {"env_prefix": "AREA_"}


class SimulationSettings(BaseSettings):
    """Simulation engine parameters."""

    h3_resolution: int = Field(default=10, description="H3 격자 해상도")
    k_ring: int = Field(default=1, description="H3 인접 격자 탐색 범위")
    agent_count: int = Field(default=20, description="생성할 에이전트 수")
    time_slots_per_day: int = Field(default=6, description="일일 시뮬레이션 횟수")
    simulation_days: int = Field(default=7, description="시뮬레이션 기간 (일)")
    report_reception_probability: float = Field(
        default=0.3, description="리포트 수신 확률"
    )
    visit_threshold: float = Field(default=0.5, description="방문 결정 점수 임계값")

    model_config = {"env_prefix": "SIM_"}


class PathSettings(BaseSettings):
    """File path configuration."""

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "data" / "output"

    @property
    def stores_csv(self) -> Path:
        return self.data_dir / "stores.csv"

    @property
    def population_json(self) -> Path:
        return self.data_dir / "인구_DB.json"

    @property
    def agents_json(self) -> Path:
        return self.output_dir / "agents.json"

    @property
    def simulation_result_csv(self) -> Path:
        return self.output_dir / "simulation_result.csv"

    @property
    def visit_log_csv(self) -> Path:
        return self.output_dir / "visit_log.csv"

    @property
    def prompt_templates_dir(self) -> Path:
        return self.project_root / "src" / "ai_layer" / "prompt_templates"


class LLMSettings(BaseSettings):
    """LLM API configuration (future use)."""

    provider: str = Field(
        default="openai", description="LLM provider: openai | anthropic"
    )
    model_name: str = Field(default="gpt-4o-mini", description="LLM model name")
    api_key: str = Field(default="", description="API key (load from .env)")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048)

    model_config = {"env_prefix": "LLM_"}


class Settings(BaseSettings):
    """Root settings aggregator."""

    area: AreaSettings = Field(default_factory=AreaSettings)
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
