"""
Agent persona data model.
Defines the attributes of a simulated consumer agent.
"""

from dataclasses import dataclass, asdict
from typing import List


@dataclass
class AgentPersona:
    """
    A consumer agent's persona.
    11 attributes capturing demographics, preferences, and sensitivities.
    """

    id: int
    name: str
    age: int
    age_group: str
    gender: str
    occupation: str
    income_level: str  # '상', '중상', '중', '중하', '하'
    value_preference: str  # 상권 이용 성향 (natural language)
    store_preferences: List[str]  # Preferred store categories
    price_sensitivity: float  # 0~1
    trend_sensitivity: float  # 0~1
    quality_preference: float  # 0~1

    def to_dict(self):
        return asdict(self)
