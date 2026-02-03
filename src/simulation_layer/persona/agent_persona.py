"""
Agent persona data model.
Defines the attributes of a simulated consumer agent.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Any


@dataclass
class AgentPersona:
    """
    A consumer agent's persona.
    11 attributes capturing demographics, preferences, and sensitivities.
    Plus optional location tracking for street network simulation.
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

    # Street network location (set during simulation)
    location: Optional[Any] = field(default=None, repr=False)

    def to_dict(self):
        """Convert to dict, excluding location (not serializable)."""
        d = asdict(self)
        d.pop("location", None)
        return d
