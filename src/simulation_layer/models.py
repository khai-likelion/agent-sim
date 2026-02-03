"""
Shared data models for the simulation layer.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BusinessReport:
    """A business report (promotion/event) targeting specific demographics."""

    store_name: str
    report_type: str  # 'discount', 'new_menu', 'renovation', 'event'
    description: str
    target_age_groups: List[str]
    appeal_factor: str  # 'price', 'trend', 'quality'
    appeal_strength: float  # 0~1


@dataclass
class SimulationEvent:
    """A single simulation event log entry."""

    timestamp: str
    agent_id: int
    agent_name: str
    age_group: str
    gender: str
    weekday: str
    time_slot: str
    is_active: bool
    current_lat: float
    current_lng: float
    visible_stores_count: int
    report_received: Optional[str]
    decision: str  # 'visit', 'ignore', 'change_destination', 'inactive'
    decision_reason: str
    visited_store: Optional[str]
    visited_category: Optional[str]
