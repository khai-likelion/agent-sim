"""
Pydantic models for API request/response.
"""

from pydantic import BaseModel
from typing import List, Optional


class SimulationRequest(BaseModel):
    num_days: int = 7
    agent_count: int = 20
    reports: Optional[List[dict]] = None


class SimulationResponse(BaseModel):
    total_events: int
    active_events: int
    visit_events: int
    conversion_rate: float


class AgentResponse(BaseModel):
    id: int
    name: str
    age: int
    age_group: str
    gender: str
    occupation: str
    income_level: str


class AgentGenerateRequest(BaseModel):
    count: int = 20


class ReportStage1Request(BaseModel):
    store_name: str
    store_context: Optional[dict] = None


class ReportStage1Response(BaseModel):
    strategies: List[dict]


class ReportStage2Request(BaseModel):
    strategy_id: int
    strategy: dict


class ReportStage2Response(BaseModel):
    verification: dict
