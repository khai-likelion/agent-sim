"""
Two-stage business report generation system.
Stage 1 (1차 리포트): LLM generates strategy candidates from data context.
Stage 2 (2차 리포트): Simulation verifies user-selected strategy, LLM summarizes.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class StrategyCandidate:
    """Output of Stage 1: a proposed business strategy."""

    id: int
    title: str
    description: str
    target_age_groups: List[str]
    appeal_factor: str  # 'price', 'trend', 'quality'
    estimated_effectiveness: float
    rationale: str


@dataclass
class VerificationResult:
    """Output of Stage 2: simulation results for a chosen strategy."""

    strategy_id: int
    simulation_visits: int
    conversion_rate: float
    age_group_breakdown: dict
    time_slot_breakdown: dict
    summary: str


class ReportGenerator:
    """
    Orchestrates the two-stage report pipeline.
    """

    async def generate_stage1(
        self, store_name: str, store_context: dict
    ) -> List[StrategyCandidate]:
        """
        1차 리포트: Generate 3-5 strategy candidates via LLM.
        Uses prompt_templates/report_stage1.txt.
        """
        raise NotImplementedError(
            "Stage 1 report generation pending LLM integration"
        )

    async def generate_stage2(
        self, strategy: StrategyCandidate, simulation_result: dict
    ) -> VerificationResult:
        """
        2차 리포트: Run simulation with chosen strategy, then summarize via LLM.
        Uses prompt_templates/report_stage2.txt.
        """
        raise NotImplementedError(
            "Stage 2 report generation pending LLM integration"
        )
