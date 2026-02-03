"""
Decide module: Evaluates stores and decides whether/where to visit.
Extracts _evaluate_store_with_persona() and _decide_visit() from SimulationEngine.

Supports two input formats:
1. DataFrame (legacy H3 mode)
2. List[Dict] (StreetNetwork mode)

Current: rule-based scoring.
Future: LLM-driven decision with memory/reflection context.
"""

import random
from typing import Optional, Union, List, Dict, Any

import pandas as pd

from .base import CognitiveModule
from config import get_settings
from src.simulation_layer.persona.agent_persona import AgentPersona
from src.simulation_layer.models import BusinessReport


class DecideModule(CognitiveModule):
    """
    Scores visible stores against agent persona and decides on visit.
    Supports both DataFrame (H3) and List[Dict] (StreetNetwork) inputs.
    """

    def process(
        self,
        visible_stores: Union[pd.DataFrame, List[Dict[str, Any]]],
        agent: AgentPersona,
        report: Optional[BusinessReport] = None,
    ) -> dict:
        """
        Returns:
            dict with keys: 'decision', 'decision_reason',
                            'visited_store', 'visited_category'
        """
        return self._decide_visit(visible_stores, agent, report)

    def _get_store_field(
        self,
        store: Union[pd.Series, Dict[str, Any]],
        field: str,
        alt_field: str = None,
        default: str = "Unknown"
    ) -> str:
        """Get field from store (works with both Series and dict)."""
        if isinstance(store, dict):
            return store.get(field, store.get(alt_field, default) if alt_field else default)
        else:
            return store.get(field, default)

    def _evaluate_store_with_persona(
        self,
        store: Union[pd.Series, Dict[str, Any]],
        agent: AgentPersona,
        report: Optional[BusinessReport] = None,
    ) -> float:
        """
        Score a store against an agent persona (0~1).
        """
        score = 0.5  # base score

        # 1. Category preference check
        store_category = self._get_store_field(store, "카테고리", "category", "")
        store_name = self._get_store_field(store, "장소명", "store_name", "")

        for pref in agent.store_preferences:
            if pref in store_category:
                score += 0.3
                break

        # 2. Report bonus
        if report and report.store_name == store_name:
            if agent.age_group in report.target_age_groups:
                score += 0.2

            if report.appeal_factor == "price":
                score += agent.price_sensitivity * report.appeal_strength * 0.5
            elif report.appeal_factor == "trend":
                score += agent.trend_sensitivity * report.appeal_strength * 0.5
            elif report.appeal_factor == "quality":
                score += agent.quality_preference * report.appeal_strength * 0.5

        # 3. Random variation (real-world uncertainty)
        score += random.uniform(-0.1, 0.1)

        return min(1.0, max(0.0, score))

    def _decide_visit(
        self,
        visible_stores: Union[pd.DataFrame, List[Dict[str, Any]]],
        agent: AgentPersona,
        report: Optional[BusinessReport] = None,
    ) -> dict:
        """
        Decide which store to visit (if any).
        Handles both DataFrame and List[Dict] inputs.
        """
        # Check if empty
        if isinstance(visible_stores, pd.DataFrame):
            is_empty = visible_stores.empty
            store_iter = visible_stores.iterrows()
        else:
            is_empty = len(visible_stores) == 0
            store_iter = enumerate(visible_stores)

        if is_empty:
            return {
                "decision": "ignore",
                "decision_reason": "주변에 매장이 없음",
                "visited_store": None,
                "visited_category": None,
            }

        settings = get_settings()
        threshold = settings.simulation.visit_threshold

        scores = []
        for idx, store in store_iter:
            score = self._evaluate_store_with_persona(store, agent, report)
            scores.append((score, store))

        scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_store = scores[0]

        if best_score > threshold:
            visited_store = self._get_store_field(best_store, "장소명", "store_name")
            visited_category = self._get_store_field(best_store, "카테고리", "category")

            if report and report.store_name == visited_store:
                reason = f"리포트 '{report.description}'에 반응하여 방문 결정"
            else:
                reason = f"선호 카테고리 및 페르소나 부합 (점수: {best_score:.2f})"

            return {
                "decision": "visit",
                "decision_reason": reason,
                "visited_store": visited_store,
                "visited_category": visited_category,
            }
        else:
            return {
                "decision": "ignore",
                "decision_reason": f"마음에 드는 매장 없음 (최고 점수: {best_score:.2f})",
                "visited_store": None,
                "visited_category": None,
            }
