"""
Decide module: Evaluates stores and decides whether/where to visit.
Supports both rule-based scoring and LLM-driven decision making.
"""

import json
import random
import time
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import pandas as pd

from .base import CognitiveModule
from config import get_settings
from src.simulation_layer.persona.agent_persona import AgentPersona
from src.simulation_layer.models import BusinessReport
from src.ai_layer.llm_client import create_llm_client, LLMClient


class DecideModule(CognitiveModule):
    """
    Scores visible stores against agent persona and decides on visit.
    Supports both rule-based and LLM-driven decision modes.
    """

    def __init__(self, use_llm: bool = False, rate_limit_delay: float = 4.0):
        """
        Args:
            use_llm: If True, use LLM for decisions. If False, use rule-based.
            rate_limit_delay: Seconds to wait between LLM calls (Groq free: 30/min → 15/min safe)
        """
        self.use_llm = use_llm
        self.rate_limit_delay = rate_limit_delay
        self.llm_client: Optional[LLMClient] = None
        self._prompt_template: Optional[str] = None

    def _get_llm_client(self) -> LLMClient:
        if self.llm_client is None:
            self.llm_client = create_llm_client()
        return self.llm_client

    def _load_prompt_template(self) -> str:
        if self._prompt_template is None:
            settings = get_settings()
            template_path = settings.paths.prompt_templates_dir / "decision.txt"
            self._prompt_template = template_path.read_text(encoding='utf-8')
        return self._prompt_template

    def process(
        self,
        visible_stores: Union[pd.DataFrame, List[Dict[str, Any]]],
        agent: AgentPersona,
        report: Optional[BusinessReport] = None,
        time_slot: str = "",
        weekday: str = "",
        memory_context: str = "",
    ) -> dict:
        """
        Decide which store to visit.

        Returns:
            dict with keys: 'decision', 'decision_reason',
                            'visited_store', 'visited_category'
        """
        if self.use_llm:
            return self._decide_with_llm(visible_stores, agent, report, time_slot, memory_context)
        else:
            return self._decide_rule_based(visible_stores, agent, report)

    # ========== LLM-based Decision ==========

    def _format_visible_stores(
        self, visible_stores: Union[pd.DataFrame, List[Dict[str, Any]]]
    ) -> str:
        """Format visible stores for prompt."""
        lines = []

        if isinstance(visible_stores, pd.DataFrame):
            for _, store in visible_stores.iterrows():
                name = store.get("장소명", store.get("store_name", "Unknown"))
                category = store.get("카테고리", store.get("category", "Unknown"))
                lines.append(f"- {name} ({category})")
        else:
            for store in visible_stores:
                name = store.get("장소명", store.get("store_name", "Unknown"))
                category = store.get("카테고리", store.get("category", "Unknown"))
                lines.append(f"- {name} ({category})")

        return "\n".join(lines) if lines else "주변에 매장이 없습니다."

    def _format_report_info(self, report: Optional[BusinessReport]) -> str:
        """Format business report for prompt."""
        if not report:
            return "받은 리포트가 없습니다."

        return f"""- 매장: {report.store_name}
- 내용: {report.description}
- 타겟 연령: {', '.join(report.target_age_groups)}
- 어필 포인트: {report.appeal_factor} (강도: {report.appeal_strength})"""

    def _build_decision_prompt(
        self,
        visible_stores: Union[pd.DataFrame, List[Dict[str, Any]]],
        agent: AgentPersona,
        report: Optional[BusinessReport],
        time_slot: str,
        memory_context: str,
    ) -> str:
        """Build the decision prompt."""
        template = self._load_prompt_template()

        return template.format(
            agent_name=agent.name,
            age=agent.age,
            age_group=agent.age_group,
            gender=agent.gender,
            occupation=agent.occupation,
            income_level=agent.income_level,
            value_preference=agent.value_preference,
            store_preferences=", ".join(agent.store_preferences),
            time_slot=time_slot or "알 수 없음",
            visible_stores=self._format_visible_stores(visible_stores),
            report_info=self._format_report_info(report),
            memory_context=memory_context or "방문 기록이 없습니다.",
        )

    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM JSON response."""
        try:
            # Find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                decision = data.get("decision", "ignore").lower()
                store_name = data.get("store_name")
                reason = data.get("reason", "")

                return {
                    "decision": "visit" if decision == "visit" and store_name else "ignore",
                    "decision_reason": reason,
                    "visited_store": store_name if decision == "visit" else None,
                    "visited_category": None,  # Will be filled by caller if needed
                }
        except (json.JSONDecodeError, KeyError) as e:
            pass

        # Fallback
        return {
            "decision": "ignore",
            "decision_reason": "LLM 응답 파싱 실패",
            "visited_store": None,
            "visited_category": None,
        }

    def _decide_with_llm(
        self,
        visible_stores: Union[pd.DataFrame, List[Dict[str, Any]]],
        agent: AgentPersona,
        report: Optional[BusinessReport],
        time_slot: str,
        memory_context: str,
    ) -> dict:
        """Make decision using LLM."""
        # Check if empty
        if isinstance(visible_stores, pd.DataFrame):
            is_empty = visible_stores.empty
        else:
            is_empty = len(visible_stores) == 0

        if is_empty:
            return {
                "decision": "ignore",
                "decision_reason": "주변에 매장이 없음",
                "visited_store": None,
                "visited_category": None,
            }

        try:
            prompt = self._build_decision_prompt(
                visible_stores, agent, report, time_slot, memory_context
            )

            client = self._get_llm_client()
            response = client.generate_sync(prompt)

            result = self._parse_llm_response(response)

            # Fill in category if visit decision
            if result["decision"] == "visit" and result["visited_store"]:
                result["visited_category"] = self._find_store_category(
                    visible_stores, result["visited_store"]
                )

            # Rate limiting
            time.sleep(self.rate_limit_delay)

            return result

        except Exception as e:
            print(f"LLM decision error: {e}. Falling back to rule-based.")
            return self._decide_rule_based(visible_stores, agent, report)

    def _find_store_category(
        self,
        visible_stores: Union[pd.DataFrame, List[Dict[str, Any]]],
        store_name: str
    ) -> Optional[str]:
        """Find category for a store name."""
        if isinstance(visible_stores, pd.DataFrame):
            for _, store in visible_stores.iterrows():
                name = store.get("장소명", store.get("store_name", ""))
                if name == store_name:
                    return store.get("카테고리", store.get("category"))
        else:
            for store in visible_stores:
                name = store.get("장소명", store.get("store_name", ""))
                if name == store_name:
                    return store.get("카테고리", store.get("category"))
        return None

    # ========== Rule-based Decision (Legacy) ==========

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
        """Score a store against an agent persona (0~1). Rule-based."""
        score = 0.5

        store_category = self._get_store_field(store, "카테고리", "category", "")
        store_name = self._get_store_field(store, "장소명", "store_name", "")

        # Category preference
        for pref in agent.store_preferences:
            if pref in store_category:
                score += 0.3
                break

        # Report bonus
        if report and report.store_name == store_name:
            if agent.age_group in report.target_age_groups:
                score += 0.2

            if report.appeal_factor == "price":
                score += agent.price_sensitivity * report.appeal_strength * 0.5
            elif report.appeal_factor == "trend":
                score += agent.trend_sensitivity * report.appeal_strength * 0.5
            elif report.appeal_factor == "quality":
                score += agent.quality_preference * report.appeal_strength * 0.5

        # Random variation
        score += random.uniform(-0.1, 0.1)

        return min(1.0, max(0.0, score))

    def _decide_rule_based(
        self,
        visible_stores: Union[pd.DataFrame, List[Dict[str, Any]]],
        agent: AgentPersona,
        report: Optional[BusinessReport] = None,
    ) -> dict:
        """Rule-based decision (legacy mode)."""
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
