"""
Chained Decision Module: Multi-step LLM prompts for realistic behavior simulation.

Decision Chain:
1. Time-slot decision (eat or not, how)
2. Category selection (what type of food)
3. Store selection (which specific store)
4. Report reaction (respond to promotions)
5. Post-meal satisfaction (feedback loop)
"""

import json
import time
import random
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import pandas as pd

from .base import CognitiveModule
from config import get_settings
from src.simulation_layer.persona.agent_persona import AgentPersona
from src.simulation_layer.persona.agent_state import AgentState
from src.simulation_layer.models import BusinessReport
from src.ai_layer.llm_client import create_llm_client, LLMClient


class ChainedDecideModule(CognitiveModule):
    """
    Multi-step decision module using chained LLM prompts.
    Simulates realistic decision-making process:
    1. Should I eat? How? (timeslot prompt)
    2. What category? (category prompt)
    3. Which store? (store prompt)
    4. React to promotion? (report prompt)
    """

    TIMESLOT_PROMPTS = {
        "아침": "timeslot_morning.txt",
        "점심": "timeslot_lunch.txt",
        "저녁": "timeslot_dinner.txt",
        "야간": "timeslot_night.txt",
    }

    def __init__(self, use_llm: bool = True, rate_limit_delay: float = 2.0, smart_mode: bool = True):
        self.use_llm = use_llm
        self.rate_limit_delay = rate_limit_delay
        self.smart_mode = smart_mode  # Use rules for Step 1, LLM only for Step 2 & 3
        self.llm_client: Optional[LLMClient] = None
        self._prompt_cache: Dict[str, str] = {}
        self._agent_states: Dict[int, AgentState] = {}
        self._response_cache: Dict[str, Dict[str, Any]] = {}  # Cache for similar contexts

    def _get_llm_client(self) -> LLMClient:
        if self.llm_client is None:
            self.llm_client = create_llm_client()
        return self.llm_client

    def _load_prompt(self, template_name: str) -> str:
        if template_name not in self._prompt_cache:
            settings = get_settings()
            template_path = settings.paths.prompt_templates_dir / template_name
            self._prompt_cache[template_name] = template_path.read_text(encoding='utf-8')
        return self._prompt_cache[template_name]

    def _get_agent_state(self, agent: AgentPersona) -> AgentState:
        if agent.id not in self._agent_states:
            self._agent_states[agent.id] = AgentState(agent_id=agent.id)
        return self._agent_states[agent.id]

    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM with rate limiting and automatic retry on 429 errors."""
        client = self._get_llm_client()

        for attempt in range(max_retries):
            try:
                response = client.generate_sync(prompt)
                time.sleep(self.rate_limit_delay)
                return response
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "Too Many Requests" in error_str:
                    # Rate limit hit - wait and retry with exponential backoff
                    wait_time = (attempt + 1) * 30  # 30s, 60s, 90s
                    print(f"  Rate limit hit, waiting {wait_time}s before retry ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    # Other error - raise immediately
                    raise e

        # All retries exhausted
        raise Exception(f"LLM call failed after {max_retries} retries due to rate limiting")

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except json.JSONDecodeError:
            pass
        return {}

    def _get_archetype_info(self, agent: AgentPersona) -> Dict[str, str]:
        """Extract archetype info from agent."""
        if hasattr(agent, '_archetype') and agent._archetype:
            return {
                "segment": agent._archetype.segment,
                "taste": agent._archetype.taste,
                "lifestyle": agent._archetype.lifestyle,
            }
        return {
            "segment": "일반",
            "taste": "보통",
            "lifestyle": "보통",
        }

    # ==================== Step 1: Timeslot Decision ====================

    def _step1_timeslot_decision(
        self,
        agent: AgentPersona,
        state: AgentState,
        time_slot: str,
        weekday: str,
    ) -> Dict[str, Any]:
        """Step 1: Decide whether/how to eat based on time slot.

        Smart mode: Use rules first, skip LLM entirely for Step 1.
        This saves 1 LLM call per agent (biggest optimization).
        """
        archetype = self._get_archetype_info(agent)

        # Smart mode: Always use rule-based for Step 1 (saves 1 LLM call per agent)
        if self.smart_mode:
            return self._smart_timeslot_decision(state, time_slot, weekday, archetype)

        # Full LLM mode (original behavior)
        template_file = self.TIMESLOT_PROMPTS.get(time_slot, "timeslot_lunch.txt")
        template = self._load_prompt(template_file)

        # Build context
        day_context = self._get_day_context(weekday, archetype["segment"])
        companion_context = self._get_companion_context(state, archetype["segment"], weekday)

        # Get meal history
        breakfast_status = "먹음" if any(m.meal_type == "breakfast" for m in state.meals_today) else "안 먹음"
        lunch_status = "먹음" if any(m.meal_type == "lunch" for m in state.meals_today) else "안 먹음"
        dinner_status = "먹음" if any(m.meal_type == "dinner" for m in state.meals_today) else "안 먹음"
        last_dinner = state.recent_visits[-1] if state.recent_visits else "기억 없음"

        prompt = template.format(
            agent_name=agent.name,
            age=agent.age,
            age_group=agent.age_group,
            gender=agent.gender,
            occupation=agent.occupation,
            segment=archetype["segment"],
            taste=archetype["taste"],
            lifestyle=archetype["lifestyle"],
            value_preference=agent.value_preference,
            state_summary=state.get_context_summary(),
            weekday=weekday,
            day_context=day_context,
            hunger_level=state.get_hunger_level(),
            fatigue_level=state.get_fatigue_level(),
            last_dinner=last_dinner,
            breakfast_status=breakfast_status,
            lunch_status=lunch_status,
            dinner_status=dinner_status,
            budget_today=state.budget_today,
            time_pressure=state.time_pressure,
            companion_context=companion_context,
            special_context=self._get_special_context(time_slot, weekday),
            night_activity_context=self._get_night_context(state, weekday),
            tomorrow_context="평일 출근" if weekday in ["일", "월", "화", "수", "목"] else "주말",
        )

        if self.use_llm:
            try:
                response = self._call_llm(prompt)
                result = self._parse_json_response(response)
                if result:
                    return result
            except Exception as e:
                print(f"Step 1 LLM error: {e}")

        # Fallback: rule-based
        return self._fallback_timeslot_decision(state, time_slot, archetype)

    def _smart_timeslot_decision(
        self, state: AgentState, time_slot: str, weekday: str, archetype: Dict
    ) -> Dict[str, Any]:
        """Smart rule-based Step 1 decision based on segment, time, and state.

        More sophisticated than fallback - considers archetype characteristics.
        No LLM call needed, saving significant API quota.
        """
        hunger = state.hunger
        segment = archetype.get("segment", "일반")
        taste = archetype.get("taste", "편안한맛")
        lifestyle = archetype.get("lifestyle", "단조로운패턴")

        # Companion context (sets state.companion)
        self._get_companion_context(state, segment, weekday)

        # Weekend vs weekday behavior
        is_weekend = weekday in ["토", "일"]
        is_friday = weekday == "금"

        # ============ 아침 (Morning) ============
        if time_slot == "아침":
            if hunger < 30:
                return {"decision": "skip", "reason": "아침이라 배가 안 고파서", "craving": "", "preferred_price": "저렴"}

            # 1인가구, 외부출퇴근: 대부분 집에서 또는 스킵
            if segment in ["1인가구", "외부출퇴근직장인"]:
                if random.random() < 0.7:
                    return {"decision": "home", "reason": "집에서 간단히 해결", "craving": "토스트/시리얼", "preferred_price": "저렴"}
                return {"decision": "skip", "reason": "아침은 건너뜀", "craving": "", "preferred_price": "저렴"}

            # 데이트커플, 2인가구: 주말엔 브런치 외식 가능성
            if segment in ["데이트커플", "2인가구"] and is_weekend and random.random() < 0.4:
                return {"decision": "dine_in", "reason": "주말 브런치", "craving": "브런치/카페", "preferred_price": "보통"}

            return {"decision": "home", "reason": "집에서 아침", "craving": "", "preferred_price": "저렴"}

        # ============ 점심 (Lunch) ============
        elif time_slot == "점심":
            if hunger < 40:
                return {"decision": "skip", "reason": "아직 배가 덜 고파서", "craving": "", "preferred_price": "저렴"}

            # 유동인구 세그먼트는 외식 확률 높음
            if segment in ["데이트커플", "약속모임", "망원유입직장인", "혼자방문"]:
                dine_prob = 0.8
            elif segment in ["외부출퇴근직장인"]:
                dine_prob = 0.3 if not is_weekend else 0.6  # 평일엔 직장 근처에서
            else:
                dine_prob = 0.5

            if random.random() < dine_prob:
                craving = self._get_craving_by_taste(taste)
                price = "보통" if taste != "미식탐구" else "비싸도 OK"
                return {"decision": "dine_in", "reason": "점심 외식", "craving": craving, "preferred_price": price}
            else:
                return {"decision": "home", "reason": "집에서 간단히", "craving": "", "preferred_price": "저렴"}

        # ============ 저녁 (Dinner) ============
        elif time_slot == "저녁":
            if hunger < 35:
                return {"decision": "skip", "reason": "점심을 많이 먹어서", "craving": "", "preferred_price": "저렴"}

            # 금요일 저녁, 주말 저녁: 외식 확률 증가
            base_prob = 0.5
            if is_friday or is_weekend:
                base_prob = 0.7

            # 유동인구는 외식 확률 높음
            if segment in ["데이트커플", "약속모임"]:
                base_prob = 0.9
            elif segment in ["혼자방문", "망원유입직장인"]:
                base_prob = 0.7

            if random.random() < base_prob:
                craving = self._get_craving_by_taste(taste)
                price = "보통"
                if taste == "미식탐구":
                    price = "비싸도 OK"
                elif taste == "편안한맛":
                    price = "저렴"
                return {"decision": "dine_in", "reason": "저녁 외식", "craving": craving, "preferred_price": price}
            else:
                return {"decision": "home", "reason": "집에서 저녁", "craving": "", "preferred_price": "저렴"}

        # ============ 야간 (Night) ============
        else:
            if state.fatigue > 70:
                return {"decision": "home", "reason": "피곤해서 귀가", "craving": "", "preferred_price": "저렴"}

            # 금/토 야간: 활동 확률 높음
            if is_friday or weekday == "토":
                if segment in ["데이트커플", "약속모임", "혼자방문"]:
                    if random.random() < 0.6:
                        return {"decision": "dine_in", "reason": "야간 활동", "craving": "술/안주", "preferred_price": "보통"}

            return {"decision": "home", "reason": "내일 일정", "craving": "", "preferred_price": "저렴"}

    def _get_craving_by_taste(self, taste: str) -> str:
        """Get craving based on taste preference."""
        cravings = {
            "자극선호": random.choice(["마라탕", "떡볶이", "짬뽕", "매운갈비찜", "닭발"]),
            "담백건강": random.choice(["샐러드", "포케", "칼국수", "설렁탕", "일식"]),
            "미식탐구": random.choice(["오마카세", "이탈리안", "스시", "스테이크", "파스타"]),
            "편안한맛": random.choice(["국밥", "백반", "김치찌개", "비빔밥", "돈까스"]),
        }
        return cravings.get(taste, "한식")

    def _get_day_context(self, weekday: str, segment: str) -> str:
        """Get context based on day and segment."""
        if weekday in ["토", "일"]:
            contexts = {
                "1인가구": "주말이라 여유롭게 늦잠을 잤다. 브런치가 당긴다.",
                "2인가구": "주말 데이트 계획이 있다. 분위기 좋은 곳에서 식사하고 싶다.",
                "4인가구": "가족과 함께하는 주말이다. 아이들도 좋아할 만한 곳을 찾는다.",
                "데이트커플": "연인과 특별한 주말을 보내고 싶다.",
                "약속모임": "친구들과 만나기로 한 날이다.",
                "외부출퇴근직장인": "드디어 주말! 동네에서 여유롭게 시간을 보낼 수 있다.",
                "망원유입직장인": "주말이라 망원동에 놀러왔다.",
                "혼자방문": "혼자만의 여유로운 시간을 보내고 싶다.",
            }
        else:
            contexts = {
                "1인가구": "평일 일상. 효율적으로 끼니를 해결해야 한다.",
                "2인가구": "평일이라 바쁘다. 간단히 해결하거나 퇴근 후 같이 먹는다.",
                "4인가구": "아이들 학교 보내고 일상이 바쁘다.",
                "데이트커플": "평일이라 각자 바쁘다. 저녁에 만날 수도 있다.",
                "약속모임": "평일이지만 퇴근 후 약속이 있을 수도 있다.",
                "외부출퇴근직장인": "출퇴근이 바쁜 평일이다.",
                "망원유입직장인": "업무 중이다. 점심/저녁 식사가 중요하다.",
                "혼자방문": "평일에 혼자 망원동을 찾았다.",
            }
        return contexts.get(segment, "평범한 하루다.")

    def _get_companion_context(self, state: AgentState, segment: str, weekday: str) -> str:
        """Determine companion for the meal."""
        if segment == "1인가구":
            state.companion = "혼자"
            return "혼자 식사한다."
        elif segment == "2인가구":
            state.companion = random.choice(["파트너", "혼자"])
            return f"{'파트너와 함께' if state.companion == '파트너' else '혼자'} 식사한다."
        elif segment == "4인가구":
            if weekday in ["토", "일"]:
                state.companion = "가족"
                return "가족과 함께 식사한다."
            else:
                state.companion = random.choice(["혼자", "가족"])
                return f"{'가족과 함께' if state.companion == '가족' else '혼자'} 식사한다."
        elif segment == "데이트커플":
            state.companion = "연인"
            return "연인과 함께 식사한다."
        elif segment == "약속모임":
            state.companion = "친구/동료"
            return "친구나 동료와 함께 식사한다."
        elif segment in ["외부출퇴근직장인", "망원유입직장인"]:
            state.companion = random.choice(["동료", "혼자"])
            return f"{'동료와 함께' if state.companion == '동료' else '혼자'} 식사한다."
        else:
            state.companion = "혼자"
            return "혼자 식사한다."

    def _get_special_context(self, time_slot: str, weekday: str) -> str:
        """Get special context for dinner time."""
        if time_slot != "저녁":
            return ""
        if weekday == "금":
            return "금요일 저녁! 불금을 즐기고 싶은 마음이 있다."
        elif weekday in ["토", "일"]:
            return "주말 저녁이라 여유롭다."
        else:
            return "평일 저녁, 내일도 일해야 한다."

    def _get_night_context(self, state: AgentState, weekday: str) -> str:
        """Get night activity context."""
        if weekday in ["금", "토"]:
            return "주말 밤이라 활동적으로 보낼 수 있다."
        else:
            return "내일 일정이 있어서 일찍 들어가는 게 좋을 수도 있다."

    def _fallback_timeslot_decision(
        self, state: AgentState, time_slot: str, archetype: Dict
    ) -> Dict[str, Any]:
        """Fallback rule-based timeslot decision."""
        hunger = state.hunger

        if time_slot == "아침":
            if hunger < 40:
                return {"decision": "skip", "reason": "배가 안 고파서", "craving": ""}
            elif random.random() < 0.6:
                return {"decision": "home", "reason": "집에서 간단히", "craving": "토스트"}
            else:
                return {"decision": "takeout", "reason": "출근길에", "craving": "커피"}

        elif time_slot == "점심":
            if hunger < 30:
                return {"decision": "skip", "reason": "아직 배가 안 고파서", "craving": ""}
            return {"decision": "dine_in", "reason": "점심시간", "craving": archetype.get("taste", "한식")}

        elif time_slot == "저녁":
            if hunger < 40 and random.random() < 0.3:
                return {"decision": "skip", "reason": "점심을 많이 먹어서", "craving": ""}
            if random.random() < 0.4:
                return {"decision": "home", "reason": "집에서 편하게", "craving": ""}
            return {"decision": "dine_in", "reason": "저녁 외식", "craving": archetype.get("taste", "한식")}

        else:  # 야간
            if state.fatigue > 70:
                return {"decision": "home", "reason": "피곤해서 귀가", "craving": ""}
            if random.random() < 0.5:
                return {"decision": "home", "reason": "내일 일정", "craving": ""}
            return {"decision": random.choice(["cafe", "drink", "snack"]), "reason": "야간 활동", "craving": ""}

    # ==================== Step 2: Category Selection ====================

    def _step2_category_selection(
        self,
        agent: AgentPersona,
        state: AgentState,
        timeslot_result: Dict[str, Any],
        available_categories: List[str],
    ) -> Dict[str, Any]:
        """Step 2: Select food category based on craving and preferences."""
        archetype = self._get_archetype_info(agent)
        template = self._load_prompt("category_selection.txt")

        recent_cats = ", ".join(state.recent_categories[-3:]) if state.recent_categories else "없음"
        available_cats = ", ".join(list(set(available_categories))[:20])

        prompt = template.format(
            agent_name=agent.name,
            taste=archetype["taste"],
            lifestyle=archetype["lifestyle"],
            store_preferences=", ".join(agent.store_preferences),
            time_slot=timeslot_result.get("time_slot", ""),
            hunger_level=state.get_hunger_level(),
            mood=state.mood,
            companion=state.companion or "혼자",
            budget_willing=timeslot_result.get("budget_willing", "보통"),
            craving=timeslot_result.get("craving", ""),
            recent_categories=recent_cats,
            available_categories=available_cats,
        )

        if self.use_llm:
            try:
                response = self._call_llm(prompt)
                result = self._parse_json_response(response)
                if result and result.get("category"):
                    return result
            except Exception as e:
                print(f"Step 2 LLM error: {e}")

        # Fallback
        preferred = agent.store_preferences[0] if agent.store_preferences else "한식"
        return {"category": preferred, "reason": "선호 카테고리", "alternative": ""}

    # ==================== Step 3: Store Selection ====================

    def _step3_store_selection(
        self,
        agent: AgentPersona,
        state: AgentState,
        category: str,
        visible_stores: Union[pd.DataFrame, List[Dict]],
        report: Optional[BusinessReport],
        timeslot_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Step 3: Select specific store from available options."""
        archetype = self._get_archetype_info(agent)
        template = self._load_prompt("store_selection.txt")

        # Filter stores by category
        store_list = self._filter_and_format_stores(visible_stores, category)
        if not store_list:
            store_list = self._format_all_stores(visible_stores)

        recent = ", ".join(state.recent_visits[-3:]) if state.recent_visits else "없음"
        report_info = self._format_report_info(report) if report else "없음"

        prompt = template.format(
            category=category,
            agent_name=agent.name,
            segment=archetype["segment"],
            taste=archetype["taste"],
            lifestyle=archetype["lifestyle"],
            time_slot=timeslot_result.get("time_slot", ""),
            companion=state.companion or "혼자",
            budget_willing=timeslot_result.get("budget_willing", "보통"),
            time_pressure=state.time_pressure,
            store_list=store_list,
            recent_visits=recent,
            report_info=report_info,
        )

        if self.use_llm:
            try:
                response = self._call_llm(prompt)
                result = self._parse_json_response(response)
                if result and result.get("store_name"):
                    return result
            except Exception as e:
                print(f"Step 3 LLM error: {e}")

        # Fallback: pick random matching store
        return self._fallback_store_selection(visible_stores, category, archetype["lifestyle"])

    def _filter_and_format_stores(
        self, stores: Union[pd.DataFrame, List[Dict]], category: str
    ) -> str:
        """Filter stores by category and format for prompt."""
        lines = []
        if isinstance(stores, pd.DataFrame):
            for _, store in stores.iterrows():
                cat = store.get("카테고리", store.get("category", ""))
                if category.lower() in cat.lower():
                    name = store.get("장소명", store.get("store_name", "Unknown"))
                    lines.append(f"- {name} ({cat})")
        else:
            for store in stores:
                cat = store.get("카테고리", store.get("category", ""))
                if category.lower() in cat.lower():
                    name = store.get("장소명", store.get("store_name", "Unknown"))
                    lines.append(f"- {name} ({cat})")

        return "\n".join(lines[:15]) if lines else "해당 카테고리 매장 없음"

    def _format_all_stores(self, stores: Union[pd.DataFrame, List[Dict]]) -> str:
        """Format all stores for prompt."""
        lines = []
        if isinstance(stores, pd.DataFrame):
            for _, store in stores.head(15).iterrows():
                name = store.get("장소명", store.get("store_name", "Unknown"))
                cat = store.get("카테고리", store.get("category", "Unknown"))
                lines.append(f"- {name} ({cat})")
        else:
            for store in stores[:15]:
                name = store.get("장소명", store.get("store_name", "Unknown"))
                cat = store.get("카테고리", store.get("category", "Unknown"))
                lines.append(f"- {name} ({cat})")
        return "\n".join(lines)

    def _format_report_info(self, report: BusinessReport) -> str:
        """Format report for prompt."""
        return f"""- 매장: {report.store_name}
- 내용: {report.description}
- 타겟: {', '.join(report.target_age_groups)}
- 어필: {report.appeal_factor} (강도: {report.appeal_strength})"""

    def _fallback_store_selection(
        self, stores: Union[pd.DataFrame, List[Dict]], category: str, lifestyle: str
    ) -> Dict[str, Any]:
        """Fallback store selection."""
        candidates = []
        if isinstance(stores, pd.DataFrame):
            for _, store in stores.iterrows():
                cat = store.get("카테고리", store.get("category", ""))
                if category.lower() in cat.lower():
                    candidates.append(store)
        else:
            for store in stores:
                cat = store.get("카테고리", store.get("category", ""))
                if category.lower() in cat.lower():
                    candidates.append(store)

        if not candidates:
            # Use any store
            if isinstance(stores, pd.DataFrame):
                candidates = [row for _, row in stores.iterrows()]
            else:
                candidates = stores

        if candidates:
            selected = random.choice(candidates)
            name = selected.get("장소명", selected.get("store_name", "Unknown"))
            return {"store_name": name, "reason": "선택", "influenced_by_report": False}

        return {"store_name": None, "reason": "매장 없음", "influenced_by_report": False}

    # ==================== Main Process ====================

    def process(
        self,
        visible_stores: Union[pd.DataFrame, List[Dict]],
        agent: AgentPersona,
        report: Optional[BusinessReport] = None,
        time_slot: str = "",
        weekday: str = "",
        memory_context: str = "",
    ) -> dict:
        """
        Main decision process with chained prompts.
        Returns standard decision dict.
        """
        state = self._get_agent_state(agent)
        state.update_for_timeslot(time_slot, weekday)

        # Step 1: Timeslot decision
        step1 = self._step1_timeslot_decision(agent, state, time_slot, weekday)
        step1["time_slot"] = time_slot

        decision_type = step1.get("decision", "skip")

        # If not eating out, return ignore
        if decision_type in ["skip", "home", "delivery", "walk"]:
            return {
                "decision": "ignore",
                "decision_reason": step1.get("reason", "외식 안 함"),
                "visited_store": None,
                "visited_category": None,
                "decision_chain": {"step1": step1},
            }

        # Step 2: Category selection (for dine_in, takeout, cafe, drink, snack)
        available_categories = self._extract_categories(visible_stores)
        step2 = self._step2_category_selection(agent, state, step1, available_categories)

        category = step2.get("category", "")

        # Step 3: Store selection
        step3 = self._step3_store_selection(
            agent, state, category, visible_stores, report, step1
        )

        store_name = step3.get("store_name")

        if store_name:
            # Find category for the store
            store_category = self._find_store_category(visible_stores, store_name) or category

            # Record the meal
            state.record_meal(time_slot, store_name, store_category)

            return {
                "decision": "visit",
                "decision_reason": f"{step1.get('reason', '')} → {step2.get('reason', '')} → {step3.get('reason', '')}",
                "visited_store": store_name,
                "visited_category": store_category,
                "influenced_by_report": step3.get("influenced_by_report", False),
                "decision_chain": {"step1": step1, "step2": step2, "step3": step3},
            }
        else:
            return {
                "decision": "ignore",
                "decision_reason": "적합한 매장 없음",
                "visited_store": None,
                "visited_category": None,
                "decision_chain": {"step1": step1, "step2": step2, "step3": step3},
            }

    def _extract_categories(self, stores: Union[pd.DataFrame, List[Dict]]) -> List[str]:
        """Extract unique categories from stores."""
        categories = set()
        if isinstance(stores, pd.DataFrame):
            for _, store in stores.iterrows():
                cat = store.get("카테고리", store.get("category", ""))
                if cat:
                    categories.add(cat)
        else:
            for store in stores:
                cat = store.get("카테고리", store.get("category", ""))
                if cat:
                    categories.add(cat)
        return list(categories)

    def _find_store_category(
        self, stores: Union[pd.DataFrame, List[Dict]], store_name: str
    ) -> Optional[str]:
        """Find category for a store name."""
        if isinstance(stores, pd.DataFrame):
            for _, store in stores.iterrows():
                name = store.get("장소명", store.get("store_name", ""))
                if name == store_name:
                    return store.get("카테고리", store.get("category"))
        else:
            for store in stores:
                name = store.get("장소명", store.get("store_name", ""))
                if name == store_name:
                    return store.get("카테고리", store.get("category"))
        return None

    def reset_daily_states(self):
        """Reset all agent states for a new day."""
        for state in self._agent_states.values():
            state.reset_for_new_day()
