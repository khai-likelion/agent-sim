"""
Action Algorithm Module - Stanford Generative Agents 기반 4단계 의사결정.

매 timeslot마다 수행하는 4단계 로직:
- Step 1: 목적지 유형 결정 (아침/점심: 식당/카페, 저녁/야간: 식당/주점/카페)
- Step 2: 업종 선택 (메모리 + 건강추구 성향 기반)
- Step 3: 매장 선택 (가용비 + 변화추구 + 평점 기반)
- Step 4: 평가 및 피드백 (맛/가성비 평점 생성, 실시간 평점 축적)
"""

import json
import time
import random
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

import pandas as pd

from src.simulation_layer.persona.generative_agent import GenerativeAgent, HEALTH_PREFERENCES
from src.data_layer.global_store import get_global_store, GlobalStore, StoreRating
from src.ai_layer.llm_client import create_llm_client, LLMClient
from config import get_settings


# 시간대별 가능한 목적지 유형
DESTINATION_TYPES = {
    "아침": ["식당", "카페"],
    "점심": ["식당", "카페"],
    "저녁": ["식당", "주점", "카페"],
    "야간": ["식당", "주점"],  # 야간에는 카페 제외
}

# 목적지 유형과 카테고리 매핑
DESTINATION_CATEGORIES = {
    "식당": ["한식", "중식", "일식", "양식", "분식", "패스트푸드", "국밥", "찌개", "고기", "치킨"],
    "카페": ["카페", "커피", "디저트", "베이커리", "브런치"],
    "주점": ["호프", "이자카야", "포차", "와인바", "술집", "막걸리", "칵테일바", "칵테일"],
}


class ActionAlgorithm:
    """
    4단계 의사결정 알고리즘.
    LLM을 사용하여 에이전트의 페르소나와 메모리를 기반으로 결정.
    """

    def __init__(self, rate_limit_delay: float = 0.5):
        self.rate_limit_delay = rate_limit_delay
        self.llm_client: Optional[LLMClient] = None
        self.global_store: GlobalStore = get_global_store()
        self._prompt_templates: Dict[str, str] = {}

    def _get_llm_client(self) -> LLMClient:
        if self.llm_client is None:
            self.llm_client = create_llm_client()
        return self.llm_client

    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """LLM 호출 with retry"""
        client = self._get_llm_client()

        for attempt in range(max_retries):
            try:
                response = client.generate_sync(prompt)
                time.sleep(self.rate_limit_delay)
                return response
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower():
                    wait_time = (attempt + 1) * 10
                    print(f"  Rate limit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e

        raise Exception(f"LLM call failed after {max_retries} retries")

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """LLM 응답에서 JSON 추출"""
        try:
            # JSON 블록 찾기
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        return {}

    # ==================== Step 1: 목적지 유형 결정 ====================

    def step1_destination_type(
        self,
        agent: GenerativeAgent,
        time_slot: str,
        weekday: str,
    ) -> Dict[str, Any]:
        """
        Step 1: 목적지 유형 결정
        - 아침/점심: 식당, 카페 중 선택
        - 저녁/야간: 식당, 주점, 카페 중 선택 (야간은 카페 제외)

        Returns: {"go_out": bool, "destination_type": str, "reason": str}
        """
        available_types = DESTINATION_TYPES.get(time_slot, ["식당"])
        meal_prob = agent.get_meal_probability(time_slot)

        # 주말 보정
        if weekday in ["토", "일"]:
            meal_prob = min(1.0, meal_prob * 1.2)

        prompt = f"""당신은 {agent.name}입니다.

{agent.get_persona_summary()}

현재 상황:
- 요일: {weekday}요일
- 시간대: {time_slot}
- 가능한 외출 유형: {', '.join(available_types)}

{agent.get_memory_context()}

질문: 이 시간대에 외출해서 식사/음료를 할 것인지 결정하세요.
외출한다면, {', '.join(available_types)} 중 어디로 갈지 선택하세요.

반드시 아래 JSON 형식으로만 응답하세요:
{{"go_out": true/false, "destination_type": "식당/카페/주점 중 하나 또는 null", "reason": "간단한 이유"}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            if result and "go_out" in result:
                return result
        except Exception as e:
            print(f"Step1 LLM error: {e}")

        # Fallback: 확률 기반 결정
        go_out = random.random() < meal_prob
        if go_out:
            dest_type = random.choice(available_types)
            return {"go_out": True, "destination_type": dest_type, "reason": "외출 결정"}
        else:
            return {"go_out": False, "destination_type": None, "reason": "집에서 해결"}

    # ==================== Step 2: 업종 선택 ====================

    def step2_category_selection(
        self,
        agent: GenerativeAgent,
        destination_type: str,
        time_slot: str,
    ) -> Dict[str, Any]:
        """
        Step 2: 업종 선택
        메모리 모듈의 최근 먹은 이력 + 건강추구 성향을 고려

        Returns: {"category": str, "reason": str}
        """
        # 가능한 카테고리 목록
        available_categories = DESTINATION_CATEGORIES.get(destination_type, ["한식"])

        # 건강 성향에 따른 선호/기피 카테고리
        preferred = agent.preferred_categories
        avoided = agent.avoided_categories

        # 최근 방문 카테고리
        recent_categories = agent.get_recent_categories(5)
        recent_text = ", ".join(recent_categories) if recent_categories else "없음"

        prompt = f"""당신은 {agent.name}입니다.

{agent.get_persona_summary()}

현재 상황:
- 시간대: {time_slot}
- 목적지 유형: {destination_type}
- 선택 가능한 업종: {', '.join(available_categories)}

건강 성향 "{agent.health_preference}":
- 선호하는 음식: {', '.join(preferred[:5])}
- 기피하는 음식: {', '.join(avoided[:3])}

최근 먹은 음식 종류: {recent_text}

질문: 어떤 업종(종류)의 음식을 먹을지 선택하세요.
최근에 자주 먹은 음식은 피하고, 건강 성향을 고려하세요.

반드시 아래 JSON 형식으로만 응답하세요:
{{"category": "선택한 업종", "reason": "선택 이유"}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            if result and result.get("category"):
                return result
        except Exception as e:
            print(f"Step2 LLM error: {e}")

        # Fallback: 건강 성향 기반 랜덤 선택
        # 최근 먹은 카테고리 제외
        candidates = [c for c in available_categories if c not in recent_categories]
        if not candidates:
            candidates = available_categories

        # 선호 카테고리 우선
        preferred_candidates = [c for c in candidates if any(p in c for p in preferred)]
        if preferred_candidates:
            category = random.choice(preferred_candidates)
        else:
            category = random.choice(candidates)

        return {"category": category, "reason": f"{agent.health_preference} 성향 기반 선택"}

    # ==================== Step 3: 매장 선택 ====================

    def step3_store_selection(
        self,
        agent: GenerativeAgent,
        category: str,
        nearby_stores: List[StoreRating],
        time_slot: str,
    ) -> Dict[str, Any]:
        """
        Step 3: 매장 선택
        가용비 + 변화추구 성향 + 누적평점을 고려

        Returns: {"store_name": str, "reason": str}
        """
        # 예산 내 매장 필터링
        affordable_stores = [s for s in nearby_stores if s.average_price <= agent.budget_per_meal]
        if not affordable_stores:
            affordable_stores = nearby_stores[:10]  # 예산 초과해도 일부 표시

        # 카테고리 매칭 매장 우선
        category_stores = [s for s in affordable_stores if category.lower() in s.category.lower()]
        if not category_stores:
            category_stores = affordable_stores

        # 최대 10개까지만
        display_stores = category_stores[:10]

        if not display_stores:
            return {"store_name": None, "reason": "주변에 적합한 매장 없음"}

        # 매장 정보 포맷팅 (기존 평점 + 에이전트 평점 모두 표시)
        store_info_lines = []
        for store in display_stores:
            store_info_lines.append(store.get_combined_info_for_prompt())

        stores_text = "\n\n".join(store_info_lines)

        # 최근 방문 매장
        recent_stores = agent.get_recent_stores(5)
        recent_text = ", ".join(recent_stores) if recent_stores else "없음"

        prompt = f"""당신은 {agent.name}입니다.

{agent.get_persona_summary()}

현재 상황:
- 시간대: {time_slot}
- 원하는 업종: {category}
- 한끼 가용비: {agent.budget_per_meal:,}원

변화 성향 "{agent.change_preference}":
- {agent.change_description}

최근 방문한 매장: {recent_text}

주변 매장 정보 (기존 리뷰 평점 + 최근 방문자 평점 둘 다 참고하세요):
{stores_text}

질문: 어떤 매장을 방문할지 선택하세요.
- "{agent.change_preference}" 성향에 맞게 선택하세요
- 기존 리뷰 평점과 최근 방문자 평점을 모두 참고하세요
- 예산({agent.budget_per_meal:,}원) 내에서 선택하세요

반드시 아래 JSON 형식으로만 응답하세요:
{{"store_name": "선택한 매장명", "reason": "선택 이유"}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            if result and result.get("store_name"):
                # 매장명 검증
                selected_name = result["store_name"]
                valid_names = [s.store_name for s in display_stores]
                if selected_name in valid_names:
                    return result
                # 부분 매칭 시도
                for name in valid_names:
                    if selected_name in name or name in selected_name:
                        result["store_name"] = name
                        return result
                # 첫 번째 매장으로 폴백
                result["store_name"] = display_stores[0].store_name
                return result
        except Exception as e:
            print(f"Step3 LLM error: {e}")

        # Fallback: 변화 성향 기반 선택
        if agent.change_preference == "도전추구":
            # 방문하지 않은 매장 중 평점 높은 곳
            unvisited = [s for s in display_stores if s.store_name not in recent_stores]
            if unvisited:
                # 에이전트 평점이 있으면 그것 우선, 없으면 기존 평점
                unvisited.sort(key=lambda s: s.agent_overall_score or s.base_overall_score, reverse=True)
                return {"store_name": unvisited[0].store_name, "reason": "새로운 매장 도전"}
        else:
            # 안정추구: 방문했던 매장 중 평점 높은 곳
            visited = [s for s in display_stores if s.store_name in recent_stores]
            if visited:
                visited.sort(key=lambda s: s.agent_overall_score or s.base_overall_score, reverse=True)
                return {"store_name": visited[0].store_name, "reason": "익숙한 매장 재방문"}

        # 기본: 평점 높은 매장
        display_stores.sort(key=lambda s: s.base_overall_score, reverse=True)
        return {"store_name": display_stores[0].store_name, "reason": "평점 기반 선택"}

    # ==================== Step 4: 평가 및 피드백 ====================

    def _calculate_taste_rating(
        self,
        agent: GenerativeAgent,
        store: StoreRating,
    ) -> int:
        """
        맛 평점 계산 (0, 1, 2)

        기본값: 1 (보통)
        - 건강 성향과 카테고리 일치도
        - 매장의 기존 누적 평점
        """
        # 기본값: 보통(1)
        taste = 1

        # 1. 건강 성향과 카테고리 일치도 확인
        category_lower = store.category.lower()
        preferred = agent.preferred_categories
        avoided = agent.avoided_categories

        # 선호 카테고리 매칭 시 +1
        category_match = any(p.lower() in category_lower for p in preferred)
        # 기피 카테고리 매칭 시 -1
        category_avoid = any(a.lower() in category_lower for a in avoided)

        if category_match:
            taste += 1
        if category_avoid:
            taste -= 1

        # 2. 매장의 기존 누적 평점 반영
        base_score = store.base_overall_score

        if base_score >= 0.7:
            # 높은 평점 매장: +0.5 확률로 +1
            if random.random() < 0.5:
                taste += 1
        elif base_score < 0.3:
            # 낮은 평점 매장: -0.5 확률로 -1
            if random.random() < 0.5:
                taste -= 1

        # 3. 에이전트 평점이 있는 경우 추가 반영
        if store.agent_overall_score is not None:
            if store.agent_overall_score >= 0.7:
                taste += 1
            elif store.agent_overall_score < 0.3:
                taste -= 1

        # 0~2 범위로 클램핑
        return max(0, min(2, taste))

    def _calculate_value_rating(
        self,
        taste_rating: int,
        store: StoreRating,
        budget: int,
    ) -> int:
        """
        가성비 평점 계산 (0, 1, 2)

        ★ 가성비는 반드시 맛 점수에 종속됨 ★
        - 맛 0점 → 가성비 무조건 0점
        - 맛 2점 + 가격 ≤ 110% → 가성비 2점
        - 맛 1점 + 가격 ≤ 80% → 가성비 2점
        - 맛 1점 + 가격 80~110% → 가성비 1점
        - 맛 2점 + 가격 > 110% → 가성비 1점
        - 맛 1점 + 가격 > 120% → 가성비 0점
        """
        # 맛이 0점이면 가성비도 무조건 0점
        if taste_rating == 0:
            return 0

        # 가격 대비 예산 비율 계산
        price_ratio = store.average_price / budget if budget > 0 else 1.0

        if taste_rating == 2:
            # 맛 2점인 경우
            if price_ratio <= 1.1:  # 가격 ≤ 110%
                return 2  # 가성비 좋음
            else:  # 가격 > 110%
                return 1  # 가성비 보통

        elif taste_rating == 1:
            # 맛 1점인 경우
            if price_ratio <= 0.8:  # 가격 ≤ 80%
                return 2  # 가성비 좋음
            elif price_ratio <= 1.1:  # 가격 80~110%
                return 1  # 가성비 보통
            elif price_ratio <= 1.2:  # 가격 110~120%
                return 1  # 가성비 보통
            else:  # 가격 > 120%
                return 0  # 가성비 별로

        return 1  # 기본값

    def step4_evaluate_and_feedback(
        self,
        agent: GenerativeAgent,
        store: StoreRating,
        visit_datetime: str,
    ) -> Dict[str, Any]:
        """
        Step 4: 평가 및 피드백

        새로운 평가 원칙:
        1. 기본 만족도는 '보통(1점)'
        2. 맛: 건강 성향 + 카테고리 매칭 + 기존 평점
        3. 가성비: 맛 점수에 종속 (맛 0점 → 가성비 0점)

        Returns: {"taste_rating": int, "value_rating": int, "comment": str}
        """
        # 맛 평점 계산 (규칙 기반)
        taste = self._calculate_taste_rating(agent, store)

        # 가성비 평점 계산 (맛에 종속)
        value = self._calculate_value_rating(taste, store, agent.budget_per_meal)

        # 코멘트 생성 (LLM 사용)
        comment = self._generate_comment(agent, store, taste, value)

        # GlobalStore에 평점 추가
        self.global_store.add_agent_rating(
            store_name=store.store_name,
            agent_id=agent.id,
            agent_name=agent.name,
            taste_rating=taste,
            value_rating=value,
            visit_datetime=visit_datetime,
        )

        # 에이전트 메모리에도 추가
        agent.add_visit(
            store_name=store.store_name,
            category=store.category,
            taste_rating=taste,
            value_rating=value,
        )

        return {"taste_rating": taste, "value_rating": value, "comment": comment}

    def _generate_comment(
        self,
        agent: GenerativeAgent,
        store: StoreRating,
        taste: int,
        value: int,
    ) -> str:
        """평가 코멘트 생성"""
        taste_text = {0: "기대 이하", 1: "무난함", 2: "만족"}
        value_text = {0: "비쌈", 1: "적당함", 2: "가성비 좋음"}

        # 간단한 규칙 기반 코멘트
        if taste == 2 and value == 2:
            return f"맛도 좋고 가격도 합리적! {agent.health_preference} 성향에 딱 맞음"
        elif taste == 2 and value == 1:
            return f"맛은 좋은데 가격이 조금 있음"
        elif taste == 1 and value == 2:
            return f"맛은 평범한데 가격이 저렴해서 괜찮음"
        elif taste == 1 and value == 1:
            return f"무난한 식사"
        elif taste == 0:
            return f"{agent.health_preference} 성향과 맞지 않음"
        else:
            return f"맛: {taste_text[taste]}, 가성비: {value_text[value]}"

    # ==================== Main Process ====================

    def process_decision(
        self,
        agent: GenerativeAgent,
        nearby_stores: List[StoreRating],
        time_slot: str,
        weekday: str,
        current_datetime: str,
    ) -> Dict[str, Any]:
        """
        4단계 의사결정 전체 프로세스 수행.

        Returns: {
            "decision": "visit" | "stay_home",
            "steps": {step1, step2, step3, step4},
            "visited_store": str | None,
            "visited_category": str | None,
            "ratings": {taste, value} | None,
        }
        """
        # Step 1: 목적지 유형 결정
        step1 = self.step1_destination_type(agent, time_slot, weekday)

        if not step1.get("go_out", False):
            return {
                "decision": "stay_home",
                "steps": {"step1": step1},
                "visited_store": None,
                "visited_category": None,
                "ratings": None,
                "reason": step1.get("reason", "외출 안 함"),
            }

        destination_type = step1.get("destination_type", "식당")

        # Step 2: 업종 선택
        step2 = self.step2_category_selection(agent, destination_type, time_slot)
        category = step2.get("category", "한식")

        # Step 3: 매장 선택
        step3 = self.step3_store_selection(agent, category, nearby_stores, time_slot)
        store_name = step3.get("store_name")

        if not store_name:
            return {
                "decision": "stay_home",
                "steps": {"step1": step1, "step2": step2, "step3": step3},
                "visited_store": None,
                "visited_category": None,
                "ratings": None,
                "reason": "적합한 매장 없음",
            }

        # 매장 정보 가져오기
        store = self.global_store.get_by_name(store_name)
        if not store:
            # nearby_stores에서 찾기
            for s in nearby_stores:
                if s.store_name == store_name:
                    store = s
                    break

        if not store:
            return {
                "decision": "stay_home",
                "steps": {"step1": step1, "step2": step2, "step3": step3},
                "visited_store": None,
                "visited_category": None,
                "ratings": None,
                "reason": "매장 정보 없음",
            }

        # Step 4: 평가 및 피드백
        step4 = self.step4_evaluate_and_feedback(agent, store, current_datetime)

        return {
            "decision": "visit",
            "steps": {"step1": step1, "step2": step2, "step3": step3, "step4": step4},
            "visited_store": store_name,
            "visited_category": store.category,
            "ratings": {
                "taste": step4["taste_rating"],
                "value": step4["value_rating"],
            },
            "reason": f"{step1.get('reason', '')} → {step2.get('reason', '')} → {step3.get('reason', '')}",
        }


if __name__ == "__main__":
    # 테스트
    from src.simulation_layer.persona.generative_agent import GenerativeAgentFactory

    # 에이전트 생성
    factory = GenerativeAgentFactory()
    agents = factory.generate_unique_agents(max_count=3)

    # GlobalStore 초기화
    from config import get_settings
    settings = get_settings()

    GlobalStore.reset_instance()
    global_store = get_global_store()
    global_store.load_from_csv(settings.paths.stores_csv)

    # 테스트 매장
    test_stores = list(global_store.stores.values())[:10]

    # Action Algorithm 테스트
    algorithm = ActionAlgorithm(rate_limit_delay=0.5)

    print("=== Action Algorithm 테스트 ===\n")

    for agent in agents[:1]:
        print(agent.get_persona_summary())
        print()

        result = algorithm.process_decision(
            agent=agent,
            nearby_stores=test_stores,
            time_slot="점심",
            weekday="수",
            current_datetime="2025-02-05 12:00:00",
        )

        print(f"결정: {result['decision']}")
        print(f"방문 매장: {result['visited_store']}")
        print(f"카테고리: {result['visited_category']}")
        print(f"평점: {result['ratings']}")
        print(f"이유: {result['reason']}")
        print()
