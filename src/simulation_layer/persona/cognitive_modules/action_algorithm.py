"""
Action Algorithm Module - Stanford Generative Agents 기반 4단계 의사결정.

매 timeslot마다 수행하는 4단계 로직:
- Step 1: 목적지 유형 결정 (아침/점심: 식당/카페, 저녁/야식: 식당/주점/카페)
- Step 2: 업종 선택 (메모리 + 건강추구 성향 기반)
- Step 3: 매장 선택 (매장 정보 + 에이전트 평점 기반)
- Step 4: 평가 및 피드백 (맛/가성비/분위기 평점 생성, LLM 기반)

평점 체계: 0~5점
- 0: 방문 없음 (기본값)
- 1: 매우별로
- 2: 별로
- 3: 보통
- 4: 좋음
- 5: 매우좋음
"""

import asyncio
import json
import random
import time
from typing import Optional, List, Dict, Any
from pathlib import Path

from src.simulation_layer.persona.agent import GenerativeAgent
from src.data_layer.global_store import get_global_store, GlobalStore, StoreRating
from src.ai_layer.llm_client import create_llm_client, LLMClient
from src.ai_layer.prompts import render_prompt, STEP1_DESTINATION, STEP2_CATEGORY, STEP3_STORE, STEP4_EVALUATE, STEP5_NEXT_ACTION
from config import get_settings


# 시간대별 가능한 목적지 유형
DESTINATION_TYPES = {
    "아침": ["식당", "카페"],
    "점심": ["식당", "카페"],
    "저녁": ["식당", "주점", "카페"],
    "야식": ["식당", "주점"],  # 야식에는 카페 제외
}

# 시간대별 기본 외식 확률 (현실적 수치)
BASE_EATING_OUT_PROB = {
    "아침": 0.40,
    "점심": 0.70,
    "저녁": 0.60,
    "야식": 0.20,
}

# 목적지 유형과 카테고리 매핑
DESTINATION_CATEGORIES = {
    "식당": ["한식", "중식", "일식", "양식", "분식", "패스트푸드", "국밥", "찌개", "고기", "치킨"],
    "카페": ["카페", "커피", "디저트", "베이커리", "브런치"],
    "주점": ["호프", "이자카야", "포차", "와인바", "술집", "막걸리", "칵테일바", "칵테일"],
}


class LLMCallFailedError(Exception):
    """LLM 호출 실패 예외"""
    pass


class ActionAlgorithm:
    """
    4단계 의사결정 알고리즘.
    LLM을 사용하여 에이전트의 페르소나와 메모리를 기반으로 결정.
    Fallback 없음 - LLM 실패 시 예외 발생.
    """

    def __init__(self, rate_limit_delay: float = 0.5, semaphore: Optional[asyncio.Semaphore] = None):
        self.rate_limit_delay = rate_limit_delay
        self.llm_client: Optional[LLMClient] = None
        self.global_store: GlobalStore = get_global_store()
        # None이면 호출 시 생성 (asyncio.run() 내부에서 생성해야 함)
        self._semaphore = semaphore

    def _get_llm_client(self) -> LLMClient:
        if self.llm_client is None:
            self.llm_client = create_llm_client()
        return self.llm_client

    async def _call_llm_async(self, prompt: str, max_retries: int = 3) -> str:
        """비동기 LLM 호출 with retry. Semaphore로 동시 호출 수를 제한."""
        client = self._get_llm_client()
        semaphore = self._semaphore

        async def _attempt():
            for attempt in range(max_retries):
                try:
                    response = await client.generate(prompt)
                    await asyncio.sleep(self.rate_limit_delay)
                    return response
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "rate" in error_str.lower():
                        wait_time = (attempt + 1) * 10
                        print(f"  Rate limit, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        if attempt == max_retries - 1:
                            raise LLMCallFailedError(f"LLM call failed: {e}")
                        continue
            raise LLMCallFailedError(f"LLM call failed after {max_retries} retries")

        if semaphore is not None:
            async with semaphore:
                return await _attempt()
        else:
            return await _attempt()

    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """동기 LLM 호출 (하위 호환 / __main__ 테스트용). 실패 시 LLMCallFailedError 발생."""
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
                    if attempt == max_retries - 1:
                        raise LLMCallFailedError(f"LLM call failed: {e}")
                    continue

        raise LLMCallFailedError(f"LLM call failed after {max_retries} retries")

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """LLM 응답에서 JSON 추출"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        return {}

    def _get_store_info_for_prompt(self, store: StoreRating) -> str:
        """
        매장 정보를 LLM 프롬프트용으로 포맷팅.

        인식 필드: store_id, store_name, category, market_analysis, revenue_analysis,
                  customer_analysis, review_metrics.overall_sentiment.comparison,
                  raw_data_context.trend_history, metadata, top_keywords,
                  critical_feedback, rag_context + 에이전트 평점
        """
        lines = [f"[{store.store_name}]"]
        lines.append(f"  store_id: {store.store_id}")
        lines.append(f"  카테고리: {store.category}")
        lines.append(f"  평균가격: {store.average_price:,}원")

        # 키워드
        if store.top_keywords:
            keywords = store.top_keywords[:5] if isinstance(store.top_keywords, list) else store.top_keywords.split(',')[:5]
            lines.append(f"  주요키워드: {', '.join(keywords)}")

        # RAG 컨텍스트 (매장 설명)
        if store.rag_context and len(store.rag_context) > 10:
            summary = store.rag_context[:150] + "..." if len(store.rag_context) > 150 else store.rag_context
            lines.append(f"  설명: {summary}")

        # 에이전트 평점 (있으면 표시)
        if store.agent_rating_count > 0:
            lines.append(f"  에이전트 평점: {store.agent_avg_rating:.1f}/5 ({store.agent_rating_count}건), 태그: 맛 {store.taste_count}, 가성비 {store.value_count}, 분위기 {store.atmosphere_count}, 서비스 {store.service_count}")

            # 최근 3개 평가
            rating_labels = {0: "없음", 1: "매우별로", 2: "별로", 3: "보통", 4: "좋음", 5: "매우좋음"}
            for r in store.agent_ratings[-3:]:
                lines.append(f"    - {r.agent_name}: {rating_labels.get(r.rating, '?')} {r.selected_tags}")
        else:
            lines.append("  에이전트 평점: 아직 없음")

        return "\n".join(lines)

    # ==================== Step 1: 망원동 내 식사 여부 결정 ====================

    def step1_eat_in_mangwon(
        self,
        agent: GenerativeAgent,
        time_slot: str,
        weekday: str,
        current_date: str = "",
    ) -> Dict[str, Any]:
        """
        Step 1: 망원동 내 식사 여부 결정 (확률 기반)

        시간대별 기본 확률에 보정을 적용.
        - 오늘 식사 횟수가 2회 이상이면 확률 감소 (하루 2~3끼)
        - 주말이면 확률 약간 증가

        Returns: {"eat_in_mangwon": bool, "reason": str}
        """
        base_prob = BASE_EATING_OUT_PROB.get(time_slot, 0.50)

        # 오늘 식사 횟수 보정 (하루 2~3끼 제한)
        meals_today = len(agent.get_meals_today(current_date)) if current_date else len(agent.recent_history)
        if meals_today >= 3:
            base_prob *= 0.1  # 3끼 이상이면 크게 감소
        elif meals_today >= 2:
            base_prob *= 0.5  # 2끼면 절반
        elif meals_today >= 1 and time_slot == "아침":
            pass  # 아침에 이미 먹었으면 그대로

        # 주말 보정
        if weekday in ["토", "일"]:
            base_prob = min(1.0, base_prob * 1.15)

        # 확률적 결정
        eat_out = random.random() < base_prob

        if eat_out:
            reasons = {
                "아침": "아침 식사를 위해 외출",
                "점심": "점심 시간 외식",
                "저녁": "저녁 외식",
                "야식": "야식이 땡김",
            }
            reason = reasons.get(time_slot, "외식")
        else:
            if time_slot == "아침" or meals_today == 0:
                reasons_skip = [
                    "배가 안 고픔",
                    "집에서 해결",
                    "이 시간에는 안 먹음",
                ]
            else:
                reasons_skip = [
                    "배가 안 고픔",
                    "이전 식사로 충분",
                    "집에서 해결",
                    "이 시간에는 안 먹음",
                ]
            reason = random.choice(reasons_skip)

        return {"eat_in_mangwon": eat_out, "reason": reason}

    # ==================== Step 2: 업종 선택 ====================

    async def step2_category_selection(
        self,
        agent: GenerativeAgent,
        destination_type: str,
        time_slot: str,
        current_date: str = "",
    ) -> Dict[str, Any]:
        """
        Step 2: 업종 선택
        LLM 실패 시 LLMCallFailedError 발생 (Fallback 없음)

        Returns: {"category": str, "reason": str}
        """
        available_categories = DESTINATION_CATEGORIES.get(destination_type, ["한식"])

        prompt = render_prompt(
            STEP2_CATEGORY,
            agent_name=agent.persona_id,
            persona_summary=agent.get_persona_summary(),
            time_slot=time_slot,
            destination_type=destination_type,
            available_categories=", ".join(available_categories),
            memory_context=agent.get_memory_context(current_date),
        )

        response = await self._call_llm_async(prompt)
        result = self._parse_json_response(response)

        if result and result.get("category"):
            return result

        raise LLMCallFailedError("Step2: Failed to parse LLM response")

    # ==================== Step 3: 매장 선택 ====================

    async def step3_store_selection(
        self,
        agent: GenerativeAgent,
        category: str,
        nearby_stores: List[StoreRating],
        time_slot: str,
        improvement_text: Optional[str] = None,
        current_date: str = "",
    ) -> Dict[str, Any]:
        """
        Step 3: 매장 선택
        LLM 실패 시 LLMCallFailedError 발생 (Fallback 없음)

        Args:
            improvement_text: 개선사항 텍스트 (있으면 LLM에 전달)

        Returns: {"store_name": str, "reason": str}
        """
        affordable_stores = nearby_stores

        # 카테고리 매칭 매장 우선
        category_stores = [s for s in affordable_stores if category.lower() in s.category.lower()]
        if not category_stores:
            category_stores = affordable_stores

        display_stores = category_stores

        if not display_stores:
            return {"store_name": None, "reason": "주변에 적합한 매장 없음", "llm_failed": False}

        # 매장 순서 셔플 (LLM의 위치 편향 방지)
        display_stores = list(display_stores)
        random.shuffle(display_stores)

        # 매장 정보 포맷팅
        store_info_lines = []
        for store in display_stores:
            store_info_lines.append(self._get_store_info_for_prompt(store))

        stores_text = "\n\n".join(store_info_lines)

        # 개선사항 텍스트 추가
        improvement_section = ""
        if improvement_text:
            improvement_section = f"\n\n[개선사항 안내]\n{improvement_text}"

        prompt = render_prompt(
            STEP3_STORE,
            agent_name=agent.persona_id,
            persona_summary=agent.get_persona_summary(),
            time_slot=time_slot,
            category=category,
            memory_context=agent.get_memory_context(current_date),
            stores_text=stores_text,
            improvement_section=improvement_section,
        )

        response = await self._call_llm_async(prompt)
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

        raise LLMCallFailedError("Step3: Failed to parse LLM response or invalid store name")

    # ==================== Step 4: 평가 및 피드백 ====================

    async def step4_evaluate_and_feedback(
        self,
        agent: GenerativeAgent,
        store: StoreRating,
        visit_datetime: str,
        improvement_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Step 4: 평가 및 피드백
        LLM을 사용하여 맛/가성비/분위기 평점(0~5)을 생성

        평점 체계:
        - 0: 방문 없음 (기본값, 평가 시에는 사용 안 함)
        - 1: 매우별로
        - 2: 별로
        - 3: 보통
        - 4: 좋음
        - 5: 매우좋음

        Returns: {"taste_rating": int, "value_rating": int, "atmosphere_rating": int, "comment": str}
        """
        store_info = self._get_store_info_for_prompt(store)

        # 개선사항 텍스트
        improvement_section = ""
        if improvement_text:
            improvement_section = f"\n\n[이 매장의 개선사항]\n{improvement_text}"

        # 1. 말투 설정 (페르소나 기반)
        tone_instruction = "자연스럽게 작성하세요."
        if agent.generation in ["Z1", "Z2"]:
            tone_instruction = "Z세대 말투(존나, 개꿀맛, ㅋ, 이모티콘 등)를 사용해서 친구한테 말하듯이 아주 솔직하고 짧게 작성하세요."
        elif agent.generation == "Y":
            tone_instruction = "밀레니얼 세대 말투로, 적당히 트렌디하면서도 정보성 있게 작성하세요."
        elif agent.generation == "X":
            tone_instruction = "X세대 말투로, 점잖으면서도 꼼꼼하게 분석하듯이 작성하세요."
        elif agent.generation == "S":
            tone_instruction = "어르신 말투로, 구체적이고 진중하게 작성하세요."
        
        # 2. 평가 관점 랜덤 선택 (다양성 유도)
        import random
        focus_aspects = ["맛/퀄리티", "가성비", "매장 분위기/인테리어", "직원 서비스/친절도", "매장 청결/위생", "특색있는 메뉴"]
        selected_focus = random.choice(focus_aspects)

        prompt = render_prompt(
            STEP4_EVALUATE,
            agent_name=agent.persona_id,
            store_name=store.store_name,
            persona_summary=agent.get_persona_summary(),
            store_info=store_info,
            improvement_section=improvement_section,
            tone_instruction=tone_instruction,
            focus_aspect=selected_focus,
        )

        response = await self._call_llm_async(prompt)
        result = self._parse_json_response(response)

        if result and "rating" in result:
            # 점수 범위 검증 (1~5)
            rating = max(1, min(5, int(result["rating"])))
            selected_tags = result.get("selected_tags", [])
            comment = result.get("comment", "")

            # GlobalStore 평점 버퍼에 추가 (다음 타임슬롯에 반영)
            self.global_store.add_pending_rating(
                store_name=store.store_name,
                agent_id=agent.id,
                agent_name=agent.persona_id,
                rating=rating,
                selected_tags=selected_tags,
                visit_datetime=visit_datetime,
                comment=comment,  # 리뷰 코멘트 전달
            )

            # 에이전트 메모리에도 추가
            agent.add_visit(
                store_name=store.store_name,
                category=store.category,
                taste_rating=rating,
                value_rating=rating,
                atmosphere_rating=rating,
                visit_datetime=visit_datetime,
                comment=comment,
            )

            return {
                "rating": rating,
                "selected_tags": selected_tags,
                "comment": comment
            }

        raise LLMCallFailedError("Step4: Failed to parse LLM response")

    # ==================== Step 5: 다음 행동 결정 ====================

    async def step5_next_action(
        self,
        agent: GenerativeAgent,
        current_time_slot: str,
        next_time_slot: str,
        weekday: str,
        last_action: str,
        current_date: str = "",
    ) -> Dict[str, Any]:
        """
        Step 5: 다음 시간대까지 무엇을 할지 결정
        LLM 실패 시 LLMCallFailedError 발생 (Fallback 없음)

        선택지:
        - 집에서_쉬기: 집에 돌아가서 휴식
        - 카페_가기: 근처 카페에서 시간 보내기
        - 배회하기: 망원동 거리를 걸으며 구경하기
        - 한강공원_산책: 망원한강공원에서 산책
        - 망원시장_장보기: 망원시장에서 장보기
        - 회사_가기: 직장에 출근하기 (직장인만)

        Returns: {"action": str, "walking_speed": float, "reason": str, "destination": dict | None}

        walking_speed는 LLM이 페르소나 특징을 보고 직접 판단한 km/h 값
        """
        # 에이전트 유형별 선택지 분리
        if agent.is_resident:
            available_actions = (
                "- 집에서_쉬기: 집에 돌아가서 휴식\n"
                "- 카페_가기: 근처 카페에서 시간 보내기\n"
                "- 배회하기: 망원동 거리를 걸으며 구경하기\n"
                "- 한강공원_산책: 망원한강공원에서 산책\n"
                "- 망원시장_장보기: 망원시장에서 장보기\n"
                "- 회사_가기: 직장에 출근하기 (직장인만)"
            )
            valid_actions = ["집에서_쉬기", "카페_가기", "배회하기", "한강공원_산책", "망원시장_장보기", "회사_가기"]
            action_options = "|".join(valid_actions)
        else:
            # 유동 에이전트: 집에서_쉬기/회사_가기 대신 망원동_떠나기
            available_actions = (
                "- 카페_가기: 근처 카페에서 시간 보내기\n"
                "- 배회하기: 망원동 거리를 걸으며 구경하기\n"
                "- 한강공원_산책: 망원한강공원에서 산책\n"
                "- 망원시장_장보기: 망원시장에서 장보기\n"
                "- 망원동_떠나기: 볼일을 마치고 망원동을 떠남 (되돌아올 수 없음)"
            )
            valid_actions = ["카페_가기", "배회하기", "한강공원_산책", "망원시장_장보기", "망원동_떠나기"]
            action_options = "|".join(valid_actions)

        prompt = render_prompt(
            STEP5_NEXT_ACTION,
            agent_name=agent.persona_id,
            persona_summary=agent.get_persona_summary(),
            weekday=weekday,
            current_time_slot=current_time_slot,
            next_time_slot=next_time_slot,
            last_action=last_action,
            memory_context=agent.get_memory_context(current_date),
            available_actions=available_actions,
            action_options=action_options,
        )

        response = await self._call_llm_async(prompt)
        result = self._parse_json_response(response)

        if result and result.get("action"):
            action = result["action"]

            # 걷는 속도 파싱 (LLM이 직접 판단한 km/h 값)
            walking_speed_raw = result.get("walking_speed", 4.0)
            try:
                walking_speed = float(walking_speed_raw)
                # 유효 범위 검증 (1.0 ~ 7.0 km/h)
                walking_speed = max(1.0, min(7.0, walking_speed))
            except (ValueError, TypeError):
                walking_speed = 4.0  # 파싱 실패 시 기본값

            if action in valid_actions:
                # 행동에 따른 목적지 정보 추가
                destination = self._get_action_destination(action, agent)
                return {
                    "action": action,
                    "walking_speed": walking_speed,
                    "reason": result.get("reason", ""),
                    "destination": destination,
                }

            # 유사 매칭 시도
            for valid_action in valid_actions:
                if action.replace(" ", "_") == valid_action or action.replace("_", " ") in valid_action:
                    destination = self._get_action_destination(valid_action, agent)
                    return {
                        "action": valid_action,
                        "walking_speed": walking_speed,
                        "reason": result.get("reason", ""),
                        "destination": destination,
                    }

        raise LLMCallFailedError("Step5: Failed to parse LLM response or invalid action")

    def _get_action_destination(self, action: str, agent: GenerativeAgent) -> Optional[Dict[str, Any]]:
        """
        행동에 따른 목적지 좌표 반환.

        좌표 정보:
        - 집: 에이전트의 home_location
        - 카페: 카페 매장 중 하나 선택
        - 한강공원: 망원한강공원 좌표
        - 망원시장: 망원시장 좌표
        - 회사: 에이전트의 work_location
        - 배회: 현재 위치 주변
        """
        if action == "집에서_쉬기":
            return {
                "type": "home",
                "name": "집",
                "lat": agent.home_location[0],
                "lng": agent.home_location[1],
            }
        elif action == "카페_가기":
            # 카페 매장 선택
            cafe_store = self._select_random_cafe()
            if cafe_store and cafe_store.coordinates:
                return {
                    "type": "cafe",
                    "name": cafe_store.store_name,
                    "lat": cafe_store.coordinates[0],
                    "lng": cafe_store.coordinates[1],
                    "store_id": cafe_store.store_id,
                }
            return None
        elif action == "한강공원_산책":
            return {
                "type": "park",
                "name": "망원한강공원",
                "lat": 37.5530,
                "lng": 126.8950,
            }
        elif action == "망원시장_장보기":
            return {
                "type": "market",
                "name": "망원시장",
                "lat": 37.5560,
                "lng": 126.9050,
            }
        elif action == "회사_가기":
            # 에이전트의 직장 위치 (work_location이 있으면 사용, 없으면 기본값)
            work_loc = getattr(agent, 'work_location', None)
            if work_loc:
                return {
                    "type": "work",
                    "name": "회사",
                    "lat": work_loc[0],
                    "lng": work_loc[1],
                }
            # 기본 회사 위치 (망원동 외곽)
            return {
                "type": "work",
                "name": "회사",
                "lat": 37.5550,
                "lng": 126.9100,
            }
        elif action == "배회하기":
            # 배회는 목적지 없이 현재 주변 이동
            return {
                "type": "wander",
                "name": "망원동 거리",
                "lat": None,
                "lng": None,
            }
        elif action == "망원동_떠나기":
            # 유동 에이전트: 진입 지점으로 돌아가서 떠남
            if agent.entry_point:
                agent.left_mangwon = True
                return {
                    "type": "leave",
                    "name": "망원동 밖",
                    "lat": agent.entry_point[0],
                    "lng": agent.entry_point[1],
                }
            return {
                "type": "leave",
                "name": "망원동 밖",
                "lat": 37.556069,
                "lng": 126.910108,
            }
        return None

    def _select_random_cafe(self) -> Optional[StoreRating]:
        """카페 매장 목록에서 랜덤 선택"""
        import random
        from pathlib import Path

        cafe_list_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "cafe_stores.txt"

        if not cafe_list_path.exists():
            return None

        with open(cafe_list_path, 'r', encoding='utf-8') as f:
            cafe_files = [line.strip() for line in f if line.strip()]

        if not cafe_files:
            return None

        # 랜덤 카페 선택
        selected_file = random.choice(cafe_files)
        store_id = selected_file.replace('.json', '')

        # GlobalStore에서 매장 정보 가져오기
        store = self.global_store.get_store(store_id)
        return store

    # ==================== Main Process ====================

    async def process_decision(
        self,
        agent: GenerativeAgent,
        nearby_stores: List[StoreRating],
        time_slot: str,
        weekday: str,
        current_datetime: str,
        improvement_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        4단계 의사결정 전체 프로세스 수행 (비동기).
        에이전트 내 각 단계는 이전 결과에 의존하므로 await로 순차 실행.
        LLM 실패 시 LLMCallFailedError 발생 (Fallback 없음)

        Args:
            improvement_text: 개선사항 텍스트 (Step 3, 4에서 LLM에 전달)

        Returns: {
            "decision": "visit" | "stay_home" | "llm_failed",
            "steps": {step1, step2, step3, step4},
            "visited_store": str | None,
            "visited_category": str | None,
            "ratings": {taste, value, atmosphere} | None,
            "error": str | None,
        }
        """
        try:
            # current_datetime에서 날짜 추출 (예: "2026-02-16T07:00:00" → "2026-02-16")
            current_date = current_datetime[:10] if current_datetime else ""

            # Step 1: 망원동 내 식사 여부 결정 (확률 기반, LLM 불필요 → 동기)
            step1 = self.step1_eat_in_mangwon(agent, time_slot, weekday, current_date)

            if not step1.get("eat_in_mangwon", False):
                return {
                    "decision": "stay_home",
                    "steps": {"step1": step1},
                    "visited_store": None,
                    "visited_category": None,
                    "ratings": None,
                    "reason": step1.get("reason", "망원동 외부 식사"),
                    "error": None,
                }

            # 시간대에 따른 목적지 유형 결정
            destination_type = "식당"  # 기본값
            if time_slot in ["저녁", "야식"]:
                destination_type = "식당"  # 저녁/야식은 식당 위주

            # Step 2: 업종 선택 (LLM, 이전 결과 필요 → await 순차)
            step2 = await self.step2_category_selection(agent, destination_type, time_slot, current_date)
            category = step2.get("category", "한식")

            # Step 3: 매장 선택 (LLM, step2 결과 필요 → await 순차)
            step3 = await self.step3_store_selection(agent, category, nearby_stores, time_slot, improvement_text, current_date)
            store_name = step3.get("store_name")

            if not store_name:
                return {
                    "decision": "stay_home",
                    "steps": {"step1": step1, "step2": step2, "step3": step3},
                    "visited_store": None,
                    "visited_category": None,
                    "ratings": None,
                    "reason": "적합한 매장 없음",
                    "error": None,
                }

            # 매장 정보 가져오기
            store = self.global_store.get_by_name(store_name)
            if not store:
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
                    "error": None,
                }

            # Step 4: 평가 및 피드백 (LLM, step3 결과 필요 → await 순차)
            step4 = await self.step4_evaluate_and_feedback(agent, store, current_datetime, improvement_text)

            return {
                "decision": "visit",
                "steps": {"step1": step1, "step2": step2, "step3": step3, "step4": step4},
                "visited_store": store_name,
                "visited_category": store.category,
                "ratings": {
                    "taste": step4["rating"],
                    "value": step4["rating"],
                    "atmosphere": step4["rating"],
                },
                "reason": f"{step1.get('reason', '')} → {step2.get('reason', '')} → {step3.get('reason', '')}",
                "error": None,
            }

        except LLMCallFailedError as e:
            return {
                "decision": "llm_failed",
                "steps": {},
                "visited_store": None,
                "visited_category": None,
                "ratings": None,
                "reason": None,
                "error": str(e),
            }


if __name__ == "__main__":
    # 테스트
    from src.simulation_layer.persona.agent import load_personas_from_md
    from config import get_settings

    settings = get_settings()

    # 에이전트 로드
    agents = load_personas_from_md()[:3]

    # GlobalStore 초기화
    GlobalStore.reset_instance()
    global_store = get_global_store()

    json_dir = settings.paths.split_store_dir
    if json_dir.exists():
        global_store.load_from_json_files(json_dir)

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
        if result.get('error'):
            print(f"에러: {result['error']}")
        print()
