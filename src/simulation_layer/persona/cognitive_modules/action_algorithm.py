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
import os
import random
import re
import time
from collections import Counter
from typing import Optional, List, Dict, Any
from src.simulation_layer.persona.agent import GenerativeAgent
from src.data_layer.global_store import get_global_store, GlobalStore, StoreRating, match_category
from src.ai_layer.llm_client import create_llm_client, LLMClient
from src.ai_layer.prompts import render_prompt, STEP1_DESTINATION, STEP2_CATEGORY, STEP3_STORE, STEP4_EVALUATE, STEP5_NEXT_ACTION


# 시간대별 가능한 목적지 유형
DESTINATION_TYPES = {
    "아침": ["식당", "카페"],
    "점심": ["식당", "카페"],
    "저녁": ["식당", "주점", "카페"],
    "야식": ["식당", "주점"],  # 야식에는 카페 제외
}


# 목적지 유형과 카테고리 매핑
DESTINATION_CATEGORIES = {
    "식당": ["한식", "중식", "일식", "양식", "분식", "패스트푸드", "국밥", "찌개", "고기", "치킨"],
    "카페": ["카페", "커피", "디저트", "베이커리", "브런치"],
    "주점": ["호프", "이자카야", "포차", "와인바", "술집", "막걸리", "칵테일바", "칵테일"],
}

## Step 2 에 직접 사용
# 시간대별 카테고리 풀 — destination_type 분기 없이 LLM에 직접 전달
TIME_SLOT_CATEGORIES = {
    "아침": DESTINATION_CATEGORIES["식당"] + DESTINATION_CATEGORIES["카페"],
    "점심": DESTINATION_CATEGORIES["식당"] + DESTINATION_CATEGORIES["카페"],
    "저녁": DESTINATION_CATEGORIES["식당"] + DESTINATION_CATEGORIES["주점"] + DESTINATION_CATEGORIES["카페"],
    "야식": DESTINATION_CATEGORIES["식당"] + DESTINATION_CATEGORIES["주점"],
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
        self.rate_limit_delay = rate_limit_delay ## LLM 호출 후 대기 시간(초)
        self.llm_client: Optional[LLMClient] = None ## 지연 초기화 (asyncio 루프 안에서 생성)
        self.global_store: GlobalStore = get_global_store() ## 전역 매장 DB 싱글턴
        # None이면 호출 시 생성 (asyncio.run() 내부에서 생성해야 함)
        self._semaphore = semaphore ## 동시 LLM 호출 수 제한용
        self.model: str = os.getenv("STEP_MODEL", "gemini-2.5-flash-lite")  ## 전 단계 공통 모델

    ## llm_client가 None이면 그때 생성. asyncio 루프 안에서 처음 호출될 때 생성되도록 지연.
    def _get_llm_client(self) -> LLMClient:
        if self.llm_client is None:
            self.llm_client = create_llm_client()
        return self.llm_client

    ## 비동기 LLM 호출 - 최대 3회 재시도
    ## model: None이면 LLMClient 기본값 사용, 지정 시 해당 모델로 오버라이드
    ## response_format: OpenAI-compatible structured output 스키마 (None이면 미사용)
    async def _call_llm_async(
        self,
        prompt: str,
        max_retries: int = 3,
        model: Optional[str] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """비동기 LLM 호출 with retry. Semaphore로 동시 호출 수를 제한."""
        client = self._get_llm_client()
        semaphore = self._semaphore

        async def _attempt():
            for attempt in range(max_retries):
                try:
                    response = await client.generate(prompt, model=model, response_format=response_format)
                    await asyncio.sleep(self.rate_limit_delay)
                    return response
                except Exception as e:
                    error_str = str(e)
                    ## 429 또는 rate 키워드 -> Rate Limit 감지
                    if "429" in error_str or "rate" in error_str.lower():
                        ## (attemp+1)*10초 대기 (1차:10초, 2차:20초, 3차:30초)
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

    ## 동기 LLM 호출 - asyncio 없이 동기 버전. __main__ 테스트에서만 사용.
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

    @staticmethod
    def _repair_unclosed_string(s: str) -> str:
        """
        Gemini가 JSON 문자열을 닫지 않고 } 를 써버린 경우 자동 복구.
        예: {"key": "val} → {"key": "val"}
        열려있는 문자열이 감지되면 첫 번째 } 직전에 " 를 삽입한다.
        """
        in_string = False
        escape_next = False
        last_open_quote = -1
        for i, c in enumerate(s):
            if escape_next:
                escape_next = False
                continue
            if c == '\\':
                escape_next = True
                continue
            if c == '"':
                if in_string:
                    in_string = False
                else:
                    in_string = True
                    last_open_quote = i
        if not in_string:
            return s  # 이미 닫혀 있음
        # 마지막으로 열린 " 이후 첫 번째 } 앞에 " 삽입
        close_pos = s.find('}', last_open_quote)
        if close_pos != -1:
            return s[:close_pos] + '"' + s[close_pos:]
        return s

    @staticmethod
    def _fix_json_newlines(s: str) -> str:
        """JSON 문자열 값 내 리터럴 개행/CR을 이스케이프 시퀀스로 교체."""
        result = []
        in_string = False
        i = 0
        while i < len(s):
            c = s[i]
            if c == '\\' and i + 1 < len(s):  # 이미 이스케이프된 문자는 그대로
                result.append(c)
                result.append(s[i + 1])
                i += 2
                continue
            if c == '"':
                in_string = not in_string
            elif in_string and c == '\n':
                result.append('\\n')
                i += 1
                continue
            elif in_string and c == '\r':
                result.append('\\r')
                i += 1
                continue
            result.append(c)
            i += 1
        return ''.join(result)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """LLM 응답에서 JSON 추출 (thinking 모델 대응: 마지막 유효 JSON 우선)"""
        # 마크다운 코드 펜스 제거 (```json ... ``` 또는 ``` ... ```)
        if '```' in response:
            response = response.replace('```json', '').replace('```', '').strip()
        response = response.strip()
        # 1차: 직접 파싱 시도 (response_schema 적용 시 순수 JSON 반환)
        try:
            obj = json.loads(response)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        # 2차: JSON 문자열 내 리터럴 개행 수정 후 재시도
        try:
            obj = json.loads(self._fix_json_newlines(response))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        # 2.5차: 닫히지 않은 문자열 자동 복구 후 재시도
        # Gemini가 reason 문자열을 " 로 닫지 않고 } 를 쓴 경우 수정
        try:
            repaired = self._repair_unclosed_string(response)
            if repaired != response:
                obj = json.loads(repaired)
                if isinstance(obj, dict):
                    return obj
        except json.JSONDecodeError:
            pass
        # 3차: rfind 방식으로 JSON 추출
        # 마지막 } 부터 역방향으로 모든 JSON 후보를 시도
        candidates = []
        pos = len(response)
        while True:
            end = response.rfind('}', 0, pos)
            if end == -1:
                break
            start = response.rfind('{', 0, end)
            if start == -1:
                break
            try:
                obj = json.loads(response[start:end + 1])
                if isinstance(obj, dict):
                    candidates.append(obj)
            except json.JSONDecodeError:
                pass
            pos = end
        # 원하는 키가 있는 첫 번째 후보 반환 (없으면 첫 번째 dict)
        for key in ("eat_in_mangwon", "rating", "store_name", "category", "action"):
            for obj in candidates:
                if key in obj:
                    return obj
        if candidates:
            return candidates[0]
        # 4차: regex로 키-값 쌍 추출 (JSON 불완전할 때 최후 수단)
        fallback: Dict[str, Any] = {}
        for m in re.finditer(r'"(\w+)":\s*"([^"]*)"', response):
            fallback[m.group(1)] = m.group(2)
        for m in re.finditer(r'"(\w+)":\s*(\d+(?:\.\d+)?)', response):
            if m.group(1) not in fallback:
                v = m.group(2)
                fallback[m.group(1)] = int(v) if '.' not in v else float(v)
        for m in re.finditer(r'"(\w+)":\s*(true|false)', response, re.IGNORECASE):
            if m.group(1) not in fallback:
                fallback[m.group(1)] = m.group(2).lower() == 'true'
        return fallback

    ## LLM에 줄 매장 1개의 정보를 텍스트로 조립
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
            lines.append(f"  설명: {store.rag_context}")

        # 에이전트 평점 (있으면 표시)
        if store.agent_rating_count > 0:
            lines.append(f"  에이전트 평점: {store.agent_avg_rating:.1f}/5 ({store.agent_rating_count}건), 태그: 맛 {store.taste_count}, 가성비 {store.value_count}, 분위기 {store.atmosphere_count}, 서비스 {store.service_count}")

            # 리뷰 맥락: summary(누적 요약) + review_buffer(미요약 최신 리뷰) — 토큰 절약
            review_ctx = store.get_review_context()
            if review_ctx:
                lines.append(f"  리뷰 맥락: {review_ctx}")
        else:
            lines.append("  에이전트 평점: 아직 없음")

        return "\n".join(lines)

    # ==================== Step 1: 망원동 내 식사 여부 결정 ====================
    ## LLM 호출
    async def step1_eat_in_mangwon(
        self,
        agent: GenerativeAgent,
        time_slot: str,
        weekday: str,
        current_date: str = "",
    ) -> Dict[str, Any]:
        """
        Step 1: 망원동 내 식사 여부 결정 (LLM 기반)

        LLM이 페르소나와 당일 식사 이력을 참고하여 외식 여부를 직접 판단.
        LLM 실패 시 LLMCallFailedError 발생 (Fallback 없음).

        Returns: {"eat_in_mangwon": bool, "reason": str}
        """
        prompt = render_prompt(
            STEP1_DESTINATION,
            agent_name=agent.persona_id,
            persona_summary=agent.get_persona_summary(),
            weekday=weekday,
            time_slot=time_slot,
            memory_context=agent.get_memory_context(current_date),
        )

        step1_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "step1_decision",
                "schema": {
                    "type": "object",
                    "properties": {
                        "eat_in_mangwon": {"type": "boolean"},
                        "reason": {"type": "string", "maxLength": 80},
                    },
                    "required": ["eat_in_mangwon", "reason"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        response = await self._call_llm_async(prompt, model=self.model, response_format=step1_schema)
        result = self._parse_json_response(response)

        if result and "eat_in_mangwon" in result:
            result["eat_in_mangwon"] = bool(result["eat_in_mangwon"])
            return result

        raise LLMCallFailedError("Step1: Failed to parse LLM response")

    # ==================== Step 2: 업종 선택 ====================
    ## LLM 호출
    ## 시간대에 맞는 카테고리 풀 + 에이전트 페르소나 + 방문 메모리를 LLM에 주고 업종 선택.
    async def step2_category_selection(
        self,
        agent: GenerativeAgent,
        time_slot: str,
        current_date: str = "",
    ) -> Dict[str, Any]:
        """
        Step 2: 업종 선택
        LLM 실패 시 LLMCallFailedError 발생 (Fallback 없음)

        시간대별 전체 카테고리 풀(식당+카페+주점)을 LLM에 전달,
        페르소나 기반으로 자연 선택하게 함.

        Returns: {"category": str, "reason": str}
        """
        available_categories = TIME_SLOT_CATEGORIES.get(time_slot, DESTINATION_CATEGORIES["식당"])

        # 시간대별 힌트
        time_hints = {
            "아침": "아침에는 식사 외에 카페/브런치도 고려하세요.",
            "점심": "점심에는 다양한 식사와 카페 디저트도 고려하세요.",
            "저녁": "저녁에는 식사, 술자리, 카페 중 자유롭게 선택하세요.",
            "야식": "야식에는 식사나 술자리를 고려하세요.",
        }
        hint = time_hints.get(time_slot, "")

        prompt = render_prompt(
            STEP2_CATEGORY,
            agent_name=agent.persona_id,
            persona_summary=agent.get_persona_summary(),
            time_slot=time_slot,
            destination_type=time_slot,  # 하위 호환: 프롬프트 변수명 유지
            available_categories=", ".join(available_categories),
            memory_context=agent.get_memory_context(current_date),
            time_hint=hint,
        )

        # response_schema: category를 실제 목록 enum으로 제한
        category_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "category_selection",
                "schema": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": list(available_categories)},
                        "reason": {"type": "string", "maxLength": 80},
                    },
                    "required": ["category", "reason"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        response = await self._call_llm_async(prompt, model=self.model, response_format=category_schema)
        result = self._parse_json_response(response)

        ## 반환 : {"category": "한식", "reason": "..."}
        if result and result.get("category"):
            return result

        raise LLMCallFailedError("Step2: Failed to parse LLM response")

    # ==================== Step 3: 매장 선택 ====================
    ## LLM 호출
    async def step3_store_selection(
        self,
        agent: GenerativeAgent,
        category: str,
        nearby_stores: List[StoreRating],
        time_slot: str,
        current_date: str = "",
    ) -> Dict[str, Any]:
        """
        Step 3: 매장 선택
        LLM 실패 시 LLMCallFailedError 발생 (Fallback 없음)

        Returns: {"store_name": str, "reason": str}
        """
        affordable_stores = nearby_stores

        if not affordable_stores:
            return {"store_name": None, "reason": "주변에 적합한 매장 없음", "llm_failed": False}

        ## 후보 매장이 30개 초과면 search_ranked_stores로 20개 추려냄
        # 매장 수가 많으면 검색 랭킹으로 선별
        SEARCH_THRESHOLD = 30  # 30개 이상이면 검색 랭킹 적용
        if len(affordable_stores) > SEARCH_THRESHOLD:
            # Softmax 가중 샘플링: 카테고리 선필터 → 점수 비례 20개 추출
            loc = agent.current_location
            agent_lat = loc.lat if loc else 0.0
            agent_lng = loc.lng if loc else 0.0
            display_stores = self.global_store.search_ranked_stores(
                category=category,
                sample_k=20,
                candidate_stores=affordable_stores,
                agent_lat=agent_lat,
                agent_lng=agent_lng,
            )
        ## 30개 이하면 카테고리 필터링만
        else:
            # 상주 에이전트: 카테고리 매칭 후 전체 사용
            category_stores = [s for s in affordable_stores if match_category(category, s.category)]
            if not category_stores:
                category_stores = affordable_stores
            display_stores = category_stores

        # 매장 순서 셔플 (LLM의 위치 편향 방지)
        display_stores = list(display_stores)
        random.shuffle(display_stores)

        # 매장 정보 포맷팅
        store_info_lines = []
        for store in display_stores:
            store_info_lines.append(self._get_store_info_for_prompt(store))

        stores_text = "\n\n".join(store_info_lines)

        valid_names = [s.store_name for s in display_stores]

        prompt = render_prompt(
            STEP3_STORE,
            agent_name=agent.persona_id,
            persona_summary=agent.get_persona_summary(),
            time_slot=time_slot,
            category=category,
            memory_context=agent.get_memory_context(current_date),
            stores_text=stores_text,
        )

        # response_schema: store_name을 실제 목록 enum으로 제한 → hallucination 방지
        # reason maxLength: 토큰 절약 및 JSON 파싱 안정성 확보
        store_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "store_selection",
                "schema": {
                    "type": "object",
                    "properties": {
                        "store_name": {"type": "string", "enum": valid_names},
                        "reason": {"type": "string", "maxLength": 80},
                    },
                    "required": ["store_name", "reason"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        response = await self._call_llm_async(prompt, model=self.model, response_format=store_schema)
        result = self._parse_json_response(response)

        ## schema가 store_name을 enum으로 제한하므로 별도 검증 불필요
        ## 단, 파싱 자체가 실패한 경우 예외 처리
        if result and result.get("store_name"):
            return result

        from tqdm import tqdm
        tqdm.write(f"      [Step3 파싱실패] raw response: {response[:200]}")
        raise LLMCallFailedError("Step3: Failed to parse LLM response")

    # ==================== Step 4: 평가 및 피드백 ====================
    ## LLM 호출
    async def step4_evaluate_and_feedback(
        self,
        agent: GenerativeAgent,
        store: StoreRating,
        visit_datetime: str,
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
        focus_aspects = ["맛/퀄리티", "가성비", "매장 분위기/인테리어", "직원 서비스/친절도", "매장 청결/위생", "특색있는 메뉴"]
        selected_focus = random.choice(focus_aspects)

        prompt = render_prompt(
            STEP4_EVALUATE,
            agent_name=agent.persona_id,
            store_name=store.store_name,
            persona_summary=agent.get_persona_summary(),
            store_info=store_info,
            tone_instruction=tone_instruction,
            focus_aspect=selected_focus,
        )

        # response_schema: rating(1~5 정수), selected_tags(enum 배열), comment(문자열) 강제
        evaluate_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "evaluation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "rating": {"type": "integer", "minimum": 1, "maximum": 5},
                        "selected_tags": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["맛", "가성비", "분위기", "서비스"]},
                        },
                        "comment": {"type": "string", "maxLength": 100},
                    },
                    "required": ["rating", "selected_tags", "comment"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        response = await self._call_llm_async(prompt, model=self.model, response_format=evaluate_schema)
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
                rating=rating,
                visit_datetime=visit_datetime,
                comment=comment,
                selected_tags=selected_tags,
            )

            return {
                "rating": rating,
                "selected_tags": selected_tags,
                "comment": comment
            }

        from tqdm import tqdm
        tqdm.write(f"      [Step4 파싱실패] raw response: {response[:200]}")
        raise LLMCallFailedError("Step4: Failed to parse LLM response")

    # ==================== Step 5: 다음 행동 결정 ====================
    ## LLM 호출
    async def step5_next_action(
        self,
        agent: GenerativeAgent,
        current_time_slot: str,
        next_time_slot: str,
        weekday: str,
        session_visits: Optional[List[Dict[str, Any]]] = None,
        current_date: str = "",
    ) -> Dict[str, Any]:
        """
        Step 5: 다음 시간대까지 무엇을 할지 결정
        LLM 실패 시 LLMCallFailedError 발생 (Fallback 없음)

        Args:
            session_visits: 이번 타임슬롯 방문 이력 리스트
                [{"visited_store": str, "visited_category": str, "rating": int}, ...]

        Returns: {"action": str, "reason": str, "destination": dict | None}
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

        # 이번 타임슬롯 활동 이력을 자연어로 구성
        if session_visits:
            activity_lines = []
            visited_categories = []
            for i, v in enumerate(session_visits, 1):
                store = v.get("visited_store", "?")
                cat = v.get("visited_category", "")
                rating = v.get("rating", "")
                rating_str = f" -> {rating}/5" if rating else ""
                activity_lines.append(f"  {i}. {store}({cat}) 방문{rating_str}")
                if cat:
                    visited_categories.append(cat)
            session_activity = "\n".join(activity_lines)

            # 특정 카테고리 중복 방문 체크
            cat_counts = Counter(visited_categories)

            # 2회 이상 방문한 카테고리가 있으면 자연스러운 "고려사항"으로 제시
            warnings = []
            for cat, count in cat_counts.items():
                if count >= 2:
                    warnings.append(f"[고려사항] '{cat}' 일정을 이미 {count}번 소화했습니다. 3번 연속 같은 활동을 하는 것은 현실적으로 부자연스럽습니다. 이제는 산책이나 장보기 등 다른 활동으로 환기하는 것이 좋습니다.")
                elif cat == "커피-음료" and count >= 1:
                    warnings.append("[고려사항] 방금 카페를 다녀왔습니다. 보통 바로 또 카페를 가지는 않지만, 분위기가 다른 곳으로 2차를 가야 할 특별한 이유가 있다면 가능합니다.")

            if warnings:
                session_activity += "\n  " + "\n  ".join(warnings)

        else:
            session_activity = "  (아직 활동 없음)"

        prompt = render_prompt(
            STEP5_NEXT_ACTION,
            agent_name=agent.persona_id,
            persona_summary=agent.get_persona_summary(),
            weekday=weekday,
            current_time_slot=current_time_slot,
            next_time_slot=next_time_slot,
            session_activity=session_activity,
            memory_context=agent.get_memory_context(current_date),
            available_actions=available_actions,
            action_options=action_options,
        )

        # response_schema: action을 실제 유효 목록 enum으로 제한
        action_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "next_action",
                "schema": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": valid_actions},
                        "reason": {"type": "string", "maxLength": 80},
                    },
                    "required": ["action", "reason"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        response = await self._call_llm_async(prompt, model=self.model, response_format=action_schema)
        result = self._parse_json_response(response)

        if result and result.get("action"):
            action = result["action"]

            if action in valid_actions:
                # 행동에 따른 목적지 정보 추가
                destination = self._get_action_destination(action, agent)
                return {
                    "action": action,
                    "reason": result.get("reason", ""),
                    "destination": destination,
                }

            ## 유효하지 않은 action이면 유사 매칭 시도
            for valid_action in valid_actions:
                if action.replace(" ", "_") == valid_action or action.replace("_", " ") in valid_action:
                    destination = self._get_action_destination(valid_action, agent)
                    return {
                        "action": valid_action,
                        "reason": result.get("reason", ""),
                        "destination": destination,
                    }

        raise LLMCallFailedError("Step5: Failed to parse LLM response or invalid action")

    ## Step 5 행동별 목적지 좌표 반환
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
            lat, lng = agent.home_location if agent.home_location else (0.0, 0.0)
            return {
                "type": "home",
                "name": "집",
                "lat": lat,
                "lng": lng,
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

    ## data/cafe_stores.txt에서 카페 store_id 목록 읽어 랜덤 선택 → GlobalStore에서 매장 정보 반환. 파일 없으면 None.
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

    # Step 5에서 매장 방문이 수반되는 행동 → Step 3→4 루프 트리거
    STORE_VISIT_ACTIONS = {
        "카페_가기": "커피-음료",
    }
    MAX_VISIT_LOOP = 3  # 타임슬롯 내 최대 추가 방문 횟수

    async def process_decision(
        self,
        agent: GenerativeAgent,
        nearby_stores: List[StoreRating],
        time_slot: str,
        weekday: str,
        current_datetime: str,
        next_time_slot: str = "",
    ) -> Dict[str, Any]:
        """
        5단계 의사결정 전체 프로세스 수행 (비동기).
        Step 1→2→3→4→5, Step 5에서 매장 방문 행동이면 3→4→5 루프.
        다음 타임슬롯에서는 Step 1부터 재시작.

        Args:
            next_time_slot: 다음 시간대 이름 (Step 5용, 없으면 빈 문자열)

        Returns: {
            "decision": "visit" | "stay_home" | "llm_failed",
            "visits": [{"visited_store", "visited_category", "ratings", "steps"}],
            "visited_store": str | None,  (첫 번째 방문, 하위 호환)
            "visited_category": str | None,
            "ratings": {taste, value, atmosphere} | None,
            "step5": {"action", "reason", "destination"} | None,
            "error": str | None,
        }
        """
        try:
            current_date = current_datetime[:10] if current_datetime else ""

            # Step 1: 망원동 내 식사 여부 결정
            # 시간대별 소프트 게이트: LLM 편향 보정
            # - 아침: 20% 하한 보장 (하드 True) + 나머지 LLM (최대 ~20%)
            # - 점심: 25% 확률로 바로 False 처리 (LLM 과도한 외식 선택 억제)
            # - 저녁: 35% 확률로 바로 True 처리 (LLM 과소 선택 보정)
            # - 야식: 15% 확률로 바로 True 처리 (LLM 과소 선택 보정)
            r = random.random()
            if time_slot == "아침":
                if r < 0.20:
                    step1 = {"eat_in_mangwon": True, "reason": "아침 외식 하한 게이트(20%) 보정"}
                else:
                    try:
                        step1 = await self.step1_eat_in_mangwon(agent, time_slot, weekday, current_date)
                    except LLMCallFailedError as e:
                        print(f"  [{agent.persona_id}] Step1 parse 오류(stay_home 처리): {e}")
                        step1 = {"eat_in_mangwon": False, "reason": "LLM 응답 파싱 오류"}
            elif time_slot == "점심":
                if r > 0.75:
                    step1 = {"eat_in_mangwon": False, "reason": "점심 외식 상한 게이트(75%) 초과"}
                else:
                    try:
                        step1 = await self.step1_eat_in_mangwon(agent, time_slot, weekday, current_date)
                    except LLMCallFailedError as e:
                        print(f"  [{agent.persona_id}] Step1 parse 오류(stay_home 처리): {e}")
                        step1 = {"eat_in_mangwon": False, "reason": "LLM 응답 파싱 오류"}
            elif time_slot == "저녁":
                if r < 0.35:
                    step1 = {"eat_in_mangwon": True, "reason": "저녁 외식 하한 게이트(35%) 보정"}
                else:
                    try:
                        step1 = await self.step1_eat_in_mangwon(agent, time_slot, weekday, current_date)
                    except LLMCallFailedError as e:
                        print(f"  [{agent.persona_id}] Step1 parse 오류(stay_home 처리): {e}")
                        step1 = {"eat_in_mangwon": False, "reason": "LLM 응답 파싱 오류"}
            elif time_slot == "야식":
                if r < 0.15:
                    step1 = {"eat_in_mangwon": True, "reason": "야식 외식 하한 게이트(15%) 보정"}
                else:
                    try:
                        step1 = await self.step1_eat_in_mangwon(agent, time_slot, weekday, current_date)
                    except LLMCallFailedError as e:
                        print(f"  [{agent.persona_id}] Step1 parse 오류(stay_home 처리): {e}")
                        step1 = {"eat_in_mangwon": False, "reason": "LLM 응답 파싱 오류"}
            else:
                try:
                    step1 = await self.step1_eat_in_mangwon(agent, time_slot, weekday, current_date)
                except LLMCallFailedError as e:
                    print(f"  [{agent.persona_id}] Step1 parse 오류(stay_home 처리): {e}")
                    step1 = {"eat_in_mangwon": False, "reason": "LLM 응답 파싱 오류"}

            if not step1.get("eat_in_mangwon", False):
                return {
                    "decision": "stay_home",
                    "steps": {"step1": step1},
                    "visits": [],
                    "visited_store": None,
                    "visited_category": None,
                    "ratings": None,
                    "reason": step1.get("reason", "망원동 외부 식사"),
                    "step5": None,
                    "error": None,
                }

            # Step 2: 업종 선택
            step2 = await self.step2_category_selection(agent, time_slot, current_date)
            category = step2.get("category", "한식")

            # Step 3: 매장 선택
            step3 = await self.step3_store_selection(agent, category, nearby_stores, time_slot, current_date)
            store_name = step3.get("store_name")

            if not store_name:
                return {
                    "decision": "stay_home",
                    "steps": {"step1": step1, "step2": step2, "step3": step3},
                    "visits": [],
                    "visited_store": None,
                    "visited_category": None,
                    "ratings": None,
                    "reason": "적합한 매장 없음",
                    "step5": None,
                    "error": None,
                }

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
                    "visits": [],
                    "visited_store": None,
                    "visited_category": None,
                    "ratings": None,
                    "reason": "매장 정보 없음",
                    "step5": None,
                    "error": None,
                }

            # Step 4: 평가 및 피드백
            step4 = await self.step4_evaluate_and_feedback(agent, store, current_datetime)

            # 다중 방문 리스트
            visits = [{
                "visited_store": store_name,
                "visited_category": store.category,
                "rating": step4["rating"],
                "steps": {"step3": step3, "step4": step4},
            }]

            # Step 5 루프: 다음 행동 결정 → 매장 방문이면 3→4→5 반복
            last_step5 = None

            for _loop_i in range(self.MAX_VISIT_LOOP):
                # 이번 타임슬롯 방문 이력을 Step 5에 전달 (LLM이 자연스럽게 판단)
                session_visits = [
                    {
                        "visited_store": v["visited_store"],
                        "visited_category": v["visited_category"],
                        "rating": v["rating"],
                    }
                    for v in visits
                ]

                try:
                    step5 = await self.step5_next_action(
                        agent, time_slot, next_time_slot or time_slot,
                        weekday, session_visits, current_date,
                    )
                except LLMCallFailedError:
                    break  # Step 5 실패 시 루프 종료 (이전 방문은 유지)

                last_step5 = step5
                action = step5.get("action", "")

                # 매장 방문이 수반되지 않는 행동 → 루프 종료
                if action not in self.STORE_VISIT_ACTIONS:
                    break

                # 매장 방문 행동 → Step 3→4 추가 실행
                loop_category = self.STORE_VISIT_ACTIONS[action]

                try:
                    step3_loop = await self.step3_store_selection(
                        agent, loop_category, nearby_stores, time_slot, current_date,
                    )
                except LLMCallFailedError:
                    break

                loop_store_name = step3_loop.get("store_name")
                if not loop_store_name:
                    break

                loop_store = self.global_store.get_by_name(loop_store_name)
                if not loop_store:
                    for s in nearby_stores:
                        if s.store_name == loop_store_name:
                            loop_store = s
                            break
                if not loop_store:
                    break

                try:
                    step4_loop = await self.step4_evaluate_and_feedback(
                        agent, loop_store, current_datetime,
                    )
                except LLMCallFailedError:
                    break

                visits.append({
                    "visited_store": loop_store_name,
                    "visited_category": loop_store.category,
                    "rating": step4_loop["rating"],
                    "steps": {"step3": step3_loop, "step4": step4_loop},
                })

            # 첫 번째 방문 정보 (하위 호환)
            first_visit = visits[0]
            return {
                "decision": "visit",
                "steps": {"step1": step1, "step2": step2, "step3": step3, "step4": step4},
                "visits": visits,
                "visited_store": first_visit["visited_store"],
                "visited_category": first_visit["visited_category"],
                "rating": first_visit["rating"],
                "reason": f"{step1.get('reason', '')} → {step2.get('reason', '')} → {step3.get('reason', '')}",
                "step5": last_step5,
                "error": None,
            }

        except LLMCallFailedError as e:
            return {
                "decision": "llm_failed",
                "steps": {},
                "visits": [],
                "visited_store": None,
                "visited_category": None,
                "ratings": None,
                "reason": None,
                "step5": None,
                "error": str(e),
            }

## 테스트 전용. 실제 시뮬레이션에서 미사용.
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

    async def _run_test():
        for agent in agents[:1]:
            print(agent.get_persona_summary())
            print()

            result = await algorithm.process_decision(
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

    asyncio.run(_run_test())
