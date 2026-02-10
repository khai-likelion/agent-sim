"""
Ollama LLM 클라이언트.
Qwen2.5-7B-Instruct 모델을 사용한 에이전트 의사결정.
"""

import json
import re
import requests
from typing import Optional, Dict, Any, List


class OllamaClient:
    """
    Ollama REST API 클라이언트.
    로컬에서 실행되는 Ollama 서버와 통신합니다.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        """
        Args:
            model: 사용할 모델 이름
            base_url: Ollama 서버 URL
            temperature: 생성 다양성 (0.0 ~ 1.0)
            max_tokens: 최대 생성 토큰 수
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        텍스트 생성.

        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트 (선택)
            temperature: 온도 오버라이드
            max_tokens: 최대 토큰 오버라이드

        Returns:
            생성된 텍스트
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"[OllamaClient] 요청 실패: {e}")
            return ""

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        채팅 형식 생성.

        Args:
            messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
            temperature: 온도 오버라이드
            max_tokens: 최대 토큰 오버라이드

        Returns:
            생성된 응답
        """
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            print(f"[OllamaClient] 채팅 요청 실패: {e}")
            return ""

    def is_available(self) -> bool:
        """Ollama 서버 상태 확인."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


class AgentDecisionLLM:
    """
    에이전트 의사결정을 위한 LLM 래퍼.
    페르소나 기반으로 가게 선택을 수행합니다.
    """

    SYSTEM_PROMPT = """당신은 망원동에 사는 주민입니다. 당신의 페르소나에 맞게 식사할 가게를 선택해야 합니다.

중요: 반드시 한국어로만 응답하세요. 중국어나 영어를 섞지 마세요.

당신의 특성:
- 입맛: 선호하는 맛을 고려하세요
- 생활양식: "익숙한맛추구형"이면 자주 가던 곳을 선호하고, "새로운맛탐험형"이면 새로운 곳을 시도합니다
- 최근 먹은 음식: 같은 음식을 연속으로 먹는 것은 피하세요

응답은 반드시 아래 JSON 형식만 출력하세요. 다른 설명은 하지 마세요:
{"decision": "visit", "store_name": "가게이름", "reason": "이유"}
또는
{"decision": "skip", "store_name": null, "reason": "이유"}"""

    def __init__(self, client: OllamaClient):
        self.client = client

    def decide_store(
        self,
        agent_info: Dict[str, Any],
        nearby_stores: List[Dict[str, Any]],
        time_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        가게 선택 의사결정.

        Args:
            agent_info: 에이전트 페르소나 정보
            nearby_stores: 근처 가게 리스트 (이름, 카테고리, 거리)
            time_context: 시간 맥락 (요일, 시간)

        Returns:
            {decision, store_name, reason}
        """
        # 프롬프트 구성
        prompt = self._build_prompt(agent_info, nearby_stores, time_context)

        # LLM 호출
        response = self.client.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
        )

        # 응답 파싱
        return self._parse_response(response, nearby_stores)

    def _build_prompt(
        self,
        agent_info: Dict[str, Any],
        nearby_stores: List[Dict[str, Any]],
        time_context: Dict[str, Any],
    ) -> str:
        """의사결정 프롬프트 생성."""

        # 가게 목록 포맷
        store_list = "\n".join([
            f"- {s['name']} ({s['category']}, {s['distance']:.0f}m)"
            for s in nearby_stores[:10]  # 최대 10개
        ])

        # 최근 식사 기록
        recent_meals = agent_info.get("recent_meals", [])
        if recent_meals:
            recent_str = ", ".join([
                f"{m['store_name']}({m['day']} {m['time']})"
                for m in recent_meals[-5:]
            ])
        else:
            recent_str = "없음"

        prompt = f"""현재 상황:
- 요일: {time_context['day']}요일
- 시간: {time_context['hour']}시 ({time_context['meal_type']})

나의 정보:
- 이름: {agent_info['name']} ({agent_info['age']}세, {agent_info['gender']})
- 선호하는 맛: {', '.join(agent_info['taste_preference'])}
- 생활양식: {agent_info['lifestyle']}
- 최근 먹은 음식: {recent_str}

걸어갈 수 있는 가게들:
{store_list}

이 중에서 어디서 {time_context['meal_type']}을 먹을까요? 또는 집에서 먹을까요?
JSON 형식으로 답해주세요."""

        return prompt

    def _clean_response(self, response: str) -> str:
        """응답에서 제어 문자 제거 및 JSON 추출."""
        # 제어 문자 제거 (탭, 줄바꿈 제외)
        response = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', response)

        # 코드 블록에서 JSON 추출
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        # JSON 객체만 추출 (첫 번째 { 부터 매칭되는 } 까지)
        match = re.search(r'\{[^{}]*"decision"[^{}]*\}', response)
        if match:
            return match.group()

        # 그래도 없으면 첫 번째 JSON 객체 시도
        match = re.search(r'\{.*?\}', response, re.DOTALL)
        if match:
            return match.group()

        return response.strip()

    def _parse_response(
        self,
        response: str,
        nearby_stores: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """LLM 응답 파싱."""
        try:
            # 응답 정리 및 JSON 추출
            cleaned = self._clean_response(response)

            result = json.loads(cleaned)

            # 유효성 검증
            if result.get("decision") == "visit":
                store_name = result.get("store_name")
                # 가게 이름이 실제 목록에 있는지 확인
                valid_names = [s["name"] for s in nearby_stores]
                if store_name not in valid_names:
                    # 가장 비슷한 이름 찾기
                    for name in valid_names:
                        if store_name and (store_name in name or name in store_name):
                            result["store_name"] = name
                            break
                    else:
                        # 못 찾으면 첫 번째 가게 선택
                        result["store_name"] = valid_names[0] if valid_names else None

            return {
                "decision": result.get("decision", "skip"),
                "store_name": result.get("store_name"),
                "reason": result.get("reason", ""),
            }

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"[AgentDecisionLLM] 파싱 실패: {e}")
            print(f"  응답: {response[:200]}...")
            # 파싱 실패시 기본값
            return {
                "decision": "skip",
                "store_name": None,
                "reason": "응답 파싱 실패",
            }


if __name__ == "__main__":
    # 테스트
    client = OllamaClient(model="qwen2.5:7b")

    print("Ollama 서버 상태:", "연결됨" if client.is_available() else "연결 안됨")

    if client.is_available():
        response = client.generate("안녕하세요, 간단히 자기소개 해주세요.", max_tokens=100)
        print("응답:", response)
