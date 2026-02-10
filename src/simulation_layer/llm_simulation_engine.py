"""
LLM 기반 시뮬레이션 엔진.
Qwen2.5-7B-Instruct 모델을 사용하여 에이전트가 가게를 선택합니다.
"""

import math
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import pandas as pd

from src.simulation_layer.persona.test_agent_persona import TestAgentPersona
from src.ai_layer.ollama_client import OllamaClient, AgentDecisionLLM


@dataclass
class LLMSimulationEvent:
    """LLM 시뮬레이션 이벤트 기록"""
    timestamp: str
    day_of_week: str
    hour: int
    agent_id: int
    agent_name: str
    age: int
    gender: str
    taste_preference: str
    lifestyle: str
    current_lat: float
    current_lng: float
    decision: str  # 'visit' or 'skip'
    visited_store: Optional[str]
    visited_category: Optional[str]
    decision_reason: str
    distance_to_store: Optional[float]
    llm_raw_reason: Optional[str]  # LLM의 원본 이유


class LLMSimulationEngine:
    """
    LLM 기반 시뮬레이션 엔진.
    Qwen2.5-7B-Instruct 모델이 에이전트의 의사결정을 수행합니다.
    """

    WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]
    TIME_SLOTS = [7, 12, 18, 22]

    MEAL_TYPES = {
        7: "아침",
        12: "점심",
        18: "저녁",
        22: "야식",
    }

    def __init__(
        self,
        agents: List[TestAgentPersona],
        stores_df: pd.DataFrame,
        llm_client: Optional[OllamaClient] = None,
    ):
        """
        Args:
            agents: 에이전트 리스트
            stores_df: 가게 데이터프레임
            llm_client: Ollama 클라이언트 (None이면 자동 생성)
        """
        self.agents = agents
        self.stores_df = stores_df.copy()
        self.events: List[LLMSimulationEvent] = []

        # LLM 클라이언트 초기화
        if llm_client is None:
            llm_client = OllamaClient(model="qwen2.5:7b", temperature=0.7)
        self.llm_client = llm_client
        self.decision_llm = AgentDecisionLLM(llm_client)

        # 가게 데이터 전처리
        self._prepare_stores()

    def _prepare_stores(self):
        """가게 데이터 전처리"""
        self.stores_df["lat"] = self.stores_df["y"]
        self.stores_df["lng"] = self.stores_df["x"]

    def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """두 좌표 간의 거리 계산 (미터)"""
        R = 6371000
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lng2 - lng1)

        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _get_nearby_stores(self, agent: TestAgentPersona) -> List[Dict[str, Any]]:
        """에이전트 걷기 거리 내 가게 조회"""
        nearby = []

        for _, store in self.stores_df.iterrows():
            dist = self._haversine_distance(
                agent.current_lat, agent.current_lng,
                store["lat"], store["lng"]
            )
            if dist <= agent.max_walk_distance:
                nearby.append({
                    "name": store["장소명"],
                    "category": store.get("업종", ""),
                    "distance": dist,
                    "lat": store["lat"],
                    "lng": store["lng"],
                })

        # 거리순 정렬
        return sorted(nearby, key=lambda x: x["distance"])

    def _get_store_category(self, store_name: str) -> str:
        """가게 이름으로 카테고리 조회"""
        match = self.stores_df[self.stores_df["장소명"] == store_name]
        if not match.empty:
            return match.iloc[0].get("업종", "")
        return ""

    def _get_store_distance(self, agent: TestAgentPersona, store_name: str) -> Optional[float]:
        """가게까지의 거리 계산"""
        match = self.stores_df[self.stores_df["장소명"] == store_name]
        if not match.empty:
            store = match.iloc[0]
            return self._haversine_distance(
                agent.current_lat, agent.current_lng,
                store["lat"], store["lng"]
            )
        return None

    def simulate_timestep(self, current_time: datetime) -> List[LLMSimulationEvent]:
        """한 타임스텝 시뮬레이션 (LLM 사용)"""
        day_of_week = self.WEEKDAYS[current_time.weekday()]
        hour = current_time.hour
        meal_type = self.MEAL_TYPES.get(hour, "식사")
        timestamp = current_time.strftime("%Y-%m-%d %H:%M")

        step_events = []

        for agent in self.agents:
            print(f"    [{agent.name}] 의사결정 중...", end=" ", flush=True)

            # 근처 가게 조회
            nearby_stores = self._get_nearby_stores(agent)

            if not nearby_stores:
                event = LLMSimulationEvent(
                    timestamp=timestamp,
                    day_of_week=day_of_week,
                    hour=hour,
                    agent_id=agent.id,
                    agent_name=agent.name,
                    age=agent.age,
                    gender=agent.gender,
                    taste_preference=", ".join(agent.taste_preference),
                    lifestyle=agent.lifestyle,
                    current_lat=agent.current_lat,
                    current_lng=agent.current_lng,
                    decision="skip",
                    visited_store=None,
                    visited_category=None,
                    decision_reason="걷기 거리 내 가게 없음",
                    distance_to_store=None,
                    llm_raw_reason=None,
                )
                step_events.append(event)
                print("가게 없음")
                continue

            # 에이전트 정보 구성
            agent_info = {
                "name": agent.name,
                "age": agent.age,
                "gender": agent.gender,
                "taste_preference": agent.taste_preference,
                "lifestyle": agent.lifestyle,
                "recent_meals": agent.get_recent_meals(5),
            }

            time_context = {
                "day": day_of_week,
                "hour": hour,
                "meal_type": meal_type,
            }

            # LLM 의사결정
            try:
                decision = self.decision_llm.decide_store(
                    agent_info=agent_info,
                    nearby_stores=nearby_stores,
                    time_context=time_context,
                )
            except Exception as e:
                print(f"LLM 오류: {e}")
                decision = {"decision": "skip", "store_name": None, "reason": f"LLM 오류: {e}"}

            # 결과 처리
            if decision["decision"] == "visit" and decision["store_name"]:
                store_name = decision["store_name"]
                category = self._get_store_category(store_name)
                distance = self._get_store_distance(agent, store_name)

                # 메모리에 기록
                agent.add_meal(
                    store_name=store_name,
                    category=category,
                    day=day_of_week,
                    time=f"{hour}시"
                )

                event = LLMSimulationEvent(
                    timestamp=timestamp,
                    day_of_week=day_of_week,
                    hour=hour,
                    agent_id=agent.id,
                    agent_name=agent.name,
                    age=agent.age,
                    gender=agent.gender,
                    taste_preference=", ".join(agent.taste_preference),
                    lifestyle=agent.lifestyle,
                    current_lat=agent.current_lat,
                    current_lng=agent.current_lng,
                    decision="visit",
                    visited_store=store_name,
                    visited_category=category,
                    decision_reason=decision.get("reason", ""),
                    distance_to_store=distance,
                    llm_raw_reason=decision.get("reason", ""),
                )
                print(f"→ {store_name}")
            else:
                event = LLMSimulationEvent(
                    timestamp=timestamp,
                    day_of_week=day_of_week,
                    hour=hour,
                    agent_id=agent.id,
                    agent_name=agent.name,
                    age=agent.age,
                    gender=agent.gender,
                    taste_preference=", ".join(agent.taste_preference),
                    lifestyle=agent.lifestyle,
                    current_lat=agent.current_lat,
                    current_lng=agent.current_lng,
                    decision="skip",
                    visited_store=None,
                    visited_category=None,
                    decision_reason=decision.get("reason", "집에서 식사"),
                    distance_to_store=None,
                    llm_raw_reason=decision.get("reason", ""),
                )
                print("→ 집밥")

            step_events.append(event)

        return step_events

    def run_simulation(
        self,
        start_date: datetime,
        num_days: int = 7,
    ) -> pd.DataFrame:
        """
        일주일 LLM 시뮬레이션 실행.
        """
        print("=" * 70)
        print("🤖 LLM 기반 시뮬레이션 (Qwen2.5-7B-Instruct)")
        print("=" * 70)

        # LLM 연결 확인
        if not self.llm_client.is_available():
            print("❌ Ollama 서버에 연결할 수 없습니다!")
            print("   ollama serve 명령으로 서버를 시작하세요.")
            return pd.DataFrame()

        print(f"✅ Ollama 연결됨 (모델: {self.llm_client.model})")
        print(f"시작일: {start_date.strftime('%Y-%m-%d')} ({self.WEEKDAYS[start_date.weekday()]}요일)")
        print(f"에이전트 수: {len(self.agents)}명")
        print(f"시뮬레이션 기간: {num_days}일")
        print(f"타임 스텝: {self.TIME_SLOTS}")
        print(f"가게 수: {len(self.stores_df)}개")
        print("=" * 70)
        print()

        all_events = []
        current_date = start_date

        for day in range(num_days):
            day_of_week = self.WEEKDAYS[current_date.weekday()]
            print(f"📅 Day {day + 1}: {current_date.strftime('%Y-%m-%d')} ({day_of_week}요일)")

            for hour in self.TIME_SLOTS:
                meal_type = self.MEAL_TYPES.get(hour, "식사")
                print(f"  ⏰ {hour:02d}:00 ({meal_type})")

                current_time = current_date.replace(hour=hour, minute=0, second=0)
                events = self.simulate_timestep(current_time)
                all_events.extend(events)

                visit_count = sum(1 for e in events if e.decision == "visit")
                print(f"  └─ 방문: {visit_count}/{len(events)}")
                print()

            current_date += timedelta(days=1)

        print("=" * 70)
        print(f"✅ 시뮬레이션 완료: 총 {len(all_events)}개 이벤트")
        print("=" * 70)

        self.events = all_events
        return pd.DataFrame([asdict(event) for event in all_events])

    def get_summary(self) -> Dict[str, Any]:
        """시뮬레이션 결과 요약"""
        if not self.events:
            return {"error": "시뮬레이션을 먼저 실행하세요."}

        total_events = len(self.events)
        visits = [e for e in self.events if e.decision == "visit"]
        total_visits = len(visits)

        agent_visits = {}
        for e in visits:
            agent_visits[e.agent_name] = agent_visits.get(e.agent_name, 0) + 1

        store_visits = {}
        for e in visits:
            if e.visited_store:
                store_visits[e.visited_store] = store_visits.get(e.visited_store, 0) + 1

        hourly_visits = {}
        for e in visits:
            hourly_visits[e.hour] = hourly_visits.get(e.hour, 0) + 1

        daily_visits = {}
        for e in visits:
            daily_visits[e.day_of_week] = daily_visits.get(e.day_of_week, 0) + 1

        return {
            "total_events": total_events,
            "total_visits": total_visits,
            "visit_rate": total_visits / total_events if total_events > 0 else 0,
            "agent_visits": agent_visits,
            "top_stores": sorted(store_visits.items(), key=lambda x: x[1], reverse=True)[:10],
            "hourly_visits": hourly_visits,
            "daily_visits": daily_visits,
        }
