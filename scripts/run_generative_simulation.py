"""
Generative Agents 기반 시뮬레이션 실행 스크립트.

Stanford Generative Agents 논문 구조를 참조한 시뮬레이션:
- Persona Generation Module: 고유 조합의 에이전트 생성 (160명, 상주47/유동113)
- Memory Module: recent_history + memory_context(당일 식사 + 과거 평점/코멘트) 기반 의사결정
- Action Algorithm: 5단계 LLM 기반 의사결정 + 다중 방문 루프 (MAX=3)
- Global Store: 실시간 평점 축적 (Softmax 계층화 샘플링, 10개마다 LLM 요약)

사용법:
    python scripts/run_generative_simulation.py [--agents N] [--dry-run]
"""

import argparse
import asyncio
import json
import pickle
import sys
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import time as time_module
from tqdm import tqdm

# 기본 시드 (개선 전/후 비교 시 동일 에이전트 구성 보장)
DEFAULT_SEED = 42

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from config import get_settings
from src.simulation_layer.persona.agent import GenerativeAgent, load_personas_from_md
from src.data_layer.global_store import get_global_store, GlobalStore
from src.simulation_layer.persona.cognitive_modules.action_algorithm import ActionAlgorithm
from src.data_layer.street_network import StreetNetwork, StreetNetworkConfig, reset_street_network


# 시간대 정의 (4개) - 정확한 시간 기반
TIME_SLOTS = {
    "아침": 7,   # 07:00
    "점심": 12,  # 12:00
    "저녁": 18,  # 18:00
    "야식": 22,  # 22:00
}

# 요일
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]

# 시뮬레이션 속도: 현실 120배 빠름 (1분 현실 = 120분 시뮬레이션)
TIME_SPEED_MULTIPLIER = 120

# 에이전트 걷는 속도 (m/s) - 평균 보행 속도 약 1.4 m/s (5 km/h)
WALKING_SPEED_MS = 1.4

# 상주 에이전트 매장 인식 반경 (km)
RESIDENT_STORE_RADIUS_KM = 0.8

# 유동 에이전트 일일 활동 수 (요일별 차등 — 인구_DB.json 비율 기반)
# 실제 비율: 월13.5% 화13.6% 수13.9% 목13.8% 금13.9% 토15.6% 일15.7%
# 주간 평균 53명 유지 → 평일 51, 주말 58 (비율 1.14배)
DAILY_FLOATING_COUNT_BY_DAY = {
    "월": 51, "화": 51, "수": 51, "목": 51, "금": 51,
    "토": 58, "일": 58,
}
DAILY_FLOATING_AGENT_COUNT = 53  # 기본값 (fallback)
REVISIT_RATE = 0.10  # 재방문율: 전날 방문자 중 10% 다음날 포함

# OSM 네트워크 설정 (망원동 기준 — run_generative_simulation에서 실제 망원동 좌표로 재설정됨)
DEFAULT_NETWORK_CENTER_LAT = 37.4980
DEFAULT_NETWORK_CENTER_LNG = 127.0276
DEFAULT_NETWORK_RADIUS_M = 800.0  # 망원동 구역 내로 제한

## 실행 전 LLM 호출 수와 비용 예측
def estimate_simulation(agent_count: int, days: int = 7, time_slots: int = 4,
                        resident_count: int = 47, floating_count: int = 113) -> Dict[str, Any]:
    """
    시뮬레이션 전 예상치 계산.

    LLM 호출 수 계산 (타임슬롯당 기본):
    - Step 1: 모든 에이전트 × 모든 타임슬롯 (LLM 기반 외식 여부 결정)
    - Step 2: 외출하는 에이전트만 (업종 선택, 약 50%)
    - Step 3: 외출하는 에이전트만 (매장 선택, Step 2와 동일)
    - Step 4: 실제 방문한 에이전트만 (평가, 외출 중 약 90%)
    - Step 5: Step 4와 동일 (다음 행동 결정)
    ※ 다중 방문 루프(카페_가기 등) 시 Step 3~5 추가 호출됨 (최대 3회)

    가정:
    - 상주 에이전트: 매일 전원 활동
    - 유동 에이전트: 평일 51명 / 주말 58명 (DAILY_FLOATING_COUNT_BY_DAY 기준)
    - 외출 확률 약 50%
    - 방문 성공률 약 90% (외출 결정 중)
    """
    daily_floating = min(DAILY_FLOATING_AGENT_COUNT, floating_count)
    daily_active = resident_count + daily_floating
    total_slots = daily_active * days * time_slots

    # Step 1: 모든 슬롯에서 호출 (LLM 기반 외식 여부 결정)
    step1_calls = total_slots

    # Step 2, 3: 외출하는 경우만 (약 50%)
    go_out_rate = 0.5
    step2_calls = int(total_slots * go_out_rate)
    step3_calls = step2_calls

    # Step 4: 실제 방문한 경우만 (외출 중 90%)
    visit_rate = 0.9
    step4_calls = int(step2_calls * visit_rate)

    total_llm_calls = step1_calls + step2_calls + step3_calls + step4_calls

    # LLM 비용 계산 (provider별 가격)
    # - GPT-4o-mini: Input $0.15/1M, Output $0.60/1M
    # - Gemini 2.0 Flash: Input $0.10/1M, Output $0.40/1M
    avg_input_tokens = 500
    avg_output_tokens = 100

    llm_settings = get_settings().llm
    if llm_settings.provider == "gemini":
        input_cost_per_million = 0.10
        output_cost_per_million = 0.40
    else:
        input_cost_per_million = 0.15
        output_cost_per_million = 0.60

    total_input_tokens = total_llm_calls * avg_input_tokens
    total_output_tokens = total_llm_calls * avg_output_tokens

    input_cost = (total_input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (total_output_tokens / 1_000_000) * output_cost_per_million
    total_cost = input_cost + output_cost

    # 예상 시간 (API 호출당 평균 1초 + 0.5초 딜레이)
    avg_time_per_call = 1.5  # seconds
    total_time_seconds = total_llm_calls * avg_time_per_call
    total_time_minutes = total_time_seconds / 60

    return {
        "agent_count": agent_count,
        "daily_active": daily_active,
        "simulation_days": days,
        "time_slots_per_day": time_slots,
        "total_time_slots": total_slots,
        "estimated_llm_calls": {
            "step1": step1_calls,
            "step2": step2_calls,
            "step3": step3_calls,
            "step4": step4_calls,
            "total": total_llm_calls,
        },
        "estimated_cost_usd": {
            "input": round(input_cost, 4),
            "output": round(output_cost, 4),
            "total": round(total_cost, 4),
        },
        "estimated_time_minutes": round(total_time_minutes, 1),
    }


def print_estimates(estimates: Dict[str, Any]):
    """예상치 출력"""
    print("\n" + "=" * 60)
    print("시뮬레이션 예상치")
    print("=" * 60)
    print(f"에이전트 수: {estimates['agent_count']}명 (일일 활동: {estimates.get('daily_active', estimates['agent_count'])}명)")
    print(f"시뮬레이션 기간: {estimates['simulation_days']}일")
    print(f"일일 타임슬롯: {estimates['time_slots_per_day']}개")
    print(f"총 타임슬롯 수: {estimates['total_time_slots']:,}개")
    print()
    print("예상 LLM 호출 수:")
    calls = estimates["estimated_llm_calls"]
    print(f"  Step 1 (목적지 유형): {calls['step1']:,}회")
    print(f"  Step 2 (업종 선택): {calls['step2']:,}회")
    print(f"  Step 3 (매장 선택): {calls['step3']:,}회")
    print(f"  Step 4 (평가): {calls['step4']:,}회")
    print(f"  총 호출: {calls['total']:,}회")
    print()
    # LLM 모델 정보
    llm_settings = get_settings().llm
    model_name = f"{llm_settings.provider.upper()}/{llm_settings.model_name}"

    cost = estimates["estimated_cost_usd"]
    print(f"예상 비용 ({model_name}):")
    print(f"  Input: ${cost['input']:.4f}")
    print(f"  Output: ${cost['output']:.4f}")
    print(f"  Total: ${cost['total']:.4f} (약 {cost['total'] * 1400:.0f}원)")
    print()
    print(f"예상 소요 시간: 약 {estimates['estimated_time_minutes']:.1f}분")
    print("=" * 60)

## 매장 파일 로드
def load_environment(settings):
    """환경 데이터 로드"""
    print("\n[1/4] 환경 데이터 로드 중...")

    # Global Store 초기화
    GlobalStore.reset_instance()
    global_store = get_global_store()

    # JSON 매장 데이터 로드 (stores.csv 대신 JSON 파일 사용)
    json_dir = settings.paths.split_store_dir
    if json_dir.exists():
        global_store.load_from_json_files(json_dir)
        print(f"  매장 데이터 로드: {len(global_store.stores)}개")

    return global_store

## 에이전트 로드
def generate_agents(agent_count: int) -> List[GenerativeAgent]:
    """personas_160.md에서 에이전트 로드"""
    print(f"\n[2/4] 에이전트 로드 중 ({agent_count}명)...")

    agents = load_personas_from_md()

    if agent_count < len(agents):
        import random as _rng
        _rng.shuffle(agents)
        agents = agents[:agent_count]
        for i, a in enumerate(agents):
            a.id = i + 1

    print(f"  총 {len(agents)}명 로드 완료")
    type_counts = {}
    for a in agents:
        type_counts[a.agent_type] = type_counts.get(a.agent_type, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}명")

    floating_count = type_counts.get("유동", 0)
    resident_count = type_counts.get("상주", 0)
    daily_active = resident_count + min(DAILY_FLOATING_AGENT_COUNT, floating_count)
    print(f"  일일 활동 에이전트: {daily_active}명 (상주 {resident_count} + 유동 {min(DAILY_FLOATING_AGENT_COUNT, floating_count)})")

    return agents

## OSM 보행자 도로 네트워크 로드. 매장 좌표의 무게중심을 네트워크 중심점으로 사용. 반경 800m 내 도로 그래프 다운로드 (or 캐시 사용).
def initialize_street_network(global_store: GlobalStore) -> StreetNetwork:
    """OSM 거리 네트워크 초기화"""
    print("\n  OSM 거리 네트워크 로드 중...")
    reset_street_network()  # 기존 인스턴스 리셋

    # 매장 좌표에서 중심점 계산
    coords = [s.coordinates for s in global_store.stores.values() if s.coordinates]
    if coords:
        center_lat = sum(c[0] for c in coords) / len(coords)
        center_lng = sum(c[1] for c in coords) / len(coords)
        print(f"    매장 기반 중심점: ({center_lat:.4f}, {center_lng:.4f})")
    else:
        center_lat = DEFAULT_NETWORK_CENTER_LAT
        center_lng = DEFAULT_NETWORK_CENTER_LNG
        print(f"    기본 중심점 사용: ({center_lat:.4f}, {center_lng:.4f})")

    config = StreetNetworkConfig(
        center_lat=center_lat,
        center_lng=center_lng,
        radius_m=DEFAULT_NETWORK_RADIUS_M,
        network_type="walk",
    )
    network = StreetNetwork(config)
    network.load_graph()

    return network



async def agent_task(
    agent: GenerativeAgent,
    algorithm: ActionAlgorithm,
    global_store: GlobalStore,
    slot_name: str,
    weekday: str,
    slot_time: datetime,
    agent_locations: Dict[int, Any],
    next_time_slot: str = "",
) -> Optional[Dict[str, Any]]:
    """
    에이전트 한 명의 한 타임슬롯 의사결정 코루틴.

    - 에이전트 내 Step 1→2→3→4→5는 await로 순차 실행.
    - Step 5에서 매장 방문 행동이면 Step 3→4→5 루프 반복.
    - 여러 에이전트의 코루틴은 asyncio.gather()로 동시에 실행됨.
    - 평점은 add_pending_rating()에 버퍼링 → 타임슬롯 종료 후 flush.
    """
    # 유동 에이전트: 망원동 이탈 시 스킵
    if agent.is_floating and agent.left_mangwon:
        return None

    # 유동 에이전트: entry_time_slot 이전이면 스킵 (아직 미도착)
    if agent.is_floating and agent.entry_time_slot:
        entry_hour = TIME_SLOTS.get(agent.entry_time_slot, 0)
        if slot_time.hour < entry_hour:
            return None

    location = agent_locations[agent.id]
    agent.current_location = location  # action_algorithm에서 lat/lng 접근 가능하도록 주입

    # 에이전트 유형별 매장 인식 범위
    if agent.is_resident:
        nearby_stores = global_store.get_stores_in_radius(
            center_lat=location.lat,
            center_lng=location.lng,
            radius_km=RESIDENT_STORE_RADIUS_KM,
        )
    else:
        nearby_stores = list(global_store.stores.values())

    # 에이전트 내부: Step 1→2→3→4→5 (Step 5에서 매장 방문이면 3→4→5 루프)
    result = await algorithm.process_decision(
        agent=agent,
        nearby_stores=nearby_stores,
        time_slot=slot_name,
        weekday=weekday,
        current_datetime=slot_time.isoformat(),
        next_time_slot=next_time_slot,
    )

    # 실시간 결정 로그
    decision = result.get("decision", "?")
    visits = result.get("visits", [])
    steps = result.get("steps", {})

    if decision == "visit":
        for i, visit in enumerate(visits):
            v_store = visit["visited_store"]
            v_category = visit["visited_category"]
            v_rating = visit["rating"]
            v_step4 = visit["steps"].get("step4", {})
            comment = v_step4.get("comment", "")
            tags = v_step4.get("selected_tags", [])
            tags_str = " ".join(f"#{t}" for t in tags) if tags else ""
            prefix = "  " if i == 0 else "    ++"
            print(f"{prefix}[{agent.persona_id}] {slot_name} | "
                  f"방문: {v_store}({v_category}) | 별점:{v_rating}/5 {tags_str}")
            if comment:
                print(f"         리뷰: {comment[:60]}{'...' if len(comment)>60 else ''}")

        # Step 5 결과 로그
        step5 = result.get("step5")
        if step5:
            print(f"    >> 다음행동: {step5.get('action', '?')} | {step5.get('reason', '')[:40]}")
    elif decision == "stay_home":
        step2 = steps.get("step2", {})
        reason = result.get("reason", step2.get("reason", ""))
        reason_short = reason[:40] + "..." if len(reason) > 40 else reason
        print(f"  [{agent.persona_id}] {slot_name} | 외출안함 | {reason_short}")
    elif decision == "llm_failed":
        err_msg = result.get("error", "")
        print(f"  [{agent.persona_id}] {slot_name} | LLM오류 | {err_msg}")

    return {
        "agent": agent,
        "location": location,
        "nearby_store_count": len(nearby_stores),
        "result": result,
    }

## 메인 시뮬레이션 루프
async def run_simulation(
    agents: List[GenerativeAgent],
    global_store: GlobalStore,
    settings,
    days: int = 7,
    max_concurrent_llm_calls: int = 10,
    checkpoint_dir: Optional[Path] = None,
    resume_from_day: int = 0,
) -> pd.DataFrame:
    """
    OSM 네트워크 기반 시뮬레이션 실행.

    에이전트들이 거리를 걸으며 타임슬롯(07:00, 12:00, 18:00, 22:00)마다
    의사결정을 수행합니다.

    시간 시스템:
    - 현실보다 120배 빠름 (시뮬레이션 1시간 = 현실 30초)
    - 하루 24시간 시뮬레이션
    """
    sim_start_time = time_module.perf_counter()
    print(f"\n[3/4] 시뮬레이션 실행 중...")
    print(f"  시간 배속: {TIME_SPEED_MULTIPLIER}x")
    print(f"  매장 인식 반경: 상주 {RESIDENT_STORE_RADIUS_KM}km / 유동 제한없음")
    print(f"  최대 동시 LLM 호출: {max_concurrent_llm_calls}개")

    # OSM 거리 네트워크 초기화
    network = initialize_street_network(global_store)

    # 상주/유동 에이전트 분리
    resident_agents = [a for a in agents if a.is_resident]
    floating_agents = [a for a in agents if a.is_floating]
    print(f"\n  상주 에이전트: {len(resident_agents)}명 (매일 활동)")
    print(f"  유동 에이전트: {len(floating_agents)}명 (요일별 차등: 평일 {DAILY_FLOATING_COUNT_BY_DAY['월']} / 금 {DAILY_FLOATING_COUNT_BY_DAY['금']} / 주말 {DAILY_FLOATING_COUNT_BY_DAY['토']}명)")

    # Semaphore: 동시에 실행되는 LLM 호출 수를 제한 (API rate limit 대응)
    semaphore = asyncio.Semaphore(max_concurrent_llm_calls)
    algorithm = ActionAlgorithm(rate_limit_delay=0.5, semaphore=semaphore)

    # 결과 저장
    results = []

    # 시작 날짜: 오늘 기준 가장 가까운 직전 월요일 (한국 시간)
    from datetime import timezone
    _today = datetime.now(timezone(timedelta(hours=9))).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None
    )
    _days_since_monday = _today.weekday()  # 월=0
    start_date = _today - timedelta(days=_days_since_monday)

    # 체크포인트 디렉토리 준비
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume: 체크포인트에서 상태 복원
    if resume_from_day > 0 and checkpoint_dir:
        cp_path = checkpoint_dir / f"checkpoint_day{resume_from_day:03d}"
        if cp_path.exists():
            print(f"  ♻️  체크포인트 복원: {cp_path.name} (Day {resume_from_day + 1}부터 재개)")
            with open(cp_path / "results.pkl", "rb") as f:
                results = pickle.load(f)
            with open(cp_path / "agents.pkl", "rb") as f:
                saved_agents = pickle.load(f)
                agent_map = {a.id: a for a in saved_agents}
                for agent in agents:
                    if agent.id in agent_map:
                        agent.__dict__.update(agent_map[agent.id].__dict__)
            with open(cp_path / "global_store.pkl", "rb") as f:
                saved_gs = pickle.load(f)
                global_store.__dict__.update(saved_gs.__dict__)
            with open(cp_path / "metadata.json", encoding="utf-8") as f:
                meta = json.load(f)
                start_date = datetime.fromisoformat(meta["start_date"])
                prev_ids = set(meta.get("prev_visitor_ids", []))
                run_simulation._prev_day_visitors = [a for a in agents if a.id in prev_ids]
        else:
            print(f"  [경고] 체크포인트 없음: {cp_path} → Day 1부터 시작")
            resume_from_day = 0

    # 타임슬롯 리스트 (시간 순서대로 정렬)
    time_slot_list = sorted(TIME_SLOTS.items(), key=lambda x: x[1])

    # 에이전트 초기 위치 딕셔너리
    agent_locations = {}

    day_pbar = tqdm(range(resume_from_day, days), desc="전체 진행", unit="day", position=0)
    for day_idx in day_pbar:
        day_start_time = time_module.perf_counter()
        current_date = start_date + timedelta(days=day_idx)
        weekday = WEEKDAYS[current_date.weekday()]

        # 요일별 유동 에이전트 수 차등 + 재방문율 적용
        base_count = DAILY_FLOATING_COUNT_BY_DAY.get(weekday, DAILY_FLOATING_AGENT_COUNT)
        daily_floating_count = min(base_count, len(floating_agents))

        # 재방문: 전날 방문한 유동 에이전트마다 REVISIT_RATE 확률로 재방문
        revisit_agents = []
        if day_idx > 0 and hasattr(run_simulation, '_prev_day_visitors'):
            prev_visitors = run_simulation._prev_day_visitors
            revisit_agents = [a for a in prev_visitors if random.random() < REVISIT_RATE]

        # 나머지는 새로운 에이전트에서 샘플링
        remaining_pool = [a for a in floating_agents if a not in revisit_agents]
        new_count = daily_floating_count - len(revisit_agents)
        new_agents = random.sample(remaining_pool, min(new_count, len(remaining_pool)))
        daily_floating = revisit_agents + new_agents

        # 유동 에이전트: 매일 초기화 (entry_point에서 시작, left_mangwon 리셋)
        for agent in daily_floating:
            agent.left_mangwon = False
            if agent.entry_point:
                lat, lng = agent.entry_point
            else:
                from src.simulation_layer.persona.agent import FLOATING_LOCATIONS
                loc = random.choice(list(FLOATING_LOCATIONS.values()))
                lat, lng = loc["lat"], loc["lng"]
            agent_locations[agent.id] = network.initialize_agent_location(lat, lng)

        # 상주 에이전트: 매일 home_location에서 시작
        for agent in resident_agents:
            lat, lng = agent.home_location
            agent_locations[agent.id] = network.initialize_agent_location(lat, lng)

        # 오늘 활동할 에이전트 = 상주(전원) + 유동(샘플링)
        daily_agents = resident_agents + daily_floating

        day_pbar.set_description(f"Day {day_idx+1}/{days} ({weekday})")
        tqdm.write(f"\n  === Day {day_idx + 1}/{days}: {current_date.strftime('%Y-%m-%d')} ({weekday}요일) ===")
        tqdm.write(f"      활동 에이전트: {len(daily_agents)}명 (상주 {len(resident_agents)} + 유동 {len(daily_floating)})")

        day_processed = 0

        # 하루의 시뮬레이션 시간 (00:00 시작)
        sim_hour = 0  # 시뮬레이션 시간 (시)

        for slot_name, slot_hour in time_slot_list:
            # 해당 타임슬롯까지 시간 이동 (에이전트들이 걸어다님)
            hours_to_walk = slot_hour - sim_hour
            if hours_to_walk > 0:
                # 현실 시간으로 걷는 시간 계산 (120배 빠름)
                # 시뮬레이션 1시간 = 현실 30초
                ## 타임슬롯 간격만큼 시간이 흘렀을 때 에이전트가 이동한 거리 계산 후 OSM 네트워크 위에서 이동
                walk_time_real_seconds = (hours_to_walk * 3600) / TIME_SPEED_MULTIPLIER
                walk_distance_m = walk_time_real_seconds * WALKING_SPEED_MS

                # 오늘 활동하는 에이전트만 이동
                for agent in daily_agents:
                    location = agent_locations[agent.id]
                    agent_locations[agent.id] = network.move_agent(location, walk_distance_m)

            sim_hour = slot_hour
            slot_time = current_date.replace(hour=slot_hour, minute=0, second=0)

            # 이전 타임슬롯 평점 반영 + review_buffer >= 10인 매장 자동 요약
            flush_result = await global_store.flush_and_summarize_async()
            flushed = flush_result["flushed"]
            summarized = flush_result["summarized"]
            if flushed > 0:
                msg = f"      평점 반영: {flushed}건"
                if summarized > 0:
                    msg += f" | 리뷰 요약: {summarized}개 매장"
                tqdm.write(msg)

            tqdm.write(f"    [{slot_name}] {slot_time.strftime('%H:%M')} - 에이전트 {len(daily_agents)}명 병렬 의사결정 중...")

            # 다음 타임슬롯 이름 계산 (Step 5용)
            current_slot_idx = [i for i, (name, _) in enumerate(time_slot_list) if name == slot_name][0]
            next_slot_name = time_slot_list[current_slot_idx + 1][0] if current_slot_idx + 1 < len(time_slot_list) else ""

            # 타임슬롯 내 모든 에이전트의 의사결정을 동시에 실행.
            # - 에이전트 간: asyncio.gather()로 병렬 실행 (LLM 응답 대기 중 다른 에이전트 요청 전송)
            # - 에이전트 내: process_decision() 안에서 Step1→2→3→4→5 순차 await (Step5 루프 포함)
            tasks = [
                agent_task(agent, algorithm, global_store, slot_name, weekday, slot_time, agent_locations, next_slot_name)
                for agent in daily_agents
            ]
            task_outputs = await asyncio.gather(*tasks, return_exceptions=True)

            slot_pbar = tqdm(
                zip(daily_agents, task_outputs),
                total=len(daily_agents),
                desc=f"  {slot_name}",
                unit="agent",
                position=1,
                leave=False,
            )
            slot_llm_failed = 0
            for agent, output in slot_pbar:
                day_processed += 1

                # gather의 예외 또는 스킵(None) 처리
                if isinstance(output, Exception):
                    tqdm.write(f"      [경고] 에이전트 {agent.persona_id} 오류: {output}")
                    continue
                if output is None:
                    continue

                location = output["location"]
                result = output["result"]
                visits = result.get("visits", [])

                # 공통 필드
                common = {
                    "timestamp": slot_time.strftime("%Y-%m-%d %H:%M"),
                    "agent_id": agent.id,
                    "persona_id": agent.persona_id,
                    "generation": agent.generation,
                    "gender_composition": agent.gender_composition,
                    "segment": agent.segment,
                    "weekday": weekday,
                    "time_slot": slot_name,
                    "agent_lat": location.lat,
                    "agent_lng": location.lng,
                    "nearby_store_count": output["nearby_store_count"],
                }

                if visits:
                    # 다중 방문: 방문당 1개 레코드
                    for visit in visits:
                        record = {
                            **common,
                            "decision": "visit",
                            "visited_store": visit["visited_store"],
                            "visited_category": visit["visited_category"],
                            "rating": visit["rating"],
                            "reason": result.get("reason", ""),
                            "comment": visit["steps"].get("step4", {}).get("comment", ""),
                        }
                        results.append(record)
                else:
                    # 비방문 (stay_home, llm_failed)
                    record = {
                        **common,
                        "decision": result["decision"],
                        "visited_store": None,
                        "visited_category": None,
                        "rating": None,
                        "reason": result.get("reason", ""),
                    }
                    results.append(record)
                    if result["decision"] == "llm_failed":
                        slot_llm_failed += 1

            # llm_failed 감지: 1건이라도 발생 시 즉시 중단
            if slot_llm_failed > 0:
                raise RuntimeError(
                    f"\n[시뮬레이션 중단] LLM 오류 발생!\n"
                    f"  Day {day_idx+1} / {slot_name} 슬롯\n"
                    f"  실패: {slot_llm_failed}/{len(daily_agents)}명\n"
                    f"  마지막 오류는 위 로그의 'LLM오류' 항목을 확인하세요."
                )

        # 하루 종료: 소요시간 출력
        day_elapsed = time_module.perf_counter() - day_start_time
        tqdm.write(f"      Day {day_idx+1} 완료: {day_elapsed:.1f}초 ({day_elapsed/60:.1f}분)")

        ## 함수 객체에 속성으로 전날 방문자 기록 저장 (간단한 상태 보존)
        # 하루 종료: 오늘 방문한 유동 에이전트 기록 (다음 날 재방문 풀)
        today_visitors = set()
        for rec in results:
            if rec.get("decision") == "visit" and rec.get("timestamp", "").startswith(current_date.strftime("%Y-%m-%d")):
                today_visitors.add(rec["agent_id"])
        run_simulation._prev_day_visitors = [
            a for a in daily_floating if a.id in today_visitors
        ]
        if run_simulation._prev_day_visitors:
            tqdm.write(f"      재방문 풀: 유동 {len(run_simulation._prev_day_visitors)}명 (내일 {REVISIT_RATE*100:.0f}% 확률로 재방문)")

        # 7일마다 체크포인트 저장 (+ 마지막 날)
        if checkpoint_dir and ((day_idx + 1) % 7 == 0 or day_idx + 1 == days):
            cp_dir = checkpoint_dir / f"checkpoint_day{day_idx+1:03d}"
            cp_dir.mkdir(parents=True, exist_ok=True)
            with open(cp_dir / "results.pkl", "wb") as f:
                pickle.dump(results, f)
            with open(cp_dir / "agents.pkl", "wb") as f:
                pickle.dump(agents, f)
            with open(cp_dir / "global_store.pkl", "wb") as f:
                pickle.dump(global_store, f)
            with open(cp_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump({
                    "day_idx_next": day_idx + 1,
                    "start_date": start_date.isoformat(),
                    "prev_visitor_ids": [a.id for a in run_simulation._prev_day_visitors] if hasattr(run_simulation, '_prev_day_visitors') else []
                }, f, ensure_ascii=False)
            tqdm.write(f"  💾 체크포인트 저장: {cp_dir.name} (Day {day_idx+1})")

    # 마지막 타임슬롯 평점 반영 + 잔여 buffer 요약
    flush_result = await global_store.flush_and_summarize_async()
    if flush_result["flushed"] > 0:
        tqdm.write(f"      마지막 평점 반영: {flush_result['flushed']}건"
                   + (f" | 리뷰 요약: {flush_result['summarized']}개 매장" if flush_result["summarized"] > 0 else ""))

    day_pbar.close()
    sim_elapsed = time_module.perf_counter() - sim_start_time
    print(f"\n  시뮬레이션 완료: 총 {sim_elapsed:.1f}초 ({sim_elapsed/60:.1f}분)")

    return pd.DataFrame(results)


def save_results(results_df: pd.DataFrame, global_store: GlobalStore, agents: List[GenerativeAgent], settings):
    """결과 저장"""
    print(f"\n[4/4] 결과 저장 중...")

    output_dir = settings.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 전체 결과 CSV
    results_path = output_dir / "generative_simulation_result.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    print(f"  시뮬레이션 결과: {results_path}")

    # 방문 로그만 추출
    visit_df = results_df[results_df["decision"] == "visit"]
    visit_path = output_dir / "generative_visit_log.csv"
    visit_df.to_csv(visit_path, index=False, encoding="utf-8-sig")
    print(f"  방문 로그: {visit_path}")

    # 매장 평점 현황 저장
    store_path = output_dir / "store_ratings.json"
    global_store.save_to_json(store_path)
    print(f"  매장 평점: {store_path}")

    # 에이전트 최종 상태 저장
    agents_data = [a.to_dict() for a in agents]
    agents_path = output_dir / "agents_final_state.json"
    with open(agents_path, "w", encoding="utf-8") as f:
        json.dump(agents_data, f, ensure_ascii=False, indent=2)
    print(f"  에이전트 상태: {agents_path}")

    # 통계 요약
    print("\n" + "=" * 60)
    print("시뮬레이션 결과 요약")
    print("=" * 60)
    total = len(results_df)
    visits = len(visit_df)
    print(f"총 이벤트: {total:,}건")
    print(f"방문 이벤트: {visits:,}건 ({visits/total*100:.1f}%)")

    if visits > 0:
        # 매장별 방문 수
        print("\n매장별 방문 TOP 10:")
        store_visits = visit_df["visited_store"].value_counts().head(10)
        for store, count in store_visits.items():
            print(f"  {store}: {count}회")

        # 평균 평점 (1~5 스케일)
        avg_rating = visit_df["rating"].mean()
        print(f"\n평균 평점 (1~5점): {avg_rating:.2f}")

    # GlobalStore 통계
    stats = global_store.get_statistics()
    print(f"\n매장 평점 현황:")
    print(f"  총 매장: {stats['total_stores']}개")
    print(f"  에이전트 평점 있는 매장: {stats['stores_with_agent_ratings']}개")
    print(f"  총 에이전트 평점: {stats['total_agent_ratings']}건")



async def async_main():
    parser = argparse.ArgumentParser(description="Generative Agents 시뮬레이션")
    parser.add_argument(
        "--agents",
        type=int,
        default=160,
        help="생성할 에이전트 수 (기본: 160, 최대: 160)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="시뮬레이션 기간 (기본: 7일)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="예상치만 출력하고 실행하지 않음",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="--dry-run과 동일",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="확인 없이 바로 실행",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"랜덤 시드 (기본: {DEFAULT_SEED}, 개선 전/후 비교 시 동일 시드 사용)",
    )
    args = parser.parse_args()

    # 시드 고정 (개선 전/후 비교 시 동일 에이전트 구성 보장)
    random.seed(args.seed)
    np.random.seed(args.seed)

    agent_count = min(160, max(1, args.agents))
    days = max(1, args.days)

    settings = get_settings()

    print("=" * 60)
    print("Generative Agents 시뮬레이션")
    print(f"시드: {args.seed}")
    print("=" * 60)
    print(f"LLM: {settings.llm.provider} / {settings.llm.model_name}")

    # 예상치 계산 및 출력 (160명 기준: 상주 47 + 유동 113)
    estimates = estimate_simulation(agent_count, days, resident_count=47, floating_count=113)
    print_estimates(estimates)

    if args.dry_run or args.estimate_only:
        print("\n[dry-run] 예상치만 출력하고 종료합니다.")
        print("실제 실행하려면 --dry-run 플래그를 제거하세요.")
        return

    # 사용자 확인 (-y 옵션이 없을 때만)
    if not args.yes:
        print("\n시뮬레이션을 시작하시겠습니까?")
        print("(Ctrl+C로 취소)")
        try:
            input("Enter를 눌러 계속...")
        except KeyboardInterrupt:
            print("\n취소되었습니다.")
            return

    # 실행
    global_store = load_environment(settings)
    agents = generate_agents(agent_count)

    results_df = await run_simulation(
        agents, global_store, settings, days,
        max_concurrent_llm_calls=5,
    )
    save_results(results_df, global_store, agents, settings)

    print("\n시뮬레이션 완료!")

## Windows에서 asyncio 이벤트 루프 정책 설정 후 async_main() 실행. finally로 루프 정상 종료 보장.
def main():
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(async_main())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
