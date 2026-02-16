"""
Generative Agents 기반 시뮬레이션 실행 스크립트.

Stanford Generative Agents 논문 구조를 참조한 시뮬레이션:
- Persona Generation Module: 고유 조합의 에이전트 생성
- Memory Module: recent_history 기반 의사결정
- Action Algorithm: 4단계 LLM 기반 의사결정
- Global Store: 실시간 평점 축적

사용법:
    python scripts/run_generative_simulation.py [--agents N] [--dry-run]
"""

import argparse
import json
import sys
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

import pandas as pd

# 기본 시드 (개선 전/후 비교 시 동일 에이전트 구성 보장)
DEFAULT_SEED = 42

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from src.simulation_layer.persona.agent import GenerativeAgent, load_personas_from_md
from src.data_layer.global_store import get_global_store, GlobalStore
from src.simulation_layer.persona.cognitive_modules.action_algorithm import ActionAlgorithm
from src.data_layer.street_network import StreetNetwork, StreetNetworkConfig, AgentLocation, reset_street_network


# 시간대 정의 (4개) - 정확한 시간 기반
TIME_SLOTS = {
    "아침": 7,   # 07:00
    "점심": 12,  # 12:00
    "저녁": 18,  # 18:00
    "야식": 22,  # 22:00
}

# 요일
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]

# 시뮬레이션 속도: 현실 24배 빠름 (1분 현실 = 24분 시뮬레이션)
TIME_SPEED_MULTIPLIER = 24

# 에이전트 걷는 속도 (m/s) - 평균 보행 속도 약 1.4 m/s (5 km/h)
WALKING_SPEED_MS = 1.4

# 매장 인식 반경 (km)
STORE_RECOGNITION_RADIUS_KM = 3.0        # 유동 에이전트
RESIDENT_STORE_RADIUS_KM = 0.8           # 상주 에이전트 (800m)

# 유동 에이전트 일일 활동 수 (113명 중 53명만 매일 활동)
DAILY_FLOATING_AGENT_COUNT = 53

# OSM 네트워크 설정 (서울 강남역 주변 기준)
DEFAULT_NETWORK_CENTER_LAT = 37.4980
DEFAULT_NETWORK_CENTER_LNG = 127.0276
DEFAULT_NETWORK_RADIUS_M = 800.0  # 망원동 구역 내로 제한


def estimate_simulation(agent_count: int, days: int = 7, time_slots: int = 4,
                        resident_count: int = 47, floating_count: int = 113) -> Dict[str, Any]:
    """
    시뮬레이션 전 예상치 계산.

    LLM 호출 수 계산:
    - Step 1: 모든 에이전트 × 모든 타임슬롯 (외출 여부)
    - Step 2: 외출하는 에이전트만 (업종 선택)
    - Step 3: 외출하는 에이전트만 (매장 선택)
    - Step 4: 방문한 에이전트만 (평가)

    가정:
    - 상주 에이전트: 매일 전원 활동
    - 유동 에이전트: 매일 53명만 활동 (하루짜리)
    - 외출 확률 약 50%
    - 방문 성공률 약 90% (외출 결정 중)
    """
    daily_floating = min(DAILY_FLOATING_AGENT_COUNT, floating_count)
    daily_active = resident_count + daily_floating
    total_slots = daily_active * days * time_slots

    # Step 1: 모든 슬롯에서 호출
    step1_calls = total_slots

    # Step 2, 3: 외출하는 경우만 (약 50%)
    go_out_rate = 0.5
    step2_calls = int(total_slots * go_out_rate)
    step3_calls = step2_calls

    # Step 4: 실제 방문한 경우만 (외출 중 90%)
    visit_rate = 0.9
    step4_calls = int(step2_calls * visit_rate)

    total_llm_calls = step1_calls + step2_calls + step3_calls + step4_calls

    # GPT-4o-mini 비용 계산 (2024년 기준)
    # Input: $0.15 / 1M tokens, Output: $0.6 / 1M tokens
    # 평균 프롬프트 ~500 tokens, 응답 ~100 tokens
    avg_input_tokens = 500
    avg_output_tokens = 100
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
    cost = estimates["estimated_cost_usd"]
    print(f"예상 비용 (GPT-4o-mini):")
    print(f"  Input: ${cost['input']:.4f}")
    print(f"  Output: ${cost['output']:.4f}")
    print(f"  Total: ${cost['total']:.4f} (약 {cost['total'] * 1400:.0f}원)")
    print()
    print(f"예상 소요 시간: 약 {estimates['estimated_time_minutes']:.1f}분")
    print("=" * 60)


def load_environment(settings):
    """환경 데이터 로드"""
    print("\n[1/4] 환경 데이터 로드 중...")

    # Global Store 초기화
    GlobalStore.reset_instance()
    global_store = get_global_store()

    # JSON 매장 데이터 로드 (stores.csv 대신 JSON 파일 사용)
    json_dir = settings.paths.data_dir / "split_by_store_id_ver3"
    if json_dir.exists():
        global_store.load_from_json_files(json_dir)
        print(f"  매장 데이터 로드: {len(global_store.stores)}개")

    return global_store


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


def initialize_agent_locations(
    agents: List[GenerativeAgent],
    network: StreetNetwork
) -> Dict[int, AgentLocation]:
    """에이전트들의 초기 위치를 home_location 기반으로 OSM 네트워크 상에 배치"""
    agent_locations = {}
    for agent in agents:
        # home_location이 있으면 해당 좌표 사용, 없으면 랜덤
        lat, lng = agent.home_location
        location = network.initialize_agent_location(lat, lng)
        agent_locations[agent.id] = location

    return agent_locations


def run_simulation(
    agents: List[GenerativeAgent],
    global_store: GlobalStore,
    settings,
    days: int = 7,
) -> pd.DataFrame:
    """
    OSM 네트워크 기반 시뮬레이션 실행.

    에이전트들이 거리를 걸으며 타임슬롯(07:00, 12:00, 18:00, 22:00)마다
    반경 내 매장을 인식하고 의사결정을 수행합니다.

    시간 시스템:
    - 현실보다 24배 빠름 (1분 현실 = 24분 시뮬레이션)
    - 하루 24시간 시뮬레이션
    """
    print(f"\n[3/4] 시뮬레이션 실행 중...")
    print(f"  시간 배속: {TIME_SPEED_MULTIPLIER}x")
    print(f"  매장 인식 반경: 상주 {RESIDENT_STORE_RADIUS_KM}km / 유동 제한없음")

    # OSM 거리 네트워크 초기화
    network = initialize_street_network(global_store)

    # 상주/유동 에이전트 분리
    resident_agents = [a for a in agents if a.is_resident]
    floating_agents = [a for a in agents if a.is_floating]
    print(f"\n  상주 에이전트: {len(resident_agents)}명 (매일 활동)")
    print(f"  유동 에이전트: {len(floating_agents)}명 (매일 {DAILY_FLOATING_AGENT_COUNT}명 샘플링)")

    algorithm = ActionAlgorithm(rate_limit_delay=0.5)

    # 결과 저장
    results = []

    # 시작 날짜 (금요일 - 일식 점심 수요 높은 요일)
    start_date = datetime(2025, 2, 7)  # 금요일 시작

    # 타임슬롯 리스트 (시간 순서대로 정렬)
    time_slot_list = sorted(TIME_SLOTS.items(), key=lambda x: x[1])

    # 에이전트 초기 위치 딕셔너리
    agent_locations = {}

    for day_idx in range(days):
        current_date = start_date + timedelta(days=day_idx)
        weekday = WEEKDAYS[current_date.weekday()]

        # 매일 유동 에이전트 53명 랜덤 샘플링
        daily_floating_count = min(DAILY_FLOATING_AGENT_COUNT, len(floating_agents))
        daily_floating = random.sample(floating_agents, daily_floating_count)

        # 유동 에이전트: 매일 새로운 초기 위치(FLOATING_LOCATIONS)에서 시작
        from src.simulation_layer.persona.agent import FLOATING_LOCATIONS
        for agent in daily_floating:
            loc = random.choice(list(FLOATING_LOCATIONS.values()))
            agent_locations[agent.id] = network.initialize_agent_location(loc["lat"], loc["lng"])

        # 상주 에이전트: 매일 home_location에서 시작
        for agent in resident_agents:
            lat, lng = agent.home_location
            agent_locations[agent.id] = network.initialize_agent_location(lat, lng)

        # 오늘 활동할 에이전트 = 상주(전원) + 유동(샘플링)
        daily_agents = resident_agents + daily_floating

        print(f"\n  === Day {day_idx + 1}/{days}: {current_date.strftime('%Y-%m-%d')} ({weekday}요일) ===")
        print(f"      활동 에이전트: {len(daily_agents)}명 (상주 {len(resident_agents)} + 유동 {len(daily_floating)})")

        # 전체 진행률 계산용
        total_day_slots = len(daily_agents) * len(TIME_SLOTS)
        day_processed = 0

        # 하루의 시뮬레이션 시간 (00:00 시작)
        sim_hour = 0  # 시뮬레이션 시간 (시)

        for slot_name, slot_hour in time_slot_list:
            # 해당 타임슬롯까지 시간 이동 (에이전트들이 걸어다님)
            hours_to_walk = slot_hour - sim_hour
            if hours_to_walk > 0:
                # 현실 시간으로 걷는 시간 계산 (24배 빠름)
                # 시뮬레이션 1시간 = 현실 2.5분
                walk_time_real_seconds = (hours_to_walk * 3600) / TIME_SPEED_MULTIPLIER
                walk_distance_m = walk_time_real_seconds * WALKING_SPEED_MS

                # 오늘 활동하는 에이전트만 이동
                for agent in daily_agents:
                    location = agent_locations[agent.id]
                    agent_locations[agent.id] = network.move_agent(location, walk_distance_m)

            sim_hour = slot_hour
            slot_time = current_date.replace(hour=slot_hour, minute=0, second=0)

            # 이전 타임슬롯에서 쌓인 평점을 GlobalStore에 반영
            flushed = global_store.flush_pending_ratings()
            if flushed > 0:
                print(f"      평점 반영: {flushed}건")

            print(f"    [{slot_name}] {slot_time.strftime('%H:%M')} - 에이전트 의사결정 중...")

            for agent in daily_agents:
                day_processed += 1

                # 진행 상황 출력 (10% 단위)
                if day_processed % max(1, total_day_slots // 10) == 0:
                    pct = day_processed / total_day_slots * 100
                    print(f"      Day {day_idx+1} 진행: {day_processed}/{total_day_slots} ({pct:.0f}%)")

                # 에이전트 현재 위치
                location = agent_locations[agent.id]

                # 에이전트 유형별 매장 인식 범위
                if agent.is_resident:
                    # 상주 에이전트: 반경 800m 이내 매장만
                    nearby_stores = global_store.get_stores_in_radius(
                        center_lat=location.lat,
                        center_lng=location.lng,
                        radius_km=RESIDENT_STORE_RADIUS_KM
                    )
                else:
                    # 유동 에이전트: 전체 매장
                    nearby_stores = list(global_store.stores.values())

                # 4단계 의사결정
                result = algorithm.process_decision(
                    agent=agent,
                    nearby_stores=nearby_stores,
                    time_slot=slot_name,
                    weekday=weekday,
                    current_datetime=slot_time.isoformat(),
                )

                # 결과 기록
                record = {
                    "timestamp": slot_time.strftime("%Y-%m-%d %H:%M"),
                    "agent_id": agent.id,
                    "persona_id": agent.persona_id,
                    "generation": agent.generation,
                    "gender_composition": agent.gender_composition,
                    "segment": agent.segment,
                    "weekday": weekday,
                    "time_slot": slot_name,
                    "decision": result["decision"],
                    "visited_store": result.get("visited_store"),
                    "visited_category": result.get("visited_category"),
                    # 위치 정보 추가
                    "agent_lat": location.lat,
                    "agent_lng": location.lng,
                    # 평점 정보
                    "taste_rating": result.get("ratings", {}).get("taste") if result.get("ratings") else None,
                    "value_rating": result.get("ratings", {}).get("value") if result.get("ratings") else None,
                    "atmosphere_rating": result.get("ratings", {}).get("atmosphere") if result.get("ratings") else None,
                    "comment": result.get("comment", ""),
                    "reason": result.get("reason", ""),
                    "nearby_store_count": len(nearby_stores),
                }
                results.append(record)

    # 마지막 타임슬롯의 평점 반영
    flushed = global_store.flush_pending_ratings()
    if flushed > 0:
        print(f"      마지막 평점 반영: {flushed}건")

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
        avg_taste = visit_df["taste_rating"].mean()
        avg_value = visit_df["value_rating"].mean()
        avg_atmosphere = visit_df["atmosphere_rating"].mean() if "atmosphere_rating" in visit_df else 0
        print(f"\n평균 평점 (1~5점):")
        print(f"  맛: {avg_taste:.2f}")
        print(f"  가성비: {avg_value:.2f}")
        print(f"  분위기: {avg_atmosphere:.2f}")

    # GlobalStore 통계
    stats = global_store.get_statistics()
    print(f"\n매장 평점 현황:")
    print(f"  총 매장: {stats['total_stores']}개")
    print(f"  에이전트 평점 있는 매장: {stats['stores_with_agent_ratings']}개")
    print(f"  총 에이전트 평점: {stats['total_agent_ratings']}건")




def main():
    parser = argparse.ArgumentParser(description="Generative Agents 시뮬레이션")
    parser.add_argument(
        "--agents",
        type=int,
        default=96,
        help="생성할 에이전트 수 (기본: 96, 최대: 99)",
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

    results_df = run_simulation(agents, global_store, settings, days)
    save_results(results_df, global_store, agents, settings)

    print("\n시뮬레이션 완료!")


if __name__ == "__main__":
    main()
