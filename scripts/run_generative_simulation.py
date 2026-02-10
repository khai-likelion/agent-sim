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
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

import pandas as pd

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from src.simulation_layer.persona.generative_agent import (
    GenerativeAgent,
    GenerativeAgentFactory,
)
from src.data_layer.global_store import get_global_store, GlobalStore
from src.simulation_layer.persona.cognitive_modules.action_algorithm import ActionAlgorithm
from src.data_layer.street_network import StreetNetwork, StreetNetworkConfig


# 시간대 정의 (4개)
TIME_SLOTS = ["아침", "점심", "저녁", "야간"]
TIME_SLOT_HOURS = {
    "아침": (7, 10),
    "점심": (11, 14),
    "저녁": (17, 20),
    "야간": (21, 24),
}

# 요일
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]


def estimate_simulation(agent_count: int, days: int = 7, time_slots: int = 4) -> Dict[str, Any]:
    """
    시뮬레이션 전 예상치 계산.

    LLM 호출 수 계산:
    - Step 1: 모든 에이전트 × 모든 타임슬롯 (외출 여부)
    - Step 2: 외출하는 에이전트만 (업종 선택)
    - Step 3: 외출하는 에이전트만 (매장 선택)
    - Step 4: 방문한 에이전트만 (평가)

    가정:
    - 외출 확률 약 50%
    - 방문 성공률 약 90% (외출 결정 중)
    """
    total_slots = agent_count * days * time_slots

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
    print(f"에이전트 수: {estimates['agent_count']}명")
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


def load_environment(settings, improved: bool = False, target_store: str = None):
    """환경 데이터 로드"""
    print("\n[1/4] 환경 데이터 로드 중...")

    # Global Store 초기화
    GlobalStore.reset_instance()
    global_store = get_global_store()
    global_store.load_from_csv(settings.paths.stores_csv)
    print(f"  매장 데이터 로드: {len(global_store.stores)}개")

    # JSON 리뷰 데이터 로드
    json_dir = settings.paths.data_dir / "split_by_store_id"
    if json_dir.exists():
        global_store.load_from_json_files(json_dir)
        print(f"  리뷰 데이터 로드 완료")

    # 타겟 매장 개선 버전 적용
    if improved and target_store:
        print(f"\n  ⭐ {target_store} 매장 개선 적용:")
        for store_id, store in global_store.stores.items():
            if store.store_name == target_store:
                # 개선 전 정보 출력
                print(f"    개선 전: 가격={store.average_price}원, 맛={store.base_taste_score}, 가성비={store.base_value_score}")

                # 매장별 맞춤 개선 적용
                if target_store == "아루감":
                    # 아루감: 객단가 상승 + 프리미엄화
                    store.average_price = 55000
                    store.base_taste_score = 0.85
                    store.base_value_score = 0.80
                    improvement_desc = "프리미엄 세트메뉴, 빠른회전, 추가주문 트리거"
                elif target_store == "류진":
                    # 류진: TOP 1, 2, 3 통합 개선
                    # TOP 1: 웨이팅 병목 제거 → 회전율 개선, 대기 이탈 감소
                    # TOP 2: 객단가 세트 전략 → 2인 세트 도입, 객단가 +10%
                    # TOP 3: 온라인 최적화 → 리뷰 키워드 유도, 평판 관리
                    store.average_price = 11000  # 10,000 → 11,000 (세트 전략 객단가 +10%)
                    store.base_taste_score = 0.85  # 0.5 → 0.85 (웨이팅 해소 + 맛 관리 + 리뷰 개선)
                    store.base_value_score = 0.85  # 0.5 → 0.85 (빠른 서비스 + 세트 가성비)
                    improvement_desc = "TOP1: 웨이팅 병목 제거 / TOP2: 2인 세트 도입(객단가+10%) / TOP3: 리뷰 키워드 유도"
                else:
                    # 기본 개선: 평점 소폭 상승
                    store.base_taste_score = min(1.0, store.base_taste_score + 0.2)
                    store.base_value_score = min(1.0, store.base_value_score + 0.2)
                    improvement_desc = "서비스 품질 개선"

                print(f"    개선 후: 가격={store.average_price}원, 맛={store.base_taste_score}, 가성비={store.base_value_score}")
                print(f"    ({improvement_desc})")
                break

    return global_store


def generate_agents(agent_count: int) -> List[GenerativeAgent]:
    """에이전트 생성"""
    print(f"\n[2/4] 에이전트 생성 중 ({agent_count}명)...")

    factory = GenerativeAgentFactory()
    agents = factory.generate_unique_agents(max_count=agent_count)

    # 분포 출력
    factory.print_distribution(agents)

    return agents


def run_simulation(
    agents: List[GenerativeAgent],
    global_store: GlobalStore,
    settings,
    days: int = 7,
    target_store: str = "류진",
) -> pd.DataFrame:
    """
    시뮬레이션 실행.
    """
    print(f"\n[3/4] 시뮬레이션 실행 중...")
    print(f"  타겟 매장: {target_store}")

    algorithm = ActionAlgorithm(rate_limit_delay=0.5)

    # 모든 매장 목록
    all_stores = list(global_store.stores.values())

    # 결과 저장
    results = []

    # 시작 날짜 (금요일 - 일식 점심 수요 높은 요일)
    start_date = datetime(2025, 2, 7)  # 금요일 시작

    total_slots = len(agents) * days * len(TIME_SLOTS)
    processed = 0

    for day_idx in range(days):
        current_date = start_date + timedelta(days=day_idx)
        # 실제 날짜의 요일 계산 (월=0, 일=6)
        weekday = WEEKDAYS[current_date.weekday()]

        # 새로운 날 시작 시 상태 리셋은 하지 않음 (메모리 유지)

        for time_slot in TIME_SLOTS:
            hour_start, hour_end = TIME_SLOT_HOURS[time_slot]
            slot_time = current_date.replace(hour=random.randint(hour_start, hour_end - 1))

            for agent in agents:
                processed += 1

                # 진행 상황 출력 (10% 단위)
                if processed % max(1, total_slots // 10) == 0:
                    pct = processed / total_slots * 100
                    print(f"  진행: {processed}/{total_slots} ({pct:.0f}%)")

                # 주변 매장 (실제로는 위치 기반이지만, 여기서는 전체 매장 사용)
                # 타겟 매장을 항상 포함하여 방문 기회 제공
                target_store_obj = next((s for s in all_stores if s.store_name == target_store), None)
                other_stores = [s for s in all_stores if s.store_name != target_store]
                nearby_stores = random.sample(other_stores, min(19, len(other_stores)))
                if target_store_obj:
                    nearby_stores.append(target_store_obj)

                # 4단계 의사결정
                result = algorithm.process_decision(
                    agent=agent,
                    nearby_stores=nearby_stores,
                    time_slot=time_slot,
                    weekday=weekday,
                    current_datetime=slot_time.isoformat(),
                )

                # 결과 기록
                record = {
                    "timestamp": slot_time.strftime("%Y-%m-%d %H:%M"),
                    "agent_id": agent.id,
                    "agent_name": agent.name,
                    "generation": agent.generation,
                    "age": agent.age,
                    "gender": agent.gender,
                    "segment": agent.segment,
                    "health_preference": agent.health_preference,
                    "change_preference": agent.change_preference,
                    "budget": agent.budget_per_meal,
                    "weekday": weekday,
                    "time_slot": time_slot,
                    "decision": result["decision"],
                    "visited_store": result.get("visited_store"),
                    "visited_category": result.get("visited_category"),
                    "taste_rating": result.get("ratings", {}).get("taste") if result.get("ratings") else None,
                    "value_rating": result.get("ratings", {}).get("value") if result.get("ratings") else None,
                    "reason": result.get("reason", ""),
                }
                results.append(record)

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

        # 평균 평점
        avg_taste = visit_df["taste_rating"].mean()
        avg_value = visit_df["value_rating"].mean()
        print(f"\n평균 평점:")
        print(f"  맛: {avg_taste:.2f} / 2.00")
        print(f"  가성비: {avg_value:.2f} / 2.00")

    # GlobalStore 통계
    stats = global_store.get_statistics()
    print(f"\n매장 평점 현황:")
    print(f"  총 매장: {stats['total_stores']}개")
    print(f"  에이전트 평점 있는 매장: {stats['stores_with_agent_ratings']}개")
    print(f"  총 에이전트 평점: {stats['total_agent_ratings']}건")


def save_target_store(target_store: str, output_dir: Path):
    """타겟 매장 설정 저장 (대시보드와 공유)"""
    config = {
        "target_store": target_store,
        "description": "현재 시뮬레이션 및 대시보드에서 추적 중인 타겟 매장",
        "updated_at": datetime.now().isoformat()
    }
    config_path = output_dir / "target_store.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  타겟 매장 설정: {config_path}")


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
        "--target-store",
        type=str,
        default="류진",
        help="추적할 타겟 매장 이름 (기본: 류진)",
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
        "--improved",
        action="store_true",
        help="타겟 매장 개선 버전으로 시뮬레이션 (--target-store와 함께 사용)",
    )
    args = parser.parse_args()

    agent_count = min(99, max(1, args.agents))
    days = max(1, args.days)

    settings = get_settings()

    print("=" * 60)
    print("Generative Agents 시뮬레이션")
    print("=" * 60)
    print(f"LLM: {settings.llm.provider} / {settings.llm.model_name}")
    print(f"타겟 매장: {args.target_store}")

    # 예상치 계산 및 출력
    estimates = estimate_simulation(agent_count, days)
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
    global_store = load_environment(settings, improved=args.improved, target_store=args.target_store)
    agents = generate_agents(agent_count)

    # 타겟 매장 설정 저장
    save_target_store(args.target_store, settings.paths.output_dir)

    results_df = run_simulation(agents, global_store, settings, days, target_store=args.target_store)
    save_results(results_df, global_store, agents, settings)

    print("\n시뮬레이션 완료!")


if __name__ == "__main__":
    main()
