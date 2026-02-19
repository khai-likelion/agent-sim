"""
전략 적용 전/후 비교 시뮬레이션

흐름:
  1. 전략 적용 전 (Baseline): 원본 돼지야.json으로 7일 시뮬레이션
  2. StrategyBridge로 돼지야_report.md 전략 적용 → 돼지야.json 업데이트
  3. 전략 적용 후 (After): 동일 에이전트/시드로 7일 시뮬레이션
  4. 결과 비교 출력

사용법:
    python scripts/run_before_after_sim.py [--agents N] [--days D] [--target-store 돼지야]
"""

import argparse
import asyncio
import json
import shutil
import sys
import os
import random
import numpy as np
from pathlib import Path
from datetime import datetime

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from config import get_settings
from src.data_layer.global_store import GlobalStore, get_global_store

# 시뮬레이션 실행 함수 임포트
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_generative_simulation import (
    run_simulation, generate_agents,
    estimate_simulation, print_estimates,
    DEFAULT_SEED,
)

# StrategyBridge 임포트
from X_to_Sim import apply_x_report_strategy_async


# ============================================================
# 환경 로드 (커스텀 데이터 디렉토리 지원)
# ============================================================

def load_environment_from_dir(store_json_dir: Path, target_store: str = None):
    """지정된 디렉토리에서 매장 데이터 로드"""
    print(f"\n매장 데이터 로드: {store_json_dir}")
    GlobalStore.reset_instance()
    global_store = get_global_store()

    if store_json_dir.exists():
        global_store.load_from_json_files(store_json_dir)
        print(f"  매장 {len(global_store.stores)}개 로드 완료")
    else:
        print(f"  ⚠️ 디렉토리 없음: {store_json_dir}")

    if target_store:
        target = global_store.get_by_name(target_store)
        if target:
            print(f"  [TARGET] 타겟 매장: {target_store}")
        else:
            print(f"  [WARN] 타겟 매장 '{target_store}' 없음")

    return global_store


def save_results_to(results_df, global_store, agents, output_dir: Path, label: str):
    """결과를 지정 디렉토리에 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "simulation_result.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    visit_df = results_df[results_df["decision"] == "visit"]
    visit_path = output_dir / "visit_log.csv"
    visit_df.to_csv(visit_path, index=False, encoding="utf-8-sig")

    store_path = output_dir / "store_ratings.json"
    global_store.save_to_json(store_path)

    agents_data = [a.to_dict() for a in agents]
    with open(output_dir / "agents_final.json", "w", encoding="utf-8") as f:
        json.dump(agents_data, f, ensure_ascii=False, indent=2)

    print(f"\n[{label}] 결과 저장 완료: {output_dir}")
    total = len(results_df)
    visits = len(visit_df)
    print(f"  총 이벤트: {total:,}건 | 방문: {visits:,}건 ({visits/total*100:.1f}%)")

    if visits > 0:
        top10 = visit_df["visited_store"].value_counts().head(5)
        print(f"  방문 TOP 5:")
        for store, count in top10.items():
            print(f"    {store}: {count}회")

    return visit_df


def compare_results(before_visit_df, after_visit_df, target_store: str):
    """전/후 비교 출력"""
    print("\n" + "=" * 60)
    print("📊 전략 적용 전/후 비교 결과")
    print("=" * 60)

    for label, df in [("전략 전", before_visit_df), ("전략 후", after_visit_df)]:
        target_visits = df[df["visited_store"] == target_store]
        total_visits = len(df)
        target_count = len(target_visits)
        share = target_count / total_visits * 100 if total_visits > 0 else 0

        avg_taste = target_visits["taste_rating"].mean() if len(target_visits) > 0 else 0
        avg_value = target_visits["value_rating"].mean() if len(target_visits) > 0 else 0

        print(f"\n[{label}]")
        print(f"  '{target_store}' 방문 횟수: {target_count}회 (전체 방문 중 {share:.1f}%)")
        print(f"  평균 맛 평점: {avg_taste:.2f}")
        print(f"  평균 가성비 평점: {avg_value:.2f}")

    before_count = len(before_visit_df[before_visit_df["visited_store"] == target_store])
    after_count = len(after_visit_df[after_visit_df["visited_store"] == target_store])
    diff = after_count - before_count
    pct = (diff / before_count * 100) if before_count > 0 else 0

    print(f"\n🎯 '{target_store}' 방문 변화: {before_count} → {after_count} ({pct:+.1f}%)")


# ============================================================
# 메인
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="전략 전/후 비교 시뮬레이션")
    parser.add_argument("--agents", type=int, default=160, help="에이전트 수 (기본: 160)")
    parser.add_argument("--days", type=int, default=7, help="시뮬레이션 기간 (기본: 7일)")
    parser.add_argument("--target-store", type=str, default="돼지야", help="분석 대상 매장")
    parser.add_argument("--report", type=str,
                        default=str(PROJECT_ROOT / "돼지야_report.md"),
                        help="X-Report 경로")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="랜덤 시드")
    parser.add_argument("--skip-bridge", action="store_true",
                        help="StrategyBridge 건너뜀 (이미 적용된 경우)")
    parser.add_argument("-y", "--yes", action="store_true", help="확인 없이 바로 실행")
    parser.add_argument("--output-prefix", type=str, default=None,
                        help="결과 폴더 접두사 (예: '돼지야' → 돼지야_before, 돼지야_after)")
    args = parser.parse_args()

    settings = get_settings()
    store_dir = settings.paths.split_store_dir
    output_base = settings.paths.output_dir

    target_store_json = store_dir / f"{args.target_store}.json"
    backup_path = store_dir / f"{args.target_store}.json.bak"
    applied_path = store_dir / f"{args.target_store}_전략적용.json"

    api_key = os.getenv("LLM_API_KEY", "")

    print("=" * 60)
    print(f"전략 전/후 비교 시뮬레이션")
    print(f"타겟 매장: {args.target_store}")
    print(f"시뮬레이션: {args.days}일 / {args.agents}명 / 시드: {args.seed}")
    print(f"데이터: {store_dir}")
    print(f"X-Report: {args.report}")
    print("=" * 60)

    if not args.yes:
        print("\n시뮬레이션을 시작하시겠습니까?")
        try:
            input("Enter를 눌러 계속...")
        except KeyboardInterrupt:
            print("\n취소되었습니다.")
            return

    # 백업 확인 (없으면 생성)
    if not backup_path.exists() and target_store_json.exists():
        shutil.copy2(target_store_json, backup_path)
        print(f"✅ 원본 백업: {backup_path.name}")

    # ============================================================
    # 시뮬레이션 1: 전략 적용 전 (원본)
    # ============================================================
    print("\n" + "=" * 60)
    print("▶ 시뮬레이션 1/2: 전략 적용 전 (Baseline)")
    print("=" * 60)

    # 원본 복원 (백업에서)
    if backup_path.exists():
        shutil.copy2(backup_path, target_store_json)
        print(f"  원본 데이터 사용: {args.target_store}.json")

    # 예상치 출력 (160명 기준: 상주 47 + 유동 113, 병렬 60)
    est = estimate_simulation(args.agents, args.days, resident_count=47, floating_count=113,
                              max_concurrent_llm_calls=60)
    print_estimates(est)
    print("※ 시뮬레이션 2회 실행 시 총 예상 시간은 위의 약 2배입니다.\n")

    random.seed(args.seed)
    np.random.seed(args.seed)

    agents_before = generate_agents(args.agents)
    global_store_before = load_environment_from_dir(store_dir, args.target_store)

    results_before = await run_simulation(
        agents_before, global_store_before, settings, args.days,
        target_store=args.target_store,
        max_concurrent_llm_calls=60,
    )

    prefix = args.output_prefix or args.target_store
    before_dir = output_base / f"{prefix}_before"
    after_dir = output_base / f"{prefix}_after"

    before_visit_df = save_results_to(
        results_before, global_store_before, agents_before,
        before_dir, "전략 전"
    )

    # ============================================================
    # StrategyBridge 전략 적용
    # ============================================================
    if not args.skip_bridge:
        print("\n" + "=" * 60)
        print("⚙️  StrategyBridge: 전략 적용 중...")
        print("=" * 60)

        # 모든 전략 ID (S1~S3, 솔루션 A/B/C)
        strategy_ids = ["S1_A", "S1_B", "S1_C", "S2_A", "S2_B", "S2_C", "S3_A", "S3_B", "S3_C"]

        await apply_x_report_strategy_async(
            store_json_path=str(target_store_json),
            x_report_path=args.report,
            selected_strategy_ids=strategy_ids,
            api_key=api_key,
            output_path=str(target_store_json),  # 같은 파일에 덮어씀
        )
        # 전략 적용본 별도 보관dk
        shutil.copy2(target_store_json, applied_path)
        print(f"✅ 전략 적용본 저장: {applied_path.name}")
    else:
        print("\n⚙️  --skip-bridge: StrategyBridge 건너뜀")
        if applied_path.exists():
            shutil.copy2(applied_path, target_store_json)
            print(f"  기존 전략 적용본 사용: {applied_path.name}")

    # ============================================================
    # 시뮬레이션 2: 전략 적용 후
    # ============================================================
    print("\n" + "=" * 60)
    print("▶ 시뮬레이션 2/2: 전략 적용 후 (After Strategy)")
    print("=" * 60)

    # 동일 시드로 재시작 (같은 에이전트 구성 보장)
    random.seed(args.seed)
    np.random.seed(args.seed)

    agents_after = generate_agents(args.agents)
    global_store_after = load_environment_from_dir(store_dir, args.target_store)

    results_after = await run_simulation(
        agents_after, global_store_after, settings, args.days,
        target_store=args.target_store,
        max_concurrent_llm_calls=60,
    )

    after_visit_df = save_results_to(
        results_after, global_store_after, agents_after,
        after_dir, "전략 후"
    )

    # ============================================================
    # 비교 결과 출력
    # ============================================================
    compare_results(before_visit_df, after_visit_df, args.target_store)

    # 비교 요약 JSON 저장
    summary = {
        "target_store": args.target_store,
        "simulation_days": args.days,
        "seed": args.seed,
        "before": {
            "total_visits": int(len(before_visit_df)),
            "target_visits": int(len(before_visit_df[before_visit_df["visited_store"] == args.target_store])),
        },
        "after": {
            "total_visits": int(len(after_visit_df)),
            "target_visits": int(len(after_visit_df[after_visit_df["visited_store"] == args.target_store])),
        },
        "run_at": datetime.now().isoformat(),
    }
    summary_path = output_base / "before_after_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n📄 비교 요약 저장: {summary_path}")
    print("\n✅ 전략 전/후 비교 시뮬레이션 완료!")


if __name__ == "__main__":
    asyncio.run(main())
