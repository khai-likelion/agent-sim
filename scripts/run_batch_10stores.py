"""
10개 매장 전략 전/후 비교 시뮬레이션 배치 실행.

160명 에이전트, 7일, 시드 42
이미 완료된 매장은 건너뜀 → 중단 후 재실행 시 이어서 진행
실패 시 30초 대기 후 최대 3회 재시도
"""

import asyncio
import json
import shutil
import sys
import os
import random
import time
import io

# Windows cp949 이모지 인코딩 에러 방지
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from config import get_settings
from src.data_layer.global_store import GlobalStore, get_global_store
from run_generative_simulation import run_simulation, generate_agents, DEFAULT_SEED
from run_before_after_sim import (
    load_environment_from_dir, save_results_to, compare_results
)
from X_to_Sim import apply_x_report_strategy_async

AGENTS = 160
DAYS = 7
SEED = DEFAULT_SEED

# (JSON파일명, report파일명, 출력폴더명)
STORES = [
    ("돼지야", "돼지야_report.md", "돼지야"),
    ("망원부자부대찌개", "망원부자부대찌개_report.md", "망원부자부대찌개"),
    ("메가MGC커피 망원망리단길점", "메가MGC커피망원망리단길점_report.md", "메가MGC커피망원망리단길점"),
    ("반했닭옛날통닭 망원점", "반했닭옛날통닭망원점_report.md", "반했닭옛날통닭망원점"),
    ("오늘요거", "오늘요거_report.md", "오늘요거"),
    ("육장", "육장_report.md", "육장"),
    ("육회by유신 망원점", "육회by유신망원점_report.md", "육회by유신망원점"),
    ("전조", "전조_report.md", "전조"),
    ("정드린치킨 망원점", "정드린치킨망원점_report.md", "정드린치킨망원점"),
    ("크리머리", "크리머리_report.md", "크리머리"),
]


async def run_one_store(store_name: str, report_file: str, output_prefix: str,
                        settings, store_dir: Path, output_base: Path):
    """한 매장의 before/after 시뮬레이션 실행"""
    target_store_json = store_dir / f"{store_name}.json"
    backup_path = store_dir / f"{store_name}.json.bak"
    applied_path = store_dir / f"{store_name}_전략적용.json"
    report_path = PROJECT_ROOT / report_file

    before_dir = output_base / f"{output_prefix}_before"
    after_dir = output_base / f"{output_prefix}_after"

    api_key = os.getenv("LLM_API_KEY", "")

    # 백업
    if not backup_path.exists() and target_store_json.exists():
        shutil.copy2(target_store_json, backup_path)
        print(f"  원본 백업: {backup_path.name}")

    # ── Before 시뮬레이션 ──
    print(f"\n  [Before] 전략 적용 전 시뮬레이션")
    if backup_path.exists():
        shutil.copy2(backup_path, target_store_json)

    random.seed(SEED)
    np.random.seed(SEED)

    agents_before = generate_agents(AGENTS)
    global_store_before = load_environment_from_dir(store_dir, store_name)

    results_before = await run_simulation(
        agents_before, global_store_before, settings, DAYS,
        target_store=store_name, max_concurrent_llm_calls=20,
    )
    before_visit_df = save_results_to(
        results_before, global_store_before, agents_before,
        before_dir, "전략 전"
    )

    # ── StrategyBridge 전략 적용 ──
    print(f"\n  [Bridge] 전략 적용 중...")
    strategy_ids = ["S1_A", "S1_B", "S1_C", "S2_A", "S2_B", "S2_C", "S3_A", "S3_B", "S3_C"]

    await apply_x_report_strategy_async(
        store_json_path=str(target_store_json),
        x_report_path=str(report_path),
        selected_strategy_ids=strategy_ids,
        api_key=api_key,
        output_path=str(target_store_json),
    )
    shutil.copy2(target_store_json, applied_path)

    # ── After 시뮬레이션 ──
    print(f"\n  [After] 전략 적용 후 시뮬레이션")
    random.seed(SEED)
    np.random.seed(SEED)

    agents_after = generate_agents(AGENTS)
    global_store_after = load_environment_from_dir(store_dir, store_name)

    results_after = await run_simulation(
        agents_after, global_store_after, settings, DAYS,
        target_store=store_name, max_concurrent_llm_calls=20,
    )
    after_visit_df = save_results_to(
        results_after, global_store_after, agents_after,
        after_dir, "전략 후"
    )

    # ── 비교 ──
    compare_results(before_visit_df, after_visit_df, store_name)

    # 요약 저장
    summary = {
        "target_store": store_name,
        "output_prefix": output_prefix,
        "simulation_days": DAYS,
        "seed": SEED,
        "before": {
            "total_visits": int(len(before_visit_df)),
            "target_visits": int(len(before_visit_df[before_visit_df["visited_store"] == store_name])),
        },
        "after": {
            "total_visits": int(len(after_visit_df)),
            "target_visits": int(len(after_visit_df[after_visit_df["visited_store"] == store_name])),
        },
        "run_at": datetime.now().isoformat(),
    }
    summary_path = output_base / f"{output_prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 원본 복원
    if backup_path.exists():
        shutil.copy2(backup_path, target_store_json)

    return summary


async def main():
    settings = get_settings()
    store_dir = settings.paths.split_store_dir
    output_base = settings.paths.output_dir

    total = len(STORES)
    completed = 0
    failed = 0

    print("=" * 60)
    print(f"10개 매장 전략 전/후 비교 시뮬레이션 배치")
    print(f"에이전트: {AGENTS}명 / 기간: {DAYS}일 / 시드: {SEED}")
    print("=" * 60)

    batch_start = time.perf_counter()

    for i, (store_name, report_file, output_prefix) in enumerate(STORES):
        idx = i + 1

        # 이미 완료된 매장 건너뛰기
        before_log = output_base / f"{output_prefix}_before" / "visit_log.csv"
        after_log = output_base / f"{output_prefix}_after" / "visit_log.csv"

        if before_log.exists() and after_log.exists():
            print(f"\n[{idx}/{total}] {store_name} - 이미 완료됨, 건너뜀")
            completed += 1
            continue

        print(f"\n{'=' * 60}")
        print(f"[{idx}/{total}] {store_name} 시뮬레이션 시작")
        print(f"{'=' * 60}")

        store_start = time.perf_counter()

        for retry in range(1, 4):
            try:
                await run_one_store(store_name, report_file, output_prefix,
                                    settings, store_dir, output_base)
                elapsed = time.perf_counter() - store_start
                print(f"\n[{idx}/{total}] {store_name} 완료! ({elapsed/60:.1f}분)")
                completed += 1
                break
            except Exception as e:
                print(f"\n[{idx}/{total}] {store_name} 실패 (시도 {retry}/3): {e}")
                if retry < 3:
                    print("30초 후 재시도...")
                    await asyncio.sleep(30)
                else:
                    failed += 1
                    print(f"[{idx}/{total}] {store_name} 3회 실패, 다음 매장으로 이동")

    batch_elapsed = time.perf_counter() - batch_start
    print(f"\n{'=' * 60}")
    print(f"배치 완료: 성공 {completed}/{total}, 실패 {failed}/{total}")
    print(f"총 소요시간: {batch_elapsed/60:.1f}분 ({batch_elapsed/3600:.1f}시간)")
    print(f"결과 폴더: data/output/{{매장명}}_before, data/output/{{매장명}}_after")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
