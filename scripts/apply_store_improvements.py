"""
매장 개선사항 적용 스크립트

JSON 스키마를 유지하면서 개선사항을 반영합니다.
수정 가능 필드: revenue_analysis, top_keywords, rag_context, category
수정 금지 필드: review_metrics, raw_data_context, metadata, store_id, store_name
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "split_by_store_id"
BACKUP_DIR = PROJECT_ROOT / "data" / "raw" / "split_by_store_id_backup"


# 류진 개선사항 (TOP 1, 2, 3 통합)
RYUJIN_IMPROVEMENTS = {
    "store_name": "류진",
    "store_id": "1484901440",

    # TOP 2: 객단가 세트 전략 반영
    "revenue_analysis": (
        "객단가는 41,000원으로 상승했습니다(기존 37,685원 대비 +9%). "
        "2인 세트(면+사이드+음료) 도입으로 객단가가 개선되었습니다. "
        "주말 비중 36.1%로 높으며, 저녁 시간대(17:00-21:00)에 고객이 집중됩니다. "
        "세트 주문 시 사이드/주류 부착률이 +10%p 증가했습니다."
    ),

    # TOP 1 + TOP 3: 웨이팅 개선 + 리뷰 키워드 유도
    "top_keywords": [
        "진한 육수",
        "쫄깃한 면",
        "푸짐한 차슈",
        "청결한 매장",
        "빠른 서비스",
        "2인 세트 추천",
        "원격줄서기",
        "데이트 맛집"
    ],

    # TOP 1, 2, 3 통합 개선 내용
    "rag_context": (
        "류진은 지로 라멘을 중심으로 한 라멘 전문점으로, 최근 서비스 개선을 통해 "
        "고객 만족도가 크게 향상되었습니다. "
        "[웨이팅 개선] 원격줄서기 시스템 도입으로 대기 시간 체감이 크게 줄었으며, "
        "피크타임 메뉴 최적화로 회전율이 15% 개선되었습니다. "
        "[세트 메뉴] 2인 세트(면+사이드+음료)를 도입하여 선택 피로를 줄이고 "
        "객단가를 자연스럽게 상승시켰습니다. "
        "[맛 커스텀] 주문 시 '기본/덜짭게' 선택이 가능하며, 두꺼운 쫄깃한 면은 "
        "류진만의 시그니처로 호평받고 있습니다. "
        "청결한 매장과 친절한 직원 서비스는 꾸준히 좋은 평가를 받고 있으며, "
        "데이트나 모임 장소로도 인기가 높아지고 있습니다."
    ),
}


def backup_original_data():
    """원본 JSON 데이터 백업"""
    if not BACKUP_DIR.exists():
        BACKUP_DIR.mkdir(parents=True)
        for json_file in DATA_DIR.glob("*.json"):
            shutil.copy2(json_file, BACKUP_DIR / json_file.name)
        print(f"원본 데이터 백업 완료: {BACKUP_DIR}")
    else:
        print(f"백업 이미 존재: {BACKUP_DIR}")


def restore_original_data():
    """원본 JSON 데이터 복원"""
    if BACKUP_DIR.exists():
        for json_file in BACKUP_DIR.glob("*.json"):
            shutil.copy2(json_file, DATA_DIR / json_file.name)
        print("원본 데이터 복원 완료")
    else:
        print("백업 데이터가 없습니다")


def apply_improvements(store_id: str, improvements: dict):
    """매장 JSON에 개선사항 적용 (허용된 필드만)"""
    json_path = DATA_DIR / f"{store_id}.json"

    if not json_path.exists():
        print(f"매장 JSON 없음: {json_path}")
        return False

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    store_data = data[0] if isinstance(data, list) else data

    # 허용된 필드만 수정
    allowed_fields = ['revenue_analysis', 'top_keywords', 'rag_context', 'category']

    print(f"\n⭐ {store_data['store_name']} 매장 개선 적용:")
    for field in allowed_fields:
        if field in improvements:
            old_value = store_data.get(field, 'N/A')
            store_data[field] = improvements[field]
            print(f"  [{field}] 수정됨")

    # 개선 적용 메타데이터 추가
    store_data['improvement_applied'] = {
        'applied_at': datetime.now().isoformat(),
        'improvements': list(improvements.keys()),
        'description': 'TOP 1: 웨이팅 병목 제거 / TOP 2: 2인 세트 도입 / TOP 3: 리뷰 키워드 유도'
    }

    # 저장
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data if isinstance(data, list) else [store_data], f, ensure_ascii=False, indent=2)

    print(f"  개선사항 저장 완료: {json_path}")
    return True


def add_simulation_results(store_id: str, visit_count: int = 0,
                          sum_taste: float = 0.0, sum_price_value: float = 0.0):
    """simulation_results 필드 추가/업데이트"""
    json_path = DATA_DIR / f"{store_id}.json"

    if not json_path.exists():
        return False

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    store_data = data[0] if isinstance(data, list) else data

    # simulation_results 초기화 또는 업데이트
    if 'simulation_results' not in store_data:
        store_data['simulation_results'] = {
            'visit_count': 0,
            'sum_taste': 0.0,
            'sum_price_value': 0.0,
            'avg_taste': 0.0,
            'avg_price_value': 0.0
        }

    sim = store_data['simulation_results']
    sim['visit_count'] += visit_count
    sim['sum_taste'] += sum_taste
    sim['sum_price_value'] += sum_price_value

    if sim['visit_count'] > 0:
        sim['avg_taste'] = sim['sum_taste'] / sim['visit_count']
        sim['avg_price_value'] = sim['sum_price_value'] / sim['visit_count']

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data if isinstance(data, list) else [store_data], f, ensure_ascii=False, indent=2)

    return True


def reset_simulation_results(store_id: str):
    """simulation_results 초기화"""
    json_path = DATA_DIR / f"{store_id}.json"

    if not json_path.exists():
        return False

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    store_data = data[0] if isinstance(data, list) else data

    if 'simulation_results' in store_data:
        del store_data['simulation_results']

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data if isinstance(data, list) else [store_data], f, ensure_ascii=False, indent=2)

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="매장 개선사항 적용")
    parser.add_argument("--apply", action="store_true", help="류진 개선사항 적용")
    parser.add_argument("--restore", action="store_true", help="원본 데이터 복원")
    parser.add_argument("--backup", action="store_true", help="원본 데이터 백업")

    args = parser.parse_args()

    if args.backup:
        backup_original_data()
    elif args.restore:
        restore_original_data()
    elif args.apply:
        backup_original_data()  # 먼저 백업
        apply_improvements(RYUJIN_IMPROVEMENTS['store_id'], RYUJIN_IMPROVEMENTS)
    else:
        parser.print_help()
