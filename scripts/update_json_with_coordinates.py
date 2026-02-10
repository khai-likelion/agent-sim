"""
stores.csv의 좌표를 JSON 파일들에 추가하는 스크립트
"""
import json
import pandas as pd
from pathlib import Path
import sys

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import get_settings

settings = get_settings()


def load_stores_csv():
    """stores.csv를 읽어서 ID를 키로 하는 딕셔너리 생성"""
    stores_df = pd.read_csv(settings.paths.stores_csv, encoding='utf-8-sig')

    # ID를 문자열로 변환하여 키로 사용
    stores_dict = {}
    for _, row in stores_df.iterrows():
        store_id = str(row['ID'])
        stores_dict[store_id] = {
            'x': row['x'],
            'y': row['y'],
            'address': row['주소'],
            'category': row['카테고리'],
            'phone': row['전화번호'] if pd.notna(row['전화번호']) else None,
            'detail_url': row['상세URL'] if pd.notna(row['상세URL']) else None,
        }

    print(f"✅ stores.csv에서 {len(stores_dict)}개 매장 로드")
    return stores_dict


def update_json_files(stores_dict):
    """모든 JSON 파일에 좌표 추가"""
    json_dir = settings.paths.data_dir / "split_by_store_id"
    json_files = list(json_dir.glob("*.json"))

    print(f"📁 {len(json_files)}개 JSON 파일 발견")

    matched = 0
    not_matched = 0
    updated_files = []
    total = len(json_files)

    for idx, json_file in enumerate(json_files, 1):
        if idx % 100 == 0:
            print(f"진행 중... {idx}/{total} ({idx/total*100:.1f}%)")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 각 JSON 파일은 배열로 1개 객체를 포함
        if len(data) == 0:
            continue

        store_data = data[0]
        store_id = store_data.get('store_id')

        if not store_id:
            not_matched += 1
            continue

        # stores.csv에서 매칭
        if store_id in stores_dict:
            store_info = stores_dict[store_id]

            # 좌표 및 추가 정보 추가
            store_data['x'] = store_info['x']
            store_data['y'] = store_info['y']
            store_data['address'] = store_info['address']
            store_data['category'] = store_info['category']

            if store_info['phone']:
                store_data['phone'] = store_info['phone']
            if store_info['detail_url']:
                store_data['detail_url'] = store_info['detail_url']

            # 업데이트된 데이터를 다시 저장
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            matched += 1
            updated_files.append(json_file.name)
        else:
            not_matched += 1

    print(f"\n✅ 매칭 성공: {matched}개")
    print(f"❌ 매칭 실패: {not_matched}개")

    # 매칭 실패한 파일 샘플 출력
    if not_matched > 0:
        not_matched_files = [f.stem for f in json_files if f.name not in updated_files]
        print(f"\n매칭 실패 파일 샘플 (처음 5개):")
        for file_id in not_matched_files[:5]:
            print(f"  - {file_id}")

    return matched, not_matched


def main():
    print("=" * 60)
    print("JSON 파일에 좌표 추가 스크립트")
    print("=" * 60)

    # 1. stores.csv 로드
    stores_dict = load_stores_csv()

    # 2. JSON 파일 업데이트
    matched, not_matched = update_json_files(stores_dict)

    # 3. 결과 확인
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)

    # 샘플 파일 확인
    json_dir = settings.paths.data_dir / "split_by_store_id"
    sample_file = list(json_dir.glob("*.json"))[0]

    print(f"\n샘플 파일 확인: {sample_file.name}")
    with open(sample_file, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)

    if 'x' in sample_data[0] and 'y' in sample_data[0]:
        print(f"✅ 좌표 추가 확인:")
        print(f"   - store_name: {sample_data[0].get('store_name')}")
        print(f"   - x: {sample_data[0].get('x')}")
        print(f"   - y: {sample_data[0].get('y')}")
        print(f"   - address: {sample_data[0].get('address')}")
    else:
        print("⚠️  좌표가 추가되지 않았습니다.")


if __name__ == "__main__":
    main()
