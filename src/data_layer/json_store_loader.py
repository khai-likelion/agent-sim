"""
JSON 매장 데이터 로더
- 721개 JSON 파일에서 매장 정보를 읽어 DataFrame으로 변환
- 리뷰 점수, 객단가 등 분석 데이터 포함
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import get_settings


settings = get_settings()


def parse_average_price(revenue_analysis: str) -> Optional[int]:
    """
    revenue_analysis 텍스트에서 객단가 파싱

    예: "객단가는 36,100원으로 나타났으며..." -> 36100
    """
    if not revenue_analysis:
        return None

    # 정규식으로 "객단가는 XX,XXX원" 패턴 찾기
    pattern = r'객단가는?\s*([\d,]+)원'
    match = re.search(pattern, revenue_analysis)

    if match:
        price_str = match.group(1).replace(',', '')
        try:
            return int(price_str)
        except ValueError:
            return None

    return None


def load_json_stores(json_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    모든 JSON 파일을 읽어서 DataFrame으로 변환

    Returns:
        DataFrame with columns:
        - store_id, store_name, x, y, address, category
        - market_phase, competition_intensity
        - avg_price (객단가)
        - taste_score, price_value_score, cleanliness_score, service_score
        - overall_sentiment_score
        - top_keywords (리스트)
        - rag_context
    """
    if json_dir is None:
        json_dir = settings.paths.data_dir / "split_by_store_id_ver3"

    json_files = list(json_dir.glob("*.json"))
    print(f"📂 {len(json_files)}개 JSON 파일 로드 중...")

    stores_data = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if len(data) == 0:
                continue

            store = data[0]

            # 필수 필드 확인
            if 'store_id' not in store or 'x' not in store or 'y' not in store:
                continue

            # 기본 정보
            store_info = {
                'store_id': store['store_id'],
                'store_name': store.get('store_name', ''),
                'x': store['x'],
                'y': store['y'],
                'address': store.get('address', ''),
                'category': store.get('category', ''),
                'phone': store.get('phone'),
                'detail_url': store.get('detail_url'),
            }

            # 메타데이터
            metadata = store.get('metadata', {})
            store_info['area'] = metadata.get('area')
            store_info['section'] = metadata.get('section')
            store_info['sector'] = metadata.get('sector')

            # 시장 분석
            market_analysis = store.get('market_analysis', {})
            store_info['market_phase'] = market_analysis.get('phase')  # 성장/보합/쇠퇴
            store_info['competition_intensity'] = market_analysis.get('competition_intensity')  # 높음/중간/낮음

            # 객단가 파싱
            revenue_analysis = store.get('revenue_analysis', '')
            store_info['avg_price'] = parse_average_price(revenue_analysis)

            # 리뷰 점수
            review_metrics = store.get('review_metrics', {})
            overall_sentiment = review_metrics.get('overall_sentiment', {})
            store_info['overall_sentiment_score'] = overall_sentiment.get('score')  # -1 ~ 1
            store_info['overall_sentiment_label'] = overall_sentiment.get('label')

            feature_scores = review_metrics.get('feature_scores', {})
            store_info['taste_score'] = feature_scores.get('taste', {}).get('score')
            store_info['price_value_score'] = feature_scores.get('price_value', {}).get('score')
            store_info['cleanliness_score'] = feature_scores.get('cleanliness', {}).get('score')
            store_info['service_score'] = feature_scores.get('service', {}).get('score')

            # 키워드 & 컨텍스트
            store_info['top_keywords'] = ','.join(store.get('top_keywords', []))  # 리스트를 문자열로
            store_info['rag_context'] = store.get('rag_context', '')

            # 원본 데이터 메트릭
            raw_data = store.get('raw_data_context', {})
            metrics = raw_data.get('metrics', {})
            store_info['store_count_in_area'] = metrics.get('store_count')  # 해당 상권의 총 점포 수

            stores_data.append(store_info)

        except Exception as e:
            print(f"⚠️  {json_file.name} 읽기 실패: {e}")
            continue

    df = pd.DataFrame(stores_data)
    print(f"✅ {len(df)}개 매장 로드 완료")

    # 데이터 타입 변환
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # 좌표가 없는 행 제거
    df = df.dropna(subset=['x', 'y'])

    print(f"📍 좌표 유효한 매장: {len(df)}개")

    # 통계 출력
    print("\n📊 데이터 통계:")
    print(f"  - 객단가 평균: {df['avg_price'].mean():.0f}원")
    print(f"  - 맛 점수 평균: {df['taste_score'].mean():.2f}")
    print(f"  - 위생 점수 평균: {df['cleanliness_score'].mean():.2f}")
    print(f"  - 서비스 점수 평균: {df['service_score'].mean():.2f}")
    print(f"  - 시장 단계: {df['market_phase'].value_counts().to_dict()}")

    return df


def load_and_index_json_stores(
    json_dir: Optional[Path] = None,
    h3_resolution: int = 10
) -> pd.DataFrame:
    """
    JSON 매장 데이터를 로드하고 H3 인덱스 추가
    (기존 load_and_index_stores와 호환)
    """
    df = load_json_stores(json_dir)

    # H3 인덱스 추가
    try:
        import h3
        df['h3_index'] = df.apply(
            lambda row: h3.latlng_to_cell(row['y'], row['x'], h3_resolution),
            axis=1
        )
        print(f"✅ H3 인덱스 추가 완료 (resolution: {h3_resolution})")
    except ImportError:
        print("⚠️  h3 라이브러리가 없어 H3 인덱스를 추가하지 않았습니다.")

    return df


if __name__ == "__main__":
    # 테스트
    print("=" * 60)
    print("JSON Store Loader 테스트")
    print("=" * 60)

    df = load_json_stores()

    print("\n샘플 데이터:")
    print(df[['store_name', 'x', 'y', 'avg_price', 'taste_score', 'market_phase']].head(10))

    print("\n컬럼 목록:")
    for col in df.columns:
        print(f"  - {col}")
