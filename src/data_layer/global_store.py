"""
Global Store - 에이전트 평점 시스템.

평점 체계 (0~5):
- 0: 방문 없음/평가 없음 (기본값)
- 1: 매우별로
- 2: 별로
- 3: 보통
- 4: 좋음
- 5: 매우좋음

에이전트가 매장 인식 시 사용하는 필드:
- store_id, store_name, category
- market_analysis, revenue_analysis, customer_analysis
- review_metrics.overall_sentiment.comparison
- raw_data_context.trend_history
- metadata, top_keywords, critical_feedback, rag_context
- 에이전트 평점 (agent_ratings)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import re
from pathlib import Path
import threading


def parse_average_price(revenue_analysis: str) -> Optional[int]:
    """revenue_analysis 텍스트에서 객단가 파싱"""
    if not revenue_analysis:
        return None
    pattern = r'객단가는?\s*([\d,]+)원'
    match = re.search(pattern, revenue_analysis)
    if match:
        price_str = match.group(1).replace(',', '')
        try:
            return int(price_str)
        except ValueError:
            return None
    return None


@dataclass
class AgentRatingRecord:
    """에이전트가 남긴 개별 평점 기록 (1~5점)"""
    agent_id: int
    agent_name: str
    visit_datetime: str
    taste_rating: int      # 1~5: 매우별로~매우좋음
    value_rating: int      # 1~5: 매우별로~매우좋음
    atmosphere_rating: int # 1~5: 매우별로~매우좋음
    comment: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "visit_datetime": self.visit_datetime,
            "taste_rating": self.taste_rating,
            "value_rating": self.value_rating,
            "atmosphere_rating": self.atmosphere_rating,
            "comment": self.comment,
        }


@dataclass
class StoreRating:
    """
    개별 매장의 평점 정보.

    매장 인식 필드:
    - store_id, store_name, category
    - market_analysis, revenue_analysis, customer_analysis
    - top_keywords, critical_feedback, rag_context
    - 에이전트 평점 (agent_ratings)
    """
    store_id: str
    store_name: str
    category: str

    # === 에이전트 평점 (시뮬레이션 중 축적, 1~5점) ===
    agent_ratings: List[AgentRatingRecord] = field(default_factory=list)

    # 가격 정보
    average_price: int = 10000

    # 메타데이터
    address: Optional[str] = None
    coordinates: Optional[Tuple[float, float]] = None  # (lat, lng)

    # === 매장 정보 필드 (LLM 인식용) ===
    market_analysis: Optional[Dict[str, Any]] = None
    revenue_analysis: str = ""
    customer_analysis: str = ""
    top_keywords: List[str] = field(default_factory=list)
    critical_feedback: str = ""
    rag_context: str = ""
    metadata: Optional[Dict[str, Any]] = None

    # 리뷰 메트릭 (overall_sentiment.comparison 등)
    review_metrics: Optional[Dict[str, Any]] = None

    # 트렌드 히스토리
    trend_history: Optional[List[Dict[str, Any]]] = None

    # === 에이전트 평점 접근자 (0~5 스케일, 0은 미평가) ===
    @property
    def agent_taste_score(self) -> float:
        """에이전트들이 남긴 맛 평점 평균 (0이면 평가 없음)"""
        if not self.agent_ratings:
            return 0.0
        total = sum(r.taste_rating for r in self.agent_ratings)
        return total / len(self.agent_ratings)

    @property
    def agent_value_score(self) -> float:
        """에이전트들이 남긴 가성비 평점 평균 (0이면 평가 없음)"""
        if not self.agent_ratings:
            return 0.0
        total = sum(r.value_rating for r in self.agent_ratings)
        return total / len(self.agent_ratings)

    @property
    def agent_atmosphere_score(self) -> float:
        """에이전트들이 남긴 분위기 평점 평균 (0이면 평가 없음)"""
        if not self.agent_ratings:
            return 0.0
        total = sum(r.atmosphere_rating for r in self.agent_ratings)
        return total / len(self.agent_ratings)

    @property
    def agent_overall_score(self) -> float:
        """에이전트 평점 기반 종합 점수 (0이면 평가 없음)"""
        if not self.agent_ratings:
            return 0.0
        return (self.agent_taste_score + self.agent_value_score + self.agent_atmosphere_score) / 3

    @property
    def agent_rating_count(self) -> int:
        """에이전트 평점 수"""
        return len(self.agent_ratings)

    def add_agent_rating(
        self,
        agent_id: int,
        agent_name: str,
        taste_rating: int,
        value_rating: int,
        atmosphere_rating: int,
        visit_datetime: Optional[str] = None,
        comment: str = "",
    ):
        """
        에이전트 평점 추가 (1~5점)
        """
        record = AgentRatingRecord(
            agent_id=agent_id,
            agent_name=agent_name,
            visit_datetime=visit_datetime or datetime.now().isoformat(),
            taste_rating=taste_rating,
            value_rating=value_rating,
            atmosphere_rating=atmosphere_rating,
            comment=comment,
        )
        self.agent_ratings.append(record)

    def get_agent_score_text(self) -> str:
        """에이전트 평점 텍스트"""
        if not self.agent_ratings:
            return "아직 평가 없음"
        return f"맛:{self.agent_taste_score:.1f}/5, 가성비:{self.agent_value_score:.1f}/5, 분위기:{self.agent_atmosphere_score:.1f}/5 ({self.agent_rating_count}건)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "store_id": self.store_id,
            "store_name": self.store_name,
            "category": self.category,
            # 에이전트 평점
            "agent_taste_score": self.agent_taste_score,
            "agent_value_score": self.agent_value_score,
            "agent_atmosphere_score": self.agent_atmosphere_score,
            "agent_overall_score": self.agent_overall_score,
            "agent_rating_count": self.agent_rating_count,
            # 기타
            "average_price": self.average_price,
            "agent_ratings": [r.to_dict() for r in self.agent_ratings],
            "top_keywords": self.top_keywords,
            "rag_context": self.rag_context[:200] if self.rag_context else "",
        }


class GlobalStore:
    """
    전역 매장 정보 관리자.
    싱글톤 패턴으로 모든 에이전트가 공유하는 매장 정보를 관리합니다.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.stores: Dict[str, StoreRating] = {}
        self._store_name_to_id: Dict[str, str] = {}
        self._pending_ratings: List[Dict[str, Any]] = []

    @classmethod
    def reset_instance(cls):
        """싱글톤 인스턴스 리셋 (테스트용)"""
        with cls._lock:
            cls._instance = None

    def _load_stores_csv(self, json_dir: Path) -> Dict[str, Dict[str, Any]]:
        """stores.csv에서 좌표/카테고리/주소 정보를 store_id 기준으로 로드"""
        import csv
        stores_csv = json_dir.parent / "stores.csv"
        result: Dict[str, Dict[str, Any]] = {}
        if not stores_csv.exists():
            return result
        with open(stores_csv, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                sid = str(row.get("ID", ""))
                if sid:
                    result[sid] = {
                        "store_name": row.get("장소명", ""),
                        "x": row.get("x"),
                        "y": row.get("y"),
                        "category": row.get("카테고리", ""),
                        "address": row.get("주소", ""),
                    }
        return result

    def load_from_json_files(self, json_dir: Path):
        """
        JSON 파일들에서 매장 데이터 로드.
        좌표/카테고리/주소는 stores.csv에서 보조 로딩.

        로드하는 필드:
        - store_id, store_name, category, x, y, address (stores.csv 우선)
        - market_analysis, revenue_analysis, customer_analysis
        - review_metrics, top_keywords, critical_feedback, rag_context
        - raw_data_context.trend_history, metadata
        """
        if not json_dir.exists():
            return

        # stores.csv에서 좌표/카테고리/주소 로드
        csv_data = self._load_stores_csv(json_dir)

        for json_file in json_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # JSON 파일 형식에 따라 파싱
                if isinstance(data, list):
                    data = data[0] if data else {}

                store_id = str(data.get("store_id", json_file.stem))
                store_name = data.get("store_name", "Unknown")

                # stores.csv에서 좌표/카테고리/주소 보충
                csv_info = csv_data.get(store_id, {})
                category = data.get("category") or csv_info.get("category", "")
                address = data.get("address") or csv_info.get("address", "")

                # 좌표 추출: JSON 우선, 없으면 stores.csv
                x = data.get("x") or csv_info.get("x")
                y = data.get("y") or csv_info.get("y")
                coordinates = None
                if x and y:
                    try:
                        coordinates = (float(y), float(x))
                    except (ValueError, TypeError):
                        pass

                # 매장 찾기 또는 새로 생성
                store = None
                if store_id in self.stores:
                    store = self.stores[store_id]
                elif store_name in self._store_name_to_id:
                    store = self.stores[self._store_name_to_id[store_name]]
                else:
                    store = StoreRating(
                        store_id=store_id,
                        store_name=store_name,
                        category=category,
                        coordinates=coordinates,
                        address=address,
                    )
                    self.stores[store_id] = store
                    self._store_name_to_id[store_name] = store_id

                # 기존 매장이면 누락된 정보 보충
                if store.coordinates is None and coordinates:
                    store.coordinates = coordinates
                if not store.category and category:
                    store.category = category
                if not store.address and address:
                    store.address = address

                # === 매장 정보 필드 로드 ===
                store.market_analysis = data.get("market_analysis")
                store.revenue_analysis = data.get("revenue_analysis", "")
                store.customer_analysis = data.get("customer_analysis", "")
                store.review_metrics = data.get("review_metrics")
                store.metadata = data.get("metadata")

                if "top_keywords" in data:
                    store.top_keywords = data["top_keywords"]
                if "critical_feedback" in data:
                    store.critical_feedback = data["critical_feedback"]
                if "rag_context" in data:
                    store.rag_context = data["rag_context"]

                # raw_data_context에서 trend_history 추출
                raw_data = data.get("raw_data_context", {})
                if "trend_history" in raw_data:
                    store.trend_history = raw_data["trend_history"]

                # 객단가 파싱
                if store.revenue_analysis:
                    parsed_price = parse_average_price(store.revenue_analysis)
                    if parsed_price:
                        store.average_price = parsed_price

            except Exception as e:
                print(f"JSON 로드 오류 ({json_file}): {e}")

    def get_store(self, store_id: str) -> Optional[StoreRating]:
        """매장 ID로 조회"""
        return self.stores.get(store_id)

    def get_by_name(self, store_name: str) -> Optional[StoreRating]:
        """매장명으로 조회"""
        store_id = self._store_name_to_id.get(store_name)
        if store_id:
            return self.stores.get(store_id)
        return None

    def get_stores_in_radius(
        self,
        center_lat: float,
        center_lng: float,
        radius_km: float = 3.0
    ) -> List[StoreRating]:
        """
        특정 좌표 기준 반경 내 매장 조회.

        Args:
            center_lat: 중심 위도
            center_lng: 중심 경도
            radius_km: 반경 (km), 기본 3km

        Returns:
            반경 내 매장 목록
        """
        from math import radians, cos, sin, asin, sqrt

        def haversine(lat1, lon1, lat2, lon2):
            """두 좌표 간 거리 계산 (km)"""
            R = 6371  # 지구 반경 (km)
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            return R * c

        nearby = []
        for store in self.stores.values():
            if store.coordinates:
                lat, lng = store.coordinates
                distance = haversine(center_lat, center_lng, lat, lng)
                if distance <= radius_km:
                    nearby.append(store)

        return nearby

    def add_agent_rating(
        self,
        store_name: str,
        agent_id: int,
        agent_name: str,
        taste_rating: int,
        value_rating: int,
        atmosphere_rating: int,
        visit_datetime: Optional[str] = None,
        comment: str = "",
    ) -> bool:
        """
        매장에 에이전트 평점 추가 (1~5점).
        Returns: 성공 여부
        """
        store = self.get_by_name(store_name)
        if store:
            store.add_agent_rating(
                agent_id=agent_id,
                agent_name=agent_name,
                taste_rating=taste_rating,
                value_rating=value_rating,
                atmosphere_rating=atmosphere_rating,
                visit_datetime=visit_datetime,
                comment=comment,
            )
            return True
        return False

    def add_pending_rating(
        self,
        store_name: str,
        agent_id: int,
        agent_name: str,
        taste_rating: int,
        value_rating: int,
        atmosphere_rating: int,
        visit_datetime: Optional[str] = None,
        comment: str = "",
    ) -> bool:
        """
        평점을 버퍼에 저장 (같은 타임슬롯 내 다른 에이전트에게 영향 안 줌).
        flush_pending_ratings() 호출 시 실제 반영.
        """
        store = self.get_by_name(store_name)
        if store:
            self._pending_ratings.append({
                "store_name": store_name,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "taste_rating": taste_rating,
                "value_rating": value_rating,
                "atmosphere_rating": atmosphere_rating,
                "visit_datetime": visit_datetime,
                "comment": comment,
            })
            return True
        return False

    def flush_pending_ratings(self) -> int:
        """
        버퍼에 쌓인 평점을 실제 매장에 일괄 반영.
        Returns: 반영된 평점 수
        """
        count = 0
        for rating in self._pending_ratings:
            success = self.add_agent_rating(
                store_name=rating["store_name"],
                agent_id=rating["agent_id"],
                agent_name=rating["agent_name"],
                taste_rating=rating["taste_rating"],
                value_rating=rating["value_rating"],
                atmosphere_rating=rating["atmosphere_rating"],
                visit_datetime=rating["visit_datetime"],
                comment=rating.get("comment", ""),
            )
            if success:
                count += 1
        self._pending_ratings.clear()
        return count

    def get_stores_by_category(self, category_keyword: str) -> List[StoreRating]:
        """카테고리로 매장 검색"""
        result = []
        for store in self.stores.values():
            if category_keyword.lower() in store.category.lower():
                result.append(store)
        return result

    def get_stores_in_budget(self, max_budget: int, category: Optional[str] = None) -> List[StoreRating]:
        """예산 내 매장 조회"""
        stores = list(self.stores.values())
        if category:
            stores = [s for s in stores if category.lower() in s.category.lower()]
        return [s for s in stores if s.average_price <= max_budget]

    def get_top_stores_by_agent_rating(self, n: int = 10, category: Optional[str] = None) -> List[StoreRating]:
        """에이전트 평점 기준 상위 매장 (평점 있는 매장만)"""
        stores = [s for s in self.stores.values() if s.agent_rating_count > 0]
        if category:
            stores = [s for s in stores if category.lower() in s.category.lower()]
        stores.sort(key=lambda s: s.agent_overall_score, reverse=True)
        return stores[:n]

    def get_statistics(self) -> Dict[str, Any]:
        """전체 통계"""
        total_agent_ratings = sum(s.agent_rating_count for s in self.stores.values())
        rated_stores = sum(1 for s in self.stores.values() if s.agent_rating_count > 0)

        # 에이전트 평점 평균 (평점 있는 매장만)
        rated_stores_list = [s for s in self.stores.values() if s.agent_rating_count > 0]
        if rated_stores_list:
            avg_agent_taste = sum(s.agent_taste_score for s in rated_stores_list) / len(rated_stores_list)
            avg_agent_value = sum(s.agent_value_score for s in rated_stores_list) / len(rated_stores_list)
            avg_agent_atmosphere = sum(s.agent_atmosphere_score for s in rated_stores_list) / len(rated_stores_list)
        else:
            avg_agent_taste = 0.0
            avg_agent_value = 0.0
            avg_agent_atmosphere = 0.0

        return {
            "total_stores": len(self.stores),
            "stores_with_agent_ratings": rated_stores,
            "total_agent_ratings": total_agent_ratings,
            "avg_agent_taste_score": avg_agent_taste,
            "avg_agent_value_score": avg_agent_value,
            "avg_agent_atmosphere_score": avg_agent_atmosphere,
        }

    def reset_agent_ratings(self):
        """에이전트 평점 초기화"""
        for store in self.stores.values():
            store.agent_ratings = []

    def save_to_json(self, output_path: Path):
        """현재 상태를 JSON으로 저장"""
        data = {
            "statistics": self.get_statistics(),
            "stores": [s.to_dict() for s in self.stores.values()],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def get_global_store() -> GlobalStore:
    """싱글톤 GlobalStore 인스턴스 반환"""
    return GlobalStore()


if __name__ == "__main__":
    from config import get_settings

    settings = get_settings()

    # 싱글톤 리셋
    GlobalStore.reset_instance()
    store = get_global_store()

    # JSON 매장 데이터 로드
    json_dir = settings.paths.data_dir / "split_by_store_id_ver3"
    store.load_from_json_files(json_dir)

    print(f"로드된 매장 수: {len(store.stores)}")

    # 통계 출력
    stats = store.get_statistics()
    print(f"\n초기 통계: {stats}")

    # 에이전트 평점 추가 테스트
    test_store = list(store.stores.values())[0] if store.stores else None
    if test_store:
        print(f"\n=== 테스트 매장: {test_store.store_name} ===")
        print(f"에이전트 평점: {test_store.get_agent_score_text()}")

        # 에이전트 평점 추가 (1~5점)
        store.add_agent_rating(test_store.store_name, 1, "김민준", 4, 3, 4)  # 맛 좋음, 가성비 보통, 분위기 좋음
        store.add_agent_rating(test_store.store_name, 2, "이서연", 5, 4, 5)  # 맛 매우좋음, 가성비 좋음, 분위기 매우좋음
        store.add_agent_rating(test_store.store_name, 3, "박지우", 3, 5, 3)  # 맛 보통, 가성비 매우좋음, 분위기 보통

        print(f"\n=== 평점 추가 후 ===")
        print(f"에이전트 평점: {test_store.get_agent_score_text()}")
