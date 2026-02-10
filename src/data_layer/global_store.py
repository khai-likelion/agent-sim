"""
Global Store - 실시간 평점 업데이트 시스템.

평점 체계:
1. 기존 평점 (base_score): 리뷰 데이터 기반 원본 평점 (불변)
2. 에이전트 평점 (agent_score): 시뮬레이션 중 에이전트가 남긴 평점들의 평균

에이전트는 두 종류의 평점을 모두 보고 판단합니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
from pathlib import Path
import threading


@dataclass
class AgentRatingRecord:
    """에이전트가 남긴 개별 평점 기록"""
    agent_id: int
    agent_name: str
    visit_datetime: str
    taste_rating: int  # 0: 별로, 1: 보통, 2: 좋음
    value_rating: int  # 0: 별로, 1: 보통, 2: 좋음

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "visit_datetime": self.visit_datetime,
            "taste_rating": self.taste_rating,
            "value_rating": self.value_rating,
        }


@dataclass
class StoreRating:
    """개별 매장의 평점 정보"""
    store_id: str
    store_name: str
    category: str

    # === 기존 평점 (리뷰 데이터 기반, 불변) ===
    base_taste_score: float = 0.5  # 0~1 스케일
    base_value_score: float = 0.5  # 0~1 스케일
    base_cleanliness_score: float = 0.5
    base_service_score: float = 0.5

    # === 에이전트 평점 (시뮬레이션 중 축적) ===
    agent_ratings: List[AgentRatingRecord] = field(default_factory=list)

    # 가격 정보
    average_price: int = 10000

    # 메타데이터
    address: Optional[str] = None
    coordinates: Optional[Tuple[float, float]] = None  # (lat, lng)

    # === 개선 가능 필드 (AI가 수정 가능) ===
    revenue_analysis: str = ""
    top_keywords: List[str] = field(default_factory=list)
    rag_context: str = ""
    improvement_applied: Optional[Dict[str, Any]] = None

    # === 시뮬레이션 결과 (simulation_results) ===
    simulation_results: Optional[Dict[str, float]] = None

    # === 기존 평점 접근자 (불변) ===
    @property
    def base_overall_score(self) -> float:
        """기존 리뷰 기반 종합 점수"""
        return (self.base_taste_score + self.base_value_score) / 2

    # === 에이전트 평점 접근자 ===
    @property
    def agent_taste_score(self) -> Optional[float]:
        """에이전트들이 남긴 맛 평점 평균 (0~1 스케일, 없으면 None)"""
        if not self.agent_ratings:
            return None
        total = sum(r.taste_rating for r in self.agent_ratings)
        return total / len(self.agent_ratings) / 2.0  # 0-2 -> 0-1 변환

    @property
    def agent_value_score(self) -> Optional[float]:
        """에이전트들이 남긴 가성비 평점 평균 (0~1 스케일, 없으면 None)"""
        if not self.agent_ratings:
            return None
        total = sum(r.value_rating for r in self.agent_ratings)
        return total / len(self.agent_ratings) / 2.0

    @property
    def agent_overall_score(self) -> Optional[float]:
        """에이전트 평점 기반 종합 점수"""
        taste = self.agent_taste_score
        value = self.agent_value_score
        if taste is None or value is None:
            return None
        return (taste + value) / 2

    @property
    def agent_rating_count(self) -> int:
        """에이전트 평점 수"""
        return len(self.agent_ratings)

    # === 시뮬레이션 결과 기반 점수 ===
    @property
    def sim_taste_score(self) -> Optional[float]:
        """simulation_results 기반 맛 점수"""
        if self.simulation_results and self.simulation_results.get('visit_count', 0) > 0:
            return self.simulation_results.get('avg_taste')
        return None

    @property
    def sim_value_score(self) -> Optional[float]:
        """simulation_results 기반 가성비 점수"""
        if self.simulation_results and self.simulation_results.get('visit_count', 0) > 0:
            return self.simulation_results.get('avg_price_value')
        return None

    def get_effective_taste_score(self) -> float:
        """
        의사결정용 맛 점수 (simulation_results가 있으면 가중치 부여)
        - 시뮬레이션 결과가 축적될수록 해당 값에 가중치를 더 줌
        """
        sim_score = self.sim_taste_score
        if sim_score is not None:
            visit_count = self.simulation_results.get('visit_count', 0)
            # 방문 횟수에 따라 가중치 증가 (최대 70%)
            sim_weight = min(0.7, visit_count * 0.1)
            return self.base_taste_score * (1 - sim_weight) + sim_score * sim_weight
        return self.base_taste_score

    def get_effective_value_score(self) -> float:
        """
        의사결정용 가성비 점수 (simulation_results가 있으면 가중치 부여)
        """
        sim_score = self.sim_value_score
        if sim_score is not None:
            visit_count = self.simulation_results.get('visit_count', 0)
            sim_weight = min(0.7, visit_count * 0.1)
            return self.base_value_score * (1 - sim_weight) + sim_score * sim_weight
        return self.base_value_score

    def is_improved(self) -> bool:
        """개선사항이 적용되었는지 확인"""
        return self.improvement_applied is not None

    def add_agent_rating(
        self,
        agent_id: int,
        agent_name: str,
        taste_rating: int,
        value_rating: int,
        visit_datetime: Optional[str] = None
    ):
        """
        에이전트 평점 추가 (0: 별로, 1: 보통, 2: 좋음)
        """
        record = AgentRatingRecord(
            agent_id=agent_id,
            agent_name=agent_name,
            visit_datetime=visit_datetime or datetime.now().isoformat(),
            taste_rating=taste_rating,
            value_rating=value_rating,
        )
        self.agent_ratings.append(record)

    def get_base_score_text(self) -> str:
        """기존 평점 텍스트"""
        return f"맛:{self.base_taste_score:.2f}, 가성비:{self.base_value_score:.2f}"

    def get_agent_score_text(self) -> str:
        """에이전트 평점 텍스트"""
        if not self.agent_ratings:
            return "아직 평가 없음"
        taste = self.agent_taste_score
        value = self.agent_value_score
        return f"맛:{taste:.2f}, 가성비:{value:.2f} ({self.agent_rating_count}건)"

    def get_combined_info_for_prompt(self) -> str:
        """LLM 프롬프트용 통합 정보 (두 종류 평점 모두 표시)"""
        lines = [f"[{self.store_name}]"]
        lines.append(f"  카테고리: {self.category}")
        lines.append(f"  평균가격: {self.average_price:,}원")
        lines.append(f"  기존리뷰 평점: {self.get_base_score_text()}")

        # 개선된 키워드가 있으면 표시
        if self.top_keywords:
            keywords = self.top_keywords[:5] if isinstance(self.top_keywords, list) else self.top_keywords.split(',')[:5]
            lines.append(f"  주요키워드: {', '.join(keywords)}")

        # 개선 설명이 있으면 요약 표시
        if self.rag_context and len(self.rag_context) > 50:
            summary = self.rag_context[:100] + "..."
            lines.append(f"  설명: {summary}")

        # 시뮬레이션 결과가 있으면 표시
        if self.simulation_results and self.simulation_results.get('visit_count', 0) > 0:
            sim = self.simulation_results
            lines.append(f"  시뮬레이션 평가: 맛 {sim.get('avg_taste', 0):.2f}, 가성비 {sim.get('avg_price_value', 0):.2f} ({sim.get('visit_count', 0)}건)")

        if self.agent_ratings:
            lines.append(f"  최근방문자 평점: {self.get_agent_score_text()}")

            # 최근 3개 평가 코멘트
            recent = self.agent_ratings[-3:]
            rating_text = {0: "별로", 1: "보통", 2: "좋음"}
            for r in recent:
                lines.append(f"    - {r.agent_name}: 맛 {rating_text[r.taste_rating]}, 가성비 {rating_text[r.value_rating]}")
        else:
            lines.append("  최근방문자 평점: 아직 없음")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "store_id": self.store_id,
            "store_name": self.store_name,
            "category": self.category,
            # 기존 평점
            "base_taste_score": self.base_taste_score,
            "base_value_score": self.base_value_score,
            "base_overall_score": self.base_overall_score,
            # 에이전트 평점
            "agent_taste_score": self.agent_taste_score,
            "agent_value_score": self.agent_value_score,
            "agent_overall_score": self.agent_overall_score,
            "agent_rating_count": self.agent_rating_count,
            # 효과적 점수 (simulation_results 반영)
            "effective_taste_score": self.get_effective_taste_score(),
            "effective_value_score": self.get_effective_value_score(),
            # 기타
            "average_price": self.average_price,
            "agent_ratings": [r.to_dict() for r in self.agent_ratings],
            # 개선 필드
            "top_keywords": self.top_keywords,
            "is_improved": self.is_improved(),
        }
        if self.simulation_results:
            result["simulation_results"] = self.simulation_results
        return result


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

    @classmethod
    def reset_instance(cls):
        """싱글톤 인스턴스 리셋 (테스트용)"""
        with cls._lock:
            cls._instance = None

    def load_from_csv(self, csv_path: Path):
        """CSV 파일에서 매장 데이터 로드"""
        import pandas as pd

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            store_id = str(row.get("store_id", row.get("id", "")))
            if not store_id:
                store_id = str(row.name)  # 인덱스 사용

            store_name = row.get("장소명", row.get("store_name", "Unknown"))
            category = row.get("카테고리", row.get("category", "기타"))

            # 좌표
            lat = row.get("y", row.get("lat", 0))
            lng = row.get("x", row.get("lng", 0))

            store = StoreRating(
                store_id=store_id,
                store_name=store_name,
                category=category,
                coordinates=(lat, lng) if lat and lng else None,
                address=row.get("주소", row.get("address", "")),
            )

            self.stores[store_id] = store
            self._store_name_to_id[store_name] = store_id

    def load_from_json_files(self, json_dir: Path):
        """JSON 파일들에서 리뷰 데이터 로드하여 기본 평점 설정"""
        if not json_dir.exists():
            return

        for json_file in json_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # JSON 파일 형식에 따라 파싱
                if isinstance(data, list):
                    data = data[0] if data else {}

                store_id = str(data.get("store_id", json_file.stem))
                store_name = data.get("store_name", "Unknown")

                # 매장 찾기
                store = None
                if store_id in self.stores:
                    store = self.stores[store_id]
                elif store_name in self._store_name_to_id:
                    store = self.stores[self._store_name_to_id[store_name]]

                if not store:
                    continue

                # === 리뷰 점수 로드 (review_metrics 구조 지원) ===
                review_metrics = data.get("review_metrics", {})
                feature_scores = review_metrics.get("feature_scores", {})

                # 기존 구조도 호환 (review_analysis)
                if not feature_scores:
                    analysis = data.get("review_analysis", {})
                    feature_scores = analysis.get("feature_scores", {})

                if "taste" in feature_scores:
                    store.base_taste_score = feature_scores["taste"].get("score", 0.5)
                if "price_value" in feature_scores:
                    store.base_value_score = feature_scores["price_value"].get("score", 0.5)
                if "cleanliness" in feature_scores:
                    store.base_cleanliness_score = feature_scores["cleanliness"].get("score", 0.5)
                if "service" in feature_scores:
                    store.base_service_score = feature_scores["service"].get("score", 0.5)

                # === 개선 가능 필드 로드 ===
                if "revenue_analysis" in data:
                    store.revenue_analysis = data["revenue_analysis"]
                if "top_keywords" in data:
                    store.top_keywords = data["top_keywords"]
                if "rag_context" in data:
                    store.rag_context = data["rag_context"]
                if "improvement_applied" in data:
                    store.improvement_applied = data["improvement_applied"]

                # === simulation_results 로드 ===
                if "simulation_results" in data:
                    store.simulation_results = data["simulation_results"]

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

    def add_agent_rating(
        self,
        store_name: str,
        agent_id: int,
        agent_name: str,
        taste_rating: int,
        value_rating: int,
        visit_datetime: Optional[str] = None
    ) -> bool:
        """
        매장에 에이전트 평점 추가.
        Returns: 성공 여부
        """
        store = self.get_by_name(store_name)
        if store:
            store.add_agent_rating(
                agent_id=agent_id,
                agent_name=agent_name,
                taste_rating=taste_rating,
                value_rating=value_rating,
                visit_datetime=visit_datetime,
            )
            return True
        return False

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

    def get_top_stores_by_base_rating(self, n: int = 10, category: Optional[str] = None) -> List[StoreRating]:
        """기존 평점 기준 상위 매장"""
        stores = list(self.stores.values())
        if category:
            stores = [s for s in stores if category.lower() in s.category.lower()]
        stores.sort(key=lambda s: s.base_overall_score, reverse=True)
        return stores[:n]

    def get_top_stores_by_agent_rating(self, n: int = 10, category: Optional[str] = None) -> List[StoreRating]:
        """에이전트 평점 기준 상위 매장 (평점 있는 매장만)"""
        stores = [s for s in self.stores.values() if s.agent_rating_count > 0]
        if category:
            stores = [s for s in stores if category.lower() in s.category.lower()]
        stores.sort(key=lambda s: s.agent_overall_score or 0, reverse=True)
        return stores[:n]

    def get_store_info_for_prompt(self, store_name: str) -> str:
        """LLM 프롬프트용 매장 정보"""
        store = self.get_by_name(store_name)
        if not store:
            return f"[{store_name}] 정보 없음"
        return store.get_combined_info_for_prompt()

    def get_stores_list_for_prompt(self, store_names: List[str]) -> str:
        """여러 매장 정보를 프롬프트용으로 포맷"""
        lines = []
        for name in store_names:
            store = self.get_by_name(name)
            if store:
                lines.append(store.get_combined_info_for_prompt())
            else:
                lines.append(f"[{name}] 정보 없음")
        return "\n\n".join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        """전체 통계"""
        total_agent_ratings = sum(s.agent_rating_count for s in self.stores.values())
        rated_stores = sum(1 for s in self.stores.values() if s.agent_rating_count > 0)

        avg_base_taste = sum(s.base_taste_score for s in self.stores.values()) / max(1, len(self.stores))
        avg_base_value = sum(s.base_value_score for s in self.stores.values()) / max(1, len(self.stores))

        # 에이전트 평점 평균 (평점 있는 매장만)
        rated_stores_list = [s for s in self.stores.values() if s.agent_rating_count > 0]
        if rated_stores_list:
            avg_agent_taste = sum(s.agent_taste_score or 0 for s in rated_stores_list) / len(rated_stores_list)
            avg_agent_value = sum(s.agent_value_score or 0 for s in rated_stores_list) / len(rated_stores_list)
        else:
            avg_agent_taste = None
            avg_agent_value = None

        return {
            "total_stores": len(self.stores),
            "stores_with_agent_ratings": rated_stores,
            "total_agent_ratings": total_agent_ratings,
            "avg_base_taste_score": avg_base_taste,
            "avg_base_value_score": avg_base_value,
            "avg_agent_taste_score": avg_agent_taste,
            "avg_agent_value_score": avg_agent_value,
        }

    def reset_agent_ratings(self):
        """에이전트 평점만 초기화 (기존 평점은 유지)"""
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
    # 테스트
    from config import get_settings

    settings = get_settings()

    # 싱글톤 리셋
    GlobalStore.reset_instance()
    store = get_global_store()

    # CSV 로드
    store.load_from_csv(settings.paths.stores_csv)
    print(f"로드된 매장 수: {len(store.stores)}")

    # JSON 리뷰 데이터 로드
    json_dir = settings.paths.data_dir / "split_by_store_id"
    store.load_from_json_files(json_dir)

    # 통계 출력
    stats = store.get_statistics()
    print(f"\n초기 통계: {stats}")

    # 에이전트 평점 추가 테스트
    test_store = list(store.stores.values())[0] if store.stores else None
    if test_store:
        print(f"\n=== 테스트 매장: {test_store.store_name} ===")
        print(f"기존 평점: {test_store.get_base_score_text()}")
        print(f"에이전트 평점: {test_store.get_agent_score_text()}")

        # 에이전트 평점 추가
        store.add_agent_rating(test_store.store_name, 1, "김민준", 2, 1)  # 맛 좋음, 가성비 보통
        store.add_agent_rating(test_store.store_name, 2, "이서연", 2, 2)  # 맛 좋음, 가성비 좋음
        store.add_agent_rating(test_store.store_name, 3, "박지우", 1, 2)  # 맛 보통, 가성비 좋음

        print(f"\n=== 평점 추가 후 ===")
        print(f"기존 평점: {test_store.get_base_score_text()} (불변)")
        print(f"에이전트 평점: {test_store.get_agent_score_text()}")

        print(f"\n=== 프롬프트용 통합 정보 ===")
        print(test_store.get_combined_info_for_prompt())
