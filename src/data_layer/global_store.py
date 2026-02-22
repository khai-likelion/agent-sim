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
import math
import os
import random
import re
from pathlib import Path
import threading
from src.ai_layer.llm_client import LLMClient

# LLM 선택값 → JSON 실제 카테고리 매핑 (부분 문자열 매칭 실패 보완)
CATEGORY_ALIAS = {
    "카페": "커피-음료",
    "커피": "커피-음료",
    "디저트": "제과점",
    "베이커리": "제과점",
    "브런치": "양식음식점",
    "이자카야": "호프-간이주점",
    "포차": "호프-간이주점",
    "와인바": "호프-간이주점",
    "술집": "호프-간이주점",
    "막걸리": "호프-간이주점",
    "칵테일바": "호프-간이주점",
    "칵테일": "호프-간이주점",
}


def match_category(query: str, store_category: str) -> bool:
    """LLM 선택 카테고리와 매장 카테고리 매칭.

    1차: 부분 문자열 매칭 (예: "한식" in "한식음식점")
    2차: CATEGORY_ALIAS 변환 후 매칭 (예: "카페" → "커피-음료" in "커피-음료")
    """
    q = query.lower()
    sc = (store_category or "").lower()
    if q in sc:
        return True
    alias = CATEGORY_ALIAS.get(query, "").lower()
    if alias and alias in sc:
        return True
    return False

## 객단가 정보는 이걸로 파싱
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

## 에이전트 리뷰/평점 남기기
@dataclass
class AgentRatingRecord:
    """
    에이전트가 남긴 개별 평점 기록.

    - rating: 0~5점 별점
    - selected_tags: 에이전트가 선택한 1-2개 태그 (맛/가성비/분위기/서비스)
    - comment: 리뷰 코멘트
    """
    agent_id: int
    agent_name: str
    visit_datetime: str
    rating: int                              # 0~5 별점
    selected_tags: List[str] = field(default_factory=list)  # ["맛", "가성비"] 등
    comment: str = ""                        # 에이전트 리뷰 코멘트

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "visit_datetime": self.visit_datetime,
            "rating": self.rating,
            "selected_tags": self.selected_tags,
            "comment": self.comment,
        }

## 에이전트 리뷰 요약 관리
@dataclass
class AgentReview:
    """
    에이전트 리뷰 요약 관리 (Summary + Buffer 구조).

    Fields:
        summary      : 과거 리뷰들을 LLM이 누적 요약한 텍스트.
        review_buffer: 아직 summary에 반영되지 않은 최근 리뷰 (최대 10개).
                       10개가 채워지면 LLM 요약 후 비워짐.
        all_comments : 요약 여부와 무관한 원본 리뷰 전체 로그.
        total_review_count: 전체 리뷰 누적 수.
        summary_threshold : 버퍼가 이 값 이상이면 자동 요약 실행 (기본 10).
    """
    summary: str = ""
    review_buffer: List[str] = field(default_factory=list)
    all_comments: List[str] = field(default_factory=list)
    total_review_count: int = 0
    summary_threshold: int = 10

    def get_context_for_agent(self) -> str:
        """에이전트에게 제공할 리뷰 맥락: summary + 현재 buffer 결합."""
        parts: List[str] = []
        if self.summary:
            parts.append(f"[요약] {self.summary}")
        if self.review_buffer:
            recent = " / ".join(self.review_buffer[-5:])  # 최신 5개만 노출
            parts.append(f"[최근 리뷰] {recent}")
        return " ".join(parts) if parts else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "review_buffer": self.review_buffer,
            "all_comments": self.all_comments,
            "total_review_count": self.total_review_count,
        }


@dataclass
class StoreRating:
    """
    개별 매장의 평점 정보.

    JSON 필드와 매핑:
    - 별점: star_rating (에이전트 평점 반영 시 평균 계산)
    - 맛/가성비/분위기/서비스: 태그 카운트 (에이전트 선택 시 증가)
    - agent_review: 에이전트 리뷰 (10개마다 요약)
    """
    store_id: str
    store_name: str
    category: str

    # === JSON에서 로드되는 별점/태그 필드 ===
    star_rating: float = 0.0           # JSON의 "별점" 필드 (시뮬레이션 중 갱신됨)
    star_rating_count: int = 0         # 별점 평가 수 (review_count 기반)
    taste_count: int = 0               # JSON의 "맛" 필드 (태그 "맛" 누적수)
    value_count: int = 0               # JSON의 "가성비" 필드 (태그 "가성비" 누적수)
    atmosphere_count: int = 0          # JSON의 "분위기" 필드 (태그 "분위기" 누적수)
    service_count: int = 0             # JSON의 "서비스" 필드 (태그 "서비스" 누적수)

    # === 에이전트 평점 기록 ===
    agent_ratings: List[AgentRatingRecord] = field(default_factory=list)

    # === 에이전트 리뷰 요약 (10개마다 Recursive Summarization) ===
    agent_review: AgentReview = field(default_factory=AgentReview)

    # 가격 정보
    average_price: int = 13000

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

    # 원본 JSON 파일 경로 (시뮬레이션 종료 시 업데이트용)
    _source_json_path: Optional[Path] = field(default=None, repr=False)

    # === 에이전트 평점 접근자 ===
    @property
    def agent_rating_count(self) -> int:
        """에이전트 평점 수"""
        return len(self.agent_ratings)

    @property
    def agent_avg_rating(self) -> float:
        """에이전트들이 남긴 별점 평균"""
        if not self.agent_ratings:
            return 0.0
        total = sum(r.rating for r in self.agent_ratings)
        return total / len(self.agent_ratings)

    def add_agent_rating(
        self,
        agent_id: int,
        agent_name: str,
        rating: int,
        selected_tags: List[str],
        visit_datetime: Optional[str] = None,
        comment: str = ""
    ):
        """
        에이전트 평점 추가.

        Args:
            agent_id: 에이전트 ID
            agent_name: 에이전트 이름
            rating: 별점 (0~5)
            selected_tags: 선택한 태그들 (1-2개, 예: ["맛", "가성비"])
            visit_datetime: 방문 시간
            comment: 리뷰 코멘트
        """
        record = AgentRatingRecord(
            agent_id=agent_id,
            agent_name=agent_name,
            visit_datetime=visit_datetime or datetime.now().isoformat(),
            rating=rating,
            selected_tags=selected_tags,
            comment=comment,
        )
        self.agent_ratings.append(record)

        # === 별점 업데이트 (평균 계산) ===
        # 공식: (기존 별점 * 기존 수 + 새 별점) / (기존 수 + 1)
        old_total = self.star_rating * self.star_rating_count
        self.star_rating_count += 1
        self.star_rating = round((old_total + rating) / self.star_rating_count, 1)

        # === 태그 카운트 증가 ===
        for tag in selected_tags:
            if tag == "맛":
                self.taste_count += 1
            elif tag == "가성비":
                self.value_count += 1
            elif tag == "분위기":
                self.atmosphere_count += 1
            elif tag == "서비스":
                self.service_count += 1

        # === 리뷰 코멘트 추가 ===
        if comment:
            self.agent_review.review_buffer.append(comment)
            self.agent_review.all_comments.append(comment)
            self.agent_review.total_review_count += 1

    def get_agent_score_text(self) -> str:
        """에이전트 평점 텍스트"""
        if not self.agent_ratings:
            return "아직 평가 없음"
        return f"별점:{self.star_rating:.1f}/5 ({self.agent_rating_count}건), 맛:{self.taste_count}, 가성비:{self.value_count}, 분위기:{self.atmosphere_count}, 서비스:{self.service_count}"

    def get_review_context(self) -> str:
        """
        에이전트 매장 조회 시 제공하는 리뷰 맥락.
        summary(누적 요약) + review_buffer(미요약 최근 리뷰)만 반환하여 토큰 절약.
        """
        return self.agent_review.get_context_for_agent()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "store_id": self.store_id,
            "store_name": self.store_name,
            "category": self.category,
            # 별점/태그
            "star_rating": self.star_rating,
            "star_rating_count": self.star_rating_count,
            "taste_count": self.taste_count,
            "value_count": self.value_count,
            "atmosphere_count": self.atmosphere_count,
            "service_count": self.service_count,
            # 에이전트 데이터
            "agent_rating_count": self.agent_rating_count,
            "agent_avg_rating": round(self.agent_avg_rating, 2),
            "agent_review": self.agent_review.to_dict(),
            "agent_ratings": [r.to_dict() for r in self.agent_ratings],
            # 기타
            "average_price": self.average_price,
            "top_keywords": self.top_keywords,
            "rag_context": self.rag_context[:250] if self.rag_context else "",
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

    def load_from_json_files(self, json_dir: Path):
        """
        JSON 파일들에서 매장 데이터 로드.

        로드하는 필드:
        - store_id, store_name, category, x, y, address
        - market_analysis, revenue_analysis, customer_analysis
        - review_metrics, top_keywords, critical_feedback, rag_context
        - raw_data_context.trend_history, metadata
        - metadata.x, metadata.y (좌표)
        """
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

                # 카테고리: JSON category → metadata.sector
                category = data.get("category", "")
                if not category:
                    metadata = data.get("metadata", {})
                    category = metadata.get("sector", "")

                # 좌표: JSON x,y → metadata.x,y
                x = data.get("x")  # longitude
                y = data.get("y")  # latitude
                if not x or not y:
                    metadata = data.get("metadata", {})
                    x = metadata.get("x", "")
                    y = metadata.get("y", "")
                coordinates = (float(y), float(x)) if x and y else None

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
                        address=data.get("address", ""),
                    )
                    self.stores[store_id] = store
                    self._store_name_to_id[store_name] = store_id

                # 원본 JSON 파일 경로 저장 (시뮬레이션 종료 시 업데이트용)
                store._source_json_path = json_file

                # === 별점/태그 필드 로드 ===
                if "별점" in data:
                    store.star_rating = float(data["별점"])
                if "review_count" in data:
                    store.star_rating_count = int(data["review_count"])

                # 태그: 새 형식 "태깅":{맛,가성비,...} 우선, 없으면 최상위 키 시도
                tagging = data.get("태깅", {})
                tag_source = tagging if tagging else data
                if "맛" in tag_source:
                    store.taste_count = int(tag_source["맛"])
                if "가성비" in tag_source:
                    store.value_count = int(tag_source["가성비"])
                if "분위기" in tag_source:
                    store.atmosphere_count = int(tag_source["분위기"])
                if "서비스" in tag_source:
                    store.service_count = int(tag_source["서비스"])

                # === 매장 정보 필드 로드 ===
                store.market_analysis = data.get("market_analysis")
                store.revenue_analysis = data.get("revenue_analysis", "")
                store.customer_analysis = data.get("customer_analysis", "")
                store.review_metrics = data.get("review_metrics")
                store.metadata = data.get("metadata")

                if "top_keywords" in data:
                    store.top_keywords = data["top_keywords"]
                if "critical_feedback" in data:
                    cf = data["critical_feedback"]
                    # 리스트이면 줄바꿈으로 합침, 문자열이면 그대로
                    store.critical_feedback = "\n".join(cf) if isinstance(cf, list) else cf
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

    def search_ranked_stores(
        self,
        category: str = "",
        agent_lat: float = 0.0,
        agent_lng: float = 0.0,
        sample_k: int = 20,
        temperature: float = 0.5,
        candidate_stores: Optional[List["StoreRating"]] = None,
    ) -> List[StoreRating]:
        """
        Softmax 가중 확률 샘플링 기반 매장 선별.

        카테고리 선필터 → 스코어링 → Softmax 확률 비례 비복원 추출.
        인기매장은 높은 확률로, 롱테일 매장도 일정 확률로 선택됨.

        개선 이력:
        - v1: 전체 매장 + 카테고리 보너스 + exploit 10/explore 5
        - v2 (1차): 카테고리 선필터 + candidate_stores + explore_k 증가
        - v3 (2차): 리뷰 수 캡(50건) + 가중치 완화 (0.5→0.3)
        - v4 (3차): 점수 지터(σ=0.8) → 인기매장 잔류율 급락 문제
        - v5 (4차): exploit/explore 비율 조정 → 인기/비인기 갭 여전
        - v6 (5차): Softmax 가중 샘플링 (T=0.5)으로 자연스러운 분포

        스코어링: 별점×1.5 + 에이전트평점×2.0 + min(리뷰,50)로그×0.3 - 거리×0.3
        선택: softmax(score / T) 확률 비례 비복원 추출
        """
        from math import radians, cos, sin, asin, sqrt, exp

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            return R * c

        # 검색 대상: candidate_stores가 있으면 해당 목록, 없으면 전체
        pool = candidate_stores if candidate_stores is not None else list(self.stores.values())

        # 카테고리 선필터링
        if category:
            category_pool = [s for s in pool if match_category(category, s.category)]
            if len(category_pool) >= 5:
                pool = category_pool

        # 스코어링
        scores = []
        for store in pool:
            score = 0.0
            score += (store.star_rating or 0) * 1.5
            base_rating = store.agent_avg_rating if store.agent_rating_count > 0 else (store.star_rating or 0)
            score += base_rating * 2.0
            capped_reviews = min(store.star_rating_count or 0, 50)
            score += math.log1p(capped_reviews) * 0.3
            if store.coordinates and agent_lat and agent_lng:
                lat, lng = store.coordinates
                dist = haversine(agent_lat, agent_lng, lat, lng)
                score -= dist * 0.3
            # feature_scores 품질 지표 반영 (X_to_Sim 전략 효과를 랭킹에 반영)
            if store.review_metrics:
                fs = store.review_metrics.get("feature_scores", {})
                if fs:
                    taste   = fs.get("taste",      {}).get("score", 0)
                    price   = fs.get("price_value", {}).get("score", 0)
                    clean   = fs.get("cleanliness", {}).get("score", 0)
                    service = fs.get("service",     {}).get("score", 0)
                    atm     = fs.get("atmosphere",  {}).get("score", 0)
                    fq = taste*0.30 + price*0.25 + clean*0.20 + service*0.15 + atm*0.10
                    score += fq * 1.5
            scores.append(score)

        # Softmax 확률 계산
        if not scores:
            return []
        max_score = max(scores)
        exp_scores = [exp((s - max_score) / temperature) for s in scores]
        total_exp = sum(exp_scores)
        probs = [e / total_exp for e in exp_scores]

        # 계층화 샘플링: 상위(60%) + 중위(25%) + 하위(15%)
        # 롱테일 매장도 LLM 후보에 포함되도록 다양성 보장
        k = min(sample_k, len(pool))
        sorted_indices = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)

        n = len(sorted_indices)
        top_end = max(1, n // 3)
        mid_end = max(top_end + 1, 2 * n // 3)

        tier_top = sorted_indices[:top_end]
        tier_mid = sorted_indices[top_end:mid_end]
        tier_low = sorted_indices[mid_end:]

        k_top = max(1, round(k * 0.60))
        k_mid = max(1, round(k * 0.20))
        k_low = k - k_top - k_mid

        def _sample_tier(tier_indices, count):
            if not tier_indices or count <= 0:
                return []
            tier_probs = [probs[i] for i in tier_indices]
            total_p = sum(tier_probs)
            if total_p == 0:
                return random.sample(tier_indices, min(count, len(tier_indices)))
            norm_p = [p / total_p for p in tier_probs]
            chosen = []
            idxs = list(tier_indices)
            ws = list(norm_p)
            for _ in range(min(count, len(idxs))):
                tw = sum(ws)
                nw = [w / tw for w in ws]
                pick = random.choices(range(len(idxs)), weights=nw, k=1)[0]
                chosen.append(idxs[pick])
                idxs.pop(pick)
                ws.pop(pick)
            return chosen

        selected_indices = (
            _sample_tier(tier_top, k_top)
            + _sample_tier(tier_mid, k_mid)
            + _sample_tier(tier_low, k_low)
        )
        selected = [pool[i] for i in selected_indices]

        return selected

    def add_agent_rating(
        self,
        store_name: str,
        agent_id: int,
        agent_name: str,
        rating: int,
        selected_tags: List[str],
        visit_datetime: Optional[str] = None,
        comment: str = ""
    ) -> bool:
        """
        매장에 에이전트 평점 추가.

        Args:
            store_name: 매장명
            agent_id: 에이전트 ID
            agent_name: 에이전트 이름
            rating: 별점 (0~5)
            selected_tags: 선택한 태그들 (1-2개, 예: ["맛", "가성비"])
            visit_datetime: 방문 시간
            comment: 리뷰 코멘트

        Returns: 성공 여부
        """
        store = self.get_by_name(store_name)
        if store:
            store.add_agent_rating(
                agent_id=agent_id,
                agent_name=agent_name,
                rating=rating,
                selected_tags=selected_tags,
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
        rating: int,
        selected_tags: List[str],
        visit_datetime: Optional[str] = None,
        comment: str = ""
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
                "rating": rating,
                "selected_tags": selected_tags,
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
                rating=rating["rating"],
                selected_tags=rating["selected_tags"],
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
            if match_category(category_keyword, store.category):
                result.append(store)
        return result

    def get_top_stores_by_agent_rating(self, n: int = 10, category: Optional[str] = None) -> List[StoreRating]:
        """에이전트 평점 기준 상위 매장 (평점 있는 매장만)"""
        stores = [s for s in self.stores.values() if s.agent_rating_count > 0]
        if category:
            stores = [s for s in stores if match_category(category, s.category)]
        stores.sort(key=lambda s: s.star_rating, reverse=True)
        return stores[:n]

    def get_statistics(self) -> Dict[str, Any]:
        """전체 통계"""
        total_agent_ratings = sum(s.agent_rating_count for s in self.stores.values())
        rated_stores = sum(1 for s in self.stores.values() if s.agent_rating_count > 0)

        # 평균 별점 (평점 있는 매장만)
        rated_stores_list = [s for s in self.stores.values() if s.agent_rating_count > 0]
        if rated_stores_list:
            avg_star_rating = sum(s.star_rating for s in rated_stores_list) / len(rated_stores_list)
        else:
            avg_star_rating = 0.0

        # 태그 총합
        total_taste = sum(s.taste_count for s in self.stores.values())
        total_value = sum(s.value_count for s in self.stores.values())
        total_atmosphere = sum(s.atmosphere_count for s in self.stores.values())
        total_service = sum(s.service_count for s in self.stores.values())

        return {
            "total_stores": len(self.stores),
            "stores_with_agent_ratings": rated_stores,
            "total_agent_ratings": total_agent_ratings,
            "avg_star_rating": round(avg_star_rating, 2),
            "total_taste_tags": total_taste,
            "total_value_tags": total_value,
            "total_atmosphere_tags": total_atmosphere,
            "total_service_tags": total_service,
        }

    def reset_agent_ratings(self):
        """에이전트 평점 초기화"""
        for store in self.stores.values():
            store.agent_ratings = []
            store.agent_review = AgentReview()

    def save_to_json(self, output_path: Path):
        """현재 상태를 JSON으로 저장"""
        data = {
            "statistics": self.get_statistics(),
            "stores": [s.to_dict() for s in self.stores.values()],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    async def auto_summarize_store_async(
        self,
        store: StoreRating,
    ) -> bool:
        """
        review_buffer >= threshold 일 때 LLMClient로 자동 요약 (비동기).

        요약 형식:
            기존 summary + buffer 리뷰들을 합쳐 핵심만 3문장 이내로 재요약.

        Args:
            store: 대상 매장 StoreRating

        Returns:
            요약 실행 여부
        """
        review = store.agent_review
        if len(review.review_buffer) < review.summary_threshold:
            return False

        try:
            llm = LLMClient()

            existing  = review.summary
            new_items = review.review_buffer
            reviews_text = "\n".join(f"- {r}" for r in new_items)

            system_msg = (
                "당신은 식당 리뷰 요약 전문가입니다. "
                "제공된 리뷰들을 핵심 정보만 남겨 간결하게 요약해 주세요."
            )
            user_msg = (
                f"매장명: {store.store_name}\n\n"
                + (f"기존 요약:\n{existing}\n\n" if existing else "")
                + f"새 리뷰 {len(new_items)}건:\n{reviews_text}\n\n"
                "위 내용을 통합하여 핵심 인상(맛·분위기·가성비·서비스)을 "
                "3문장 이내 한국어로 요약해 주세요."
            )

            review.summary = await llm.generate(user_msg, system_msg)
            review.review_buffer = []   # buffer 비움
            return True

        except Exception as e:
            # 요약 실패 시 buffer는 유지 (다음 flush 때 재시도)
            print(f"  [리뷰 요약 오류] {store.store_name}: {e}")
            return False

    async def flush_and_summarize_async(self) -> Dict[str, int]:
        """
        pending_ratings flush 후, buffer >= threshold 매장을 비동기 요약.

        시뮬레이션 타임슬롯 종료 시 flush_pending_ratings() 대신 호출하면
        평점 반영 + 리뷰 자동 요약을 한 번에 처리.

        Returns:
            {"flushed": int, "summarized": int}
        """
        flushed = self.flush_pending_ratings()

        import asyncio
        tasks = [
            self.auto_summarize_store_async(store)
            for store in self.stores.values()
            if len(store.agent_review.review_buffer) >= store.agent_review.summary_threshold
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        summarized = sum(1 for r in results if r is True)

        return {"flushed": flushed, "summarized": summarized}

    # ==================== 원본 JSON 파일 업데이트 ====================

    def update_original_json_files(self) -> int:
        """
        시뮬레이션 종료 시 원본 JSON 파일에 에이전트 데이터 반영.

        업데이트되는 필드:
        - 별점: 에이전트 평점 반영 평균
        - 맛/가성비/분위기/서비스: 태그 카운트
        - agent_review: 리뷰 요약 및 코멘트 (review_count 뒤에 위치)
        - agent_ratings: 개별 평점 기록

        Returns:
            업데이트된 파일 수
        """
        updated_count = 0

        for store in self.stores.values():
            if not store._source_json_path or not store._source_json_path.exists():
                continue

            # 에이전트 데이터가 없으면 스킵
            if not store.agent_ratings:
                continue

            try:
                # 원본 JSON 로드
                with open(store._source_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 리스트 형태인 경우 첫 번째 요소
                if isinstance(data, list):
                    data = data[0] if data else {}

                # === 별점 업데이트 (평균 계산됨) ===
                data["별점"] = store.star_rating

                # === 태그 카운트 업데이트 ===
                # 새 형식: "태깅" 중첩 딕셔너리 / 구 형식: 최상위 키
                if "태깅" in data:
                    data["태깅"]["맛"] = store.taste_count
                    data["태깅"]["가성비"] = store.value_count
                    data["태깅"]["분위기"] = store.atmosphere_count
                    data["태깅"]["서비스"] = store.service_count
                else:
                    data["맛"] = store.taste_count
                    data["가성비"] = store.value_count
                    data["분위기"] = store.atmosphere_count
                    data["서비스"] = store.service_count

                # === agent_review 추가 (review_count 뒤) ===
                data["agent_review"] = store.agent_review.to_dict()

                # === agent_ratings 추가 ===
                data["agent_ratings"] = [r.to_dict() for r in store.agent_ratings]

                # 저장
                with open(store._source_json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                updated_count += 1

            except Exception as e:
                print(f"JSON 업데이트 오류 ({store.store_name}): {e}")

        return updated_count


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
    json_dir = settings.paths.split_store_dir
    store.load_from_json_files(json_dir)

    print(f"로드된 매장 수: {len(store.stores)}")

    # 통계 출력
    stats = store.get_statistics()
    print(f"\n초기 통계: {stats}")

    # 에이전트 평점 추가 테스트
    test_store = list(store.stores.values())[0] if store.stores else None
    if test_store:
        print(f"\n=== 테스트 매장: {test_store.store_name} ===")
        print(f"초기 별점: {test_store.star_rating}, 맛:{test_store.taste_count}, 가성비:{test_store.value_count}")
        print(f"에이전트 평점: {test_store.get_agent_score_text()}")

        # 에이전트 평점 추가 (별점 + 태그 선택)
        store.add_agent_rating(
            test_store.store_name, 1, "김민준",
            rating=4, selected_tags=["맛", "가성비"],
            comment="국물이 진하고 맛있었어요!"
        )
        store.add_agent_rating(
            test_store.store_name, 2, "이서연",
            rating=5, selected_tags=["맛"],
            comment="정말 맛있는 닭한마리! 또 오고 싶어요."
        )
        store.add_agent_rating(
            test_store.store_name, 3, "박지우",
            rating=3, selected_tags=["가성비", "분위기"],
            comment="가격 대비 괜찮아요. 분위기도 좋고요."
        )

        print(f"\n=== 평점 추가 후 ===")
        print(f"업데이트된 별점: {test_store.star_rating}")
        print(f"태그: 맛:{test_store.taste_count}, 가성비:{test_store.value_count}, 분위기:{test_store.atmosphere_count}, 서비스:{test_store.service_count}")
        print(f"에이전트 평점: {test_store.get_agent_score_text()}")
        print(f"리뷰 버퍼: {test_store.agent_review.review_buffer}")
