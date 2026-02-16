"""
Store Rating System: Agent experiences → Ratings → Store info update cycle.

Flow:
1. Agent perceives store: 기존 매장정보 (existing store info)
2. Agent visits: X-report 주입 (X-report injection)
3. Agent rates: 본인 평점 매김 (self-rating: 별점 + 태깅)
4. Rating update: 평점이 기존 매장정보에 누적 반영
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

from src.simulation_layer.models import BusinessReport


@dataclass
class AgentRating:
    """
    Agent's self-rating after visiting a store.
    평점 = 별점 + 태깅 (맛, 가성비, 분위기)
    """
    agent_id: int
    store_name: str
    timestamp: datetime
    
    # 별점 (0.0 ~ 1.0, 0.0=최악, 1.0=최고)
    star_rating: float
    
    # 태깅 (맛, 가성비, 분위기)
    taste_tag: float = 0.5  # 0.0 ~ 1.0
    value_tag: float = 0.5  # 가성비
    atmosphere_tag: float = 0.5  # 분위기
    
    # 추가 태그 (선택적)
    service_tag: Optional[float] = None
    cleanliness_tag: Optional[float] = None
    
    # 메모
    comment: str = ""
    
    # X-report 영향 여부
    influenced_by_report: bool = False
    report_description: Optional[str] = None


@dataclass
class StoreRatingState:
    """
    Dynamic store rating state that accumulates agent ratings.
    기존 매장정보 = 이 누적된 평점들로부터 계산됨
    """
    store_name: str
    
    # 누적된 평점들
    ratings: List[AgentRating] = field(default_factory=list)
    
    # 계산된 평균 점수 (기존 매장정보)
    avg_star_rating: float = 0.5
    avg_taste: float = 0.5
    avg_value: float = 0.5
    avg_atmosphere: float = 0.5
    avg_service: float = 0.5
    avg_cleanliness: float = 0.5
    
    # 누적된 키워드 (태깅에서 추출)
    positive_keywords: List[str] = field(default_factory=list)
    negative_keywords: List[str] = field(default_factory=list)
    
    # 리뷰 수
    review_count: int = 0
    
    def add_rating(self, rating: AgentRating):
        """Add agent rating and update averages."""
        self.ratings.append(rating)
        self.review_count = len(self.ratings)
        
        # Recalculate averages
        if self.ratings:
            self.avg_star_rating = sum(r.star_rating for r in self.ratings) / len(self.ratings)
            self.avg_taste = sum(r.taste_tag for r in self.ratings) / len(self.ratings)
            self.avg_value = sum(r.value_tag for r in self.ratings) / len(self.ratings)
            self.avg_atmosphere = sum(r.atmosphere_tag for r in self.ratings) / len(self.ratings)
            
            service_ratings = [r.service_tag for r in self.ratings if r.service_tag is not None]
            if service_ratings:
                self.avg_service = sum(service_ratings) / len(service_ratings)
            
            cleanliness_ratings = [r.cleanliness_tag for r in self.ratings if r.cleanliness_tag is not None]
            if cleanliness_ratings:
                self.avg_cleanliness = sum(cleanliness_ratings) / len(cleanliness_ratings)
            
            # Extract keywords from comments
            if rating.comment:
                # Simple keyword extraction (can be enhanced)
                if rating.star_rating >= 0.7:
                    self.positive_keywords.append(rating.comment[:20])
                elif rating.star_rating <= 0.3:
                    self.negative_keywords.append(rating.comment[:20])
    
    def get_summary(self) -> str:
        """Get summary for agent perception (기존 매장정보)."""
        if self.review_count == 0:
            return f"[{self.store_name}] 리뷰 정보 없음"
        
        # Calculate sentiment label
        if self.avg_star_rating >= 0.8:
            sentiment = "매우 긍정적"
        elif self.avg_star_rating >= 0.6:
            sentiment = "긍정적"
        elif self.avg_star_rating >= 0.4:
            sentiment = "보통"
        else:
            sentiment = "부정적"
        
        # Get recent keywords
        recent_keywords = ", ".join(self.positive_keywords[-5:]) if self.positive_keywords else "없음"
        negatives = ", ".join(self.negative_keywords[-3:]) if self.negative_keywords else "없음"
        
        return (
            f"[{self.store_name}] 평점:{sentiment} | "
            f"맛:{self.avg_taste:.1f} 가성비:{self.avg_value:.1f} "
            f"청결:{self.avg_cleanliness:.1f} 서비스:{self.avg_service:.1f} | "
            f"키워드: {recent_keywords} | 단점: {negatives}"
        )


class StoreRatingManager:
    """
    Manages the rating cycle:
    Agent ratings → Store rating state → 기존 매장정보 업데이트
    """
    
    def __init__(self):
        # store_name -> StoreRatingState
        self._store_states: Dict[str, StoreRatingState] = {}
        
        # Initial review data (from JSON files)
        self._initial_reviews: Dict[str, StoreRatingState] = {}
        self._initialized = False
    
    def initialize_from_reviews(self, review_loader=None):
        """Initialize store states from existing review data."""
        if self._initialized:
            return
        
        # Load initial reviews from JSON files
        # This represents "기존 매장정보" at simulation start
        from src.data_layer.store_reviews import get_store_review_loader
        
        loader = review_loader or get_store_review_loader()
        loader._load_all()
        
        # Convert existing StoreReview data into initial rating states
        # This simulates accumulated ratings from before simulation
        for store_id, review in loader._cache.items():
            store_name = review.store_name
            state = self.get_store_state(store_name)
            
            # Set initial averages from review data
            state.avg_star_rating = review.overall_score
            state.avg_taste = review.taste_score
            state.avg_value = review.price_value_score
            state.avg_atmosphere = 0.5  # Default if not in review
            state.avg_service = review.service_score
            state.avg_cleanliness = review.cleanliness_score
            
            # Set keywords from review
            state.positive_keywords = review.top_keywords[:10]
            state.negative_keywords = review.critical_feedback[:5]
            
            # Estimate review count from sentiment
            # Higher sentiment = more reviews (rough estimate)
            state.review_count = max(10, int(review.overall_score * 50))
        
        self._initialized = True
    
    def get_store_state(self, store_name: str) -> StoreRatingState:
        """Get or create store rating state."""
        if store_name not in self._store_states:
            self._store_states[store_name] = StoreRatingState(store_name=store_name)
        return self._store_states[store_name]
    
    def add_agent_rating(
        self,
        agent_id: int,
        store_name: str,
        star_rating: float,
        taste_tag: float,
        value_tag: float,
        atmosphere_tag: float,
        timestamp: datetime,
        comment: str = "",
        service_tag: Optional[float] = None,
        cleanliness_tag: Optional[float] = None,
        influenced_by_report: bool = False,
        report_description: Optional[str] = None,
    ):
        """
        Add agent's self-rating (본인 평점 매김).
        This updates the store's rating state (평점 update).
        """
        rating = AgentRating(
            agent_id=agent_id,
            store_name=store_name,
            timestamp=timestamp,
            star_rating=star_rating,
            taste_tag=taste_tag,
            value_tag=value_tag,
            atmosphere_tag=atmosphere_tag,
            service_tag=service_tag,
            cleanliness_tag=cleanliness_tag,
            comment=comment,
            influenced_by_report=influenced_by_report,
            report_description=report_description,
        )
        
        state = self.get_store_state(store_name)
        state.add_rating(rating)
    
    def get_store_summary(self, store_name: str) -> str:
        """
        Get store summary for agent perception (기존 매장정보).
        This is what agents see when they recognize a store.
        """
        state = self.get_store_state(store_name)
        return state.get_summary()
    
    def get_all_summaries(self) -> Dict[str, str]:
        """Get summaries for all stores."""
        return {
            name: state.get_summary()
            for name, state in self._store_states.items()
        }


# Singleton
_rating_manager: Optional[StoreRatingManager] = None


def get_store_rating_manager() -> StoreRatingManager:
    """Get singleton instance of StoreRatingManager."""
    global _rating_manager
    if _rating_manager is None:
        _rating_manager = StoreRatingManager()
    return _rating_manager
