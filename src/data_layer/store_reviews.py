"""
Store review data loader.
Loads review/analysis data from split_by_store_id JSON files.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from config import get_settings


@dataclass
class StoreReview:
    """Parsed store review data for agent decision-making."""
    store_id: str
    store_name: str
    category: str  # metadata.sector

    # Scores (0-1)
    overall_score: float
    taste_score: float
    price_value_score: float
    cleanliness_score: float
    service_score: float

    # Text info
    sentiment_label: str  # "매우 긍정적", "긍정적", etc.
    top_keywords: List[str]
    critical_feedback: List[str]
    rag_context: str  # Review summary

    def get_brief_summary(self) -> str:
        """Get brief summary for LLM prompt (token-efficient)."""
        keywords = ", ".join(self.top_keywords[:5])
        negatives = self.critical_feedback[0][:50] if self.critical_feedback else "없음"
        return (
            f"[{self.store_name}] 평점:{self.sentiment_label} | "
            f"맛:{self.taste_score:.1f} 가성비:{self.price_value_score:.1f} "
            f"청결:{self.cleanliness_score:.1f} 서비스:{self.service_score:.1f} | "
            f"키워드: {keywords} | 단점: {negatives}"
        )

    def get_detailed_summary(self) -> str:
        """Get detailed summary for important decisions."""
        return self.rag_context


class StoreReviewLoader:
    """Loads and caches store review data."""

    def __init__(self):
        settings = get_settings()
        self.review_dir = settings.paths.data_dir / "split_by_store_id"
        self._cache: Dict[str, StoreReview] = {}
        self._name_to_id: Dict[str, str] = {}
        self._loaded = False

    def _load_all(self) -> None:
        """Load all store reviews into cache."""
        if self._loaded:
            return

        if not self.review_dir.exists():
            print(f"Warning: Store review directory not found: {self.review_dir}")
            self._loaded = True
            return

        for json_file in self.review_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Each file is a list with one item
                if isinstance(data, list) and len(data) > 0:
                    item = data[0]
                else:
                    item = data

                review = self._parse_review(item)
                if review:
                    self._cache[review.store_id] = review
                    # Also index by name (normalized)
                    name_key = review.store_name.strip().lower()
                    self._name_to_id[name_key] = review.store_id
            except Exception as e:
                # Silently skip problematic files
                continue

        self._loaded = True
        print(f"  Loaded {len(self._cache)} store reviews")

    def _parse_review(self, data: Dict[str, Any]) -> Optional[StoreReview]:
        """Parse JSON data into StoreReview object."""
        try:
            store_id = str(data.get("store_id", ""))
            store_name = data.get("store_name", "")

            if not store_id or not store_name:
                return None

            # Get category from metadata
            metadata = data.get("metadata", {})
            category = metadata.get("sector", "기타")

            # Get review metrics
            review_metrics = data.get("review_metrics", {})
            overall = review_metrics.get("overall_sentiment", {})
            features = review_metrics.get("feature_scores", {})

            return StoreReview(
                store_id=store_id,
                store_name=store_name,
                category=category,
                overall_score=overall.get("score", 0.5),
                taste_score=features.get("taste", {}).get("score", 0.5),
                price_value_score=features.get("price_value", {}).get("score", 0.5),
                cleanliness_score=features.get("cleanliness", {}).get("score", 0.5),
                service_score=features.get("service", {}).get("score", 0.5),
                sentiment_label=overall.get("label", "보통"),
                top_keywords=data.get("top_keywords", []),
                critical_feedback=data.get("critical_feedback", []),
                rag_context=data.get("rag_context", ""),
            )
        except Exception:
            return None

    def get_by_id(self, store_id: str) -> Optional[StoreReview]:
        """Get store review by ID."""
        self._load_all()
        return self._cache.get(str(store_id))

    def get_by_name(self, store_name: str) -> Optional[StoreReview]:
        """Get store review by name (fuzzy match)."""
        self._load_all()

        # Exact match first
        name_key = store_name.strip().lower()
        if name_key in self._name_to_id:
            store_id = self._name_to_id[name_key]
            return self._cache.get(store_id)

        # Partial match
        for cached_name, store_id in self._name_to_id.items():
            if name_key in cached_name or cached_name in name_key:
                return self._cache.get(store_id)

        return None

    def get_reviews_for_stores(self, store_names: List[str]) -> Dict[str, StoreReview]:
        """Get reviews for multiple stores."""
        self._load_all()
        result = {}
        for name in store_names:
            review = self.get_by_name(name)
            if review:
                result[name] = review
        return result

    def format_for_prompt(self, store_names: List[str], max_stores: int = 10) -> str:
        """Format store reviews for LLM prompt."""
        self._load_all()

        lines = []
        count = 0
        for name in store_names:
            if count >= max_stores:
                break
            review = self.get_by_name(name)
            if review:
                lines.append(review.get_brief_summary())
                count += 1
            else:
                # No review data
                lines.append(f"[{name}] 리뷰 정보 없음")
                count += 1

        return "\n".join(lines) if lines else "리뷰 정보 없음"


# Global singleton
_review_loader: Optional[StoreReviewLoader] = None


def get_store_review_loader() -> StoreReviewLoader:
    """Get singleton instance of StoreReviewLoader."""
    global _review_loader
    if _review_loader is None:
        _review_loader = StoreReviewLoader()
    return _review_loader
