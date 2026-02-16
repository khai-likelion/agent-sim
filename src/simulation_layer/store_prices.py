"""
Store price system: Derives price tiers from review data + category defaults.
Maps each store to a price tier and estimated per-person cost (객단가).
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.data_layer.store_reviews import StoreReviewLoader, get_store_review_loader


@dataclass
class StorePriceInfo:
    """Price information for a single store."""
    store_name: str
    price_tier: str         # "저렴"/"보통"/"비쌈"/"고급"
    estimated_price: int    # 예상 객단가 (원)
    price_value_score: float  # 0~1, from review data (higher = better value)


# Category-based default prices (per person, in KRW)
CATEGORY_BASE_PRICE: Dict[str, int] = {
    "한식": 9000,
    "국밥": 8000,
    "백반": 7000,
    "분식": 5000,
    "중식": 10000,
    "일식": 15000,
    "양식": 14000,
    "카페": 5500,
    "디저트": 6000,
    "베이커리": 4500,
    "패스트푸드": 7000,
    "치킨": 18000,
    "피자": 15000,
    "고기/구이": 22000,
    "술집/호프": 16000,
    "샐러드": 10000,
    "아시안": 11000,
    "브런치": 13000,
}

DEFAULT_BASE_PRICE = 10000

# Price tier thresholds
TIER_THRESHOLDS = {
    "저렴": (0, 8000),
    "보통": (8000, 15000),
    "비쌈": (15000, 25000),
    "고급": (25000, 999999),
}


def _price_to_tier(price: int) -> str:
    """Convert price to tier label."""
    for tier, (low, high) in TIER_THRESHOLDS.items():
        if low <= price < high:
            return tier
    return "보통"


class StorePriceManager:
    """Manages store price information derived from reviews + category defaults."""

    def __init__(self, review_loader: Optional[StoreReviewLoader] = None):
        self._review_loader = review_loader
        self._cache: Dict[str, StorePriceInfo] = {}

    def _get_review_loader(self) -> StoreReviewLoader:
        if self._review_loader is None:
            self._review_loader = get_store_review_loader()
        return self._review_loader

    def get_price_info(self, store_name: str, category: str = "") -> StorePriceInfo:
        """Get price info for a store.

        Logic:
        - Start with category base price
        - If review data exists, adjust based on price_value score:
          - High price_value (>0.7) = good value = lower effective price
          - Low price_value (<0.3) = expensive = higher effective price
        """
        if store_name in self._cache:
            return self._cache[store_name]

        loader = self._get_review_loader()
        review = loader.get_by_name(store_name)

        # Determine base price from category
        base_price = DEFAULT_BASE_PRICE
        if category:
            # Try exact match first, then partial
            for cat_key, price in CATEGORY_BASE_PRICE.items():
                if cat_key in category or category in cat_key:
                    base_price = price
                    break
        elif review and review.category:
            for cat_key, price in CATEGORY_BASE_PRICE.items():
                if cat_key in review.category or review.category in cat_key:
                    base_price = price
                    break

        # Adjust based on review price_value score
        price_value_score = 0.5  # default neutral
        if review:
            price_value_score = review.price_value_score

            # price_value > 0.7 means good value → actual price is lower
            # price_value < 0.3 means bad value → actual price is higher
            if price_value_score > 0.7:
                price_modifier = 0.8  # 20% cheaper than category avg
            elif price_value_score > 0.5:
                price_modifier = 0.95
            elif price_value_score > 0.3:
                price_modifier = 1.1
            else:
                price_modifier = 1.3  # 30% more expensive
        else:
            price_modifier = 1.0

        estimated_price = int(base_price * price_modifier)
        # Round to nearest 500
        estimated_price = round(estimated_price / 500) * 500

        tier = _price_to_tier(estimated_price)

        info = StorePriceInfo(
            store_name=store_name,
            price_tier=tier,
            estimated_price=estimated_price,
            price_value_score=price_value_score,
        )
        self._cache[store_name] = info
        return info

    def get_price_context(self, budget_today: str, price_sensitivity: float) -> str:
        """Generate budget context text for prompt injection."""
        if budget_today == "tight":
            return "오늘은 예산이 빠듯합니다. 저렴한 곳을 우선으로 고려하세요."
        elif budget_today == "generous":
            if price_sensitivity < 0.3:
                return "오늘은 여유가 있어 비싼 곳도 괜찮습니다."
            return "오늘은 여유가 있지만, 가성비도 따져봅니다."
        else:
            if price_sensitivity > 0.6:
                return "보통 예산이지만 가성비를 중시합니다."
            return "보통 예산입니다."

    def format_price_for_store(self, store_name: str, category: str = "") -> str:
        """Format price info for a single store in prompt."""
        info = self.get_price_info(store_name, category)
        return f"[가격:{info.price_tier} ~{info.estimated_price:,}원]"

    def calculate_price_match_score(
        self, store_name: str, category: str, budget_today: str, price_sensitivity: float
    ) -> float:
        """Calculate how well a store's price matches agent's budget.

        Returns 0.0~1.0 where 1.0 = perfect match.
        """
        info = self.get_price_info(store_name, category)
        tier = info.price_tier

        # Base score by budget-tier alignment
        if budget_today == "tight":
            scores = {"저렴": 1.0, "보통": 0.6, "비쌈": 0.2, "고급": 0.0}
        elif budget_today == "generous":
            scores = {"저렴": 0.6, "보통": 0.8, "비쌈": 1.0, "고급": 0.9}
        else:  # normal
            scores = {"저렴": 0.8, "보통": 1.0, "비쌈": 0.5, "고급": 0.2}

        base = scores.get(tier, 0.5)

        # Price sensitivity amplifies the effect
        # High sensitivity → bigger penalty for expensive, bigger reward for cheap
        sensitivity_factor = 0.5 + price_sensitivity * 0.5  # 0.5 ~ 1.0
        # Pull score toward extreme based on sensitivity
        return base * sensitivity_factor + (1 - sensitivity_factor) * 0.5


# Singleton
_price_manager: Optional[StorePriceManager] = None


def get_store_price_manager() -> StorePriceManager:
    global _price_manager
    if _price_manager is None:
        _price_manager = StorePriceManager()
    return _price_manager
