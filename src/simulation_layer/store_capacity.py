"""
Store capacity system: Limits simultaneous visits per store per timeslot.
Provides waiting logic when stores are full.
"""

import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class StoreCapacityInfo:
    """Capacity status for a store in current timeslot."""
    store_name: str
    max_capacity: int
    current_occupancy: int
    wait_time_minutes: int

    @property
    def is_full(self) -> bool:
        return self.current_occupancy >= self.max_capacity

    @property
    def occupancy_ratio(self) -> float:
        return self.current_occupancy / self.max_capacity if self.max_capacity > 0 else 1.0


# Category-based default capacity per timeslot
# Represents how many agents can visit in one time period
CATEGORY_CAPACITY: Dict[str, int] = {
    "카페": 8,
    "디저트": 6,
    "베이커리": 5,
    "한식": 10,
    "국밥": 10,
    "백반": 10,
    "일식": 7,
    "중식": 10,
    "양식": 8,
    "분식": 12,
    "패스트푸드": 15,
    "치킨": 8,
    "피자": 8,
    "고기/구이": 8,
    "술집/호프": 12,
    "브런치": 6,
    "샐러드": 6,
    "아시안": 8,
}

DEFAULT_CAPACITY = 8


class StoreCapacityManager:
    """Manages per-store capacity tracking within a timeslot."""

    def __init__(self):
        # store_name -> current occupancy count
        self._occupancy: Dict[str, int] = {}
        # store_name -> max capacity (cached after first lookup)
        self._max_capacity: Dict[str, int] = {}

    def _get_max_capacity(self, store_name: str, category: str = "") -> int:
        """Get max capacity for a store based on its category."""
        if store_name in self._max_capacity:
            return self._max_capacity[store_name]

        capacity = DEFAULT_CAPACITY
        if category:
            for cat_key, cap in CATEGORY_CAPACITY.items():
                if cat_key in category or category in cat_key:
                    capacity = cap
                    break

        self._max_capacity[store_name] = capacity
        return capacity

    def try_visit(self, store_name: str, category: str = "") -> Tuple[bool, int]:
        """Attempt to visit a store.

        Returns:
            (success: bool, wait_time_minutes: int)
            - success=True, wait=0: Immediate entry
            - success=False, wait=N: Store full, estimated N min wait
        """
        max_cap = self._get_max_capacity(store_name, category)
        current = self._occupancy.get(store_name, 0)

        if current < max_cap:
            # Space available - register visit
            self._occupancy[store_name] = current + 1
            return True, 0
        else:
            # Full - estimate wait time based on overflow
            overflow = current - max_cap + 1
            wait_minutes = overflow * random.randint(8, 15)  # 8-15 min per person ahead
            return False, wait_minutes

    def register_visit(self, store_name: str, category: str = "") -> None:
        """Register a visit (for waited agents who got in)."""
        current = self._occupancy.get(store_name, 0)
        self._occupancy[store_name] = current + 1

    def get_capacity_info(self, store_name: str, category: str = "") -> StoreCapacityInfo:
        """Get current capacity status for a store."""
        max_cap = self._get_max_capacity(store_name, category)
        current = self._occupancy.get(store_name, 0)
        wait = 0
        if current >= max_cap:
            wait = (current - max_cap + 1) * 10

        return StoreCapacityInfo(
            store_name=store_name,
            max_capacity=max_cap,
            current_occupancy=current,
            wait_time_minutes=wait,
        )

    def reset_timeslot(self) -> None:
        """Reset occupancy for a new timeslot."""
        self._occupancy.clear()

    def get_full_stores(self) -> Dict[str, StoreCapacityInfo]:
        """Get all stores that are at or over capacity."""
        result = {}
        for store_name, occ in self._occupancy.items():
            max_cap = self._max_capacity.get(store_name, DEFAULT_CAPACITY)
            if occ >= max_cap:
                result[store_name] = StoreCapacityInfo(
                    store_name=store_name,
                    max_capacity=max_cap,
                    current_occupancy=occ,
                    wait_time_minutes=(occ - max_cap + 1) * 10,
                )
        return result

    def get_stats(self) -> Dict[str, int]:
        """Get summary stats for logging."""
        total_stores = len(self._occupancy)
        full_stores = sum(
            1 for name, occ in self._occupancy.items()
            if occ >= self._max_capacity.get(name, DEFAULT_CAPACITY)
        )
        total_visits = sum(self._occupancy.values())
        return {
            "stores_visited": total_stores,
            "stores_full": full_stores,
            "total_visits": total_visits,
        }


def should_wait(
    wait_minutes: int,
    time_pressure: str,
    hunger: float = 50.0,
) -> bool:
    """Decide whether an agent should wait or find an alternative.

    Args:
        wait_minutes: Estimated wait time
        time_pressure: "rushed" / "normal" / "relaxed"
        hunger: Agent hunger level (0-100)
    """
    if time_pressure == "rushed" or wait_minutes > 20:
        return False
    if time_pressure == "relaxed" and wait_minutes <= 15:
        return True
    # Normal pressure + moderate wait → coin flip weighted by hunger
    # Higher hunger → less willing to wait
    wait_prob = 0.5 - (hunger - 50) * 0.005  # hunger 80 → prob 0.35
    return random.random() < max(0.2, min(0.8, wait_prob))
