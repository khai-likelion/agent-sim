"""
Statistics-based spawn point selection for agents.
Replaces random spawning with POI-weighted location assignment
based on agent segment type (resident vs floating population).
"""

import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class SpawnPOI:
    """A point of interest used as agent spawn location."""
    name: str
    lat: float
    lng: float
    poi_type: str  # "transit", "market", "residential", "commercial"


# Mangwon-dong major POI coordinates (hardcoded from actual locations)
MANGWON_POIS: List[SpawnPOI] = [
    # Transit hubs
    SpawnPOI("망원역 1번출구", 37.5564, 126.9104, "transit"),
    SpawnPOI("망원역 2번출구", 37.5560, 126.9098, "transit"),
    SpawnPOI("합정역 1번출구", 37.5500, 126.9140, "transit"),
    SpawnPOI("합정역 7번출구", 37.5497, 126.9120, "transit"),
    SpawnPOI("망원동 버스정류장(망원역)", 37.5558, 126.9110, "transit"),
    SpawnPOI("망원동 버스정류장(월드컵로)", 37.5570, 126.9085, "transit"),
    # Markets
    SpawnPOI("망원시장 입구", 37.5565, 126.9065, "market"),
    SpawnPOI("망원시장 중앙", 37.5568, 126.9050, "market"),
    # Residential areas
    SpawnPOI("주거밀집A(망원동 주택가)", 37.5545, 126.9070, "residential"),
    SpawnPOI("주거밀집B(성산동 방향)", 37.5555, 126.9050, "residential"),
    SpawnPOI("주거밀집C(합정 방향)", 37.5530, 126.9100, "residential"),
    # Commercial areas
    SpawnPOI("망리단길 입구", 37.5555, 126.9090, "commercial"),
]


# Segment -> POI type spawn probability weights
SEGMENT_SPAWN_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Residents (상주): mostly residential
    "1인가구":           {"residential": 0.70, "market": 0.15, "transit": 0.10, "commercial": 0.05},
    "2인가구":           {"residential": 0.65, "market": 0.10, "transit": 0.10, "commercial": 0.15},
    "4인가구":           {"residential": 0.75, "market": 0.15, "transit": 0.05, "commercial": 0.05},
    "외부출퇴근직장인":   {"transit": 0.60, "residential": 0.25, "market": 0.05, "commercial": 0.10},
    # Floating (유동): mostly transit/commercial
    "데이트커플":         {"transit": 0.50, "commercial": 0.30, "market": 0.15, "residential": 0.05},
    "약속모임":           {"transit": 0.55, "commercial": 0.25, "market": 0.10, "residential": 0.10},
    "망원유입직장인":     {"transit": 0.65, "commercial": 0.20, "market": 0.10, "residential": 0.05},
    "혼자방문":           {"transit": 0.40, "commercial": 0.30, "market": 0.20, "residential": 0.10},
}


class SpawnPointSelector:
    """Select spawn locations based on agent segment type and POI distribution."""

    def __init__(self, pois: Optional[List[SpawnPOI]] = None):
        self.pois = pois or MANGWON_POIS
        self._pois_by_type: Dict[str, List[SpawnPOI]] = {}
        for poi in self.pois:
            self._pois_by_type.setdefault(poi.poi_type, []).append(poi)

    def get_spawn_location(self, segment: str) -> Tuple[float, float]:
        """Return (lat, lng) spawn point based on segment type.

        Uses weighted random to select POI type, then picks a random POI
        of that type, adding Gaussian jitter (~50m) for spatial diversity.
        """
        weights = SEGMENT_SPAWN_WEIGHTS.get(
            segment, {"transit": 0.4, "residential": 0.3, "market": 0.15, "commercial": 0.15}
        )

        poi_types = list(weights.keys())
        type_weights = [weights[t] for t in poi_types]
        chosen_type = random.choices(poi_types, weights=type_weights, k=1)[0]

        candidates = self._pois_by_type.get(chosen_type, self.pois)
        chosen_poi = random.choice(candidates)

        # Add Gaussian jitter (~50m radius)
        lat_jitter = random.gauss(0, 0.0003)  # ~33m at this latitude
        lng_jitter = random.gauss(0, 0.0004)

        return chosen_poi.lat + lat_jitter, chosen_poi.lng + lng_jitter
