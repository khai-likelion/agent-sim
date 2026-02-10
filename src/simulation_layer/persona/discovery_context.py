"""
Discovery context: Determines how agents find stores and why they visit.
Two axes of prompt branching:
1. Discovery channel (네이버검색, 카카오지도, 길거리발견, SNS추천, 지인추천)
2. Visit purpose (맛집탐방, 회식, 혼밥, 데이트, 간편식사, 카페/디저트)
"""

import random
from dataclasses import dataclass
from typing import Dict


@dataclass
class DiscoveryContext:
    """Encapsulates how an agent discovered/chose a destination."""
    channel: str       # Discovery channel
    purpose: str       # Visit purpose
    context_text: str  # Natural language description for prompt injection


# Channel descriptions for prompt injection
CHANNEL_TEXTS: Dict[str, str] = {
    "네이버검색": "네이버에서 검색해서 찾은 맛집 정보를 바탕으로 방문합니다. 블로그 리뷰와 평점을 많이 참고합니다.",
    "카카오지도": "카카오맵에서 주변 맛집을 검색했습니다. 거리와 영업시간, 평점이 중요합니다.",
    "길거리발견": "길을 걸으며 눈에 띈 가게에 관심을 갖습니다. 외관, 줄 서있는 사람, 냄새에 영향을 받습니다.",
    "SNS추천": "인스타그램/유튜브에서 본 맛집을 방문합니다. 비주얼과 트렌디함이 중요합니다.",
    "지인추천": "친구나 지인이 추천한 곳을 방문합니다. 믿을 수 있는 추천이라 기대가 높습니다.",
}

# Purpose descriptions for prompt injection
PURPOSE_TEXTS: Dict[str, str] = {
    "맛집탐방": "새로운 맛집을 경험하는 것이 목적입니다. 맛과 분위기를 중시합니다.",
    "회식": "동료와의 회식입니다. 단체석, 메뉴 다양성, 주류가 중요합니다.",
    "혼밥": "혼자 간편하게 식사합니다. 1인석, 빠른 서비스, 부담없는 분위기가 중요합니다.",
    "데이트": "연인과의 식사입니다. 분위기, 인테리어, 맛이 모두 중요합니다.",
    "간편식사": "빠르고 간단하게 끼니를 해결하고 싶습니다. 속도와 접근성이 우선입니다.",
    "카페/디저트": "커피나 디저트를 즐기려 합니다. 공간 분위기, 디저트 퀄리티가 중요합니다.",
}


def assign_discovery_channel(segment: str, lifestyle: str, time_slot: str) -> str:
    """Determine how the agent discovered stores, based on archetype.

    변화추구 agents use more online channels.
    단조로운패턴 agents rely on walking and word-of-mouth.
    Floating population uses map apps more.
    """
    if lifestyle == "변화추구":
        weights = {
            "네이버검색": 0.30, "카카오지도": 0.20,
            "SNS추천": 0.25, "길거리발견": 0.15, "지인추천": 0.10,
        }
    else:
        weights = {
            "길거리발견": 0.40, "지인추천": 0.20,
            "네이버검색": 0.15, "카카오지도": 0.15, "SNS추천": 0.10,
        }

    # Floating population more likely to use maps
    if segment in ["데이트커플", "약속모임", "망원유입직장인", "혼자방문"]:
        weights["카카오지도"] = weights.get("카카오지도", 0.15) + 0.15
        weights["길거리발견"] = max(0.05, weights.get("길거리발견", 0.15) - 0.10)

    # Night time → more planned (map/search), less walk-by
    if time_slot == "야간":
        weights["카카오지도"] = weights.get("카카오지도", 0.15) + 0.10
        weights["길거리발견"] = max(0.05, weights.get("길거리발견", 0.15) - 0.10)

    channels = list(weights.keys())
    w = [max(0, weights[c]) for c in channels]
    return random.choices(channels, weights=w, k=1)[0]


def assign_visit_purpose(segment: str, companion: str, time_slot: str) -> str:
    """Determine the purpose of the visit based on context.

    Uses segment + companion + time to infer purpose.
    """
    # Direct mappings from segment + companion
    if segment == "데이트커플" and companion == "연인":
        return "데이트"
    if segment == "약속모임" and companion in ["친구/동료", "동료"]:
        return random.choice(["회식", "맛집탐방"])
    if segment == "1인가구" and companion == "혼자":
        return random.choice(["혼밥", "간편식사", "맛집탐방"])
    if segment == "혼자방문" and companion == "혼자":
        return random.choice(["맛집탐방", "혼밥", "카페/디저트"])
    if segment == "4인가구" and companion == "가족":
        return "간편식사"

    # Time-based defaults
    if time_slot == "아침":
        return "간편식사"
    if time_slot == "야간":
        return random.choice(["회식", "카페/디저트", "맛집탐방"])

    # Companion-based fallbacks
    if companion == "혼자":
        return random.choice(["혼밥", "간편식사", "맛집탐방"])
    if companion in ["동료"]:
        return random.choice(["회식", "간편식사"])
    if companion in ["연인", "파트너"]:
        return "데이트"
    if companion in ["가족"]:
        return "간편식사"
    if companion in ["친구/동료"]:
        return random.choice(["회식", "맛집탐방"])

    return "맛집탐방"


def build_discovery_context(channel: str, purpose: str) -> DiscoveryContext:
    """Build natural language context for prompt injection."""
    channel_text = CHANNEL_TEXTS.get(channel, "")
    purpose_text = PURPOSE_TEXTS.get(purpose, "")

    context_text = ""
    if channel_text:
        context_text += f"[발견경로: {channel}] {channel_text}"
    if purpose_text:
        if context_text:
            context_text += "\n"
        context_text += f"[방문목적: {purpose}] {purpose_text}"

    return DiscoveryContext(
        channel=channel,
        purpose=purpose,
        context_text=context_text,
    )
