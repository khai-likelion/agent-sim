"""
Mangwon-dong specific scenario setup.
Default business reports and simulation parameters.
"""

from datetime import datetime
from typing import List

from src.simulation_layer.models import BusinessReport


def create_default_reports() -> List[BusinessReport]:
    """The three sample business reports for the Mangwon-dong scenario."""
    return [
        BusinessReport(
            store_name="맥도날드 망원점",
            report_type="discount",
            description="빅맥 세트 20% 할인 이벤트",
            target_age_groups=["10대", "20대", "30대"],
            appeal_factor="price",
            appeal_strength=0.8,
        ),
        BusinessReport(
            store_name="몬스터스토리지 메종망원점",
            report_type="new_menu",
            description="인스타그램 화제! 신메뉴 '딸기 크로플' 출시",
            target_age_groups=["20대", "30대"],
            appeal_factor="trend",
            appeal_strength=0.9,
        ),
        BusinessReport(
            store_name="망원시장손칼국수",
            report_type="event",
            description="개업 20주년 기념 특별 서비스",
            target_age_groups=["40대", "50대", "60대+"],
            appeal_factor="quality",
            appeal_strength=0.7,
        ),
    ]


def get_default_start_date() -> datetime:
    """Default simulation start: Monday 06:00."""
    return datetime(2025, 2, 3, 6, 0)
