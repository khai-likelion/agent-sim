"""
Population-statistics-based agent generator.
Creates realistic consumer personas reflecting Mangwon-dong demographics.
"""

import random
from typing import List, Tuple

from config import get_settings
from src.data_layer.population_stats import PopulationStatistics
from src.simulation_layer.persona.agent_persona import AgentPersona


class AgentGenerator:
    """Generates agent personas weighted by real population statistics."""

    SURNAMES = [
        "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임",
        "오", "한", "신", "서", "권", "황", "안", "송", "류", "홍",
    ]
    MALE_NAMES = [
        "민준", "서준", "예준", "도윤", "시우", "주원", "하준", "지호", "준서", "건우",
        "우진", "현우", "선우", "연우", "유준", "정우", "승우", "승현", "시윤", "준혁",
    ]
    FEMALE_NAMES = [
        "서연", "하은", "민서", "지우", "서현", "수아", "지아", "윤서", "채원", "지유",
        "예은", "소율", "다은", "예린", "시은", "하윤", "민지", "유나", "예원", "수빈",
    ]

    def __init__(self, stats: PopulationStatistics):
        self.stats = stats

    def _generate_name(self, gender: str) -> str:
        surname = random.choice(self.SURNAMES)
        if gender == "남성":
            given_name = random.choice(self.MALE_NAMES)
        else:
            given_name = random.choice(self.FEMALE_NAMES)
        return f"{surname}{given_name}"

    def _age_group_to_age(self, age_group: str) -> int:
        mapping = {
            "10대": (15, 19),
            "20대": (20, 29),
            "30대": (30, 39),
            "40대": (40, 49),
            "50대": (50, 59),
            "60대+": (60, 75),
        }
        low, high = mapping.get(age_group, (25, 35))
        return random.randint(low, high)

    def _generate_occupation(self, age: int, gender: str) -> str:
        if age < 20:
            return random.choice(["고등학생", "대학생(신입)"])
        elif age < 25:
            return random.choice(["대학생", "취준생", "인턴", "신입사원"])
        elif age < 35:
            return random.choice(
                ["회사원", "IT 개발자", "디자이너", "마케터", "프리랜서", "스타트업 직원"]
            )
        elif age < 45:
            return random.choice(
                ["회사원(중간관리자)", "자영업자", "전문직", "공무원", "교사"]
            )
        elif age < 60:
            return random.choice(
                ["회사원(관리직)", "자영업자", "전문직", "공무원", "주부"]
            )
        else:
            return random.choice(["은퇴자", "자영업자", "주부", "프리랜서"])

    def _generate_income_level(self, age: int, occupation: str) -> str:
        if "학생" in occupation or "취준생" in occupation or "인턴" in occupation:
            return random.choice(["하", "중하"])
        elif "신입" in occupation:
            return random.choice(["중하", "중"])
        elif "관리직" in occupation or "전문직" in occupation:
            return random.choice(["중상", "상"])
        elif "자영업" in occupation:
            return random.choice(["중", "중상"])
        elif "은퇴자" in occupation or "주부" in occupation:
            return random.choice(["중하", "중"])
        else:
            return random.choice(["중하", "중", "중상"])

    def _generate_value_preference(
        self, age: int, gender: str, income_level: str
    ) -> str:
        if age < 30:
            if gender == "여성":
                return "트렌디한 카페, SNS 핫플, 분위기 좋은 식당 선호. 인스타그램 감성 중시."
            else:
                return "가성비 좋은 식당, 새로운 경험 추구. 친구들과 함께할 수 있는 공간 선호."
        elif age < 40:
            if income_level in ["상", "중상"]:
                return "품질 좋은 브런치 카페, 퀄리티 있는 레스토랑 선호. 시간 효율 중시."
            else:
                return "가성비 식당, 직장 근처 편의성 중시. 점심/저녁 빠른 식사 선호."
        elif age < 50:
            return "가족 단위 식사, 익숙한 맛집 선호. 주차 편의성과 메뉴 다양성 중시."
        elif age < 65:
            return "건강식, 조용한 분위기, 친절한 서비스 중시. 단골 가게 선호."
        else:
            return "전통시장, 익숙한 노포, 저렴하고 푸짐한 식당 선호. 건강과 경제성 중시."

    def _generate_store_preferences(self, age: int, gender: str) -> List[str]:
        if age < 30:
            if gender == "여성":
                return ["카페", "디저트카페", "브런치", "이탈리안", "일식", "베이커리"]
            else:
                return ["한식", "일식", "중식", "양식", "햄버거", "치킨", "호프"]
        elif age < 40:
            return ["카페", "한식", "일식", "브런치", "베이커리", "샐러드", "돈까스"]
        elif age < 50:
            return ["한식", "중식", "일식", "패밀리레스토랑", "고기", "해물"]
        elif age < 65:
            return ["한식", "칼국수", "설렁탕", "백반", "해물", "전통음식"]
        else:
            return ["한식", "국밥", "칼국수", "순대", "전통시장", "빈대떡"]

    def _calculate_sensitivities(
        self, age: int, gender: str, income_level: str
    ) -> Tuple[float, float, float]:
        """
        Calculate price/trend/quality sensitivities.
        Returns: (price_sensitivity, trend_sensitivity, quality_preference)
        """
        price = 0.5
        trend = 0.5
        quality = 0.5

        if age < 30:
            trend, price, quality = 0.8, 0.6, 0.4
        elif age < 40:
            trend, price, quality = 0.6, 0.5, 0.6
        elif age < 50:
            trend, price, quality = 0.3, 0.5, 0.7
        elif age < 65:
            trend, price, quality = 0.2, 0.6, 0.6
        else:
            trend, price, quality = 0.1, 0.8, 0.5

        if income_level in ["상", "중상"]:
            price *= 0.7
            quality *= 1.2
        elif income_level in ["하", "중하"]:
            price *= 1.3
            quality *= 0.8

        if gender == "여성":
            trend *= 1.1

        price = min(1.0, max(0.0, price))
        trend = min(1.0, max(0.0, trend))
        quality = min(1.0, max(0.0, quality))

        return price, trend, quality

    def generate_agent(self, agent_id: int) -> AgentPersona:
        """Generate a single agent persona."""
        age_group = self.stats.get_random_age_group()
        gender = self.stats.get_random_gender()
        age = self._age_group_to_age(age_group)
        name = self._generate_name(gender)
        occupation = self._generate_occupation(age, gender)
        income_level = self._generate_income_level(age, occupation)
        value_preference = self._generate_value_preference(age, gender, income_level)
        store_preferences = self._generate_store_preferences(age, gender)
        price_sens, trend_sens, quality_pref = self._calculate_sensitivities(
            age, gender, income_level
        )

        return AgentPersona(
            id=agent_id,
            name=name,
            age=age,
            age_group=age_group,
            gender=gender,
            occupation=occupation,
            income_level=income_level,
            value_preference=value_preference,
            store_preferences=store_preferences,
            price_sensitivity=price_sens,
            trend_sensitivity=trend_sens,
            quality_preference=quality_pref,
        )

    def generate_agents(self, num_agents: int | None = None) -> List[AgentPersona]:
        """Generate multiple agent personas."""
        settings = get_settings()
        num_agents = num_agents or settings.simulation.agent_count
        return [self.generate_agent(i) for i in range(1, num_agents + 1)]
