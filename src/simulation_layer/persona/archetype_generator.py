"""
64-Archetype based agent generator.
8 segments × 4 taste preferences × 2 lifestyles = 64 fixed archetypes.

Segments (8):
- Resident: 1인가구, 2인가구, 4인가구, 외부출퇴근직장인
- Floating: 데이트커플, 약속모임, 망원유입직장인, 혼자방문

Taste preferences (4) - 미각 성향:
- 자극선호: 맵고 자극적인 맛, 강한 양념, 매운맛 도전
- 담백건강: 건강하고 담백한 맛, 재료 본연의 맛
- 미식탐구: 새로운 맛 경험, 고급 음식, 오마카세/파인다이닝
- 편안한맛: 익숙하고 편안한 맛, 한식, 국밥류, 어머니 손맛

Lifestyles (2):
- 단조로운패턴: 익숙한 곳 반복 방문
- 변화추구: 새로운 곳 탐험 선호
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
from src.simulation_layer.persona.agent_persona import AgentPersona


@dataclass
class Archetype:
    """Defines a single archetype with its characteristics."""
    segment: str
    taste: str
    lifestyle: str

    # Derived attributes
    weight: float  # Population weight for result aggregation

    # Demographics
    age_range: Tuple[int, int]
    gender_ratio: float  # 0.5 = balanced, >0.5 = more female
    income_level: str

    # Preferences
    store_preferences: List[str]
    price_sensitivity: float
    trend_sensitivity: float
    quality_preference: float

    @property
    def id(self) -> str:
        return f"{self.segment}_{self.taste}_{self.lifestyle}"

    @property
    def description(self) -> str:
        return f"{self.segment} / {self.taste} / {self.lifestyle}"


# 8 Segments definition
SEGMENTS = {
    # 상주인구 (Residents)
    "1인가구": {
        "type": "resident",
        "age_range": (25, 40),
        "gender_ratio": 0.5,
        "income_level": "중",
        "base_weight": 0.15,
        "description": "망원동 거주 1인가구. 혼밥/혼술 문화에 익숙하고 배달앱 사용 빈도 높음.",
    },
    "2인가구": {
        "type": "resident",
        "age_range": (28, 45),
        "gender_ratio": 0.5,
        "income_level": "중상",
        "base_weight": 0.12,
        "description": "신혼부부 또는 동거커플. 주말 브런치, 분위기 좋은 식당 선호.",
    },
    "4인가구": {
        "type": "resident",
        "age_range": (35, 50),
        "gender_ratio": 0.5,
        "income_level": "중상",
        "base_weight": 0.10,
        "description": "자녀가 있는 가정. 가족 외식, 아이 메뉴 있는 식당 선호.",
    },
    "외부출퇴근직장인": {
        "type": "resident",
        "age_range": (25, 45),
        "gender_ratio": 0.5,
        "income_level": "중",
        "base_weight": 0.13,
        "description": "망원동 거주, 타 지역 출퇴근. 평일 저녁/주말에 동네 상권 이용.",
    },
    # 유동인구 (Floating)
    "데이트커플": {
        "type": "floating",
        "age_range": (22, 35),
        "gender_ratio": 0.5,
        "income_level": "중",
        "base_weight": 0.15,
        "description": "연인과 함께 망원동 방문. 분위기, SNS 감성, 디저트 카페 선호.",
    },
    "약속모임": {
        "type": "floating",
        "age_range": (25, 40),
        "gender_ratio": 0.5,
        "income_level": "중",
        "base_weight": 0.12,
        "description": "친구/동료와 약속. 단체석, 2차 갈 수 있는 동선 고려.",
    },
    "망원유입직장인": {
        "type": "floating",
        "age_range": (25, 40),
        "gender_ratio": 0.45,
        "income_level": "중",
        "base_weight": 0.13,
        "description": "망원동 인근 직장인. 점심/저녁 식사 위해 유입. 빠른 식사 선호.",
    },
    "혼자방문": {
        "type": "floating",
        "age_range": (20, 45),
        "gender_ratio": 0.55,
        "income_level": "중",
        "base_weight": 0.10,
        "description": "혼자 망원동 방문. 카페, 서점, 혼밥 가능한 식당 선호.",
    },
}

# 4 Taste preferences (미각 성향)
TASTES = {
    "자극선호": {
        "store_preferences": ["마라탕", "떡볶이", "닭발", "곱창", "매운갈비찜", "불닭", "짬뽕", "마라샹궈", "엽떡", "신전떡볶이"],
        "price_sensitivity": 0.5,
        "trend_sensitivity": 0.7,
        "quality_preference": 0.5,
        "weight_modifier": 0.25,
        "description": "맵고 자극적인 맛 선호. 강한 양념과 매운맛 도전을 즐김. 얼얼한 마라, 화끈한 매운맛에 끌림.",
    },
    "담백건강": {
        "store_preferences": ["샐러드", "포케", "비건", "일식", "칼국수", "설렁탕", "죽", "두부요리", "쌈밥", "현미밥"],
        "price_sensitivity": 0.4,
        "trend_sensitivity": 0.5,
        "quality_preference": 0.8,
        "weight_modifier": 0.25,
        "description": "건강하고 담백한 맛 선호. 재료 본연의 맛을 중시. 자극적이지 않고 깔끔한 음식 선호.",
    },
    "미식탐구": {
        "store_preferences": ["오마카세", "파인다이닝", "와인바", "이탈리안", "프렌치", "스시", "스테이크", "퓨전요리", "코스요리"],
        "price_sensitivity": 0.2,
        "trend_sensitivity": 0.9,
        "quality_preference": 0.9,
        "weight_modifier": 0.20,
        "description": "새로운 맛 경험 추구. 고급 음식과 셰프의 요리에 관심. 맛집 탐방과 미식 경험을 즐김.",
    },
    "편안한맛": {
        "store_preferences": ["국밥", "백반", "김치찌개", "된장찌개", "순대국", "설렁탕", "칼국수", "비빔밥", "제육볶음", "돈까스"],
        "price_sensitivity": 0.6,
        "trend_sensitivity": 0.3,
        "quality_preference": 0.5,
        "weight_modifier": 0.30,
        "description": "익숙하고 편안한 맛 선호. 어머니 손맛 같은 한식, 푸근한 국물요리. 모험보다 안정적인 맛 선택.",
    },
}

# 2 Lifestyles
LIFESTYLES = {
    "단조로운패턴": {
        "trend_modifier": -0.2,
        "loyalty_bonus": 0.3,
        "weight_modifier": 0.55,
        "description": "익숙한 단골집 반복 방문. 새로운 곳 시도에 소극적.",
    },
    "변화추구": {
        "trend_modifier": 0.2,
        "loyalty_bonus": -0.2,
        "weight_modifier": 0.45,
        "description": "새로운 맛집 탐방 즐김. SNS 핫플 민감. 리뷰 적극 참고.",
    },
}


class ArchetypeGenerator:
    """Generates 64 fixed archetypes with population weights (8 segments × 4 tastes × 2 lifestyles)."""

    def __init__(self):
        self.archetypes: List[Archetype] = []
        self._generate_all_archetypes()

    def _generate_all_archetypes(self):
        """Generate all 64 archetype combinations (8 × 4 × 2)."""
        archetype_id = 0

        for seg_name, seg_data in SEGMENTS.items():
            for taste_name, taste_data in TASTES.items():
                for life_name, life_data in LIFESTYLES.items():
                    # Calculate combined weight
                    weight = (
                        seg_data["base_weight"]
                        * taste_data["weight_modifier"]
                        * life_data["weight_modifier"]
                    )

                    # Calculate sensitivities with lifestyle modifier
                    trend_sens = min(1.0, max(0.0,
                        taste_data["trend_sensitivity"] + life_data["trend_modifier"]
                    ))

                    archetype = Archetype(
                        segment=seg_name,
                        taste=taste_name,
                        lifestyle=life_name,
                        weight=weight,
                        age_range=seg_data["age_range"],
                        gender_ratio=seg_data["gender_ratio"],
                        income_level=seg_data["income_level"],
                        store_preferences=taste_data["store_preferences"],
                        price_sensitivity=taste_data["price_sensitivity"],
                        trend_sensitivity=trend_sens,
                        quality_preference=taste_data["quality_preference"],
                    )
                    self.archetypes.append(archetype)
                    archetype_id += 1

        # Normalize weights to sum to 1
        total_weight = sum(a.weight for a in self.archetypes)
        for a in self.archetypes:
            a.weight = a.weight / total_weight

    def get_archetype(self, segment: str, taste: str, lifestyle: str) -> Archetype:
        """Get a specific archetype by its components."""
        for a in self.archetypes:
            if a.segment == segment and a.taste == taste and a.lifestyle == lifestyle:
                return a
        raise ValueError(f"Archetype not found: {segment}/{taste}/{lifestyle}")

    def generate_agents(self) -> List[AgentPersona]:
        """Generate one agent per archetype (64 agents total)."""
        import random

        agents = []
        for idx, archetype in enumerate(self.archetypes):
            # Generate demographics based on archetype
            age = random.randint(*archetype.age_range)
            gender = "여성" if random.random() < archetype.gender_ratio else "남성"

            # Generate name
            name = self._generate_name(gender, archetype.segment)

            # Build value preference description
            seg_desc = SEGMENTS[archetype.segment]["description"]
            taste_desc = TASTES[archetype.taste]["description"]
            life_desc = LIFESTYLES[archetype.lifestyle]["description"]
            value_pref = f"{seg_desc} {taste_desc} {life_desc}"

            # Determine occupation based on segment
            occupation = self._get_occupation(archetype.segment, age)

            # Determine household type
            household_map = {
                "1인가구": "1인가구",
                "2인가구": "2세대가구",
                "4인가구": "2세대가구",
                "외부출퇴근직장인": "1인가구",
                "데이트커플": None,
                "약속모임": None,
                "망원유입직장인": None,
                "혼자방문": "1인가구",
            }

            agent = AgentPersona(
                id=idx + 1,
                name=name,
                age=age,
                age_group=self._age_to_group(age),
                gender=gender,
                occupation=occupation,
                income_level=archetype.income_level,
                value_preference=value_pref,
                store_preferences=archetype.store_preferences.copy(),
                price_sensitivity=archetype.price_sensitivity,
                trend_sensitivity=archetype.trend_sensitivity,
                quality_preference=archetype.quality_preference,
                household_type=household_map.get(archetype.segment),
            )

            # Attach archetype info for weight-based aggregation
            agent._archetype = archetype

            agents.append(agent)

        return agents

    def _generate_name(self, gender: str, segment: str) -> str:
        """Generate a representative name for the archetype."""
        import random

        surnames = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임"]
        male_names = ["민준", "서준", "예준", "도윤", "시우", "주원", "하준", "지호"]
        female_names = ["서연", "하은", "민서", "지우", "서현", "수아", "지아", "윤서"]

        surname = random.choice(surnames)
        given = random.choice(female_names if gender == "여성" else male_names)

        # Add segment suffix for clarity
        segment_short = {
            "1인가구": "1인",
            "2인가구": "2인",
            "4인가구": "4인",
            "외부출퇴근직장인": "출퇴근",
            "데이트커플": "데이트",
            "약속모임": "약속",
            "망원유입직장인": "유입",
            "혼자방문": "혼자",
        }

        return f"{surname}{given}({segment_short.get(segment, segment)[:2]})"

    def _age_to_group(self, age: int) -> str:
        if age < 20:
            return "10대"
        elif age < 30:
            return "20대"
        elif age < 40:
            return "30대"
        elif age < 50:
            return "40대"
        elif age < 60:
            return "50대"
        else:
            return "60대+"

    def _get_occupation(self, segment: str, age: int) -> str:
        import random

        occupation_map = {
            "1인가구": ["회사원", "프리랜서", "자영업자", "대학원생"],
            "2인가구": ["회사원", "전문직", "공무원", "디자이너"],
            "4인가구": ["회사원(관리직)", "자영업자", "전문직", "공무원"],
            "외부출퇴근직장인": ["회사원", "IT개발자", "마케터", "영업직"],
            "데이트커플": ["회사원", "대학생", "프리랜서", "스타트업"],
            "약속모임": ["회사원", "프리랜서", "자영업자", "전문직"],
            "망원유입직장인": ["회사원", "스타트업", "IT개발자", "디자이너"],
            "혼자방문": ["프리랜서", "작가", "대학원생", "회사원"],
        }

        return random.choice(occupation_map.get(segment, ["회사원"]))

    def get_weight_dict(self) -> Dict[str, float]:
        """Return archetype ID -> weight mapping for result aggregation."""
        return {a.id: a.weight for a in self.archetypes}

    def print_summary(self):
        """Print archetype summary."""
        print(f"Total archetypes: {len(self.archetypes)}")
        print(f"Total weight: {sum(a.weight for a in self.archetypes):.4f}")
        print()

        # Group by segment
        for seg_name in SEGMENTS:
            seg_archetypes = [a for a in self.archetypes if a.segment == seg_name]
            seg_weight = sum(a.weight for a in seg_archetypes)
            print(f"{seg_name}: {len(seg_archetypes)} archetypes, weight={seg_weight:.3f}")
