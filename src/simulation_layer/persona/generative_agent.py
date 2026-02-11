"""
Generative Agent Persona - Stanford Generative Agents 논문 기반 구조.

에이전트 분류:
1. 개인 단위 세그먼트 (성별/나이 포함):
   - 상주_1인가구, 상주_외부출퇴근, 유동_망원유입직장인, 유동_나홀로방문

2. 그룹 단위 세그먼트 (세대 중심, 성별/나이 제외):
   - 상주_2인가구, 상주_4인가구, 유동_데이트, 유동_약속모임
   - 개인보다 그룹/가구 단위의 소비 행태가 더 중요
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
import random


# 세대 정의 (Generation)
GENERATIONS = {
    "Alpha": {
        "birth_range": (2010, 2025),
        "age_range": (0, 15),
        "description": "디지털 네이티브, 태블릿과 함께 성장",
        "group_traits": {
            "상주_2인가구": "젊은 부모와 함께하는 2인가구, 키즈메뉴/분위기 중시",
            "상주_4인가구": "아이 중심의 가족 외식, 놀이시설 있는 식당 선호",
            "유동_데이트": "10대 커플, 저렴하고 인스타그래머블한 장소 선호",
            "유동_약속모임": "친구 모임, 디저트카페/분식 중심",
        },
    },
    "Z": {
        "birth_range": (1997, 2009),
        "age_range": (16, 28),
        "description": "SNS 세대, 트렌드에 민감, 경험 중시",
        "group_traits": {
            "상주_2인가구": "연인/룸메이트와 동거, SNS 감성 맛집 탐방",
            "상주_4인가구": "형제자매 포함 가족, 가성비+분위기 모두 중시",
            "유동_데이트": "인스타 핫플, 디저트카페, 이색 데이트 선호",
            "유동_약속모임": "대학생/직장인 모임, 2차 문화, 가성비 중시",
        },
    },
    "Y": {
        "birth_range": (1981, 1996),
        "age_range": (29, 44),
        "description": "밀레니얼, 가성비와 경험 모두 중시",
        "group_traits": {
            "상주_2인가구": "신혼부부/딩크족, 주말 브런치, 와인바 선호",
            "상주_4인가구": "어린 자녀 동반 가족, 키즈존/패밀리레스토랑 선호",
            "유동_데이트": "분위기 좋은 레스토랑, 오마카세, 경험 중시",
            "유동_약속모임": "동창모임/직장 회식, 단체석/프라이빗룸 선호",
        },
    },
    "X": {
        "birth_range": (1965, 1980),
        "age_range": (45, 60),
        "description": "실용적, 가족 중심, 품질 중시",
        "group_traits": {
            "상주_2인가구": "자녀 독립 후 부부, 건강식/한정식 선호",
            "상주_4인가구": "중고등학생 자녀 가족, 든든한 한식/고기집 선호",
            "유동_데이트": "부부 외출, 한정식/일식 등 고급 식당 선호",
            "유동_약속모임": "동창회/동호회, 한정식/고기집/룸 있는 곳 선호",
        },
    },
    "BB": {
        "birth_range": (1946, 1964),
        "age_range": (61, 79),
        "description": "베이비붐, 건강 관심, 전통 선호",
        "group_traits": {
            "상주_2인가구": "노부부, 건강식/전통 한식 선호, 조용한 분위기",
            "상주_4인가구": "3세대 가족 외식, 한정식/뷔페 선호",
            "유동_데이트": "노년 커플, 산책 후 식사, 전통 맛집 선호",
            "유동_약속모임": "동창회/경조사, 한정식/고급 한식 선호",
        },
    },
    "S": {
        "birth_range": (1928, 1945),
        "age_range": (80, 97),
        "description": "시니어, 익숙한 맛 선호, 건강 우선",
        "group_traits": {
            "상주_2인가구": "노부부, 익숙한 동네 식당, 자극적이지 않은 음식",
            "상주_4인가구": "3세대 가족 모임, 한정식/백반 선호",
            "유동_데이트": "노년 커플, 단골 식당, 추억의 맛집",
            "유동_약속모임": "경조사/명절 모임, 한정식/예식장 식당",
        },
    },
}

# 그룹 단위 세그먼트 (성별/나이 제외)
GROUP_BASED_SEGMENTS = ["상주_2인가구", "상주_4인가구", "유동_데이트", "유동_약속모임"]

# 개인 단위 세그먼트 (성별/나이 포함)
INDIVIDUAL_SEGMENTS = ["상주_1인가구", "상주_외부출퇴근", "유동_망원유입직장인", "유동_나홀로방문"]

# 세그먼트 정의 (8종)
SEGMENTS = {
    # === 개인 단위 세그먼트 (성별/나이 포함) ===
    "상주_1인가구": {
        "type": "resident",
        "persona_type": "individual",
        "description": "혼밥/혼술 문화에 익숙, 배달앱 사용 빈도 높음, 간편식 선호",
        "meal_frequency": {"아침": 0.3, "점심": 0.7, "저녁": 0.8, "야간": 0.4},
    },
    "상주_외부출퇴근": {
        "type": "resident",
        "persona_type": "individual",
        "description": "망원동 거주, 타 지역 출퇴근, 평일 저녁/주말에 동네 상권 이용",
        "meal_frequency": {"아침": 0.2, "점심": 0.2, "저녁": 0.6, "야간": 0.3},
    },
    "유동_망원유입직장인": {
        "type": "floating",
        "persona_type": "individual",
        "description": "망원동 인근 직장인, 점심/저녁 식사 위해 유입, 빠른 식사 선호",
        "meal_frequency": {"아침": 0.1, "점심": 0.9, "저녁": 0.7, "야간": 0.2},
    },
    "유동_나홀로방문": {
        "type": "floating",
        "persona_type": "individual",
        "description": "혼자 방문, 카페/서점/혼밥 가능 식당 선호",
        "meal_frequency": {"아침": 0.2, "점심": 0.6, "저녁": 0.5, "야간": 0.3},
    },
    # === 그룹 단위 세그먼트 (세대 중심, 성별/나이 제외) ===
    "상주_2인가구": {
        "type": "resident",
        "persona_type": "group",
        "description": "2인 가구(커플/부부/룸메이트), 함께하는 식사 문화, 분위기 중시",
        "meal_frequency": {"아침": 0.4, "점심": 0.6, "저녁": 0.9, "야간": 0.3},
    },
    "상주_4인가구": {
        "type": "resident",
        "persona_type": "group",
        "description": "가족 단위 외식, 아이 메뉴/넓은 좌석 있는 식당 선호",
        "meal_frequency": {"아침": 0.2, "점심": 0.5, "저녁": 0.7, "야간": 0.1},
    },
    "유동_데이트": {
        "type": "floating",
        "persona_type": "group",
        "description": "커플 데이트, 분위기/SNS 감성/디저트 카페 선호",
        "meal_frequency": {"아침": 0.3, "점심": 0.8, "저녁": 0.9, "야간": 0.5},
    },
    "유동_약속모임": {
        "type": "floating",
        "persona_type": "group",
        "description": "친구/동료 모임, 단체석/2차 동선 고려, 분위기 중시",
        "meal_frequency": {"아침": 0.1, "점심": 0.7, "저녁": 0.9, "야간": 0.6},
    },
}

# 건강 추구 성향
HEALTH_PREFERENCES = {
    "건강선호": {
        "preferred_categories": ["샐러드", "포케", "칼국수", "설렁탕", "국밥", "비건", "일식", "죽"],
        "avoided_categories": ["마라탕", "닭발", "곱창", "치킨", "피자", "햄버거"],
        "description": "건강한 음식 선호, 자극적인 음식 기피",
    },
    "자극선호": {
        "preferred_categories": ["마라탕", "떡볶이", "닭발", "곱창", "매운갈비찜", "짬뽕", "불닭"],
        "avoided_categories": ["샐러드", "죽", "비건"],
        "description": "맵고 자극적인 맛 선호, 강한 양념",
    },
}

# 변화 추구 성향
CHANGE_PREFERENCES = {
    "안정추구": {
        "new_store_probability": 0.2,
        "repeat_bonus": 0.3,
        "description": "기존에 가던 매장 반복 방문 선호",
    },
    "도전추구": {
        "new_store_probability": 0.7,
        "repeat_bonus": -0.1,
        "description": "새로운 매장 탐험 선호, SNS 핫플 민감",
    },
}

# 한끼 가용비 범위 (세대별)
BUDGET_RANGES = {
    "Alpha": (5000, 10000),
    "Z": (8000, 15000),
    "Y": (10000, 20000),
    "X": (12000, 25000),
    "BB": (8000, 18000),
    "S": (6000, 12000),
}

# 그룹 세그먼트용 예산 배수 (그룹 인원 고려)
GROUP_BUDGET_MULTIPLIER = {
    "상주_2인가구": 1.8,  # 2인분
    "상주_4인가구": 3.5,  # 4인분 (가족 할인 고려)
    "유동_데이트": 2.2,   # 2인분 + 분위기 비용
    "유동_약속모임": 1.2, # 1인당 (n빵 가정)
}


@dataclass
class VisitRecord:
    """방문 기록 (Memory Module용)"""
    visit_datetime: str  # ISO format
    store_name: str
    category: str
    taste_rating: int  # 1: 매우별로, 2: 별로, 3: 보통, 4: 좋음, 5: 매우좋음
    value_rating: int  # 1: 매우별로, 2: 별로, 3: 보통, 4: 좋음, 5: 매우좋음

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerativeAgent:
    """
    Stanford Generative Agents 스타일의 에이전트.
    페르소나 + 메모리 모듈 통합.

    - 개인 단위 세그먼트: gender, age 필수
    - 그룹 단위 세그먼트: gender=None, age=None (세대 중심)
    """
    # 기본 정보
    id: int
    name: str
    generation: str  # Alpha, Z, Y, X, BB, S (핵심 변수)

    # 세그먼트
    segment: str  # 8종 중 하나

    # 생활양식
    health_preference: str  # "건강선호" or "자극선호"
    change_preference: str  # "안정추구" or "도전추구"

    # 경제적 특성
    budget_per_meal: int  # 한끼 가용비 (원)

    # 개인 정보 (개인 세그먼트만 해당, 그룹은 None)
    gender: Optional[str] = None  # "남성" or "여성" or None
    age: Optional[int] = None  # 구체적 나이 or None

    # 메모리 모듈
    recent_history: List[VisitRecord] = field(default_factory=list)

    # 위치 (시뮬레이션 중 설정)
    current_location: Optional[Any] = field(default=None, repr=False)
    home_location: Optional[Any] = field(default=None, repr=False)

    @property
    def is_group_based(self) -> bool:
        """그룹 단위 세그먼트 여부"""
        return self.segment in GROUP_BASED_SEGMENTS

    @property
    def age_group(self) -> Optional[str]:
        """연령대 반환 (개인 세그먼트만)"""
        if self.age is None:
            return None
        if self.age < 20:
            return "10대"
        elif self.age < 30:
            return "20대"
        elif self.age < 40:
            return "30대"
        elif self.age < 50:
            return "40대"
        elif self.age < 60:
            return "50대"
        else:
            return "60대+"

    @property
    def generation_description(self) -> str:
        """세대 설명"""
        return GENERATIONS[self.generation]["description"]

    @property
    def group_trait(self) -> Optional[str]:
        """그룹 세그먼트의 세대별 특성"""
        if not self.is_group_based:
            return None
        return GENERATIONS[self.generation]["group_traits"].get(self.segment, "")

    @property
    def segment_type(self) -> str:
        """상주/유동 구분"""
        return SEGMENTS[self.segment]["type"]

    @property
    def segment_description(self) -> str:
        """세그먼트 설명"""
        return SEGMENTS[self.segment]["description"]

    @property
    def health_description(self) -> str:
        """건강 추구 성향 설명"""
        return HEALTH_PREFERENCES[self.health_preference]["description"]

    @property
    def change_description(self) -> str:
        """변화 추구 성향 설명"""
        return CHANGE_PREFERENCES[self.change_preference]["description"]

    @property
    def preferred_categories(self) -> List[str]:
        """선호 카테고리"""
        return HEALTH_PREFERENCES[self.health_preference]["preferred_categories"]

    @property
    def avoided_categories(self) -> List[str]:
        """기피 카테고리"""
        return HEALTH_PREFERENCES[self.health_preference]["avoided_categories"]

    def get_meal_probability(self, time_slot: str) -> float:
        """시간대별 외식 확률"""
        # 유동인구는 외식 확률 100%
        if self.segment.startswith("유동_"):
            return 1.0
        # 상주인구는 기존 확률값 사용
        return SEGMENTS[self.segment]["meal_frequency"].get(time_slot, 0.5)

    def get_new_store_probability(self) -> float:
        """새 매장 방문 확률"""
        return CHANGE_PREFERENCES[self.change_preference]["new_store_probability"]

    def get_repeat_bonus(self) -> float:
        """재방문 보너스"""
        return CHANGE_PREFERENCES[self.change_preference]["repeat_bonus"]

    def add_visit(self, store_name: str, category: str, taste_rating: int, value_rating: int):
        """방문 기록 추가"""
        record = VisitRecord(
            visit_datetime=datetime.now().isoformat(),
            store_name=store_name,
            category=category,
            taste_rating=taste_rating,
            value_rating=value_rating,
        )
        self.recent_history.append(record)
        # 최근 15개만 유지
        if len(self.recent_history) > 15:
            self.recent_history = self.recent_history[-15:]

    def get_recent_categories(self, n: int = 5) -> List[str]:
        """최근 방문한 카테고리 목록"""
        return [v.category for v in self.recent_history[-n:]]

    def get_recent_stores(self, n: int = 5) -> List[str]:
        """최근 방문한 매장 목록"""
        return [v.store_name for v in self.recent_history[-n:]]

    def has_visited(self, store_name: str) -> bool:
        """매장 방문 여부"""
        return any(v.store_name == store_name for v in self.recent_history)

    def get_visit_count(self, store_name: str) -> int:
        """특정 매장 방문 횟수"""
        return sum(1 for v in self.recent_history if v.store_name == store_name)

    def get_memory_context(self) -> str:
        """LLM 프롬프트용 메모리 컨텍스트"""
        if not self.recent_history:
            return "최근 방문 기록이 없습니다."

        lines = ["최근 방문 기록:"]
        for v in self.recent_history[-5:]:
            rating_text = {1: "매우별로", 2: "별로", 3: "보통", 4: "좋음", 5: "매우좋음"}
            lines.append(
                f"  - {v.store_name} ({v.category}): "
                f"맛 {rating_text.get(v.taste_rating, '?')}, 가성비 {rating_text.get(v.value_rating, '?')}"
            )
        return "\n".join(lines)

    def get_persona_summary(self) -> str:
        """LLM 프롬프트용 페르소나 요약"""
        if self.is_group_based:
            # 그룹 단위 세그먼트: 세대 중심, 성별/나이 제외
            return f"""[{self.name} - {self.segment}]
- 세대: {self.generation} ({self.generation_description})
- 유형: {self.segment} ({self.segment_type})
- 그룹 특성: {self.group_trait}
- 건강 성향: {self.health_preference} - {self.health_description}
- 변화 성향: {self.change_preference} - {self.change_description}
- 1회 외식 예산: {self.budget_per_meal:,}원
- 세그먼트 특성: {self.segment_description}"""
        else:
            # 개인 단위 세그먼트: 성별/나이 포함
            return f"""[{self.name}의 프로필]
- 세대: {self.generation} ({self.age}세, {self.gender})
- 유형: {self.segment} ({self.segment_type})
- 건강 성향: {self.health_preference} - {self.health_description}
- 변화 성향: {self.change_preference} - {self.change_description}
- 한끼 가용비: {self.budget_per_meal:,}원
- 세그먼트 특성: {self.segment_description}"""

    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화용"""
        data = {
            "id": self.id,
            "name": self.name,
            "generation": self.generation,
            "segment": self.segment,
            "is_group_based": self.is_group_based,
            "health_preference": self.health_preference,
            "change_preference": self.change_preference,
            "budget_per_meal": self.budget_per_meal,
            "recent_history": [v.to_dict() for v in self.recent_history],
        }
        # 개인 세그먼트만 성별/나이 포함
        if not self.is_group_based:
            data["gender"] = self.gender
            data["age"] = self.age
            data["age_group"] = self.age_group
        return data


class GenerativeAgentFactory:
    """
    고유한 에이전트 조합 생성.

    개인 세그먼트: 세대 × 성별 × 건강성향 × 변화성향
    그룹 세그먼트: 세대 × 건강성향 × 변화성향 (성별 제외)
    """

    # 이름 풀
    SURNAMES = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한", "오", "서", "신", "권", "황", "안", "송", "류", "홍"]
    MALE_NAMES = ["민준", "서준", "예준", "도윤", "시우", "주원", "하준", "지호", "준서", "건우", "현우", "우진", "지훈", "준혁", "성민"]
    FEMALE_NAMES = ["서연", "하은", "민서", "지우", "서현", "수아", "지아", "윤서", "채원", "소율", "예은", "다은", "수빈", "지유", "예서"]
    GROUP_NAMES = ["망원", "합정", "상수", "연남", "성산", "홍대", "연희", "서교"]

    def __init__(self):
        self.used_combinations = set()
        self.agent_count = 0

    def _generate_individual_name(self, gender: str) -> str:
        """개인 세그먼트용 이름 생성"""
        surname = random.choice(self.SURNAMES)
        given = random.choice(self.FEMALE_NAMES if gender == "여성" else self.MALE_NAMES)
        return f"{surname}{given}"

    def _generate_group_name(self, segment: str, generation: str) -> str:
        """그룹 세그먼트용 이름 생성 (개인명 대신 그룹 식별자)"""
        segment_short = {
            "상주_2인가구": "2인",
            "상주_4인가구": "가족",
            "유동_데이트": "커플",
            "유동_약속모임": "모임",
        }
        prefix = random.choice(self.GROUP_NAMES)
        seg_name = segment_short.get(segment, "그룹")
        return f"{prefix}_{generation}세대_{seg_name}"

    def _get_age_for_generation(self, generation: str) -> int:
        """세대에 맞는 나이 생성"""
        min_age, max_age = GENERATIONS[generation]["age_range"]
        # 시뮬레이션 대상은 최소 15세 이상
        min_age = max(min_age, 15)
        return random.randint(min_age, max_age)

    def _get_budget(self, generation: str, segment: str) -> int:
        """세대별/세그먼트별 한끼 가용비"""
        # 유동인구는 가용비 제한 없음 (1,000,000원 = 사실상 무제한)
        if segment.startswith("유동_"):
            return 1_000_000

        min_budget, max_budget = BUDGET_RANGES[generation]
        budget = random.randint(min_budget, max_budget)

        # 그룹 세그먼트는 인원수 고려
        if segment in GROUP_BUDGET_MULTIPLIER:
            budget = int(budget * GROUP_BUDGET_MULTIPLIER[segment])

        # 1000원 단위로 반올림
        return round(budget / 1000) * 1000

    def generate_unique_agents(self, max_count: int = 96) -> List[GenerativeAgent]:
        """
        고유한 조합의 에이전트들 생성.

        개인 세그먼트 (4종): 세대6 × 성별2 × 건강2 × 변화2 = 48 조합/세그먼트
        그룹 세그먼트 (4종): 세대6 × 건강2 × 변화2 = 24 조합/세그먼트

        총 가능 조합: 4×48 + 4×24 = 192 + 96 = 288
        """
        agents = []
        all_combinations = []

        # 개인 세그먼트 조합 생성
        for seg in INDIVIDUAL_SEGMENTS:
            for gen in GENERATIONS.keys():
                for gender in ["남성", "여성"]:
                    for health in HEALTH_PREFERENCES.keys():
                        for change in CHANGE_PREFERENCES.keys():
                            all_combinations.append({
                                "segment": seg,
                                "generation": gen,
                                "gender": gender,
                                "health": health,
                                "change": change,
                                "is_group": False,
                            })

        # 그룹 세그먼트 조합 생성 (성별 제외)
        for seg in GROUP_BASED_SEGMENTS:
            for gen in GENERATIONS.keys():
                for health in HEALTH_PREFERENCES.keys():
                    for change in CHANGE_PREFERENCES.keys():
                        all_combinations.append({
                            "segment": seg,
                            "generation": gen,
                            "gender": None,
                            "health": health,
                            "change": change,
                            "is_group": True,
                        })

        # 현실적이지 않은 조합 필터링
        filtered_combinations = []
        for combo in all_combinations:
            gen = combo["generation"]
            seg = combo["segment"]

            # Alpha 세대는 상주인구 제외 (부모에게 의존)
            if gen == "Alpha" and seg.startswith("상주_"):
                continue

            # S 세대는 야간 활동 세그먼트 제외
            if gen == "S" and seg in ["유동_약속모임"]:
                continue

            # Alpha 세대는 4인가구 핵심 결정자 아님
            if gen == "Alpha" and seg == "상주_4인가구":
                continue

            filtered_combinations.append(combo)

        # 랜덤하게 max_count개 선택
        random.shuffle(filtered_combinations)
        selected = filtered_combinations[:max_count]

        for idx, combo in enumerate(selected):
            seg = combo["segment"]
            gen = combo["generation"]
            is_group = combo["is_group"]

            budget = self._get_budget(gen, seg)

            if is_group:
                # 그룹 세그먼트: 성별/나이 없음
                name = self._generate_group_name(seg, gen)
                agent = GenerativeAgent(
                    id=idx + 1,
                    name=name,
                    generation=gen,
                    segment=seg,
                    gender=None,
                    age=None,
                    health_preference=combo["health"],
                    change_preference=combo["change"],
                    budget_per_meal=budget,
                )
            else:
                # 개인 세그먼트: 성별/나이 포함
                gender = combo["gender"]
                age = self._get_age_for_generation(gen)
                name = self._generate_individual_name(gender)
                agent = GenerativeAgent(
                    id=idx + 1,
                    name=name,
                    generation=gen,
                    segment=seg,
                    gender=gender,
                    age=age,
                    health_preference=combo["health"],
                    change_preference=combo["change"],
                    budget_per_meal=budget,
                )

            agents.append(agent)

        return agents

    def print_distribution(self, agents: List[GenerativeAgent]):
        """에이전트 분포 출력"""
        print(f"\n총 에이전트 수: {len(agents)}")

        # 세대별 분포
        gen_counts = {}
        for a in agents:
            gen_counts[a.generation] = gen_counts.get(a.generation, 0) + 1
        print("\n세대별 분포:")
        for gen, count in sorted(gen_counts.items()):
            print(f"  {gen}: {count}명")

        # 세그먼트별 분포
        seg_counts = {}
        for a in agents:
            seg_counts[a.segment] = seg_counts.get(a.segment, 0) + 1
        print("\n세그먼트별 분포:")
        for seg, count in sorted(seg_counts.items()):
            seg_type = "그룹" if seg in GROUP_BASED_SEGMENTS else "개인"
            print(f"  {seg} [{seg_type}]: {count}")

        # 개인 세그먼트 성별 분포
        individual_agents = [a for a in agents if not a.is_group_based]
        if individual_agents:
            gender_counts = {"남성": 0, "여성": 0}
            for a in individual_agents:
                if a.gender:
                    gender_counts[a.gender] += 1
            print(f"\n개인 세그먼트 성별 분포: 남성 {gender_counts['남성']}명, 여성 {gender_counts['여성']}명")

        # 그룹 세그먼트 수
        group_agents = [a for a in agents if a.is_group_based]
        print(f"그룹 세그먼트: {len(group_agents)}개 그룹")

        # 생활양식 분포
        health_counts = {"건강선호": 0, "자극선호": 0}
        change_counts = {"안정추구": 0, "도전추구": 0}
        for a in agents:
            health_counts[a.health_preference] += 1
            change_counts[a.change_preference] += 1
        print(f"\n건강 성향: 건강선호 {health_counts['건강선호']}, 자극선호 {health_counts['자극선호']}")
        print(f"변화 성향: 안정추구 {change_counts['안정추구']}, 도전추구 {change_counts['도전추구']}")


if __name__ == "__main__":
    # 테스트
    factory = GenerativeAgentFactory()
    agents = factory.generate_unique_agents(max_count=96)
    factory.print_distribution(agents)

    # 샘플 에이전트 출력
    print("\n\n=== 샘플 개인 세그먼트 에이전트 ===")
    for agent in [a for a in agents if not a.is_group_based][:2]:
        print(agent.get_persona_summary())
        print()

    print("\n=== 샘플 그룹 세그먼트 에이전트 ===")
    for agent in [a for a in agents if a.is_group_based][:2]:
        print(agent.get_persona_summary())
        print()
