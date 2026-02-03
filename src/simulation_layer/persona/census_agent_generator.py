"""
Census-based agent generator.
Creates realistic consumer personas reflecting actual census data from area_summary.csv.
"""

import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from config import get_settings
from src.simulation_layer.persona.agent_persona import AgentPersona


class CensusBasedAgentGenerator:
    """Generates agent personas weighted by real census data."""

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

    # Age range mappings
    AGE_RANGES = [
        ("4세이하", 0, 4),
        ("5세이상~9세이하", 5, 9),
        ("10세이상~14세이하", 10, 14),
        ("15세이상~19세이하", 15, 19),
        ("20세이상~24세이하", 20, 24),
        ("25세이상~29세이하", 25, 29),
        ("30세이상~34세이하", 30, 34),
        ("35세이상~39세이하", 35, 39),
        ("40세이상~44세이하", 40, 44),
        ("45세이상~49세이하", 45, 49),
        ("50세이상~54세이하", 50, 54),
        ("55세이상~59세이하", 55, 59),
        ("60세이상~64세이하", 60, 64),
        ("65세이상~69세이하", 65, 69),
        ("70세이상~74세이하", 70, 74),
        ("75세이상~79세이하", 75, 79),
        ("80세이상~84세이하", 80, 84),
        ("85세이상~89세이하", 85, 89),
        ("90세이상~94세이하", 90, 94),
        ("95세이상~99세이하", 95, 99),
        ("100세이상", 100, 100),
    ]

    # Residence types in CSV order
    RESIDENCE_TYPES = ["다세대", "단독주택", "아파트", "연립주택", "영업용 건물 내 주택"]

    # Household types (note: 1인가구 is separate from generational household types)
    HOUSEHOLD_TYPES = ["1인가구", "1세대가구", "2세대가구", "3세대가구"]

    def __init__(self, csv_path: str | None = None, target_agents: int = 3500):
        """
        Initialize census-based agent generator.

        Args:
            csv_path: Path to area_summary.csv
            target_agents: Target number of agents to generate (default: 3500)
        """
        settings = get_settings()
        if csv_path is None:
            csv_path = str(settings.paths.data_dir / "area_summary.csv")

        self.df = pd.read_csv(csv_path, encoding='cp949')
        self.target_agents = target_agents
        self.total_population = self.df['선택범위_합계'].sum()
        self.scale_factor = target_agents / self.total_population

        print(f"Total population: {self.total_population:.0f}")
        print(f"Target agents: {target_agents}")
        print(f"Scale factor: 1/{self.total_population/target_agents:.1f}")

        self._build_population_pool()

    def _build_population_pool(self):
        """Build a weighted population pool from census data."""
        self.population_pool = []

        for _, row in self.df.iterrows():
            area_code = str(row['TOT_OA_CD'])

            # Calculate residence type distribution for this area
            residence_counts = {
                "다세대": row['다세대'],
                "단독주택": row['단독주택'],
                "아파트": row['아파트'],
                "연립주택": row['연립주택'],
                "영업용 건물 내 주택": row['영업용 건물 내 주택'],
            }
            residence_total = sum(v for v in residence_counts.values() if pd.notna(v) and v > 0)

            # Calculate household type distribution
            # Note: 1인가구 represents individuals, while others represent household generations
            household_counts = {
                "1인가구": row['1인가구'],
                "1세대가구": row['1세대가구'],
                "2세대가구": row['2세대가구'],
                "3세대가구": row['3세대가구'],
            }

            # Process each age/gender combination
            for age_range_name, age_min, age_max in self.AGE_RANGES:
                for gender in ["남자", "여자"]:
                    # Get count for this age/gender
                    if gender == "남자":
                        col_name = f"{age_range_name}_{gender}" if age_range_name != "100세이상" else "100세이상"
                    else:
                        col_name = f"{age_range_name}_{gender}" if age_range_name != "100세이상" else "100세이상_여자"

                    # Handle special case for total age column
                    if col_name not in self.df.columns:
                        if age_range_name == "100세이상" and gender == "남자":
                            # For male 100+, use total minus female
                            total_100 = row.get("100세이상", 0)
                            female_100 = row.get("100세이상_여자", 0)
                            count = total_100 - female_100 if pd.notna(total_100) and pd.notna(female_100) else 0
                        else:
                            count = 0
                    else:
                        count = row[col_name]

                    if pd.isna(count) or count <= 0:
                        continue

                    # Add to population pool
                    self.population_pool.append({
                        'area_code': area_code,
                        'age_min': age_min,
                        'age_max': age_max,
                        'age_range': age_range_name,
                        'gender': "남성" if gender == "남자" else "여성",
                        'count': count,
                        'residence_counts': residence_counts,
                        'residence_total': residence_total,
                        'household_counts': household_counts,
                    })

        print(f"Built population pool with {len(self.population_pool)} age/gender/area segments")

    def _generate_name(self, gender: str) -> str:
        surname = random.choice(self.SURNAMES)
        if gender == "남성":
            given_name = random.choice(self.MALE_NAMES)
        else:
            given_name = random.choice(self.FEMALE_NAMES)
        return f"{surname}{given_name}"

    def _age_to_age_group(self, age: int) -> str:
        """Convert exact age to age group category."""
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

    def _generate_income_level(self, age: int, occupation: str, residence_type: str) -> str:
        """Generate income level based on age, occupation, and residence type."""
        # Base income by occupation
        if "학생" in occupation or "취준생" in occupation or "인턴" in occupation:
            base = random.choice(["하", "중하"])
        elif "신입" in occupation:
            base = random.choice(["중하", "중"])
        elif "관리직" in occupation or "전문직" in occupation:
            base = random.choice(["중상", "상"])
        elif "자영업" in occupation:
            base = random.choice(["중", "중상"])
        elif "은퇴자" in occupation or "주부" in occupation:
            base = random.choice(["중하", "중"])
        else:
            base = random.choice(["중하", "중", "중상"])

        # Adjust by residence type (아파트 > 연립주택 > 다세대 > 단독주택 > 영업용)
        if residence_type == "아파트":
            if base == "하":
                base = "중하"
            elif base == "중하":
                base = random.choice(["중하", "중"])
        elif residence_type == "영업용 건물 내 주택":
            if base == "상":
                base = "중상"
            elif base == "중상":
                base = random.choice(["중상", "중"])

        return base

    def _generate_value_preference(self, age: int, gender: str, income_level: str) -> str:
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
        """Calculate price/trend/quality sensitivities."""
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

    def _sample_residence_type(self, residence_counts: dict, residence_total: float) -> str:
        """Sample residence type weighted by counts."""
        if residence_total <= 0:
            return random.choice(self.RESIDENCE_TYPES)

        types = []
        weights = []
        for rtype in self.RESIDENCE_TYPES:
            count = residence_counts.get(rtype, 0)
            if pd.notna(count) and count > 0:
                types.append(rtype)
                weights.append(count)

        if not types:
            return random.choice(self.RESIDENCE_TYPES)

        return random.choices(types, weights=weights)[0]

    def _sample_household_type(self, household_counts: dict, age: int, residence_type: str) -> str:
        """Sample household type weighted by counts and adjusted by age."""
        # Base weights from census
        types = []
        weights = []
        for htype in self.HOUSEHOLD_TYPES:
            count = household_counts.get(htype, 0)
            if pd.notna(count) and count > 0:
                types.append(htype)
                weights.append(count)

        if not types:
            # Fallback: guess by age
            if age < 30:
                return "1인가구"
            elif age < 50:
                return random.choice(["1인가구", "2세대가구"])
            else:
                return random.choice(["2세대가구", "3세대가구"])

        # Adjust weights by age (younger → more 1인가구)
        adjusted_weights = []
        for htype, weight in zip(types, weights):
            if htype == "1인가구":
                if age < 30:
                    weight *= 1.5
                elif age < 40:
                    weight *= 1.2
            elif htype in ["3세대가구", "4세대가구"]:
                if age >= 50:
                    weight *= 1.3
            adjusted_weights.append(weight)

        return random.choices(types, weights=adjusted_weights)[0]

    def generate_agents(self, num_agents: int | None = None) -> List[AgentPersona]:
        """Generate agents based on census data."""
        num_agents = num_agents or self.target_agents

        # Calculate how many agents to generate from each segment
        agents = []

        # Calculate scaled counts
        pool_with_scaled_counts = []
        for segment in self.population_pool:
            scaled_count = segment['count'] * self.scale_factor
            pool_with_scaled_counts.append({
                **segment,
                'scaled_count': scaled_count
            })

        # Generate agents proportionally using fractional allocation to avoid rounding errors
        total_scaled = sum(s['scaled_count'] for s in pool_with_scaled_counts)

        # First pass: allocate integer parts
        remainders = []
        for segment in pool_with_scaled_counts:
            proportion = segment['scaled_count'] / total_scaled
            exact_count = proportion * num_agents
            integer_part = int(exact_count)
            fractional_part = exact_count - integer_part

            segment['allocated_count'] = integer_part
            remainders.append((segment, fractional_part))

        # Second pass: distribute remaining agents based on fractional parts
        total_allocated = sum(s['allocated_count'] for s in pool_with_scaled_counts)
        remaining = num_agents - total_allocated

        # Sort by fractional part (largest first) and allocate remaining
        remainders.sort(key=lambda x: x[1], reverse=True)
        for i in range(remaining):
            if i < len(remainders):
                remainders[i][0]['allocated_count'] += 1

        # Generate agents
        for segment in pool_with_scaled_counts:
            for _ in range(segment['allocated_count']):
                # Generate exact age within range
                age = random.randint(segment['age_min'], segment['age_max'])
                gender = segment['gender']
                age_group = self._age_to_age_group(age)

                # Sample residence and household type
                residence_type = self._sample_residence_type(
                    segment['residence_counts'],
                    segment['residence_total']
                )
                household_type = self._sample_household_type(
                    segment['household_counts'],
                    age,
                    residence_type
                )

                # Generate other attributes
                name = self._generate_name(gender)
                occupation = self._generate_occupation(age, gender)
                income_level = self._generate_income_level(age, occupation, residence_type)
                value_preference = self._generate_value_preference(age, gender, income_level)
                store_preferences = self._generate_store_preferences(age, gender)
                price_sens, trend_sens, quality_pref = self._calculate_sensitivities(
                    age, gender, income_level
                )

                agent = AgentPersona(
                    id=len(agents) + 1,
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
                    residence_type=residence_type,
                    household_type=household_type,
                    census_area_code=segment['area_code'],
                )
                agents.append(agent)

        # Shuffle to randomize order
        random.shuffle(agents)

        # Re-assign IDs
        for i, agent in enumerate(agents, 1):
            agent.id = i

        print(f"\nGenerated {len(agents)} agents")
        return agents
