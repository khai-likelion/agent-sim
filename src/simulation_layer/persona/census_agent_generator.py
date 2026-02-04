"""
Census-based agent generator with LLM persona generation.
Creates realistic consumer personas using Groq LLM + census data weighting.
"""

import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from config import get_settings
from src.simulation_layer.persona.agent_persona import AgentPersona
from src.ai_layer.llm_client import create_llm_client, LLMClient


class CensusBasedAgentGenerator:
    """Generates agent personas using LLM + census data weighting."""

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

    RESIDENCE_TYPES = ["다세대", "단독주택", "아파트", "연립주택", "영업용 건물 내 주택"]
    HOUSEHOLD_TYPES = ["1인가구", "1세대가구", "2세대가구", "3세대가구"]

    # Income levels for LLM prompt
    INCOME_LEVELS = ["하", "중하", "중", "중상", "상"]

    def __init__(
        self,
        csv_path: str | None = None,
        target_agents: int = 3500,
        use_llm: bool = True,
        rate_limit_delay: float = 2.5,  # Groq free: 30 req/min → ~2초 간격
    ):
        settings = get_settings()
        if csv_path is None:
            csv_path = str(settings.paths.data_dir / "area_summary.csv")

        self.df = pd.read_csv(csv_path, encoding='cp949')
        self.target_agents = target_agents
        self.total_population = self.df['선택범위_합계'].sum()
        self.scale_factor = target_agents / self.total_population

        self.use_llm = use_llm
        self.rate_limit_delay = rate_limit_delay
        self.llm_client: Optional[LLMClient] = None
        self.persona_cache: Dict[str, Dict[str, Any]] = {}
        self._prompt_template: Optional[str] = None

        print(f"Total population: {self.total_population:.0f}")
        print(f"Target agents: {target_agents}")
        print(f"Scale factor: 1/{self.total_population/target_agents:.1f}")
        print(f"LLM mode: {'ON' if use_llm else 'OFF (fallback)'}")

        self._build_population_pool()

    def _load_prompt_template(self) -> str:
        if self._prompt_template is None:
            settings = get_settings()
            template_path = settings.paths.prompt_templates_dir / "persona_generation.txt"
            self._prompt_template = template_path.read_text(encoding='utf-8')
        return self._prompt_template

    def _get_llm_client(self) -> LLMClient:
        if self.llm_client is None:
            self.llm_client = create_llm_client()
        return self.llm_client

    def _build_population_pool(self):
        """Build a weighted population pool from census data."""
        self.population_pool = []

        for _, row in self.df.iterrows():
            area_code = str(row['TOT_OA_CD'])

            residence_counts = {
                "다세대": row['다세대'],
                "단독주택": row['단독주택'],
                "아파트": row['아파트'],
                "연립주택": row['연립주택'],
                "영업용 건물 내 주택": row['영업용 건물 내 주택'],
            }
            residence_total = sum(v for v in residence_counts.values() if pd.notna(v) and v > 0)

            household_counts = {
                "1인가구": row['1인가구'],
                "1세대가구": row['1세대가구'],
                "2세대가구": row['2세대가구'],
                "3세대가구": row['3세대가구'],
            }

            for age_range_name, age_min, age_max in self.AGE_RANGES:
                for gender in ["남자", "여자"]:
                    if gender == "남자":
                        col_name = f"{age_range_name}_{gender}" if age_range_name != "100세이상" else "100세이상"
                    else:
                        col_name = f"{age_range_name}_{gender}" if age_range_name != "100세이상" else "100세이상_여자"

                    if col_name not in self.df.columns:
                        if age_range_name == "100세이상" and gender == "남자":
                            total_100 = row.get("100세이상", 0)
                            female_100 = row.get("100세이상_여자", 0)
                            count = total_100 - female_100 if pd.notna(total_100) and pd.notna(female_100) else 0
                        else:
                            count = 0
                    else:
                        count = row[col_name]

                    if pd.isna(count) or count <= 0:
                        continue

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

        print(f"Built population pool with {len(self.population_pool)} segments")

    def _generate_name(self, gender: str) -> str:
        surname = random.choice(self.SURNAMES)
        if gender == "남성":
            given_name = random.choice(self.MALE_NAMES)
        else:
            given_name = random.choice(self.FEMALE_NAMES)
        return f"{surname}{given_name}"

    def _age_to_age_group(self, age: int) -> str:
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

    def _estimate_income_level(self, age: int, residence_type: str) -> str:
        """Estimate income level based on age and residence type for LLM prompt."""
        # Age-based base income
        if age < 25:
            base_weights = [0.3, 0.4, 0.2, 0.08, 0.02]  # 하, 중하, 중, 중상, 상
        elif age < 35:
            base_weights = [0.1, 0.25, 0.35, 0.2, 0.1]
        elif age < 50:
            base_weights = [0.05, 0.15, 0.35, 0.3, 0.15]
        elif age < 65:
            base_weights = [0.1, 0.2, 0.35, 0.25, 0.1]
        else:
            base_weights = [0.2, 0.3, 0.3, 0.15, 0.05]

        # Adjust by residence type
        if residence_type == "아파트":
            base_weights = [w * 0.7 if i < 2 else w * 1.3 for i, w in enumerate(base_weights)]
        elif residence_type == "영업용 건물 내 주택":
            base_weights = [w * 1.3 if i < 2 else w * 0.7 for i, w in enumerate(base_weights)]

        # Normalize
        total = sum(base_weights)
        base_weights = [w / total for w in base_weights]

        return random.choices(self.INCOME_LEVELS, weights=base_weights)[0]

    def _sample_residence_type(self, residence_counts: dict, residence_total: float) -> str:
        if residence_total <= 0:
            return random.choice(self.RESIDENCE_TYPES)

        types, weights = [], []
        for rtype in self.RESIDENCE_TYPES:
            count = residence_counts.get(rtype, 0)
            if pd.notna(count) and count > 0:
                types.append(rtype)
                weights.append(count)

        return random.choices(types, weights=weights)[0] if types else random.choice(self.RESIDENCE_TYPES)

    def _sample_household_type(self, household_counts: dict, age: int) -> str:
        types, weights = [], []
        for htype in self.HOUSEHOLD_TYPES:
            count = household_counts.get(htype, 0)
            if pd.notna(count) and count > 0:
                types.append(htype)
                weights.append(count)

        if not types:
            if age < 30:
                return "1인가구"
            elif age < 50:
                return random.choice(["1인가구", "2세대가구"])
            else:
                return random.choice(["2세대가구", "3세대가구"])

        # Age adjustment
        adjusted_weights = []
        for htype, weight in zip(types, weights):
            if htype == "1인가구" and age < 35:
                weight *= 1.5
            elif htype in ["3세대가구"] and age >= 50:
                weight *= 1.3
            adjusted_weights.append(weight)

        return random.choices(types, weights=adjusted_weights)[0]

    def _get_cache_key(
        self, age_group: str, gender: str, residence_type: str,
        household_type: str, income_level: str
    ) -> str:
        """Generate cache key for persona archetype."""
        return f"{age_group}_{gender}_{residence_type}_{household_type}_{income_level}"

    def _generate_persona_via_llm(
        self,
        age_group: str,
        gender: str,
        residence_type: str,
        household_type: str,
        income_level: str,
    ) -> Dict[str, Any]:
        """Generate persona attributes using LLM."""
        cache_key = self._get_cache_key(age_group, gender, residence_type, household_type, income_level)

        # Check cache
        if cache_key in self.persona_cache:
            return self.persona_cache[cache_key]

        template = self._load_prompt_template()
        prompt = template.format(
            age_group=age_group,
            gender=gender,
            residence_type=residence_type,
            household_type=household_type,
            income_level=income_level,
        )

        try:
            client = self._get_llm_client()
            response = client.generate_sync(prompt)

            # Parse JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                persona_data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            # Validate and normalize
            persona_data = self._validate_persona_data(persona_data)

            # Cache the result
            self.persona_cache[cache_key] = persona_data

            # Rate limiting
            time.sleep(self.rate_limit_delay)

            return persona_data

        except Exception as e:
            print(f"LLM error for {cache_key}: {e}. Using fallback.")
            return self._generate_persona_fallback(age_group, gender, income_level)

    def _validate_persona_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize persona data from LLM."""
        return {
            "occupation": str(data.get("occupation", "회사원")),
            "value_preference": str(data.get("value_preference", "다양한 음식점 방문을 즐김")),
            "store_preferences": list(data.get("store_preferences", ["한식", "카페", "편의점"])),
            "price_sensitivity": max(0.0, min(1.0, float(data.get("price_sensitivity", 0.5)))),
            "trend_sensitivity": max(0.0, min(1.0, float(data.get("trend_sensitivity", 0.5)))),
            "quality_preference": max(0.0, min(1.0, float(data.get("quality_preference", 0.5)))),
        }

    def _generate_persona_fallback(
        self, age_group: str, gender: str, income_level: str
    ) -> Dict[str, Any]:
        """Fallback persona generation without LLM."""
        # Occupation by age
        age_occupations = {
            "10대": ["고등학생", "대학생"],
            "20대": ["대학생", "취준생", "회사원", "프리랜서"],
            "30대": ["회사원", "IT 개발자", "마케터", "자영업자"],
            "40대": ["회사원", "관리자", "자영업자", "전문직"],
            "50대": ["회사원", "자영업자", "공무원", "주부"],
            "60대+": ["은퇴자", "자영업자", "주부"],
        }

        # Store preferences by age/gender
        store_prefs = {
            ("10대", "여성"): ["카페", "디저트", "분식", "패스트푸드"],
            ("10대", "남성"): ["패스트푸드", "분식", "PC방", "편의점"],
            ("20대", "여성"): ["카페", "브런치", "이탈리안", "디저트카페", "베이커리"],
            ("20대", "남성"): ["한식", "일식", "햄버거", "치킨", "호프"],
            ("30대", "여성"): ["카페", "한식", "브런치", "샐러드", "베이커리"],
            ("30대", "남성"): ["한식", "일식", "중식", "고기", "호프"],
            ("40대", "여성"): ["한식", "카페", "베이커리", "샐러드"],
            ("40대", "남성"): ["한식", "중식", "고기", "일식"],
            ("50대", "여성"): ["한식", "칼국수", "백반", "전통시장"],
            ("50대", "남성"): ["한식", "중식", "설렁탕", "고기"],
            ("60대+", "여성"): ["한식", "칼국수", "국밥", "전통시장"],
            ("60대+", "남성"): ["한식", "설렁탕", "국밥", "중식"],
        }

        # Sensitivities
        age_sens = {
            "10대": (0.7, 0.8, 0.3),
            "20대": (0.6, 0.8, 0.4),
            "30대": (0.5, 0.6, 0.6),
            "40대": (0.5, 0.4, 0.7),
            "50대": (0.6, 0.2, 0.6),
            "60대+": (0.8, 0.1, 0.5),
        }

        occupation = random.choice(age_occupations.get(age_group, ["회사원"]))
        prefs = store_prefs.get((age_group, gender), ["한식", "카페", "편의점"])
        price_s, trend_s, quality_p = age_sens.get(age_group, (0.5, 0.5, 0.5))

        # Adjust by income
        if income_level in ["상", "중상"]:
            price_s *= 0.7
            quality_p = min(1.0, quality_p * 1.2)
        elif income_level in ["하", "중하"]:
            price_s = min(1.0, price_s * 1.3)

        value_templates = {
            "10대": f"학교 근처 {prefs[0]}나 {prefs[1]}을 자주 방문. 친구들과 함께 갈 수 있는 곳 선호.",
            "20대": f"{'SNS 핫플레이스와 트렌디한 ' if gender == '여성' else '가성비 좋은 '}{prefs[0]} 선호. {'분위기' if gender == '여성' else '양'}을 중시.",
            "30대": f"{'퀄리티 있는 브런치와 ' if gender == '여성' else '직장 동료와 함께하는 '}{prefs[0]} 방문. 시간 효율 중시.",
            "40대": f"가족 단위 외식이나 {'건강을 고려한' if gender == '여성' else '푸짐한'} 식사 선호. 주차 편의성 고려.",
            "50대": f"익숙한 {prefs[0]}집 위주로 방문. 건강식과 적당한 가격대 선호.",
            "60대+": f"전통시장이나 오래된 맛집 선호. {prefs[0]}이나 {prefs[1]} 즐겨 찾음. 경제성 중시.",
        }

        return {
            "occupation": occupation,
            "value_preference": value_templates.get(age_group, "다양한 음식점 방문"),
            "store_preferences": prefs,
            "price_sensitivity": round(price_s, 2),
            "trend_sensitivity": round(trend_s, 2),
            "quality_preference": round(quality_p, 2),
        }

    def generate_agents(self, num_agents: int | None = None) -> List[AgentPersona]:
        """Generate agents using LLM + census weighting."""
        num_agents = num_agents or self.target_agents
        agents = []

        # Calculate scaled counts
        pool_with_scaled_counts = []
        for segment in self.population_pool:
            scaled_count = segment['count'] * self.scale_factor
            pool_with_scaled_counts.append({**segment, 'scaled_count': scaled_count})

        total_scaled = sum(s['scaled_count'] for s in pool_with_scaled_counts)

        # Allocate agents proportionally
        remainders = []
        for segment in pool_with_scaled_counts:
            proportion = segment['scaled_count'] / total_scaled
            exact_count = proportion * num_agents
            integer_part = int(exact_count)
            fractional_part = exact_count - integer_part

            segment['allocated_count'] = integer_part
            remainders.append((segment, fractional_part))

        total_allocated = sum(s['allocated_count'] for s in pool_with_scaled_counts)
        remaining = num_agents - total_allocated

        remainders.sort(key=lambda x: x[1], reverse=True)
        for i in range(remaining):
            if i < len(remainders):
                remainders[i][0]['allocated_count'] += 1

        # Generate agents
        total_segments = sum(1 for s in pool_with_scaled_counts if s['allocated_count'] > 0)
        processed_segments = 0

        for segment in pool_with_scaled_counts:
            if segment['allocated_count'] <= 0:
                continue

            processed_segments += 1

            for _ in range(segment['allocated_count']):
                age = random.randint(segment['age_min'], segment['age_max'])
                gender = segment['gender']
                age_group = self._age_to_age_group(age)

                residence_type = self._sample_residence_type(
                    segment['residence_counts'], segment['residence_total']
                )
                household_type = self._sample_household_type(segment['household_counts'], age)
                income_level = self._estimate_income_level(age, residence_type)

                # Generate persona via LLM or fallback
                if self.use_llm:
                    persona_data = self._generate_persona_via_llm(
                        age_group, gender, residence_type, household_type, income_level
                    )
                else:
                    persona_data = self._generate_persona_fallback(age_group, gender, income_level)

                name = self._generate_name(gender)

                agent = AgentPersona(
                    id=len(agents) + 1,
                    name=name,
                    age=age,
                    age_group=age_group,
                    gender=gender,
                    occupation=persona_data["occupation"],
                    income_level=income_level,
                    value_preference=persona_data["value_preference"],
                    store_preferences=persona_data["store_preferences"],
                    price_sensitivity=persona_data["price_sensitivity"],
                    trend_sensitivity=persona_data["trend_sensitivity"],
                    quality_preference=persona_data["quality_preference"],
                    residence_type=residence_type,
                    household_type=household_type,
                    census_area_code=segment['area_code'],
                )
                agents.append(agent)

            # Progress update
            if processed_segments % 10 == 0:
                print(f"Progress: {processed_segments}/{total_segments} segments, {len(agents)} agents, {len(self.persona_cache)} cached personas")

        random.shuffle(agents)
        for i, agent in enumerate(agents, 1):
            agent.id = i

        print(f"\nGenerated {len(agents)} agents")
        print(f"LLM persona archetypes cached: {len(self.persona_cache)}")
        return agents
