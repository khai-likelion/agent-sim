"""
망원동 상권 ABM 시뮬레이션 (Pure LLM 버전)
완전한 LLM 의사결정 - 하드코딩 제거, 페르소나만 주입

LLM이 모든 것을 판단:
- 어느 식당 갈지
- 만족도 어떤지
- 리뷰 쓸지 말지
- 리뷰 내용
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.stats import truncnorm
import pandas as pd
import json
import os
from openai import OpenAI
from pathlib import Path
import config

# ============================================================================
# LLM CLIENT
# ============================================================================

client = None

def init_llm_client(api_key: Optional[str] = None):
    """OpenAI API 클라이언트 초기화"""
    global client
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 필요")
    client = OpenAI(api_key=api_key)


# ============================================================================
# SEGMENT & TIME CONFIGURATION
# ============================================================================

# 시간대: 아침/점심/저녁/야식
TIMEBLOCKS = ["breakfast", "lunch", "dinner", "late_night"]

TIMEBLOCK_DESC = {
    "breakfast": "아침 (7-10시)",
    "lunch": "점심 (11시-14시)",
    "dinner": "저녁 (17시-21시)",
    "late_night": "야식 (21시-24시)"
}

# 8개 Persona 세그먼트
SEGMENT_DATA = {
    "R1_OnePerson": {
        "name": "1인 가구 거주민",
        "description": "혼자 사는 직장인 또는 학생. 혼밥 선호, 가까운 거리, 루틴 중시",
        "characteristics": [
            "집 근처 단골집 선호",
            "혼밥 가능한 곳 필수",
            "간편하고 빠른 식사",
            "가성비 중요"
        ],
        "beta_traits": {
            "solo_pref": (9.0, 1.5),
            "novelty_seeking": (2.0, 5.0),
            "repeat_tolerance": (6.0, 2.0),
            "hygiene_threshold": (2.0, 3.0),
            "review_propensity": (2.0, 3.0),
            "influence_sensitivity": (2.0, 2.5)
        },
        "presence_by_timeblock": {
            "breakfast": 0.15, "lunch": 0.25, "dinner": 0.55, "late_night": 0.30
        },
        "time_pressure_profile": {
            "lunch": 0.45, "dinner": 0.25
        }
    },
    "R2_TwoPeople": {
        "name": "2인 가구",
        "description": "부부 또는 커플 동거. 적당한 외식 빈도, 분위기 있는 곳 선호",
        "characteristics": [
            "주말 외식 빈도 높음",
            "비주얼/분위기 중시",
            "편안한 공간 선호",
            "다양한 메뉴 탐색"
        ],
        "beta_traits": {
            "visual_importance": (4.0, 2.5),
            "comfort_importance": (4.0, 2.5),
            "novelty_seeking": (3.5, 2.8),
            "repeat_tolerance": (3.0, 3.0),
            "hygiene_threshold": (3.0, 2.5),
            "review_propensity": (2.5, 2.5),
            "influence_sensitivity": (3.0, 2.5)
        },
        "presence_by_timeblock": {
            "breakfast": 0.10, "lunch": 0.20, "dinner": 0.55, "late_night": 0.20
        },
        "time_pressure_profile": {
            "lunch": 0.40, "dinner": 0.20
        }
    },
    "R3_FamilyFour": {
        "name": "4인 가족",
        "description": "부모+자녀 2명. 위생과 편안함 최우선, 새로운 시도 회피",
        "characteristics": [
            "위생 기준 매우 엄격",
            "편안하고 넓은 공간 필수",
            "아이들 먹기 좋은 메뉴",
            "검증된 식당 선호"
        ],
        "beta_traits": {
            "comfort_importance": (8.0, 1.8),
            "hygiene_threshold": (10.0, 1.5),
            "novelty_seeking": (2.0, 6.0),
            "repeat_tolerance": (4.0, 3.0),
            "review_propensity": (2.2, 2.8),
            "influence_sensitivity": (2.5, 2.8)
        },
        "presence_by_timeblock": {
            "breakfast": 0.05, "lunch": 0.40, "dinner": 0.65, "late_night": 0.05
        },
        "time_pressure_profile": {
            "dinner": 0.20
        }
    },
    "R4_CommuterResident": {
        "name": "출퇴근 거주민",
        "description": "망원 거주, 타지역 출퇴근. 저녁/주말에 망원 외식 즐김",
        "characteristics": [
            "저녁 외식 빈도 높음",
            "주말 브런치/맛집 탐방",
            "트렌디한 곳 선호",
            "리뷰 많이 참고"
        ],
        "beta_traits": {
            "novelty_seeking": (4.5, 2.5),
            "visual_importance": (4.5, 2.8),
            "repeat_tolerance": (3.0, 3.5),
            "hygiene_threshold": (4.0, 2.2),
            "review_propensity": (2.2, 2.6),
            "influence_sensitivity": (3.2, 2.4)
        },
        "presence_by_timeblock": {
            "breakfast": 0.10, "lunch": 0.05, "dinner": 0.65, "late_night": 0.40
        },
        "time_pressure_profile": {
            "dinner": 0.35
        }
    },
    "F1_DateCouple": {
        "name": "데이트 커플",
        "description": "망원을 찾는 커플. 분위기와 비주얼 최우선",
        "characteristics": [
            "인스타그래머블 필수",
            "분위기 좋은 곳",
            "거리는 덜 중요",
            "리뷰/SNS 참고 많이"
        ],
        "beta_traits": {
            "visual_importance": (12.0, 1.2),
            "comfort_importance": (6.0, 2.0),
            "novelty_seeking": (7.0, 2.0),
            "hygiene_threshold": (5.0, 2.0),
            "review_propensity": (5.0, 2.0),
            "influence_sensitivity": (6.0, 2.0),
            "repeat_tolerance": (2.0, 5.0),
            "solo_pref": (1.0, 12.0)
        },
        "presence_by_timeblock": {
            "breakfast": 0.05, "lunch": 0.30, "dinner": 0.70, "late_night": 0.25
        },
        "time_pressure_profile": {
            "dinner": 0.20
        }
    },
    "F2_SocialGroup": {
        "name": "친구 모임",
        "description": "친구들끼리 만남. 트렌디하고 공유 가능한 음식",
        "characteristics": [
            "인원수 많음 (3-5명)",
            "공유 메뉴 선호",
            "사진 찍기 좋은 곳",
            "핫플 탐방"
        ],
        "beta_traits": {
            "visual_importance": (8.0, 2.0),
            "stimulation_pref": (5.0, 2.0),
            "novelty_seeking": (6.0, 2.0),
            "review_propensity": (4.0, 2.2),
            "influence_sensitivity": (5.0, 2.2),
            "repeat_tolerance": (2.5, 4.5)
        },
        "presence_by_timeblock": {
            "breakfast": 0.03, "lunch": 0.25, "dinner": 0.65, "late_night": 0.50
        },
        "time_pressure_profile": {
            "dinner": 0.25
        }
    },
    "F3_IncomingWorker": {
        "name": "외부 출근자",
        "description": "타지역 거주, 망원 직장. 점심시간 효율 중시",
        "characteristics": [
            "점심시간 빠른 회전",
            "회사 근처 단골집",
            "반복 방문 OK",
            "간편 식사"
        ],
        "beta_traits": {
            "repeat_tolerance": (10.0, 1.5),
            "novelty_seeking": (2.0, 7.0),
            "visual_importance": (2.0, 7.0),
            "hygiene_threshold": (2.5, 3.0),
            "review_propensity": (1.8, 3.8),
            "influence_sensitivity": (2.0, 3.5),
            "solo_pref": (4.0, 2.5)
        },
        "presence_by_timeblock": {
            "breakfast": 0.20, "lunch": 0.85, "dinner": 0.15, "late_night": 0.05
        },
        "time_pressure_profile": {
            "lunch": 0.80
        }
    },
    "F4_SoloVisitor": {
        "name": "솔로 탐방객",
        "description": "혼자 망원 탐방. 숨은 맛집 찾기",
        "characteristics": [
            "혼밥 거부감 없음",
            "새로운 곳 도전",
            "리뷰 적극 작성",
            "마니아적 성향"
        ],
        "beta_traits": {
            "solo_pref": (12.0, 1.2),
            "novelty_seeking": (10.0, 1.6),
            "visual_importance": (5.0, 2.5),
            "review_propensity": (3.5, 2.5),
            "influence_sensitivity": (3.5, 2.2),
            "repeat_tolerance": (2.0, 6.0)
        },
        "presence_by_timeblock": {
            "breakfast": 0.05, "lunch": 0.35, "dinner": 0.60, "late_night": 0.25
        },
        "time_pressure_profile": {}
    }
}

GLOBAL_DEFAULT = {
    "beta_traits": {
        "spicy_pref": (2.0, 2.0),
        "stimulation_pref": (2.0, 2.0),
        "novelty_seeking": (2.0, 2.0),
        "visual_importance": (2.0, 2.0),
        "comfort_importance": (2.0, 2.0),
        "solo_pref": (2.0, 2.0),
        "hygiene_threshold": (2.5, 2.0),
        "repeat_tolerance": (2.0, 2.0),
        "memory_decay": (2.0, 2.0),
        "review_propensity": (2.0, 2.5),
        "influence_sensitivity": (2.0, 2.0)
    }
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Agent:
    """소비자 에이전트"""
    id: int
    segment: str
    home_location: Tuple[float, float]
    work_location: Optional[Tuple[float, float]]

    # Preferences [0,1]
    spicy_pref: float
    stimulation_pref: float
    novelty_seeking: float
    visual_importance: float
    comfort_importance: float
    solo_pref: float

    # Constraints
    hygiene_threshold: float
    repeat_tolerance: float
    memory_decay: float
    review_propensity: float
    influence_sensitivity: float

    # Dynamic states
    hunger_level: float = 0.5
    time_pressure: float = 0.3
    fatigue_level: float = 0.3

    # Memory
    memory: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # RNG
    rng: np.random.Generator = field(default=None, repr=False)


@dataclass
class Restaurant:
    """레스토랑 (실제 stores.csv 데이터)"""
    id: int
    name: str
    category: str
    location: Tuple[float, float]  # (x좌표, y좌표)
    address: str
    업종: str

    # Simulated attributes [0,1]
    hygiene_level: float = 0.5
    stimulation_level: float = 0.5
    visual_score: float = 0.5
    comfort_level: float = 0.5
    solo_accessibility: float = 0.5

    # Reviews
    review_count: int = 0
    avg_review_score: float = 0.5

    # Strategy
    strategy_flags: Dict[str, bool] = field(default_factory=dict)
    novelty_flag: float = 0.0


# ============================================================================
# UTILITIES
# ============================================================================

def sample_beta(rng: np.random.Generator, alpha: float, beta: float) -> float:
    return rng.beta(alpha, beta)


def euclidean_distance(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
    """두 위치 간 거리 (좌표 단위)"""
    return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)


def get_current_location(agent: Agent, timeblock: str) -> Tuple[float, float]:
    if agent.segment.startswith("R"):
        return agent.home_location
    elif agent.segment == "F3_IncomingWorker" and timeblock == "lunch":
        return agent.work_location if agent.work_location else agent.home_location
    else:
        return agent.home_location


def load_restaurants_from_csv(csv_path: str, sample_size: Optional[int] = None) -> List[Restaurant]:
    """stores.csv에서 실제 망원동 식당 데이터 로드"""
    df = pd.read_csv(csv_path)

    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    restaurants = []
    rng = np.random.default_rng(42)

    for idx, row in df.iterrows():
        # 업종별 속성 추정
        업종 = row.get('업종', '한식음식점')
        category = row.get('카테고리', '음식점')

        # 업종별 기본 속성
        if '카페' in 업종:
            base_hygiene = 0.75
            base_visual = 0.85
            base_comfort = 0.80
            base_solo = 0.90
            base_stimulation = 0.20
        elif '한식' in 업종:
            base_hygiene = 0.65
            base_visual = 0.50
            base_comfort = 0.65
            base_solo = 0.70
            base_stimulation = 0.50
        elif '일식' in 업종:
            base_hygiene = 0.80
            base_visual = 0.75
            base_comfort = 0.70
            base_solo = 0.60
            base_stimulation = 0.40
        elif '중식' in 업종:
            base_hygiene = 0.55
            base_visual = 0.45
            base_comfort = 0.60
            base_solo = 0.60
            base_stimulation = 0.70
        elif '호프' in 업종 or '술집' in 업종:
            base_hygiene = 0.60
            base_visual = 0.65
            base_comfort = 0.70
            base_solo = 0.30
            base_stimulation = 0.65
        elif '치킨' in 업종:
            base_hygiene = 0.60
            base_visual = 0.50
            base_comfort = 0.60
            base_solo = 0.50
            base_stimulation = 0.75
        elif '패스트푸드' in 업종:
            base_hygiene = 0.85
            base_visual = 0.60
            base_comfort = 0.50
            base_solo = 0.85
            base_stimulation = 0.50
        else:
            base_hygiene = 0.60
            base_visual = 0.55
            base_comfort = 0.60
            base_solo = 0.65
            base_stimulation = 0.50

        # 랜덤성 추가
        hygiene = np.clip(base_hygiene + rng.normal(0, 0.10), 0, 1)
        visual = np.clip(base_visual + rng.normal(0, 0.10), 0, 1)
        comfort = np.clip(base_comfort + rng.normal(0, 0.10), 0, 1)
        solo = np.clip(base_solo + rng.normal(0, 0.10), 0, 1)
        stimulation = np.clip(base_stimulation + rng.normal(0, 0.10), 0, 1)

        # 초기 리뷰 수
        initial_reviews = int(rng.exponential(50))
        initial_score = rng.beta(5, 2)

        restaurant = Restaurant(
            id=int(row.get('ID', idx)),
            name=row['장소명'],
            category=category,
            location=(float(row['x']), float(row['y'])),
            address=row['주소'],
            업종=업종,
            hygiene_level=hygiene,
            stimulation_level=stimulation,
            visual_score=visual,
            comfort_level=comfort,
            solo_accessibility=solo,
            review_count=initial_reviews,
            avg_review_score=initial_score
        )
        restaurants.append(restaurant)

    return restaurants


def create_agent(agent_id: int, segment: str, rng: np.random.Generator,
                home_loc: Tuple[float, float], work_loc: Optional[Tuple[float, float]] = None) -> Agent:
    """세그먼트 기반 에이전트 생성"""
    seg_data = SEGMENT_DATA[segment]

    beta_traits = {}
    for trait_name in GLOBAL_DEFAULT["beta_traits"]:
        if trait_name in seg_data.get("beta_traits", {}):
            spec = seg_data["beta_traits"][trait_name]
        else:
            spec = GLOBAL_DEFAULT["beta_traits"][trait_name]
        beta_traits[trait_name] = sample_beta(rng, spec[0], spec[1])

    return Agent(
        id=agent_id, segment=segment, home_location=home_loc, work_location=work_loc,
        rng=rng, **beta_traits
    )


def create_demo_agents(n_agents: int, seed: int, center_loc: Tuple[float, float]) -> List[Agent]:
    """에이전트 생성"""
    base_rng = np.random.default_rng(seed)
    agents = []
    segments = list(SEGMENT_DATA.keys())

    for i in range(n_agents):
        segment = segments[i % len(segments)]

        # 중심점 근처 랜덤 위치
        home_x = center_loc[0] + base_rng.normal(0, 0.005)
        home_y = center_loc[1] + base_rng.normal(0, 0.005)
        home_loc = (home_x, home_y)

        work_loc = None
        if segment in ["R4_CommuterResident", "F3_IncomingWorker"]:
            work_x = center_loc[0] + base_rng.normal(0, 0.005)
            work_y = center_loc[1] + base_rng.normal(0, 0.005)
            work_loc = (work_x, work_y)

        agent_rng = np.random.default_rng(seed + i + 1)
        agent = create_agent(i, segment, agent_rng, home_loc, work_loc)
        agents.append(agent)

    return agents


def export_agent_profiles(agents: List[Agent], output_path: str):
    """에이전트 프로필을 JSON으로 내보내기"""
    profiles = []

    for agent in agents:
        seg_data = SEGMENT_DATA[agent.segment]

        profile = {
            "id": agent.id,
            "segment": {
                "code": agent.segment,
                "name": seg_data["name"],
                "description": seg_data["description"],
                "characteristics": seg_data["characteristics"]
            },
            "location": {
                "home": {"x": agent.home_location[0], "y": agent.home_location[1]},
                "work": {"x": agent.work_location[0], "y": agent.work_location[1]} if agent.work_location else None
            },
            "preferences": {
                "spicy_pref": round(agent.spicy_pref, 3),
                "stimulation_pref": round(agent.stimulation_pref, 3),
                "novelty_seeking": round(agent.novelty_seeking, 3),
                "visual_importance": round(agent.visual_importance, 3),
                "comfort_importance": round(agent.comfort_importance, 3),
                "solo_pref": round(agent.solo_pref, 3)
            },
            "constraints": {
                "hygiene_threshold": round(agent.hygiene_threshold, 3)
            },
            "habits": {
                "repeat_tolerance": round(agent.repeat_tolerance, 3),
                "memory_decay": round(agent.memory_decay, 3),
                "review_propensity": round(agent.review_propensity, 3),
                "influence_sensitivity": round(agent.influence_sensitivity, 3)
            },
            "timeblock_presence": seg_data.get("presence_by_timeblock", {}),
            "time_pressure_profile": seg_data.get("time_pressure_profile", {})
        }

        profiles.append(profile)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    print(f"[OK] {len(profiles)} agent profiles exported to {output_path}\n")


# ============================================================================
# PURE LLM DECISION - 완전한 자유 판단
# ============================================================================

def build_persona_prompt(agent: Agent, restaurants: List[Restaurant],
                         current_loc: Tuple[float, float], timeblock: str, t: int) -> str:
    """
    페르소나만 주입, 규칙/가이드 없음
    LLM이 완전히 자유롭게 판단
    """
    seg_data = SEGMENT_DATA[agent.segment]

    # Persona 설명
    persona = f"""# 당신의 정체성
당신은 {seg_data['name']}입니다.
{seg_data['description']}

## 성격 및 특성
{chr(10).join(f"- {c}" for c in seg_data['characteristics'])}

## 개인적 성향 (0~1 척도)
- 비주얼/분위기 중요도: {agent.visual_importance:.2f}
- 자극적인 음식 선호: {agent.stimulation_pref:.2f}
- 편안함 중요도: {agent.comfort_importance:.2f}
- 혼밥 선호도: {agent.solo_pref:.2f}
- 새로운 것 추구: {agent.novelty_seeking:.2f}
- 위생 민감도: {agent.hygiene_threshold:.2f}
- 같은 곳 재방문 허용도: {agent.repeat_tolerance:.2f}
- 리뷰 작성 성향: {agent.review_propensity:.2f}
- 리뷰 영향 민감도: {agent.influence_sensitivity:.2f}

## 현재 상태
- 시간: {TIMEBLOCK_DESC[timeblock]}
- 배고픔: {agent.hunger_level:.2f}
- 시간 여유: {1 - agent.time_pressure:.2f} (시간 압박: {agent.time_pressure:.2f})
- 피로도: {agent.fatigue_level:.2f}
- 현재 위치: ({current_loc[0]:.4f}, {current_loc[1]:.4f})
"""

    # 과거 경험
    if agent.memory:
        memory_str = "\n## 과거 방문 경험\n"
        for rest in restaurants:
            if rest.id in agent.memory:
                mem = agent.memory[rest.id]
                days_ago = t - mem['last_visit_t']
                memory_str += f"- {rest.name}: {mem['visit_count']}회 방문, 만족도 {mem['satisfaction_ema']:.2f}, {days_ago}일 전\n"
        persona += memory_str

    # 식당 목록 (Top 10으로 제한)
    rest_with_dist = [(r, euclidean_distance(current_loc, r.location)) for r in restaurants]
    rest_with_dist.sort(key=lambda x: x[1])
    nearby_restaurants = [r for r, _ in rest_with_dist[:10]]

    rest_list = "\n# 근처 식당 목록 (가까운 순)\n"
    for i, r in enumerate(nearby_restaurants):
        dist = euclidean_distance(current_loc, r.location) * 100  # meters
        rest_list += f"\n{i}. **{r.name}** ({r.업종})\n"
        rest_list += f"   - 거리: {dist:.0f}m\n"
        rest_list += f"   - 위생: {r.hygiene_level:.2f}, 비주얼: {r.visual_score:.2f}, 편안함: {r.comfort_level:.2f}\n"
        rest_list += f"   - 혼밥 적합도: {r.solo_accessibility:.2f}, 자극성: {r.stimulation_level:.2f}\n"
        rest_list += f"   - 리뷰: {r.review_count}개, 평점: {r.avg_review_score:.2f}\n"
        if r.novelty_flag > 0:
            rest_list += f"   - 🆕 최근 신메뉴/리뉴얼\n"

    # 완전 개방형 질문
    decision_prompt = f"""
{persona}
{rest_list}

---

# 질문
위 상황에서 당신이라면 어떻게 하시겠습니까?

식당을 선택하고, 방문 후의 느낌과 행동을 상상해서 답변해주세요.

**출력 형식 (JSON):**
{{
  "choice": <선택한 식당 번호 (0부터 시작)>,
  "reasoning": "<왜 이 식당을 선택했는지 간단히>",
  "will_visit": <true/false, 실제로 갈지 말지>,
  "expected_satisfaction": <0~1, 방문한다면 예상 만족도>,
  "will_write_review": <true/false, 리뷰 쓸지 말지>,
  "review_text": "<리뷰 내용, 쓴다면>"
}}

**중요**:
- 당신의 성향과 현재 상태를 고려하여 자유롭게 판단하세요.
- 제약이나 규칙은 없습니다. 당신이 실제로 할 법한 선택을 하세요.
- 가고 싶지 않으면 will_visit: false로 해도 됩니다.
"""

    return decision_prompt


def llm_pure_decision(agent: Agent, restaurants: List[Restaurant],
                     t: int, timeblock: str, max_retries: int = 3) -> Optional[Dict]:
    """
    완전한 LLM 의사결정
    반환: {choice, reasoning, will_visit, expected_satisfaction, will_write_review, review_text, chosen_restaurant}
    """
    if not client:
        # Fallback: 랜덤
        current_loc = get_current_location(agent, timeblock)
        candidates = sorted(restaurants, key=lambda r: euclidean_distance(current_loc, r.location))[:10]
        chosen = candidates[agent.rng.choice(len(candidates))]
        return {
            "choice": 0,
            "reasoning": "Fallback mode",
            "will_visit": True,
            "expected_satisfaction": 0.6,
            "will_write_review": False,
            "review_text": "",
            "chosen_restaurant": chosen
        }

    current_loc = get_current_location(agent, timeblock)

    # Top 10 nearby restaurants
    rest_with_dist = [(r, euclidean_distance(current_loc, r.location)) for r in restaurants]
    rest_with_dist.sort(key=lambda x: x[1])
    nearby_restaurants = [r for r, _ in rest_with_dist[:10]]

    prompt = build_persona_prompt(agent, restaurants, current_loc, timeblock, t)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 주어진 페르소나로 행동하는 시뮬레이션 에이전트입니다. 페르소나의 특성에 맞게 자연스럽게 판단하세요. JSON 형식으로만 답변하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=200,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # 유효성 검사
            if "choice" in result and 0 <= result["choice"] < len(nearby_restaurants):
                # 기본값 설정
                result.setdefault("will_visit", True)
                result.setdefault("expected_satisfaction", 0.5)
                result.setdefault("will_write_review", False)
                result.setdefault("review_text", "")
                result.setdefault("reasoning", "")

                # 실제 선택된 레스토랑 객체 추가
                result["chosen_restaurant"] = nearby_restaurants[result["choice"]]

                return result

        except Exception as e:
            if attempt == max_retries - 1:
                # 최종 fallback
                chosen = nearby_restaurants[agent.rng.choice(len(nearby_restaurants))]
                return {
                    "choice": 0,
                    "reasoning": f"Fallback after error: {str(e)}",
                    "will_visit": True,
                    "expected_satisfaction": 0.6,
                    "will_write_review": False,
                    "review_text": "",
                    "chosen_restaurant": chosen
                }

    return None


# ============================================================================
# SIMULATION
# ============================================================================

def run_pure_llm_simulation(agents: List[Agent], restaurants: List[Restaurant],
                           n_days: int, seed: int, strategy: Optional[Dict] = None,
                           verbose: bool = False) -> Dict[str, Any]:
    """Pure LLM 시뮬레이션"""

    # Strategy 적용
    if strategy:
        target_id = strategy.get("target_restaurant_id")
        for rest in restaurants:
            if rest.id == target_id:
                if strategy.get("improve_hygiene"):
                    rest.hygiene_level = min(rest.hygiene_level + 0.15, 1.0)
                if strategy.get("new_menu"):
                    rest.novelty_flag = 1.0
                if strategy.get("improve_visual"):
                    rest.visual_score = min(rest.visual_score + 0.12, 1.0)
                rest.strategy_flags = strategy
                break

    # Metrics
    total_visits = 0
    total_decisions = 0  # LLM 호출 횟수
    visits_by_segment = {seg: 0 for seg in SEGMENT_DATA.keys()}
    visits_by_timeblock = {tb: 0 for tb in TIMEBLOCKS}
    visits_by_restaurant = {r.id: 0 for r in restaurants}
    agent_visit_counts = {a.id: 0 for a in agents}
    reviews_generated = []
    decision_logs = []

    for day in range(n_days):
        if verbose and day % 2 == 0:
            print(f"  Day {day}/{n_days}...")

        for timeblock in TIMEBLOCKS:
            for agent in agents:
                seg_data = SEGMENT_DATA[agent.segment]
                presence_prob = seg_data.get("presence_by_timeblock", {}).get(timeblock, 0.1)

                if agent.rng.random() > presence_prob:
                    continue

                # Update states
                agent.hunger_level = np.clip(agent.rng.beta(3, 2), 0, 1)
                time_pressure_mean = seg_data.get("time_pressure_profile", {}).get(timeblock, 0.3)
                agent.time_pressure = np.clip(agent.rng.normal(time_pressure_mean, 0.15), 0, 1)

                # LLM 의사결정
                total_decisions += 1
                decision = llm_pure_decision(agent, restaurants, day, timeblock)

                if decision is None or not decision.get("will_visit", False):
                    continue

                # 선택된 식당
                chosen = decision["chosen_restaurant"]

                # 방문
                total_visits += 1
                visits_by_segment[agent.segment] += 1
                visits_by_timeblock[timeblock] += 1
                visits_by_restaurant[chosen.id] += 1
                agent_visit_counts[agent.id] += 1

                # 만족도 (LLM이 예측한 값)
                satisfaction = decision.get("expected_satisfaction", 0.5)

                # 메모리 업데이트
                if chosen.id not in agent.memory:
                    agent.memory[chosen.id] = {
                        "last_visit_t": day,
                        "visit_count": 1,
                        "satisfaction_ema": satisfaction
                    }
                else:
                    mem = agent.memory[chosen.id]
                    mem["last_visit_t"] = day
                    mem["visit_count"] += 1
                    alpha = 0.3
                    mem["satisfaction_ema"] = alpha * satisfaction + (1 - alpha) * mem["satisfaction_ema"]

                # 리뷰 (LLM이 판단)
                if decision.get("will_write_review", False) and decision.get("review_text"):
                    reviews_generated.append({
                        "restaurant_id": chosen.id,
                        "restaurant_name": chosen.name,
                        "agent_id": agent.id,
                        "segment": agent.segment,
                        "satisfaction": satisfaction,
                        "review_text": decision["review_text"],
                        "day": day
                    })

                    # 리뷰 수/평점 업데이트
                    alpha_review = 0.15
                    chosen.avg_review_score = (
                        alpha_review * satisfaction + (1 - alpha_review) * chosen.avg_review_score
                    )
                    chosen.review_count += 1

                # 로그
                decision_logs.append({
                    "day": day,
                    "timeblock": timeblock,
                    "agent_id": agent.id,
                    "segment": agent.segment,
                    "restaurant": chosen.name,
                    "reasoning": decision.get("reasoning", ""),
                    "satisfaction": satisfaction
                })

    # Revisit rate
    revisit_agents = sum(1 for count in agent_visit_counts.values() if count > 1)
    revisit_rate = revisit_agents / max(len(agents), 1)

    return {
        "total_visits": total_visits,
        "total_decisions": total_decisions,
        "visits_by_segment": visits_by_segment,
        "visits_by_timeblock": visits_by_timeblock,
        "visits_by_restaurant": visits_by_restaurant,
        "revisit_rate": revisit_rate,
        "restaurant_reviews": {
            r.id: {"count": r.review_count, "avg_score": r.avg_review_score, "name": r.name}
            for r in restaurants
        },
        "reviews_generated": reviews_generated,
        "decision_logs": decision_logs
    }


# ============================================================================
# DEMO
# ============================================================================

def main():
    print("망원동 Pure LLM 시뮬레이션 (하드코딩 제거)\n")

    # API 키 필수 (config.py 사용)
    api_key = config.OPENAI_API_KEY
    if not api_key or "your_openai_api_key" in api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        
    if not api_key:
        print("API 키가 config.py에 설정되지 않았습니다.")
        return

    init_llm_client(api_key)
    print("[OK] LLM client initialized\n")

    SEED = 42
    N_AGENTS = 30  # Pure LLM으로 30명
    N_DAYS = 5

    # 데이터 로드
    script_dir = Path(__file__).parent
    csv_path = script_dir.parent / "data" / "raw" / "stores.csv"

    print(f"Loading stores.csv...")
    restaurants = load_restaurants_from_csv(str(csv_path), sample_size=50)
    print(f"[OK] {len(restaurants)} restaurants\n")

    center_loc = (126.906, 37.556)
    agents_before = create_demo_agents(N_AGENTS, SEED, center_loc)

    # Agent profiles export
    export_agent_profiles(agents_before, str(script_dir / "agent_profiles.json"))

    TARGET_RESTAURANT_ID = restaurants[0].id
    print(f"설정:")
    print(f"  - 에이전트: {N_AGENTS}명")
    print(f"  - 기간: {N_DAYS}일")
    print(f"  - 식당: {len(restaurants)}개")
    print(f"  - 타겟: {restaurants[0].name}\n")

    print("BEFORE 시뮬레이션 (Pure LLM)...")
    metrics_before = run_pure_llm_simulation(agents_before, restaurants, N_DAYS, SEED, verbose=True)

    # AFTER
    restaurants_after = load_restaurants_from_csv(str(csv_path), sample_size=50)
    agents_after = create_demo_agents(N_AGENTS, SEED, center_loc)

    strategy = {
        "target_restaurant_id": TARGET_RESTAURANT_ID,
        "improve_hygiene": True,
        "new_menu": True,
        "improve_visual": True
    }

    print("\nAFTER 시뮬레이션 (Pure LLM)...")
    metrics_after = run_pure_llm_simulation(agents_after, restaurants_after, N_DAYS, SEED, strategy=strategy, verbose=True)

    # 결과
    print("\n" + "="*70)
    print("Pure LLM 시뮬레이션 결과")
    print("="*70)

    print(f"\n[LLM 호출]")
    print(f"  Before: {metrics_before['total_decisions']}회")
    print(f"  After:  {metrics_after['total_decisions']}회")

    print(f"\n[방문]")
    print(f"  Before: {metrics_before['total_visits']} (의사결정 대비 {metrics_before['total_visits']/max(metrics_before['total_decisions'],1)*100:.1f}%)")
    print(f"  After:  {metrics_after['total_visits']} (의사결정 대비 {metrics_after['total_visits']/max(metrics_after['total_decisions'],1)*100:.1f}%)")

    print(f"\n[타겟 레스토랑: {restaurants[0].name}]")
    before_visits = metrics_before['visits_by_restaurant'].get(TARGET_RESTAURANT_ID, 0)
    after_visits = metrics_after['visits_by_restaurant'].get(TARGET_RESTAURANT_ID, 0)
    print(f"  방문: {before_visits} → {after_visits} ({after_visits - before_visits:+})")

    # 리뷰 샘플
    if metrics_after['reviews_generated']:
        print(f"\n[생성된 리뷰 샘플]")
        for rev in metrics_after['reviews_generated'][:5]:
            print(f"  - {rev['restaurant_name']} ({rev['segment']})")
            print(f"    \"{rev['review_text']}\"")

    # Decision log 샘플
    if metrics_after['decision_logs']:
        print(f"\n[의사결정 로그 샘플]")
        for log in metrics_after['decision_logs'][:3]:
            print(f"  Day {log['day']}, {log['timeblock']}: {log['segment']} → {log['restaurant']}")
            print(f"    이유: {log['reasoning']}")

    print("\n완료!")


if __name__ == "__main__":
    main()
