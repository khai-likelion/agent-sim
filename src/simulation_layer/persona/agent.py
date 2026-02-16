"""
personas_160.md 기반 에이전트 모듈.

md 파일의 '#### P001 / ... ~ ---' 단위가 에이전트 하나.
자연어 페르소나가 LLM 의사결정의 유일한 근거.
"""

import random
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# 유동 에이전트 초기 위치 후보 (망원동 주요 정류장/거점)
FLOATING_LOCATIONS = {
    "MANGWON_STATION": {"lat": 37.556069, "lng": 126.910108},
    "STATION_STOP": {"lat": 37.556097, "lng": 126.910283},
    "MARKET_STOP": {"lat": 37.557637, "lng": 126.905902},
    "ENTRANCE_STOP": {"lat": 37.557944, "lng": 126.907324},
    "RIVER_ENTRY_STOP": {"lat": 37.550704, "lng": 126.912613},
    "MANGWON_HANGANG_PARK_ENTRANCE": {"lat": 37.551025, "lng": 126.898877},
}

# 상주 에이전트 주거지 좌표
RESIDENT_LOCATIONS = {
    "아파트": [
        {"lat": 37.558682, "lng": 126.898706},  # 아파트1
        {"lat": 37.553427, "lng": 126.904841},  # 아파트2
        {"lat": 37.559734, "lng": 126.901044},  # 아파트3
    ],
    "빌라": [
        {"lat": 37.553972, "lng": 126.903356},  # 빌라1
        {"lat": 37.555740, "lng": 126.904030},  # 빌라2
        {"lat": 37.554726, "lng": 126.908740},  # 빌라3
    ],
    "주택": [
        {"lat": 37.555097, "lng": 126.907753},  # 주택1
        {"lat": 37.554986, "lng": 126.902714},  # 주택2
        {"lat": 37.552770, "lng": 126.905787},  # 주택3
    ],
}


@dataclass
class VisitRecord:
    """방문 기록 (Memory Module용)"""
    visit_datetime: str
    store_name: str
    category: str
    taste_rating: int
    value_rating: int
    atmosphere_rating: int = 3
    comment: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerativeAgent:
    """
    personas_160.md에서 로드된 에이전트.
    """
    id: int
    persona_id: str          # P001~P113, R001~R047

    group_type: str           # 생활베이스형, 사적모임형, 공적모임형, 가족모임형
    group_size: int           # 1, 2, 4
    generation: str           # Z1, Z2, Y, X, S
    gender_composition: str   # 남, 여, 혼성
    agent_type: str           # 유동, 상주

    natural_language_persona: str = ""
    housing_type: Optional[str] = None

    recent_history: List[VisitRecord] = field(default_factory=list)
    current_location: Optional[Any] = field(default=None, repr=False)
    home_location: Tuple[float, float] = field(default=(37.5565, 126.9029), repr=False)

    # 유동 에이전트 전용: 진입 지점 (home_location과 분리)
    entry_point: Optional[Tuple[float, float]] = field(default=None, repr=False)
    entry_time_slot: Optional[str] = field(default=None, repr=False)  # "아침", "점심", "저녁", "야식"
    left_mangwon: bool = field(default=False, repr=False)  # 망원동 떠남 여부

    @property
    def is_resident(self) -> bool:
        return self.agent_type == "상주"

    @property
    def is_floating(self) -> bool:
        return self.agent_type == "유동"

    @property
    def segment(self) -> str:
        return f"{self.agent_type}_{self.group_type}_{self.group_size}인"

    def add_visit(self, store_name: str, category: str, taste_rating: int, value_rating: int, atmosphere_rating: int = 3, visit_datetime: str = "", comment: str = ""):
        record = VisitRecord(
            visit_datetime=visit_datetime or datetime.now().isoformat(),
            store_name=store_name, category=category,
            taste_rating=taste_rating, value_rating=value_rating,
            atmosphere_rating=atmosphere_rating,
            comment=comment,
        )
        self.recent_history.append(record)
        if len(self.recent_history) > 15:
            self.recent_history = self.recent_history[-15:]

    def get_meals_today(self, current_date: str) -> List['VisitRecord']:
        """현재 시뮬레이션 날짜 기준 오늘 식사 기록만 반환"""
        return [v for v in self.recent_history if v.visit_datetime.startswith(current_date)]

    def get_recent_stores(self, n: int = 5) -> List[str]:
        return [v.store_name for v in self.recent_history[-n:]]

    def get_recent_categories(self, n: int = 5) -> List[str]:
        return [v.category for v in self.recent_history[-n:]]

    def has_visited(self, store_name: str) -> bool:
        return any(v.store_name == store_name for v in self.recent_history)

    def get_memory_context(self, current_date: str = "") -> str:
        lines = []

        # 오늘 식사 기록 (current_date가 있을 때)
        if current_date:
            today_meals = self.get_meals_today(current_date)
            if today_meals:
                lines.append(f"오늘의 식사 기록 (현재 {len(today_meals)}끼 외식 완료):")
                for v in today_meals:
                    time_part = v.visit_datetime[11:16] if len(v.visit_datetime) > 16 else ""
                    lines.append(f"  - {time_part} {v.store_name} ({v.category})")
            else:
                lines.append("오늘의 식사 기록: 아직 없음")
            lines.append("")

        # 최근 방문 기록 (이전 날짜 포함)
        if not self.recent_history:
            if not lines:
                return "최근 방문 기록이 없습니다."
        else:
            rating_text = {1: "매우별로", 2: "별로", 3: "보통", 4: "좋음", 5: "매우좋음"}
            lines.append("당신의 과거 경험:")
            for v in self.recent_history[-5:]:
                line = f"  - {v.store_name} ({v.category}): {rating_text.get(v.taste_rating, '?')}"
                if v.comment:
                    line += f' → "{v.comment}"'
                lines.append(line)

        return "\n".join(lines)

    def get_persona_summary(self) -> str:
        header = f"[{self.persona_id}]\n"
        header += f"- 유형: {self.agent_type} / {self.group_type} / {self.group_size}인\n"
        header += f"- 세대: {self.generation}\n"
        header += f"- 성별 구성: {self.gender_composition}\n"
        if self.housing_type:
            header += f"- 주거 유형: {self.housing_type}\n"
        header += "\n"
        if self.natural_language_persona:
            header += self.natural_language_persona
        return header

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "persona_id": self.persona_id,
            "group_type": self.group_type, "group_size": self.group_size,
            "generation": self.generation, "gender_composition": self.gender_composition,
            "agent_type": self.agent_type, "segment": self.segment,
            "housing_type": self.housing_type,
            "home_location": list(self.home_location),
            "entry_point": list(self.entry_point) if self.entry_point else None,
            "entry_time_slot": self.entry_time_slot,
            "recent_history": [v.to_dict() for v in self.recent_history],
        }


# ── md 파서 ───────────────────────────────────────────────

# #### P001 / 생활베이스형 / [유동] / 1인 / Z1 / 남
# #### R001 / 생활베이스형 / [상주] / 다세대(빌라) / 1인 / Z1 / 남
_HEADER_RE = re.compile(r"^####\s+([PR]\d+)\s*/\s*(.+?)\s*/\s*\[(.+?)\]\s*/\s*(.+)$")


def _parse_header(line: str) -> Optional[Dict[str, Any]]:
    m = _HEADER_RE.match(line.strip())
    if not m:
        return None

    persona_id = m.group(1)
    group_type = m.group(2).strip()
    agent_type = m.group(3).strip()
    rest = [s.strip() for s in m.group(4).split("/")]

    if agent_type == "상주":
        housing_type, size_str, generation, gender = rest[0], rest[1], rest[2], rest[3]
    else:
        housing_type = None
        size_str, generation, gender = rest[0], rest[1], rest[2]

    return {
        "persona_id": persona_id, "group_type": group_type,
        "agent_type": agent_type, "housing_type": housing_type,
        "group_size": int(size_str.replace("인", "")),
        "generation": generation, "gender_composition": gender,
    }


def load_personas_from_md(
    md_path: Optional[Path] = None,
    agent_type_filter: Optional[str] = None,
) -> List[GenerativeAgent]:
    """
    personas_160.md를 파싱하여 에이전트 리스트를 반환한다.

    md 구조:
        #### P001 / 그룹유형 / [유동|상주] / ...
        (본문 = 자연어 페르소나)
        ---
    """
    if md_path is None:
        md_path = Path(__file__).parent / "personas_160.md"

    lines = md_path.read_text(encoding="utf-8").split("\n")

    # 헤더 위치 수집
    headers: List[Tuple[int, Dict[str, Any]]] = []
    for i, line in enumerate(lines):
        parsed = _parse_header(line)
        if parsed:
            headers.append((i, parsed))

    agents: List[GenerativeAgent] = []
    for idx, (line_num, attrs) in enumerate(headers):
        end = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines)
        body = "\n".join(lines[line_num + 1 : end]).strip().strip("-").strip()

        if agent_type_filter and attrs["agent_type"] != agent_type_filter:
            continue

        # 에이전트 유형별 초기 위치 설정
        entry_point = None
        entry_time_slot = None

        if attrs["agent_type"] == "유동":
            # 유동 에이전트: entry_point 설정 (home_location과 분리)
            loc = random.choice(list(FLOATING_LOCATIONS.values()))
            entry_point = (loc["lat"], loc["lng"])
            home = (0.0, 0.0)  # 유동은 home 없음

            # 세그먼트별 진입 시간대
            group_type = attrs["group_type"]
            if group_type == "공적모임형":
                entry_time_slot = "점심"  # 직장인 → 점심
            elif group_type == "사적모임형":
                entry_time_slot = random.choice(["점심", "저녁"])  # 친구/데이트
            elif group_type == "가족모임형":
                entry_time_slot = random.choice(["아침", "점심"])  # 가족 나들이
            else:  # 생활베이스형
                entry_time_slot = random.choice(["아침", "점심", "저녁"])  # 자유
        elif attrs["agent_type"] == "상주" and attrs["group_type"] == "가족모임형" and attrs["group_size"] == 4:
            # 상주 + 가족모임형 + 4인 → 아파트1~3 중 랜덤
            loc = random.choice(RESIDENT_LOCATIONS["아파트"])
            home = (loc["lat"], loc["lng"])
        elif attrs["agent_type"] == "상주" and attrs.get("housing_type") == "단독·연립(주택)":
            # 상주 + 단독·연립(주택) → 주택1~3 중 랜덤
            loc = random.choice(RESIDENT_LOCATIONS["주택"])
            home = (loc["lat"], loc["lng"])
        elif attrs["agent_type"] == "상주" and attrs.get("housing_type") == "다세대(빌라)":
            # 상주 + 다세대(빌라) → 빌라1~3 중 랜덤
            loc = random.choice(RESIDENT_LOCATIONS["빌라"])
            home = (loc["lat"], loc["lng"])
        else:
            home = (37.5565, 126.9029)  # 상주 에이전트 기본 위치

        agents.append(GenerativeAgent(
            id=len(agents) + 1,
            persona_id=attrs["persona_id"],
            group_type=attrs["group_type"],
            group_size=attrs["group_size"],
            generation=attrs["generation"],
            gender_composition=attrs["gender_composition"],
            agent_type=attrs["agent_type"],
            natural_language_persona=body,
            housing_type=attrs["housing_type"],
            home_location=home,
            entry_point=entry_point,
            entry_time_slot=entry_time_slot,
        ))

    return agents


if __name__ == "__main__":
    agents = load_personas_from_md()
    print(f"총 {len(agents)}명 로드")
    for t in sorted({a.agent_type for a in agents}):
        print(f"  {t}: {sum(1 for a in agents if a.agent_type == t)}명")
    print(f"\n=== 샘플 (P001) ===")
    print(agents[0].get_persona_summary())
