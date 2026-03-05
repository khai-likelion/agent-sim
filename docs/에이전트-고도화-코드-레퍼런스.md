# 에이전트 고도화 코드 레퍼런스

> **통합 문서 안내:** 설정·코드·고도화를 수도 코드·텍스트 위주로 한 파일에 정리한 **[에이전트-레퍼런스.md](./에이전트-레퍼런스.md)** 를 권장합니다.  
> 이 문서는 상세 코드 블록을 포함하는 레거시 버전입니다.

---

## 목차

| # | 항목 | 핵심 파일 |
|---|------|----------|
| [1](#1-리뷰-멘트-다양성) | 리뷰 멘트 다양성 | action_algorithm.py, step4_evaluate.txt |
| [2](#2-4끼-전부-외식--memory_context) | Step 1 LLM 전환 + memory_context | action_algorithm.py, agent.py |
| [3](#3-유동-초기위치--entry_point) | 유동 entry_point, entry_time_slot | agent.py, run_generative_simulation.py |
| [5](#5-유동인구-교체-비율) | 요일별 유동인구 + 재방문 10% | run_generative_simulation.py |
| [6](#6-상주-초기위치-오프셋) | 상주 150m 랜덤 오프셋 | agent.py |
| [8](#8-softmax--계층화-샘플링) | Softmax + 계층화 샘플링 | global_store.py |
| [12](#12-아침-이전-식사로-충분-버그) | 아침 사유 제외 | *(프롬프트만 해당)* |
| [13](#13-category_alias--json-metadata) | CATEGORY_ALIAS + JSON fallback | global_store.py |
| [14](#14-time_slot_categories) | TIME_SLOT_CATEGORIES | action_algorithm.py |
| [15](#15-한강-위-노드-제거) | 한강 위 노드 제거 | street_network.py |
| [25](#25-step-5-루프) | Step 5 루프 + STORE_VISIT_ACTIONS | action_algorithm.py |
| [26](#26-session_activity) | session_activity | action_algorithm.py |
| [27](#27-고려사항-동적-경고) | [고려사항] 동적 경고 | action_algorithm.py |

---

## 1. 리뷰 멘트 다양성

<details>
<summary><strong>▶ tone_instruction (세대별 말투) + 평가 관점 LLM 자율 선택</strong></summary>

### 설정 이유
모든 에이전트가 동일한 "한줄 평가" 톤 → 세대별 차별화로 리뷰 다양성 확보.

### 실제 코드 (`action_algorithm.py`)

```python
# 1. 말투 설정 (페르소나 기반)
tone_instruction = "자연스럽게 작성하세요."
if agent.generation in ["Z1", "Z2"]:
    tone_instruction = "Z세대 말투(존나, 개꿀맛, ㅋ, 이모티콘 등)를 사용해서 친구한테 말하듯이 아주 솔직하고 짧게 작성하세요."
elif agent.generation == "Y":
    tone_instruction = "밀레니얼 세대 말투로, 적당히 트렌디하면서도 정보성 있게 작성하세요."
elif agent.generation == "X":
    tone_instruction = "X세대 말투로, 점잖으면서도 꼼꼼하게 분석하듯이 작성하세요."
elif agent.generation == "S":
    tone_instruction = "어르신 말투로, 구체적이고 진중하게 작성하세요."

# 2. 평가 관점: 프롬프트에 6개 관점 제시, LLM이 페르소나에 맞는 하나를 스스로 선택
# (focus_aspect 변수 제거 — step4_evaluate.txt에 관점 목록 내장)

prompt = render_prompt(
    STEP4_EVALUATE,
    ...
    tone_instruction=tone_instruction,
)
```

### 프롬프트 변수 (`step4_evaluate.txt`)
- `{tone_instruction}` — 위에서 생성된 세대별 문구
- 평가 관점: 6개 관점 목록이 프롬프트에 내장, LLM 자율 선택

</details>

---

## 2. 4끼 전부 외식 + memory_context

<details>
<summary><strong>▶ get_memory_context (당일 식사 + 과거 경험)</strong></summary>

### 설정 이유
Step 2/3에서 recent_categories, recent_stores만 전달 → 평점·코멘트 없어 "이전에 별로였다" 판단 불가.  
당일 식사 기록 + 과거 평점·코멘트를 통합해 LLM이 맥락 기반으로 판단하도록.

### 실제 코드 (`agent.py`)

```python
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
    rating_text = {1: "매우별로", 2: "별로", 3: "보통", 4: "좋음", 5: "매우좋음"}
    lines.append("당신의 과거 경험:")
    for v in self.recent_history[-5:]:
        line = f"  - {v.store_name} ({v.category}): {rating_text.get(v.taste_rating, '?')}"
        if v.comment:
            line += f' → "{v.comment}"'
        lines.append(line)

    return "\n".join(lines)
```

### 사용처
- Step 1, 2, 3, 5 모두 `agent.get_memory_context(current_date)` 전달

</details>

---

## 3. 유동 초기위치 + entry_point

<details>
<summary><strong>▶ entry_point, entry_time_slot, left_mangwon</strong></summary>

### 설정 이유
유동이 역/정류장을 "집"으로 인식하고 아침부터 전원 활동 → entry_point 분리 + 세그먼트별 진입 시간.

### 실제 코드 (`agent.py`)

```python
# 유동 에이전트 전용: 진입 지점 (home_location과 분리)
entry_point: Optional[Tuple[float, float]] = field(default=None, repr=False)
entry_time_slot: Optional[str] = field(default=None, repr=False)  # "아침", "점심", "저녁", "야식"
left_mangwon: bool = field(default=False, repr=False)  # 망원동 떠남 여부
```

### 유동 매일 초기화 (`run_generative_simulation.py`)

```python
# 유동 에이전트: 매일 초기화 (entry_point에서 시작, left_mangwon 리셋)
for agent in daily_floating:
    agent.left_mangwon = False
    if agent.entry_point:
        lat, lng = agent.entry_point
    else:
        from src.simulation_layer.persona.agent import FLOATING_LOCATIONS
        loc = random.choice(list(FLOATING_LOCATIONS.values()))
        lat, lng = loc["lat"], loc["lng"]
    agent_locations[agent.id] = network.initialize_agent_location(lat, lng)
```

### Step 5 선택지 분리 (`action_algorithm.py`)
- **상주**: 집에서_쉬기 O, 망원동_떠나기 X
- **유동**: 집에서_쉬기 X, 망원동_떠나기 O

</details>

---

## 5. 유동인구 교체 비율

<details>
<summary><strong>▶ DAILY_FLOATING_COUNT_BY_DAY + REVISIT_RATE</strong></summary>

### 설정 이유
매일 53명 고정 → 요일별(주말 증가) + 전날 방문자 10% 재방문 우선 포함.

### 실제 코드 (`run_generative_simulation.py`)

```python
DAILY_FLOATING_COUNT_BY_DAY = {
    "월": 51, "화": 51, "수": 51, "목": 51, "금": 51,
    "토": 58, "일": 58,
}
DAILY_FLOATING_AGENT_COUNT = 53  # 기본값 (fallback)
REVISIT_RATE = 0.10  # 재방문율: 전날 방문자 중 10% 다음날 포함
```

```python
# 요일별 유동 에이전트 수 차등 + 재방문율 적용
base_count = DAILY_FLOATING_COUNT_BY_DAY.get(weekday, DAILY_FLOATING_AGENT_COUNT)
daily_floating_count = min(base_count, len(floating_agents))

# 재방문: 전날 방문자 중 일부를 우선 포함
revisit_agents = []
if day_idx > 0 and hasattr(run_simulation, '_prev_day_visitors'):
    prev_visitors = run_simulation._prev_day_visitors
    revisit_count = max(1, int(len(prev_visitors) * REVISIT_RATE))
    revisit_agents = random.sample(prev_visitors, min(revisit_count, len(prev_visitors)))

# 나머지는 새로운 에이전트에서 샘플링
remaining_pool = [a for a in floating_agents if a not in revisit_agents]
new_count = daily_floating_count - len(revisit_agents)
new_agents = random.sample(remaining_pool, min(new_count, len(remaining_pool)))
daily_floating = revisit_agents + new_agents
```

</details>

---

## 6. 상주 초기위치 오프셋

<details>
<summary><strong>▶ _apply_location_offset (150m)</strong></summary>

### 설정 이유
주거유형별 3좌표 고정 → 같은 좌표에 6~7명 겹침 → 중심점 + 150m 랜덤 오프셋으로 다양화.

### 실제 코드 (`agent.py`)

```python
RESIDENT_OFFSET_RADIUS_M = 150  # 구역 중심점 기준 최대 오프셋 반경 (미터)

def _apply_location_offset(base_lat: float, base_lng: float, max_radius_m: float = RESIDENT_OFFSET_RADIUS_M) -> Tuple[float, float]:
    """구역 중심 좌표에 랜덤 오프셋을 적용해 에이전트마다 다른 위치 부여."""
    offset_m = random.uniform(0, max_radius_m)
    angle = random.uniform(0, 2 * math.pi)
    dlat = (offset_m * math.cos(angle)) / 111320
    dlng = (offset_m * math.sin(angle)) / (111320 * math.cos(math.radians(base_lat)))
    return (base_lat + dlat, base_lng + dlng)
```

### 적용 예시
```python
# 아파트1 좌표에 오프셋 적용
home = _apply_location_offset(loc["lat"], loc["lng"])
```

</details>

---

## 8. Softmax + 계층화 샘플링

<details>
<summary><strong>▶ search_ranked_stores — Softmax T=0.5 + 60/20/20 계층화</strong></summary>

### 설정 이유
exploit 10 + explore 5 고정 → 롱테일 매장 0.7% 진입.
Softmax 확률 비례 추출 + 상위60%/중위20%/하위20%로 롱테일도 LLM 후보에 포함.

### 실제 코드 (`global_store.py`)

```python
# Softmax 확률 계산
max_score = max(scores)
exp_scores = [exp((s - max_score) / temperature) for s in scores]
total_exp = sum(exp_scores)
probs = [e / total_exp for e in exp_scores]

# 계층화 샘플링: 상위(60%) + 중위(20%) + 하위(20%)
k = min(sample_k, len(pool))
sorted_indices = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
n = len(sorted_indices)
top_end = max(1, n // 3)
mid_end = max(top_end + 1, 2 * n // 3)
tier_top = sorted_indices[:top_end]
tier_mid = sorted_indices[top_end:mid_end]
tier_low = sorted_indices[mid_end:]

k_top = max(1, round(k * 0.60))
k_mid = max(1, round(k * 0.20))
k_low = k - k_top - k_mid
```

- `temperature`: 기본 0.5 (인기/롱테일 갭 완화)
- 스코어: 별점×1.5 + 에이전트평점×2.0 + log1p(min(리뷰,50))×0.3 - 거리×0.3 + feature_quality×1.5
  - `feature_quality` = 맛×0.30 + 가성비×0.25 + 청결×0.20 + 서비스×0.15 + 분위기×0.10
  - `review_metrics.feature_scores` 기반 (X_to_Sim 전략 적용 결과가 랭킹에 반영됨)

</details>

---

## 12. 아침 "이전 식사로 충분" 버그

<details>
<summary><strong>▶ 아침/첫 끼 시 "이전 식사로 충분" 사유 제외</strong></summary>

### 설정 이유
아침은 그날 첫 끼니 → "이전 식사로 충분" 출력 시 모순.

### 관련 코드
`action_algorithm.py`의 Step 1에서 `meals_today == 0` 또는 `time_slot == "아침"`일 때  
`reasons_skip` 풀에서 "이전 식사로 충분" 제외 후 LLM에 전달.

> 현재 Step 1은 **LLM 순수 의사결정**으로 전환되어, 선택지와 판단 기준만 프롬프트에 포함.  
> 사유 풀은 기존 확률 기반 로직에서 사용되었으나, LLM 전환 후에는 프롬프트의 선택지 텍스트만 참고.

### 프롬프트 (`step1_destination.txt`)
```
2. 이 시간대에는 먹지 않는다 (배가 안 고픔, 이전 식사로 충분, 다이어트 중 등)
```
→ LLM이 "아침 + 첫 끼" 맥락에서 "이전 식사로 충분"을 비현실적이라 스스로 회피하도록 유도.

</details>

---

## 13. CATEGORY_ALIAS + JSON metadata

<details>
<summary><strong>▶ match_category + CATEGORY_ALIAS</strong></summary>

### 설정 이유
LLM "카페" vs JSON "커피-음료" 등 명칭 불일치 → 매칭 실패 → 카페/주점 방문 0건.

### 실제 코드 (`global_store.py`)

```python
CATEGORY_ALIAS = {
    "카페": "커피-음료",
    "커피": "커피-음료",
    "디저트": "제과점",
    "베이커리": "제과점",
    "브런치": "양식음식점",
    "이자카야": "호프-간이주점",
    "포차": "호프-간이주점",
    "와인바": "호프-간이주점",
    "술집": "호프-간이주점",
    "막걸리": "호프-간이주점",
    "칵테일바": "호프-간이주점",
    "칵테일": "호프-간이주점",
}

def match_category(query: str, store_category: str) -> bool:
    """LLM 선택 카테고리와 매장 카테고리 매칭.
    1차: 부분 문자열 매칭 (예: "한식" in "한식음식점")
    2차: CATEGORY_ALIAS 변환 후 매칭 (예: "카페" → "커피-음료" in "커피-음료")
    """
    q = query.lower()
    sc = (store_category or "").lower()
    if q in sc:
        return True
    alias = CATEGORY_ALIAS.get(query, "").lower()
    if alias and alias in sc:
        return True
    return False
```

### 사용처
- `search_ranked_stores()` 카테고리 선필터
- `get_stores_by_category()`, `get_stores_in_budget()` 등

</details>

<details>
<summary><strong>▶ JSON metadata fallback (좌표·카테고리)</strong></summary>

### 설정 이유
JSON 최상위에 `x`, `y`, `category` 없고 `metadata.x`, `metadata.sector`에만 있음 → 반경검색 0건.

### 처리 방식
`load_from_json_dir()`에서:
- `category` 없으면 `metadata.sector` 사용
- `x`, `y` 없으면 `metadata.x`, `metadata.y` 사용

</details>

---

## 14. TIME_SLOT_CATEGORIES

<details>
<summary><strong>▶ 시간대별 전체 카테고리 풀</strong></summary>

### 설정 이유
`destination_type = "식당"` 고정 → 아침 커피, 저녁 술자리 선택 구조적 불가.  
시간대별 전체 카테고리를 Step 2에 전달해 LLM이 자유롭게 선택.

### 실제 코드 (`action_algorithm.py`)

```python
DESTINATION_CATEGORIES = {
    "식당": ["한식", "중식", "일식", "양식", "분식", "패스트푸드", "국밥", "찌개", "고기", "치킨"],
    "카페": ["카페", "커피", "디저트", "베이커리", "브런치"],
    "주점": ["호프", "이자카야", "포차", "와인바", "술집", "막걸리", "칵테일바", "칵테일"],
}

TIME_SLOT_CATEGORIES = {
    "아침": DESTINATION_CATEGORIES["식당"] + DESTINATION_CATEGORIES["카페"],
    "점심": DESTINATION_CATEGORIES["식당"] + DESTINATION_CATEGORIES["카페"],
    "저녁": DESTINATION_CATEGORIES["식당"] + DESTINATION_CATEGORIES["주점"] + DESTINATION_CATEGORIES["카페"],
    "야식": DESTINATION_CATEGORIES["식당"] + DESTINATION_CATEGORIES["주점"],
}

# Step 2에서 사용
available_categories = TIME_SLOT_CATEGORIES.get(time_slot, DESTINATION_CATEGORIES["식당"])
```

</details>

---

## 15. 한강 위 노드 제거

<details>
<summary><strong>▶ load_graph() — lat < 37.550 노드 제거</strong></summary>

### 설정 이유
OSM 반경 2km에 한강 다리 도로 포함 → 에이전트가 한강 위를 걷는 것처럼 표시.

### 실제 코드 (`street_network.py`)

```python
def load_graph(self) -> None:
    self._graph = ox.graph_from_point(...)

    # 한강 위 노드 제거 (lat < 37.550)
    river_nodes = [n for n, d in self._graph.nodes(data=True) if d.get('y', 999) < 37.550]
    if river_nodes:
        self._graph.remove_nodes_from(river_nodes)
        print(f"[StreetNetwork] Removed {len(river_nodes)} river nodes (lat < 37.550)")
```

</details>

---

## 25. Step 5 루프

<details>
<summary><strong>▶ STORE_VISIT_ACTIONS + MAX_VISIT_LOOP</strong></summary>

### 설정 이유
Step 5 미호출(죽은 코드) → 카페_가기 해도 실제 매장 선택·평가 없음.  
매장 방문 행동이면 Step 3→4→5 루프, 최대 3회 추가 방문.

### 실제 코드 (`action_algorithm.py`)

```python
STORE_VISIT_ACTIONS = {
    "카페_가기": "커피-음료",
}
MAX_VISIT_LOOP = 3  # 타임슬롯 내 최대 추가 방문 횟수
```

```python
for _loop_i in range(self.MAX_VISIT_LOOP):
    session_visits = [...]
    step5 = await self.step5_next_action(agent, ..., session_visits, ...)
    action = step5.get("action", "")

    if action not in self.STORE_VISIT_ACTIONS:
        break  # 비매장 행동 → 루프 종료

    loop_category = self.STORE_VISIT_ACTIONS[action]
    step3_loop = await self.step3_store_selection(agent, loop_category, ...)
    step4_loop = await self.step4_evaluate_and_feedback(...)
    visits.append({...})
```

</details>

---

## 26. session_activity

<details>
<summary><strong>▶ session_visits → session_activity 자연어 변환</strong></summary>

### 설정 이유
Step 5에 `last_action` 단일 문자열만 전달 → 이번 슬롯에서 뭘 했는지 모름.  
`session_visits` 전체를 자연어로 변환해 LLM에 전달.

### 실제 코드 (`action_algorithm.py`)

```python
if session_visits:
    activity_lines = []
    visited_categories = []
    for i, v in enumerate(session_visits, 1):
        store = v.get("visited_store", "?")
        cat = v.get("visited_category", "")
        rating = v.get("rating", "")
        rating_str = f" -> {rating}/5" if rating else ""
        activity_lines.append(f"  {i}. {store}({cat}) 방문{rating_str}")
        if cat:
            visited_categories.append(cat)
    session_activity = "\n".join(activity_lines)
    # + [고려사항] warnings ...
else:
    session_activity = "  (아직 활동 없음)"

prompt = render_prompt(STEP5_NEXT_ACTION, ..., session_activity=session_activity, ...)
```

### 프롬프트 예시
```
이번 시간대 활동:
  1. 스시요리하루(일식) 방문 -> 4/5
  2. 딥블루레이크(커피-음료) 방문 -> 5/5
  [고려사항] 방금 카페를 다녀왔습니다...
```

</details>

---

## 27. [고려사항] 동적 경고

<details>
<summary><strong>▶ 카테고리 중복 시 경고 주입 (하드 가드 제거)</strong></summary>

### 설정 이유
같은 카테고리면 무조건 break → 카페 2차 등 현실적 선택 불가.  
하드 가드 제거 후 `[고려사항]` 맥락만 주입해 LLM이 판단하도록.

### 실제 코드 (`action_algorithm.py`)

```python
from collections import Counter
cat_counts = Counter(visited_categories)

warnings = []
for cat, count in cat_counts.items():
    if count >= 2:
        warnings.append(
            f"[고려사항] '{cat}' 일정을 이미 {count}번 소화했습니다. "
            "3번 연속 같은 활동을 하는 것은 현실적으로 부자연스럽습니다. "
            "이제는 산책이나 장보기 등 다른 활동으로 환기하는 것이 좋습니다."
        )
    elif cat == "커피-음료" and count >= 1:
        warnings.append(
            "[고려사항] 방금 카페를 다녀왔습니다. 보통 바로 또 카페를 가지는 않지만, "
            "분위기가 다른 곳으로 2차를 가야 할 특별한 이유가 있다면 가능합니다."
        )

if warnings:
    session_activity += "\n  " + "\n  ".join(warnings)
```

### 하드 가드 제거 (주석 처리)
```python
# visited_cats = {v["visited_category"] for v in visits}
# if loop_category in visited_cats:
#     break
```

</details>

---

---

## 28. 하이브리드 LLM 모델 라우팅

<details>
<summary><strong>▶ Step별 Gemini 모델 분리 (비용·품질 균형)</strong></summary>

### 설정 이유
모든 Step에 동일 모델 사용 → 단순 선택(Step2/3/5)에 고비용 모델 낭비.
Step4(리뷰 생성)만 고품질 모델, 나머지는 경량 모델로 비용 절감.

### 실제 코드 (`action_algorithm.py`)

```python
# __init__ — env vars에서 모델 읽기
self.step_lite_model = os.getenv("STEP_LITE_MODEL", "gemini-2.5-flash-lite")
self.step4_model     = os.getenv("STEP4_MODEL",     "gemini-2.5-flash")

# Step2 (업종 선택)
response = await self._call_llm_async(prompt, model=self.step_lite_model)

# Step3 (매장 선택)
response = await self._call_llm_async(prompt, model=self.step_lite_model)

# Step4 (평가·리뷰) ← 고품질 모델
response = await self._call_llm_async(prompt, model=self.step4_model)

# Step5 (다음 행동)
response = await self._call_llm_async(prompt, model=self.step_lite_model)
```

### `.env` 설정

```env
STEP_LITE_MODEL=gemini-2.5-flash-lite   # Step2/3/5
STEP4_MODEL=gemini-2.5-flash            # Step4 (평가·리뷰)
```

### `llm_client.py` 변경 사항

```python
# generate() / generate_sync() 에 model 파라미터 추가
async def generate(self, prompt, system_prompt=None, model=None) -> str:
    json={"model": model or self.model, ...}  # None이면 기본 self.model 사용
```

</details>

---

## 관련 문서

| 문서 | 설명 |
|------|------|
| [에이전트-설정-레퍼런스.md](./에이전트-설정-레퍼런스.md) | 설정·이유 한눈에 요약 |
| [에이전트-결정-흐름.md](./에이전트-결정-흐름.md) | Step 1~5 데이터 흐름 |
| [에이전트-고도화-요약.md](./에이전트-고도화-요약.md) | 고도화 상세 (문제·원인·해결) |
