# 에이전트 설정 가이드 — 설정·이유 한눈에

`agent-decision-flow.md`(의사결정 흐름)와 `agent-upgrade-summary.md`(고도화 내용)를 통합하여  
**각 설정이 왜 그렇게 되어 있는지**를 한눈에 파악할 수 있는 레퍼런스 문서입니다.  
▶ 토글을 클릭하면 실제 반영된 코드를 펼쳐볼 수 있습니다.

---

## 목차

| 영역 | 설정/개념 | 문서 위치 |
|------|----------|----------|
| [의사결정 흐름](#1-의사결정-흐름-step-15) | Step 1~5, Loop, 종료 조건 | 아래 1절 |
| [매장 선택](#2-매장-선택) | 카테고리, 샘플링, 필터 | 아래 2절 |
| [리뷰·평가](#3-리뷰평가) | tone, focus_aspect, comment | 아래 3절 |
| [에이전트·위치](#4-에이전트위치) | 상주/유동, 초기위치, 교체 비율 | 아래 4절 |
| [루프·행동](#5-루프행동) | MAX_VISIT_LOOP, session_activity | 아래 5절 |
| [코드·환경](#6-코드환경) | LLM, 병렬 처리, 대시보드 | 아래 6절 |

---

## 1. 의사결정 흐름 (Step 1~5)

### 1.1 Step 1: LLM 전면 전환 (확률 기반 폐기)

| 항목 | 내용 |
|------|------|
| **설정** | Step 1을 `BASE_EATING_OUT_PROB` 확률이 아닌 **LLM 순수 의사결정**으로 전환 |
| **이전 방식** | 아침 40%, 점심 70% 등 확률 기반 → 4끼 전부 외식 발생 |
| **설정 이유** | 확률 보정만으로는 비현실적. LLM이 "이미 몇 끼 먹었는지" 맥락을 보고 판단하도록 변경 |
| **고도화 #** | #2 (4끼 전부 외식) |
| **참고** | `step1_destination.txt`, 선택지 3가지(외식/안먹음/망원동밖) + 시간대별 외식 경향 수치 제공 |

### 1.2 memory_context 통합 (당일 식사 + 과거 경험)

| 항목 | 내용 |
|------|------|
| **설정** | Step 1~5 공통으로 `memory_context` 전달 — **오늘 식사 기록 + 과거 방문(평점·코멘트)** |
| **이전 방식** | Step2는 recent_categories만, Step3는 recent_stores만 → 평점/코멘트 없음 |
| **설정 이유** | "이전에 여기 가서 별로였다" 같은 판단 불가 → 만족/불만족 기반 재방문·회피 가능하도록 |
| **고도화 #** | #2, #10 |
| **참고** | `agent.get_memory_context(current_date)`, `get_meals_today()` |

<details>
<summary>▶ 코드: get_memory_context</summary>

```python
# agent.py
def get_memory_context(self, current_date: str = "") -> str:
    lines = []
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
    rating_text = {1: "매우별로", 2: "별로", 3: "보통", 4: "좋음", 5: "매우좋음"}
    lines.append("당신의 과거 경험:")
    for v in self.recent_history[-5:]:
        line = f"  - {v.store_name} ({v.category}): {rating_text.get(v.taste_rating, '?')}"
        if v.comment:
            line += f' → "{v.comment}"'
        lines.append(line)
    return "\n".join(lines)
```

</details>

### 1.3 아침 "이전 식사로 충분" 사유 제외

| 항목 | 내용 |
|------|------|
| **설정** | 아침 또는 `meals_today == 0`일 때 `reasons_skip`에서 "이전 식사로 충분" 제외 |
| **설정 이유** | 아침은 그날 첫 끼니인데 이전 식사가 있을 수 없음 — 모순 방지 |
| **고도화 #** | #12 |

<details>
<summary>▶ 설명: 프롬프트 유도</summary>

Step 1은 LLM 순수 의사결정으로 전환됨. 프롬프트 `step1_destination.txt`에 선택지로  
`2. 이 시간대에는 먹지 않는다 (배가 안 고픔, 이전 식사로 충분, 다이어트 중 등)`를 포함하고,  
LLM이 "아침 + 첫 끼" 맥락에서 "이전 식사로 충분"을 비현실적이라 스스로 회피하도록 유도함.

</details>

### 1.4 Step 5 루프 (매장 방문 행동 → Step 3→4→5 반복)

| 항목 | 내용 |
|------|------|
| **설정** | Step 4 후 Step 5 호출. `action`이 `STORE_VISIT_ACTIONS`에 있으면 Step 3→4→5 루프 |
| **이전 방식** | Step 5 미호출(죽은 코드) → 카페_가기 해도 실제 매장 선택/평가 없음 |
| **설정 이유** | 점심 후 카페, 저녁 후 2차 같은 현실적 다중 방문 패턴 반영 |
| **고도화 #** | #25 |
| **참고** | `STORE_VISIT_ACTIONS = {"카페_가기": "커피-음료"}`, `MAX_VISIT_LOOP = 3` |

<details>
<summary>▶ 코드: STORE_VISIT_ACTIONS + 루프</summary>

```python
# action_algorithm.py
STORE_VISIT_ACTIONS = {"카페_가기": "커피-음료"}
MAX_VISIT_LOOP = 3

for _loop_i in range(self.MAX_VISIT_LOOP):
    step5 = await self.step5_next_action(agent, ..., session_visits, ...)
    action = step5.get("action", "")
    if action not in self.STORE_VISIT_ACTIONS:
        break
    loop_category = self.STORE_VISIT_ACTIONS[action]
    step3_loop = await self.step3_store_selection(agent, loop_category, ...)
    step4_loop = await self.step4_evaluate_and_feedback(...)
```

</details>

### 1.5 session_activity (Step 5 맥락)

| 항목 | 내용 |
|------|------|
| **설정** | Step 5에 `last_action` 대신 `session_visits` → `session_activity` 자연어로 전달 |
| **이전 방식** | "방금 한 일: 돼지야 방문" 정도만 → 이번 슬롯에서 뭘 했는지 전체 맥락 부족 |
| **설정 이유** | "한식 먹고 4점 → 카페 가자" 같은 맥락 기반 자연스러운 판단 유도 |
| **고도화 #** | #26 |
| **참고** | `이번 시간대 활동: 1. 스시요리하루(일식) 방문 -> 4/5 2. 딥블루레이크(커피-음료) 방문 -> 5/5` |

<details>
<summary>▶ 코드: session_visits → session_activity</summary>

```python
# action_algorithm.py
if session_visits:
    activity_lines = []
    for i, v in enumerate(session_visits, 1):
        store = v.get("visited_store", "?")
        cat = v.get("visited_category", "")
        rating = v.get("rating", "")
        rating_str = f" -> {rating}/5" if rating else ""
        activity_lines.append(f"  {i}. {store}({cat}) 방문{rating_str}")
    session_activity = "\n".join(activity_lines)
    # + [고려사항] warnings ...
else:
    session_activity = "  (아직 활동 없음)"
```

</details>

### 1.6 [고려사항] 동적 경고 (카테고리 중복)

| 항목 | 내용 |
|------|------|
| **설정** | 하드 가드(break) 제거. 대신 동일 카테고리 연속 방문 시 `[고려사항]` 문구 동적 주입 |
| **이전 방식** | 같은 카테고리면 무조건 break → 카페 2차 등 현실적 선택 불가 |
| **설정 이유** | "비현실적이지만 이유가 있으면 가능" — LLM 판단에 위임, 기계적 제약 완화 |
| **고도화 #** | #27 |
| **참고** | 커피 1회 → "방금 카페 다녀왔습니다... 가능합니다", 2회+ → "이미 N번... 부자연스럽습니다" |

<details>
<summary>▶ 코드: [고려사항] 동적 경고 주입</summary>

```python
# action_algorithm.py
from collections import Counter
cat_counts = Counter(visited_categories)
warnings = []
for cat, count in cat_counts.items():
    if count >= 2:
        warnings.append(f"[고려사항] '{cat}' 일정을 이미 {count}번 소화했습니다...")
    elif cat == "커피-음료" and count >= 1:
        warnings.append("[고려사항] 방금 카페를 다녀왔습니다. 보통 바로 또 카페를 가지는 않지만...")
if warnings:
    session_activity += "\n  " + "\n  ".join(warnings)
```

</details>

---

## 2. 매장 선택

### 2.1 TIME_SLOT_CATEGORIES (시간대별 전체 카테고리)

| 항목 | 내용 |
|------|------|
| **설정** | `destination_type = "식당"` 폐기. 시간대별 **전체 카테고리 풀**을 Step 2에 전달 |
| **이전 방식** | 식당만 전달 → 아침 커피, 저녁 술자리 선택 구조적 불가 |
| **설정 이유** | LLM이 페르소나 기반으로 아침 카페, 저녁 주점 등 현실적 선택 가능하도록 |
| **고도화 #** | #14 |
| **참고** | 아침: 식당+카페, 점심: 식당+카페, 저녁: 식당+주점+카페, 야식: 식당+주점 |

<details>
<summary>▶ 코드: TIME_SLOT_CATEGORIES</summary>

```python
# action_algorithm.py
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
available_categories = TIME_SLOT_CATEGORIES.get(time_slot, DESTINATION_CATEGORIES["식당"])
```

</details>

### 2.2 Softmax 가중 샘플링 (T=0.5, 20개)

| 항목 | 내용 |
|------|------|
| **설정** | exploit/explore 이분법 폐기 → **Softmax 확률 비례 비복원 추출** (temperature=0.5, sample_k=20) |
| **이전 방식** | 상위 10개 고정 + 나머지 5개 랜덤 → 롱테일 매장 0.7% 진입 |
| **설정 이유** | 인기매장 집중 + 롱테일 진입 기회. T=0.5로 인기/롱테일 갭 8.8x→3.8x로 축소 |
| **고도화 #** | #8 (5차 개선) |
| **참고** | `search_ranked_stores()`, 돼지야 포함율 0.7%→10.8% |

### 2.3 계층화 샘플링 (상위60% + 중위25% + 하위15%)

| 항목 | 내용 |
|------|------|
| **설정** | 점수순 상·중·하 1/3씩 → 상위 60% + 중위 25% + 하위 15% 비율로 추출 |
| **설정 이유** | Softmax만으로도 롱테일 부족 시, 하위 계층에서 15% 확률로 후보 진입 |
| **고도화 #** | #23 |
| **참고** | 돼지야 0건 → 13건, 메가MGC커피 0건 → 35건(카테고리 선택 가능해짐) |

<details>
<summary>▶ 코드: Softmax + 계층화 샘플링</summary>

```python
# global_store.py
max_score = max(scores)
exp_scores = [exp((s - max_score) / temperature) for s in scores]
probs = [e / sum(exp_scores) for e in exp_scores]

# 계층화: 상위60% + 중위25% + 하위15%
sorted_indices = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
tier_top = sorted_indices[:n//3]
tier_mid = sorted_indices[n//3:2*n//3]
tier_low = sorted_indices[2*n//3:]
k_top, k_mid, k_low = round(k*0.60), round(k*0.25), k - k_top - k_mid
# 각 tier 내 Softmax 확률 비례 추출
```

스코어: 별점×1.5 + 에이전트평점×2.0 + log1p(min(리뷰,50))×0.3 - 거리×0.3

</details>

### 2.4 CATEGORY_ALIAS + match_category

| 항목 | 내용 |
|------|------|
| **설정** | LLM 선택 "카페" vs JSON "커피-음료" 등 **명칭 불일치**를 매핑 테이블로 해결 |
| **설정 이유** | 카페/디저트/베이커리가 매칭 실패해 방문 0건 → alias로 매칭 보완 |
| **고도화 #** | #13 |
| **참고** | `CATEGORY_ALIAS`, `match_category()` |

<details>
<summary>▶ 코드: CATEGORY_ALIAS + match_category</summary>

```python
# global_store.py
CATEGORY_ALIAS = {
    "카페": "커피-음료", "커피": "커피-음료",
    "디저트": "제과점", "베이커리": "제과점",
    "이자카야": "호프-간이주점", "포차": "호프-간이주점", ...
}

def match_category(query: str, store_category: str) -> bool:
    q, sc = query.lower(), (store_category or "").lower()
    if q in sc:
        return True
    alias = CATEGORY_ALIAS.get(query, "").lower()
    return bool(alias and alias in sc)
```

</details>

### 2.5 JSON metadata fallback (좌표·카테고리)

| 항목 | 내용 |
|------|------|
| **설정** | JSON 최상위 `x,y`, `category` 없으면 `metadata.x`, `metadata.y`, `metadata.sector` 사용 |
| **설정 이유** | 상주 에이전트 점심 전원 "적합한 매장 없음" — 좌표/카테고리 누락으로 검색 0건 |
| **고도화 #** | #13 |
| **참고** | `global_store.load_from_json_dir()` |

<details>
<summary>▶ 설명: load_from_json_dir</summary>

`category` 없으면 `metadata.sector`, `x`/`y` 없으면 `metadata.x`/`metadata.y` 사용.

</details>

### 2.6 random.shuffle (위치 편향 방지)

| 항목 | 내용 |
|------|------|
| **설정** | 매장 후보 리스트를 LLM에 넣기 전 `random.shuffle()` |
| **설정 이유** | LLM이 리스트 앞쪽 항목을 선호하는 positional bias 완화 |
| **고도화 #** | #8 (유동 매장 선택 최적화) |

---

## 3. 리뷰·평가

### 3.1 tone_instruction (세대별 말투)

| 항목 | 내용 |
|------|------|
| **설정** | Z/Y/X/S 세대별로 서로 다른 말투 지시 (`tone_instruction`) |
| **설정 이유** | 전원 동일한 "한줄 평가" 톤 → 세대별 차별화로 리뷰 다양성 확보 |
| **고도화 #** | #1 |
| **참고** | Z: "존나, 개꿀맛, ㅋ", Y: "트렌디+정보성", X: "점잖고 꼼꼼", S: "구체적·진중" |

### 3.2 focus_aspect (평가 관점 랜덤)

| 항목 | 내용 |
|------|------|
| **설정** | 6개 관점 중 랜덤 1개: 맛/퀄리티, 가성비, 분위기, 서비스, 청결, 특색 메뉴 |
| **설정 이유** | 같은 매장이어도 리뷰 초점을 나눠서 다양성 확보 |
| **고도화 #** | #1 |
| **참고** | `random.choice(focus_aspects)` |

### 3.3 comment 작성 규칙

| 항목 | 내용 |
|------|------|
| **설정** | 키워드 나열 금지, 구체적 상황/감정 유도, 일반적 표현 대신 개인 감상 |
| **설정 이유** | "부드러운 돈카츠가 인상적..." 류의 변주만 반복 → 구체적·개인적 리뷰 유도 |
| **고도화 #** | #1 |

### 3.4 평가 항목 개편 (rating + selected_tags)

| 항목 | 내용 |
|------|------|
| **설정** | 맛/가성비/분위기 3개 → **종합 만족도(rating)** 1개 + **주요 장점 태그(selected_tags)** 0~2개 |
| **설정 이유** | 단순화하면서도 리뷰 품질 유지 |
| **고도화 #** | #1 |

<details>
<summary>▶ 코드: tone_instruction + focus_aspect</summary>

```python
# action_algorithm.py (Step 4)
tone_instruction = "자연스럽게 작성하세요."
if agent.generation in ["Z1", "Z2"]:
    tone_instruction = "Z세대 말투(존나, 개꿀맛, ㅋ, 이모티콘 등)를 사용해서..."
elif agent.generation == "Y":
    tone_instruction = "밀레니얼 세대 말투로, 적당히 트렌디하면서도 정보성 있게..."
elif agent.generation == "X":
    tone_instruction = "X세대 말투로, 점잖으면서도 꼼꼼하게 분석하듯이..."
elif agent.generation == "S":
    tone_instruction = "어르신 말투로, 구체적이고 진중하게..."

focus_aspects = ["맛/퀄리티", "가성비", "매장 분위기/인테리어", "직원 서비스/친절도", "매장 청결/위생", "특색있는 메뉴"]
selected_focus = random.choice(focus_aspects)
```

</details>

---

## 4. 에이전트·위치

### 4.1 entry_point / entry_time_slot (유동)

| 항목 | 내용 |
|------|------|
| **설정** | 유동은 `home_location` 대신 `entry_point` 사용. 세그먼트별 `entry_time_slot` 적용 |
| **이전 방식** | 역/정류장을 집으로 인식, 아침부터 전원 활동 |
| **설정 이유** | 직장인→점심, 데이트족→저녁 등 세그먼트별 현실적 진입 시간 반영 |
| **고도화 #** | #3 |
| **참고** | `entry_time_slot` 이전 타임슬롯은 스킵 |

### 4.2 Step 5 선택지 (상주/유동 분리)

| 항목 | 내용 |
|------|------|
| **설정** | 상주: 집에서_쉬기 O, 망원동_떠나기 X / 유동: 집에서_쉬기 X, 망원동_떠나기 O |
| **설정 이유** | 유동은 "집"이 없음. 떠나기 선택 시 해당 일 시뮬 종료 |
| **고도화 #** | #3 |

<details>
<summary>▶ 코드: entry_point + 유동 초기화</summary>

```python
# agent.py
entry_point: Optional[Tuple[float, float]] = field(default=None)
entry_time_slot: Optional[str] = field(default=None)
left_mangwon: bool = field(default=False)

# run_generative_simulation.py
for agent in daily_floating:
    agent.left_mangwon = False
    if agent.entry_point:
        lat, lng = agent.entry_point
    else:
        loc = random.choice(list(FLOATING_LOCATIONS.values()))
        lat, lng = loc["lat"], loc["lng"]
    agent_locations[agent.id] = network.initialize_agent_location(lat, lng)
```

</details>

### 4.3 상주 초기위치 오프셋 (150m)

| 항목 | 내용 |
|------|------|
| **설정** | 주거유형별 고정 3좌표 대신, **중심점 + 반경 150m 랜덤 오프셋** |
| **이전 방식** | 같은 좌표에 6~7명 겹침 → 동일 매장 리스트 → 다양성 저하 |
| **설정 이유** | 에이전트별 서로 다른 반경 검색 결과 → 선택 다양화 |
| **고도화 #** | #6 |
| **참고** | `RESIDENT_OFFSET_RADIUS_M = 150`, `_apply_location_offset()` |

<details>
<summary>▶ 코드: _apply_location_offset</summary>

```python
# agent.py
RESIDENT_OFFSET_RADIUS_M = 150

def _apply_location_offset(base_lat: float, base_lng: float, max_radius_m: float = RESIDENT_OFFSET_RADIUS_M):
    offset_m = random.uniform(0, max_radius_m)
    angle = random.uniform(0, 2 * math.pi)
    dlat = (offset_m * math.cos(angle)) / 111320
    dlng = (offset_m * math.sin(angle)) / (111320 * math.cos(math.radians(base_lat)))
    return (base_lat + dlat, base_lng + dlng)
```

</details>

### 4.4 유동인구 교체 비율 (요일별 + 재방문 10%)

| 항목 | 내용 |
|------|------|
| **설정** | 평일 51명 / 주말 58명 (인구DB 비율). 전날 방문자 10% 재방문 우선 포함 |
| **이전 방식** | 매일 53명 고정, 요일 무관 |
| **설정 이유** | 주말 유동인구 증가 반영. 재방문으로 연속성 부여 |
| **고도화 #** | #5 |
| **참고** | `DAILY_FLOATING_COUNT_BY_DAY`, `REVISIT_RATE = 0.10` |

<details>
<summary>▶ 코드: DAILY_FLOATING_COUNT_BY_DAY + REVISIT_RATE</summary>

```python
# run_generative_simulation.py
DAILY_FLOATING_COUNT_BY_DAY = {"월": 51, "화": 51, "수": 51, "목": 51, "금": 51, "토": 58, "일": 58}
REVISIT_RATE = 0.10

base_count = DAILY_FLOATING_COUNT_BY_DAY.get(weekday, 53)
revisit_count = max(1, int(len(prev_visitors) * REVISIT_RATE))
revisit_agents = random.sample(prev_visitors, min(revisit_count, len(prev_visitors)))
daily_floating = revisit_agents + new_agents
```

</details>

### 4.5 유동 [0,0] fallback

| 항목 | 내용 |
|------|------|
| **설정** | `home_location = [0,0]` → `entry_point` 사용, 없으면 `FLOATING_LOCATIONS` 랜덤 |
| **설정 이유** | 대시보드에서 아프리카 앞바다 등 엉뚱한 위치 표시 방지 |
| **고도화 #** | #24 |

### 4.6 한강 위 노드 제거

| 항목 | 내용 |
|------|------|
| **설정** | `load_graph()`에서 lat &lt; 37.550 노드 제거 |
| **설정 이유** | 한강 다리 도로를 경유지로 사용해 한강 위를 걷는 것처럼 표시되는 문제 |
| **고도화 #** | #15 |

<details>
<summary>▶ 코드: 한강 위 노드 제거</summary>

```python
# street_network.py
river_nodes = [n for n, d in self._graph.nodes(data=True) if d.get('y', 999) < 37.550]
if river_nodes:
    self._graph.remove_nodes_from(river_nodes)
```

</details>

---

## 5. 루프·행동

### 5.1 MAX_VISIT_LOOP = 3

| 항목 | 내용 |
|------|------|
| **설정** | 타임슬롯당 최대 3회 추가 방문 (식사 1회 + 카페 등 최대 3회) |
| **설정 이유** | 무한루프 방지 + 현실적 상한 (한 끼에 4곳 이상 방문은 드묾) |
| **고도화 #** | #25 |

### 5.2 STORE_VISIT_ACTIONS

| 항목 | 내용 |
|------|------|
| **설정** | `{"카페_가기": "커피-음료"}` — 해당 action 선택 시 `loop_category`로 Step 3 재실행 |
| **설정 이유** | Step 5 "카페_가기" → 실제 매장 선택·평가 없이 끝나던 문제 해결 |
| **고도화 #** | #25 |

---

## 6. 코드·환경

### 6.1 LLM 프로바이더 (Gemini 기본)

| 항목 | 내용 |
|------|------|
| **설정** | `LLM_PROVIDER=gemini`, `LLM_MODEL_NAME=gemini-2.0-flash` (기본값) |
| **설정 이유** | GPT-4o-mini 대비 비용 33% 절감, 성능·속도 우위 |
| **고도화 #** | #4 |
| **참고** | `.env`, `config/settings.py` |

### 6.2 asyncio 병렬 + Semaphore

| 항목 | 내용 |
|------|------|
| **설정** | `asyncio.gather` + `Semaphore`로 동시 LLM 호출 제한 (기본 5개) |
| **설정 이유** | 순차 호출 시 7일 약 2시간 51분 → 병렬로 대폭 단축 |
| **고도화 #** | #7 |
| **참고** | `max_concurrent_llm_calls` |

### 6.3 시뮬레이션 좌표 (대시보드)

| 항목 | 내용 |
|------|------|
| **설정** | 지도 에이전트 위치를 `random.uniform()`이 아닌 `results_df`의 `agent_lat`/`agent_lng` 사용 |
| **설정 이유** | 실제 시뮬레이션 좌표와 불일치해 공원/바깥에 표시되던 문제 |
| **고도화 #** | #9 |

### 6.4 시뮬 시작 날짜 (KST 금요일)

| 항목 | 내용 |
|------|------|
| **설정** | `datetime(2025, 2, 7)` 하드코딩 → 한국 시간 기준 **가장 가까운 과거 금요일** 자동 계산 |
| **설정 이유** | 실행 시점에 맞는 주간 시뮬레이션 |
| **고도화 #** | #20 |

---

## 요약표: 설정 → 고도화 # 매핑

| 고도화 # | 핵심 설정 | 한 줄 요약 |
|:---:|---|----------|
| 1 | tone_instruction, focus_aspect, comment 규칙 | 리뷰 다양성 (세대별 말투·관점 분산) |
| 2 | Step 1 LLM 전환, memory_context 당일 식사 | 4끼 전부 외식 방지 |
| 3 | entry_point, entry_time_slot, 상주/유동 선택지 | 유동 초기위치·시작시간 |
| 4 | LLM_PROVIDER=gemini | LLM 모델 (비용·성능) |
| 5 | DAILY_FLOATING_COUNT_BY_DAY, REVISIT_RATE | 유동인구 요일별·재방문 |
| 6 | RESIDENT_OFFSET_RADIUS_M | 상주 초기위치 오프셋 |
| 7 | asyncio.gather, Semaphore | 시뮬 속도 (병렬) |
| 8 | Softmax T=0.5, search_ranked_stores | 유동 매장 선택 (롱테일 진입) |
| 9 | results_df agent_lat/lng | 대시보드 좌표 일치 |
| 10 | memory_context 과거 경험 | 과거 평점·코멘트 기반 판단 |
| 12 | 아침/첫끼 "이전 식사로 충분" 제외 | 사유 모순 방지 |
| 13 | JSON metadata, CATEGORY_ALIAS | 상주 "적합한 매장 없음", 카테고리 매칭 |
| 14 | TIME_SLOT_CATEGORIES | 카페/주점 방문 가능 |
| 15 | lat < 37.550 노드 제거 | 한강 위 걷기 방지 |
| 23 | 계층화 샘플링 | 타겟 매장(돼지야 등) 방문 0건 해소 |
| 24 | [0,0] fallback | 유동 좌표 오류 방지 |
| 25 | Step 5 루프, STORE_VISIT_ACTIONS, MAX_VISIT_LOOP | 다중 방문 (카페 2차 등) |
| 26 | session_activity | Step 5 활동 맥락 |
| 27 | [고려사항] 동적 경고 | 카테고리 중복 하드가드 제거 |

---

## 관련 문서

| 문서 | 설명 |
|------|------|
| [agent-decision-flow.md](./agent-decision-flow.md) | Step 1~5 데이터 흐름, 프롬프트 예시, 변수 출처 |
| [agent-upgrade-summary.md](./agent-upgrade-summary.md) | 고도화 상세 (문제·원인·해결·변경 파일) |
| [agent-upgrade-code-reference.md](./agent-upgrade-code-reference.md) | 고도화 코드 상세 (별도 문서) |
| [agent_action_algorithm.md](./agent_action_algorithm.md) | 의사결정 알고리즘 기술 스펙 |
