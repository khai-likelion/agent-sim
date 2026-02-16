# 에이전트 행동 알고리즘 (ActionAlgorithm)

> **파일 위치:** `src/simulation_layer/persona/cognitive_modules/action_algorithm.py`

---

## 1. 개요

각 에이전트는 타임슬롯(07:00 / 12:00 / 18:00 / 22:00)마다 **5단계 의사결정**을 수행한다.
Step 1은 확률 기반, Step 2~5는 LLM(언어 모델) 기반이다.

```
타임슬롯 시작
    │
    ▼
Step 1: 망원동 내 식사 여부 결정  ──→ 아니오 ──→ stay_home (종료)
    │ 예
    ▼
Step 2: 업종 선택 (LLM)           ──→ 실패 ──→ LLMCallFailedError
    │
    ▼
Step 3: 매장 선택 (LLM)           ──→ 매장없음 ──→ stay_home (종료)
    │
    ▼
Step 4: 평가 및 피드백 (LLM)      ──→ 실패 ──→ LLMCallFailedError
    │
    ▼
결과 반환 (decision: "visit")
```

---

## 2. 비동기 실행 구조

### 2-1. 에이전트 간 병렬 (타임슬롯 단위)

같은 타임슬롯의 에이전트들은 **`asyncio.gather()`** 로 동시에 실행된다.
LLM 응답을 기다리는 동안 다른 에이전트의 요청이 함께 전송된다.

```
타임슬롯 [점심]
├── agent_task(P001) ──▶ Step1→2→3→4  (await 순차)
├── agent_task(P002) ──▶ Step1→2→3→4  (await 순차)  } 동시 실행
├── agent_task(R001) ──▶ Step1→2→3→4  (await 순차)
└── ...                                               ↑
                                          asyncio.gather(*tasks)
```

### 2-2. 에이전트 내 순차 (단계 의존성)

한 에이전트의 Step 2→3→4는 **이전 결과가 다음 입력**이 되므로 반드시 순차 실행한다.

```python
# process_decision() 내부 흐름
step1 = self.step1_eat_in_mangwon(...)          # 동기 (확률)
step2 = await self.step2_category_selection(...) # 업종 → step3의 입력
step3 = await self.step3_store_selection(        # 매장 → step4의 입력
    category=step2["category"], ...)
step4 = await self.step4_evaluate_and_feedback(  # store = step3 결과
    store=..., ...)
```

### 2-3. API Rate Limit 제어

`asyncio.Semaphore(N)`으로 **동시 LLM 호출 수를 N개로 제한**한다.
기본값: `max_concurrent_llm_calls = 10`

```python
# run_simulation() 내부
semaphore = asyncio.Semaphore(10)
algorithm = ActionAlgorithm(rate_limit_delay=0.5, semaphore=semaphore)

# _call_llm_async() 내부
async with semaphore:          # 10개 초과 시 여기서 대기
    response = await client.generate(prompt)
    await asyncio.sleep(0.5)   # 호출 간 딜레이
```

---

## 3. 각 단계 상세

### Step 1 — 망원동 내 식사 여부 결정

| 구분 | 내용 |
|------|------|
| 방식 | 확률 기반 (LLM 불필요) |
| 반환 | `{"eat_in_mangwon": bool, "reason": str}` |

**시간대별 기본 외식 확률:**

| 시간대 | 기본 확률 | 비고 |
|--------|-----------|------|
| 아침   | 40%       | |
| 점심   | 70%       | |
| 저녁   | 60%       | |
| 야식   | 20%       | |

**보정 규칙:**

- 오늘 식사 3회 이상 → 확률 × 0.1
- 오늘 식사 2회 → 확률 × 0.5
- 주말(토·일) → 확률 × 1.15 (최대 1.0)

---

### Step 2 — 업종 선택

| 구분 | 내용 |
|------|------|
| 방식 | LLM (페르소나 + 메모리 기반) |
| 입력 | `destination_type`, `time_slot`, 에이전트 메모리 |
| 반환 | `{"category": str, "reason": str}` |

**시간대별 선택 가능 업종:**

| 시간대 | 목적지 유형 | 선택 가능 카테고리 |
|--------|-------------|-------------------|
| 아침   | 식당/카페   | 한식, 중식, 일식, 양식, 분식, 패스트푸드, 국밥, 찌개, 고기, 치킨 / 카페, 커피, 디저트, 베이커리, 브런치 |
| 점심   | 식당/카페   | (동일) |
| 저녁   | 식당/주점/카페 | 위 + 호프, 이자카야, 포차, 와인바, 술집, 막걸리, 칵테일바 |
| 야식   | 식당/주점   | 카페 제외 |

---

### Step 3 — 매장 선택

| 구분 | 내용 |
|------|------|
| 방식 | LLM (매장 정보 + 에이전트 평점 기반) |
| 입력 | `category`, `nearby_stores`, 에이전트 메모리 |
| 반환 | `{"store_name": str, "reason": str}` |

**LLM에 전달하는 매장 정보 (store당):**

```
[매장명]
  store_id: S001
  카테고리: 한식
  평균가격: 12,000원
  주요키워드: 김치찌개, 된장찌개, 혼밥, 점심, 가성비
  설명: 망원동 골목 안 가정식 백반집...
  에이전트 평점: 4.2/5 (8건), 태그: 맛 5, 가성비 3, 분위기 2, 서비스 1
    - R001: 좋음 ['맛', '가성비']
    - P012: 매우좋음 ['맛']
```

**위치 편향 방지:** 매장 목록을 LLM에 전달 전 `random.shuffle()` 적용

---

### Step 4 — 평가 및 피드백

| 구분 | 내용 |
|------|------|
| 방식 | LLM (페르소나 말투 + 랜덤 평가 관점 적용) |
| 입력 | 방문 매장 정보, 에이전트 세대/페르소나 |
| 반환 | `{"rating": int, "selected_tags": list, "comment": str}` |

**평점 체계 (1~5):**

| 점수 | 의미 |
|------|------|
| 1 | 매우 별로 |
| 2 | 별로 |
| 3 | 보통 |
| 4 | 좋음 |
| 5 | 매우 좋음 |

**선택 가능 태그:** `맛` / `가성비` / `분위기` / `서비스`

**세대별 말투 지시:**

| 세대 | 말투 |
|------|------|
| Z1/Z2 | Z세대 (존나, 개꿀맛, ㅋ, 이모티콘) |
| Y | 밀레니얼 (트렌디, 정보성) |
| X | X세대 (점잖고 꼼꼼) |
| S | 시니어 (구체적, 진중) |

**랜덤 평가 관점 (다양성 유도):**
`맛/퀄리티` / `가성비` / `매장 분위기/인테리어` / `직원 서비스/친절도` / `매장 청결/위생` / `특색있는 메뉴`

**평점 버퍼링:**
`GlobalStore.add_pending_rating()` 에 버퍼링 → 타임슬롯 종료 후 `flush_pending_ratings()` 일괄 반영 (동시 기록 충돌 방지)

---

### Step 5 — 다음 행동 결정

| 구분 | 내용 |
|------|------|
| 방식 | LLM |
| 입력 | 현재/다음 타임슬롯, 요일, 마지막 행동, 에이전트 메모리 |
| 반환 | `{"action": str, "walking_speed": float, "reason": str, "destination": dict}` |

> **주의:** Step 5는 현재 메인 시뮬레이션 루프에서 호출되지 않음.
> `action_algorithm.py`의 `__main__` 테스트 블록 또는 확장 구현에서 활용 가능.

**상주 에이전트 선택지:**

| 행동 | 설명 |
|------|------|
| 집에서_쉬기 | `home_location`으로 이동 |
| 카페_가기 | `cafe_stores.txt`에서 랜덤 선택 |
| 배회하기 | 목적지 없이 현재 주변 이동 |
| 한강공원_산책 | 망원한강공원 (37.5530, 126.8950) |
| 망원시장_장보기 | 망원시장 (37.5560, 126.9050) |
| 회사_가기 | `work_location` 또는 기본값 |

**유동 에이전트 선택지:**
위에서 `집에서_쉬기` / `회사_가기` 제외, `망원동_떠나기` 추가
(`left_mangwon = True` 설정 → 이후 타임슬롯 전부 스킵)

**`walking_speed`:** LLM이 페르소나를 보고 직접 판단한 km/h 값 (유효 범위 1.0~7.0)

---

## 4. 오류 처리

| 상황 | 동작 |
|------|------|
| LLM 호출 실패 (일반) | 최대 3회 retry → `LLMCallFailedError` |
| Rate Limit (429) | `(attempt+1) × 10`초 대기 후 retry |
| 매장 없음 (Step 3) | `decision: "stay_home"` 반환 (예외 아님) |
| `process_decision` 전체 실패 | `decision: "llm_failed"`, `error: str` 반환 |

```python
# process_decision() 반환 형식
{
    "decision": "visit" | "stay_home" | "llm_failed",
    "steps": {
        "step1": {...},
        "step2": {...},   # eat_in_mangwon=True인 경우만
        "step3": {...},   # step2 이후
        "step4": {...},   # visit인 경우만
    },
    "visited_store": str | None,
    "visited_category": str | None,
    "ratings": {"taste": int, "value": int, "atmosphere": int} | None,
    "reason": str | None,
    "error": str | None,
}
```

---

## 5. 에이전트 유형별 동작 차이

| 구분 | 상주 에이전트 (상주) | 유동 에이전트 (유동) |
|------|---------------------|---------------------|
| 매장 인식 범위 | 반경 800m 이내 | 전체 매장 |
| 시작 위치 | `home_location` | `entry_point` (6개 진입 지점 중 하나) |
| 일일 활동 | 매일 전원 | 113명 중 53명 랜덤 샘플링 |
| 이탈 | 없음 | `망원동_떠나기` 선택 시 `left_mangwon=True`, 이후 스킵 |
| 진입 시간 | 없음 | `entry_time_slot` 이전 타임슬롯 스킵 |

---

## 6. 데이터 흐름 요약

```
run_simulation()
│
├── asyncio.Semaphore(10) ──▶ ActionAlgorithm(semaphore=...)
│
└── 타임슬롯마다:
    │
    ├── flush_pending_ratings()      ← 이전 슬롯 평점 일괄 반영
    │
    ├── asyncio.gather(              ← 에이전트 간 병렬
    │     agent_task(P001),
    │     agent_task(P002), ...
    │   )
    │     └── process_decision()    ← 에이전트 내 순차 await
    │           ├── step1 (동기)
    │           ├── await step2 (LLM)
    │           ├── await step3 (LLM)
    │           └── await step4 (LLM) ──▶ add_pending_rating() 버퍼
    │
    └── 결과 records 수집
```
