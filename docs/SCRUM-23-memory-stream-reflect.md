# SCRUM-23: 메모리 스트림 + 리플렉션 + 프롬프트 분기 + A/B 테스트

## 개요

에이전트 시뮬레이션의 현실성을 4가지 축에서 강화했습니다.

| Feature | 요약 | LLM 비용 |
|---------|------|----------|
| F4. 초기 위치 통계 반영 | 세그먼트별 POI 가중 스폰 | 0 (좌표 계산) |
| F3. 에이전트 메모리 | Stanford GA 스타일 Memory Stream + Reflection | ~+1/5방문 |
| F2. 프롬프트 분기 | 발견 경로 5종 × 방문 목적 6종 컨텍스트 주입 | 0 (기존 프롬프트 확장) |
| F1. X리포트 A/B 비교 | 리포트 유/무 2회 실행 비교 분석 | 2× (2회 실행) |

---

## F4. 초기 위치 통계 반영

### 문제
기존에는 모든 에이전트가 망원동 영역 내 완전 랜덤 좌표에서 시작하여, 역 근처/시장 근처 등 실제 유동인구 패턴을 반영하지 못했습니다.

### 해결

```
SpawnPointSelector → 12개 POI 중 세그먼트 가중치 기반 선택 → ±50m Gaussian jitter
```

### 생성 파일
- `src/simulation_layer/spawn_points.py`

### 구조

```python
SpawnPOI(name, lat, lng, poi_type)  # transit / market / residential / commercial

MANGWON_POIS = [
    SpawnPOI("망원역 1번출구",   37.5564, 126.9104, "transit"),
    SpawnPOI("망원역 2번출구",   37.5560, 126.9098, "transit"),
    SpawnPOI("합정역 1번출구",   37.5500, 126.9140, "transit"),
    SpawnPOI("합정역 7번출구",   37.5505, 126.9135, "transit"),
    SpawnPOI("망원시장 입구",    37.5555, 126.9085, "market"),
    SpawnPOI("망원시장 중앙",    37.5552, 126.9080, "market"),
    SpawnPOI("망원동 버스정류장1", 37.5558, 126.9115, "transit"),
    SpawnPOI("망원동 버스정류장2", 37.5545, 126.9095, "transit"),
    SpawnPOI("주거밀집 1 (망원1동)", 37.5540, 126.9070, "residential"),
    SpawnPOI("주거밀집 2 (망원2동)", 37.5535, 126.9100, "residential"),
    SpawnPOI("주거밀집 3 (합정동)", 37.5510, 126.9120, "residential"),
    SpawnPOI("망리단길 입구",    37.5548, 126.9090, "commercial"),
]
```

### 세그먼트별 POI 가중치

| 세그먼트 | residential | market | transit | commercial |
|----------|-------------|--------|---------|------------|
| 1인가구 | 70% | 15% | 10% | 5% |
| 데이트커플 | 5% | 10% | 50% | 35% |
| 약속모임 | 5% | 5% | 65% | 25% |
| 망원유입직장인 | 5% | 10% | 55% | 30% |
| 외부출퇴근직장인 | 60% | 15% | 20% | 5% |

### 수정 파일
- `src/simulation_layer/engine.py` — `_get_agent_spawn_location()` 추가
- `scripts/run_simulation.py` — `--archetype` 모드에서 자동 연결

---

## F3. 에이전트 메모리 (Memory Stream + Reflection)

### 문제
에이전트가 과거 경험을 기억하지 못하여, 매번 같은 매장에 방문하거나 만족도가 낮았던 곳을 재방문하는 비현실적 패턴이 발생했습니다.

### 해결: Stanford Generative Agents 패턴 적용

```
방문 이벤트 → EventMemory에 저장
                ↓
retrieve_relevant(recency × importance × relevance)
                ↓
프롬프트에 memory_context 주입
                ↓
5회 방문마다 Reflection 트리거 → 선호도 자동 조정
```

### 수정/생성 파일
- `src/simulation_layer/persona/memory_structures/event_memory.py` (기존 stub 확장)
- `src/simulation_layer/persona/memory_structures/reflection.py` (기존 stub 확장)
- `src/simulation_layer/persona/cognitive_modules/chained_decide.py` (통합)

### EventMemory 핵심 구조

```python
@dataclass
class MemoryEntry:
    timestamp: datetime
    event_type: str          # "visit" | "report_reception" | "reflection"
    description: str
    importance: float        # 0.0 ~ 1.0
    store_name: str = ""
    category: str = ""
    satisfaction: float = 0.5
    day_number: int = 0

class EventMemory:
    MAX_ENTRIES = 200

    def add_visit(timestamp, store_name, category, satisfaction, ...)
    def add_report_reception(timestamp, report_description)
    def retrieve_relevant(query_context, top_k=5, current_time=None)
    def format_for_prompt(entries, max_chars=500)
```

### 메모리 검색 스코어링 (Stanford GA 방식)

```
score = recency(0.4) × importance(0.3) × relevance(0.3)

recency    = e^(-0.02 × hours_elapsed)
importance = entry.importance (0.0 ~ 1.0)
relevance  = 카테고리/매장명/시간대 키워드 매칭 (0.0 ~ 1.0)
```

### Reflection 메커니즘

```
5회 방문마다 트리거 → 규칙 기반 분석:
  - 동일 카테고리 3회+ & 만족도 ≥ 0.7 → 해당 카테고리 boost +0.1
  - 동일 카테고리 3회+ & 만족도 < 0.4 → 해당 카테고리 reduce -0.1
  - 동일 매장 3회+ → 단골 패턴 감지
  - 5회+ 방문 & 4+ 고유 카테고리 → 다양성 추구 감지
```

### 프롬프트 통합 예시

```
## 최근 방문 기억 (중요도순)
[3일 전] 망원시장손칼국수 방문 (만족도: 높음) - "칼국수 맛있었음"
[5일 전] 몬스터스토리지 방문 (만족도: 보통) - "디저트 괜찮았지만 가격이..."
[반영] 한식 카테고리 선호도 증가 (최근 만족도 높음)
```

### 크로스데이 지속
- `reset_daily_states()`: meals_today만 리셋, 메모리/반영은 유지
- `_day_number` 증가로 시간 감쇠 정확도 유지

### 출력 파일
- `data/output/agent_memories.json` — 전체 에이전트 메모리 + 반영 직렬화

---

## F2. 프롬프트 분기 (발견 경로 × 방문 목적)

### 문제
모든 에이전트가 동일한 정보 접근 경로와 방문 동기를 가정하여, 매장 선택 기준이 단일했습니다.

### 해결: 2축 컨텍스트 분기

```
세그먼트 + 생활양식 + 시간대
         ↓
발견 경로 (5종) × 방문 목적 (6종) = 30가지 조합
         ↓
자연어 컨텍스트 → Step 2, 3 프롬프트에 주입
```

### 생성 파일
- `src/simulation_layer/persona/discovery_context.py`

### 발견 경로 5종

| 경로 | 설명 | 매장 선택 기준 |
|------|------|----------------|
| 네이버검색 | 블로그 리뷰 기반 탐색 | 리뷰 수, 평점 높은 곳 우선 |
| 카카오지도 | 지도 앱 기반 탐색 | 가까운 거리, 영업 중 우선 |
| 길거리발견 | 도보 중 우연 발견 | 외관, 줄, 냄새에 영향 |
| SNS추천 | 인스타/틱톡 등 | 인스타그래머블, 트렌디 우선 |
| 지인추천 | 친구/가족 추천 | 추천받은 특정 매장 우선 |

### 방문 목적 6종

| 목적 | 트리거 조건 | 매장 선택 기준 |
|------|-------------|----------------|
| 맛집탐방 | 변화추구 + 미식탐구 | 맛 평점 최우선, 새로운 곳 |
| 회식 | 약속모임 + 친구/동료 | 단체석, 술 메뉴, 다양한 메뉴 |
| 혼밥 | 1인가구 + 혼자 | 1인석, 빠른 서비스 |
| 데이트 | 데이트커플 + 연인 | 분위기, 서비스 |
| 간편식사 | 직장인 + 점심 | 빠르고 가까운 곳 |
| 카페/디저트 | 야간 + 낮은 배고픔 | 분위기, 디저트 메뉴 |

### 가중치 할당 (생활양식별)

```python
# 변화추구
{"네이버검색": 0.30, "SNS추천": 0.25, "카카오지도": 0.20, "길거리발견": 0.15, "지인추천": 0.10}

# 단조로운패턴
{"길거리발견": 0.40, "지인추천": 0.20, "네이버검색": 0.15, "카카오지도": 0.15, "SNS추천": 0.10}
```

### 수정 파일
- `src/simulation_layer/persona/cognitive_modules/chained_decide.py` — Step 2, 3에 discovery_context 주입
- `src/simulation_layer/models.py` — `SimulationEvent`에 `discovery_channel`, `visit_purpose` 필드 추가
- `src/simulation_layer/engine.py` — 이벤트 로깅에 새 필드 반영
- `src/ai_layer/prompt_templates/category_selection.txt` — `{discovery_context}` 섹션 추가
- `src/ai_layer/prompt_templates/store_selection.txt` — `{discovery_context}` + 경로/목적별 선택 기준 추가

---

## F1. X리포트 A/B 비교

### 문제
X리포트(프로모션)의 효과를 정량적으로 측정할 방법이 없었습니다.

### 해결: 동일 시드 2회 실행 비교

```
seed=42 → Run A (리포트 有) → simulation_result_A.csv
seed=42 → Run B (리포트 無) → simulation_result_B.csv
                    ↓
            ABComparator.compute(df_a, df_b)
                    ↓
            ab_comparison.json + ab_comparison_summary.md
```

### 생성 파일
- `src/simulation_layer/ab_comparison.py`

### ComparisonMetrics (24개 지표)

| 카테고리 | 지표 |
|----------|------|
| 전체 전환율 | conversion_rate_a/b, delta, pct_change |
| 방문수 | total_visits_a/b, visit_delta |
| 카테고리 분포 | category_distribution_a/b, category_delta |
| 리포트 대상 매장 | report_store_visits_a/b, report_store_delta |
| 세그먼트별 전환율 | segment_conversion_a/b, segment_conversion_delta |
| 정보 확산 | report_reception_rate, report_influenced_rate |
| 시간대별 방문 | timeslot_visits_a/b |
| 발견 경로 | discovery_channel_a/b |

### 에이전트별 리포트 수신 확률 차등화

```python
# engine.py :: _get_report_reception_prob()
base_prob = 0.3  # 설정값

# 생활양식 보정
변화추구     → × 1.5
단조로운패턴 → × 0.7

# 트렌드 민감도 보정
trend_scale = 0.7 + trend_sensitivity × 0.6  # 0.7 ~ 1.3

# 상한
min(0.9, base_prob × lifestyle_mod × trend_scale)
```

### CLI 사용법

```bash
# A/B 테스트 실행
python scripts/run_simulation.py --archetype --chained --ab-test --seed 42

# 리포트 없이 단독 실행
python scripts/run_simulation.py --archetype --chained --no-reports

# 시드 지정 단독 실행
python scripts/run_simulation.py --archetype --chained --seed 123
```

### 수정 파일
- `src/simulation_layer/engine.py` — `_get_report_reception_prob()` 메서드 추가
- `scripts/run_simulation.py` — `--no-reports`, `--ab-test`, `--seed` 플래그 + `_run_ab_test()` 함수
- `scripts/dashboard.py` — A/B 비교 시각화 섹션 추가

### 대시보드 A/B 비교 섹션

| 차트 | 내용 |
|------|------|
| 전환율 메트릭 3열 | A/B 전환율 + 변화율 |
| 정보 확산 | 수신율 + 영향 전환율 |
| 리포트 대상 매장 바 차트 | 매장별 A vs B 방문수 |
| 세그먼트별 전환율 | 그룹별 A vs B 비교 |
| 카테고리 분포 | 전체 카테고리 A vs B |
| 발견 경로 비교 | 채널별 A vs B |

---

## 전체 아키텍처 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│  run_simulation.py (CLI)                                        │
│  --archetype --chained --ab-test --seed 42                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. stores.csv → Environment (StreetNetwork)                    │
│  2. ArchetypeGenerator → 64 agents                              │
│  3. SpawnPointSelector → 세그먼트별 POI 위치                     │
│  4. SimulationEngine.run_simulation()                           │
│     │                                                           │
│     │  ┌──── Per Agent, Per Timestep ────────────────────────┐  │
│     │  │                                                     │  │
│     │  │  Report Reception (에이전트별 차등 확률)              │  │
│     │  │       │                                             │  │
│     │  │  Step 1: 시간대 의사결정 (Smart 규칙기반)             │  │
│     │  │       │                                             │  │
│     │  │  Memory.retrieve_relevant() → memory_context        │  │
│     │  │  DiscoveryContext(channel × purpose) → context_text  │  │
│     │  │       │                                             │  │
│     │  │  Step 2: 카테고리 선택 (LLM + 메모리 + 발견경로)      │  │
│     │  │       │                                             │  │
│     │  │  Step 3: 매장 선택 (LLM + 리뷰 + 메모리 + 발견경로)   │  │
│     │  │       │                                             │  │
│     │  │  Memory.add_visit() → Reflection 체크               │  │
│     │  │                                                     │  │
│     │  └─────────────────────────────────────────────────────┘  │
│     │                                                           │
│  5. Save: CSV + agent_memories.json                             │
│  6. (A/B) ABComparator → ab_comparison.json                     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  dashboard.py (Streamlit)                                       │
│  - 지도/시간대/카테고리/프로필 분석                                │
│  - 발견 경로 × 방문 목적 히트맵                                   │
│  - A/B 비교 분석 (ab_comparison.json 로드 시)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 변경 파일 목록

### 신규 생성 (4개)
| 파일 | Feature | 설명 |
|------|---------|------|
| `src/simulation_layer/spawn_points.py` | F4 | POI 기반 스폰 포인트 선택기 |
| `src/simulation_layer/persona/discovery_context.py` | F2 | 발견 경로 × 방문 목적 컨텍스트 |
| `src/simulation_layer/ab_comparison.py` | F1 | A/B 비교 메트릭 + 분석기 |
| `docs/SCRUM-23-memory-stream-reflect.md` | - | 이 문서 |

### 확장 (기존 stub → 본격 구현, 2개)
| 파일 | Feature | 설명 |
|------|---------|------|
| `src/simulation_layer/persona/memory_structures/event_memory.py` | F3 | Memory Stream 구현 |
| `src/simulation_layer/persona/memory_structures/reflection.py` | F3 | Reflection 메커니즘 구현 |

### 수정 (5개)
| 파일 | Feature | 주요 변경 |
|------|---------|----------|
| `src/simulation_layer/engine.py` | F4,F3,F2,F1 | spawn_selector, current_time 전달, 리포트 수신 확률 차등화, discovery 필드 |
| `src/simulation_layer/persona/cognitive_modules/chained_decide.py` | F3,F2 | 메모리 통합, 발견경로 컨텍스트, reflection 트리거 |
| `src/simulation_layer/models.py` | F2 | SimulationEvent에 discovery_channel, visit_purpose 추가 |
| `scripts/run_simulation.py` | F4,F3,F1 | spawn_selector, 메모리 저장, A/B 테스트 모드 |
| `scripts/dashboard.py` | F2,F1 | 발견경로/목적 차트, A/B 비교 섹션 |

### 수정 (프롬프트 템플릿, 2개)
| 파일 | Feature | 추가 섹션 |
|------|---------|----------|
| `src/ai_layer/prompt_templates/category_selection.txt` | F3,F2 | `{memory_context}`, `{discovery_context}` |
| `src/ai_layer/prompt_templates/store_selection.txt` | F3,F2 | `{memory_context}`, `{discovery_context}`, 경로/목적별 선택 기준 |

---

## 하위 호환성

- 모든 새 CLI 플래그는 **opt-in** (기존 `--archetype --chained` 동작 불변)
- `SimulationEvent` 새 필드는 `Optional[str] = None`
- 프롬프트 새 변수는 `DecideModule`에서 빈 문자열 전달
- 메모리/반영은 `ChainedDecideModule`에서만 활성화
- `SpawnPointSelector`는 archetype 모드에서만 생성
