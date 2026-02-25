# Agent 고도화 작업 목록

> Notion "고도화 작업 목록" 기준으로 정리. 코드와 일치 여부 검증 완료.

## 총괄 요약표

| # | 항목 | 현재 방식 | 개선안 | 최종 채택안 |
|---|------|----------|--------|------------|
| 1 | 리뷰 멘트 다양성 | Step4 프롬프트가 "한줄 평가"만 지시 → 전원 동일한 키워드 재조합 | 페르소나 말투 지정 + 관점 분산 + 구체적 경험 유도 | **완료** (tone_instruction + 평가 관점 LLM 자율 선택 + selected_tags 기준 + comment 규칙) |
| 2 | 4끼 전부 외식 | Step1이 확률 기반(BASE_EATING_OUT_PROB) + memory_context에 당일 식사 이력 없음 | Step1 LLM 전면 전환(순수 의사결정) + memory_context에 당일 식사 이력 추가 | **완료** (LLM 판단 + 프롬프트 강화) |
| 3 | 유동 초기위치/시작시간 | FLOATING_LOCATIONS → home_location 저장, 아침부터 전원 활동 | entry_point 분리 + entry_time_slot 도입 + "집에서_쉬기" → "망원동 떠나기" | **완료** |
| 4 | LLM 모델 선택 | GPT-4o-mini (고정) | Gemini 2.0 Flash (33% 저렴, 성능 우위, 속도 25% 빠름) | **완료** (provider 라우팅 + Gemini 전환) |
| 5 | 유동인구 교체 비율 | 매일 113명 중 53명 랜덤 샘플링 (고정) | 요일별 차등(평일 51/주말 58, 인구DB 비율 1.14배) + 재방문율 10% | **완료** |
| 6 | 상주인구 초기위치 | 주거유형별 3좌표 고정 (아파트3, 빌라3, 주택3) | 고정 좌표를 중심점으로 반경 150m 랜덤 오프셋 | **완료** |
| 7 | 시뮬레이션 속도 | 순차 호출 (호출당 1.5s, 7일 약 2시간 51분) | asyncio 병렬 호출 (Semaphore 동시 제한) | **완료** (asyncio.gather + Semaphore + max_concurrent_llm_calls, 예상시간 병렬 반영) |
| 8 | 매장/에이전트 좌표 불일치 | 개요 지도에서 에이전트 위치를 `random.uniform()`으로 생성 → 비현실적 위치 표시 | `results_df`의 실제 시뮬레이션 좌표(`agent_lat`/`agent_lng`) 사용 | **완료** |
| 9 | X-report to Sim 결합 | 전체 X-report 전달 | StrategyBridge로 JSON 매장 속성 수정 후 시뮬 전달 | **완료** (`apply_x_report_strategy_async`, run_before_after_sim) |
| 10 | 과거 경험 기반 의사결정 | Step2는 recent_categories만, Step3는 recent_stores만 → 평점/코멘트 없음 | memory_context 통합 (오늘 식사 + 과거 평점 + 코멘트) + 유도 문구 | **완료** |
| 11 | 유동 매장 선택 최적화 | 720개 전체 매장을 LLM에 전달 → positional bias + 롱테일 매장 0방문 | 카테고리 선필터 + candidate_stores + 리뷰 캡/가중치 완화 + Softmax 가중 샘플링(T=0.5) | **완료** (5차 개선: 돼지야 후보 포함율 0.7%→10.8%, 인기/롱테일 갭 8.8x→3.8x) |
| 12 | Step5 직장인 선택지 구분 | 상주/유동만 분리, 직장인(1인·2인·4인 생베 겸직)은 별도 구분 없음 | 직장인 세그먼트 판별 + Step5 선택지 3분류 (상주/유동/직장인) | 미정 |
| 13 | 아침 "이전 식사로 충분" 버그 | Step1 외출안함 사유가 아침에도 "이전 식사로 충분" 출력 | 아침/첫끼일 때는 해당 사유 제외 | **완료** |
| 14 | 상주 에이전트 "적합한 매장 없음" 대량 발생 | 상주(R*) 에이전트 점심에 전원 "적합한 매장 없음"으로 외출 실패 — JSON에 x,y 좌표/category 누락 → 반경검색 결과 0개 | JSON metadata fallback(x,y,category) + CATEGORY_ALIAS 매핑 | **완료** |
| 15 | 카페/주점 방문 불가 | destination_type이 항상 "식당" 고정 → Step2에 카페/주점 카테고리가 전달 안 됨 | `TIME_SLOT_CATEGORIES` 도입, destination_type 분기 제거 → LLM에 시간대별 전체 카테고리 풀 전달 | **완료** |
| 16 | 한강 위 에이전트 노드 | OSM 반경 2000m에 한강 다리 도로 포함 → 에이전트가 한강 위를 걸어다님 | `load_graph()`에서 lat < 37.550 노드 제거 | **완료** |
| 17 | 에이전트 추적 지도 UI | Folium 기반 + 색상 점만으로 에이전트/매장 구분 불가 | pydeck 전환 + 이모지 마커 + 텍스트 라벨 + @st.fragment 부분 렌더링 | **완료** |
| 18 | 10개 매장 배치 시뮬레이션 | 매장별 수동 실행 필요 | `run_batch_10stores.py` — 10매장 순차 before/after 비교, 완료 건너뛰기, 3회 재시도 | **완료** |
| 19 | 미사용 코드/설정 정리 | AreaSettings, 9개 SimulationSettings 필드, 5개 PathSettings 속성, 11개 .env 변수 등 미사용 | 전면 삭제 + requirements.txt 정리 (fastapi/uvicorn/geopandas 제거) | **완료** |
| 20 | 대시보드 폴더 선택 | `data/output/` 루트의 단일 CSV만 로드 | 사이드바 드롭다운으로 before/after 결과 폴더 선택 | **완료** |
| 21 | 시뮬 시작 날짜 KST 기준 | `datetime(2025, 2, 7)` 하드코딩 | 한국 시간 기준 가장 가까운 과거 금요일 자동 계산 | **완료** |
| 22 | 에이전트 추적 슬라이더 | 슬라이더 클릭/드래그 시 시간 미반영, 자동재생 속도 과다 | `value` 직접 바인딩 + 속도 곡선 완화 (sleep 0.5s) | **완료** |
| 23 | 지도 마커/뷰 개선 | 이모지만으로 에이전트/매장 구분 어려움, 경로 잘림 | 사람 마커(🧑+색 원) + 매장 핀(빨간 원+🍴) + 바운딩 박스 자동 줌 | **완료** |
| 24 | 타겟 매장 방문 0건 | 돼지야(한식) 0건, 메가커피(커피) 0건 — Softmax 랭킹 편중 + 카페 카테고리 미선택 | 계층화 샘플링(상위60%+중위20%+하위20%) → 롱테일 매장도 LLM 후보 진입 | **완료** |
| 25 | 유동 에이전트 좌표 [0,0] | home_location [0,0]이 유효값으로 처리 → 지도에서 엉뚱한 위치 표시 | entry_point fallback + `_is_valid_loc()` 체크 3중 적용 | **완료** |
| 26 | Step 5 활성화 + 다중 방문 루프 | Step 5 미호출 (죽은 코드) → ①카페_가기 등이 실제 매장 선택/평가 없이 종료 ②타임슬롯당 1회 방문만 가능 | Step 4 후 Step 5 호출 → 매장 방문 행동이면 Step 3→4→5 루프 (MAX=3) | **완료** |
| 27 | Step 5 활동 맥락 부재 | Step 5에 last_action 단일 문자열만 전달 → 이번 타임슬롯에서 뭘 했는지 모름 | session_visits → session_activity 자연어 변환하여 LLM에 전달 | **완료** |
| 28 | 카테고리 중복 방문 (카페 3연속) | 하드 가드(break)로 원천 봉쇄 → 기계적 제한 | 하드 가드 제거 + `[고려사항]` 맥락 주입 (Internal Thought) → 2회 허용, 3회 자제 유도 | **완료** |

---

## 1. 리뷰 멘트 다양성

### 현재 방식
- **파일**: `src/ai_layer/prompts/step4_evaluate.txt`
- Step4에서 LLM이 맛/가성비/분위기 평점(1~5) + `"comment": "한줄 평가"` 생성
- 모든 에이전트가 동일한 `store_info`(top_keywords, rag_context)를 입력받음
- **결과**: 같은 매장이면 "부드러운 돈카츠가 인상적이었지만 웨이팅이 아쉬웠다" 류의 변주만 반복

### 근본 원인
1. 페르소나 말투 지시 없음 — Z1 남자든 X세대 가족이든 같은 톤
2. 매장 정보가 동일 — 모든 에이전트가 같은 키워드를 보고 같은 포인트 언급
3. 리뷰 다양성 유도 없음 — "한줄 평가"라는 짧은 지시가 전부

### 최종 채택안
**완료** — 세대별 말투 지정 + 평가 관점 LLM 자율 선택 + selected_tags 기준 명시 + 구체적 경험 유도

구현 내용:
1. **말투 지정 (`tone_instruction`)**: 세대별 차별화
   - Z세대: "존나, 개꿀맛, ㅋ, 이모티콘 등 친구한테 말하듯이 솔직하고 짧게"
   - Y세대(밀레니얼): "적당히 트렌디하면서도 정보성 있게"
   - X세대: "점잖으면서도 꼼꼼하게 분석하듯이"
   - S세대(시니어): "구체적이고 진중하게"
2. **평가 관점**: 프롬프트에 6개 관점 제시 후 LLM이 페르소나에 맞는 하나를 스스로 선택 (맛/퀄리티, 가성비, 매장 분위기·인테리어, 직원 서비스·친절도, 매장 청결·위생, 특색있는 메뉴)
3. **comment 작성 규칙**: 키워드 나열 금지, 구체적 상황/감정 유도, 예시 3개 제공
4. **평가 항목 개편**: 종합 만족도(rating) 1개 + 주요 장점 태그(selected_tags) 기준 충족 시에만 선택 (0~4개, 없으면 빈 배열)

### 변경 파일
- `src/ai_layer/prompts/step4_evaluate.txt`: 프롬프트 전면 개편 (tone_instruction, 평가 관점 6개 제시 후 LLM 자율 선택, selected_tags 기준, comment 규칙, rating 단일화)
- `src/simulation_layer/persona/cognitive_modules/action_algorithm.py`: Step4 호출 시 세대별 tone_instruction 생성 (focus_aspect 변수 제거)

---

## 2. 4끼 전부 외식

### 현재 방식
- **파일**: `src/simulation_layer/persona/cognitive_modules/action_algorithm.py` (Step1)
- Step1이 **확률 기반**으로 동작 (`BASE_EATING_OUT_PROB`: 아침 40%, 점심 70%, 저녁 60%, 야식 20%)
- 식사 횟수 보정: 2끼→확률 절반, 3끼 이상→확률 10%로 감소, 주말 15% 보너스
- memory_context에 당일 식사 이력이 없어서 LLM(Step2/Step3)이 "이미 몇 끼 먹었는지" 모름
- **결과**: 확률 보정만으로는 4끼 연속 외식이 여전히 발생, LLM이 이전 식사를 고려하지 못함

### 개선안 (2가지 병행)

**A. Step1 프롬프트 선택지 강화** (완료)
- "안 먹는다" 선택지를 명확하게 3가지로 분리
- 현실적 판단 유도문 추가 ("대부분의 사람은 하루 2~3끼")
- 시간대별 외식 경향 참고 수치 제공

```
질문: 이 시간대에 외식을 할지 결정하세요.
1. 망원동 내 매장에서 식사한다
2. 이 시간대에는 먹지 않는다 (배가 안 고픔, 이전 식사로 충분, 다이어트 중 등)
3. 망원동 밖에서 해결한다 (집밥, 편의점, 배달 등)

시간대별 외식 경향: 아침(~40%), 점심(~70%), 저녁(~60%), 야식(~20%)
```

**B. memory_context에 당일 식사 이력 추가** (완료)
- `agent.get_meals_today(current_date)`로 당일 식사 횟수/매장 추적
- Step1 호출 시 `memory_context`에 오늘 식사 기록 포함
- Step2/Step3에도 동일한 `memory_context` 전달 → LLM이 이전 식사를 고려하여 업종/매장 선택

```
오늘의 식사 기록 (현재 1끼 외식 완료):
  - 12:00 류진 (한식)

당신의 과거 경험:
  - 류진 (한식): 좋음 → "국물이 진해서 좋았음"
```

### 최종 채택안
**완료** — Step1 LLM 전면 전환 + memory_context 당일 식사 이력 통합

> 기존 확률 기반(`BASE_EATING_OUT_PROB`) 로직을 폐기하고, Step 1부터 LLM(`STEP1_DESTINATION` 프롬프트)을 사용하여 순수 의사결정으로 전환함.
> 이제 에이전트는 "확률 70%니까 나감"이 아니라, "어제 파스타 먹었으니 오늘은 한식 먹으러 나가야지"라고 직접 판단함.

### 변경 파일
- `src/ai_layer/prompts/step1_destination.txt`: 선택지 3가지 분리 + 현실적 판단 유도문 + 시간대별 외식 경향 수치
- `src/simulation_layer/persona/agent.py`: `get_meals_today()` 메서드, `get_memory_context(current_date)` 메서드
- `src/simulation_layer/persona/cognitive_modules/action_algorithm.py`: **Step1 메서드(`step1_eat_in_mangwon`)를 async LLM 호출로 변경**

---

## 3. 유동 초기위치 / 시작시간

### 현재 방식
- **파일**: `src/simulation_layer/persona/agent.py` (FLOATING_LOCATIONS, home_location)
- 유동 에이전트 생성 시 FLOATING_LOCATIONS(역/정류장 6곳) 중 랜덤 → `home_location`에 저장
- 시뮬레이션은 07시(아침)부터 4타임슬롯 순차 실행 → 유동도 전원 아침부터 활동
- Step5 "집에서_쉬기" → `home_location`(=정류장) 좌표로 이동
- **결과**: 유동이 정류장을 집으로 인식, 아침부터 무조건 활동

### 최종 채택안
**완료** — entry_point 분리 + entry_time_slot(세그먼트별) + Step5 유형별 선택지 + 망원동_떠나기

구현 내용:
1. **`home_location` → `entry_point` 분리**: 유동 에이전트는 `entry_point` (진입 지점) 사용, 진입 지점별 페르소나 가중치
2. **`entry_time_slot` 속성**: 세그먼트별 진입 시간 차등 (직장인→점심, 데이트족→저녁, 관광→아침), entry_time_slot 이전 타임슬롯은 스킵
3. **Step5 선택지 유형별 분리**:
   - 상주: 집에서_쉬기, 카페_가기, 한강공원_산책, 망원시장_장보기, 배회하기
   - 유동: 카페_가기, 한강공원_산책, 망원시장_장보기, 배회하기, 망원동_떠나기
   - 유동 "망원동_떠나기" 선택 시 해당 일 시뮬레이션에서 퇴장

### 변경 파일
- `src/simulation_layer/persona/agent.py`: `entry_point`, `entry_time_slot`, `left_mangwon` 필드 추가, `load_personas_from_md()`에서 유동 에이전트 초기화 로직 분리
- `src/ai_layer/prompts/step5_next_action.txt`: `{available_actions}`, `{action_options}` 템플릿 변수 도입
- `src/simulation_layer/persona/cognitive_modules/action_algorithm.py`: Step5 유형별 선택지 생성, `_get_action_destination`에 "망원동_떠나기" 처리
- `scripts/run_generative_simulation.py`: entry_time_slot 이전 스킵 + left_mangwon 스킵 + 유동 에이전트 매일 초기화

---

## 4. LLM 모델 선택

### 현재 방식
- GPT-4o-mini 사용
- Input: $0.15/1M tokens, Output: $0.60/1M tokens
- MMLU Pro: 63.1%, MATH: 70.2%

### 벤치마크 비교

| 항목 | GPT-4o-mini | Gemini 2.0 Flash |
|------|-------------|------------------|
| Input / 1M tokens | $0.15 | $0.10 (-33%) |
| Output / 1M tokens | $0.60 | $0.40 (-33%) |
| 컨텍스트 | 128K | 1M |
| MMLU Pro | 63.1% | 77.6% (+14.5p) |
| MATH | 70.2% | 90.9% (+20.7p) |
| TTFT | ~0.35s | ~0.34s |

### 최종 채택안
**완료** — Gemini 2.0 Flash 전환 + 멀티 프로바이더 라우팅

구현 내용:
1. **provider 라우팅**: `.env`의 `LLM_PROVIDER` 설정으로 gemini/openai/groq/deepseek/sambanova 전환 가능
2. **Gemini Flash 기본 설정**: `LLM_PROVIDER=gemini`, `LLM_MODEL_NAME=gemini-2.0-flash`
3. **비용 계산 개선**: provider별 비용 계산 (Gemini 무료 tier 반영), 모델명 출력

### 변경 파일
- `src/ai_layer/llm_client.py`: Gemini API 호출 지원 (OpenAI-compatible endpoint 라우팅)
- `config/settings.py`: `LLM_PROVIDER`, `LLM_MODEL_NAME` 환경변수 로드
- `scripts/run_generative_simulation.py`: provider별 비용 계산 + 모델명 표시
- `.env`: Gemini 설정 기본값

---

## 5. 유동인구 교체 비율

### 현재 방식
- **파일**: `scripts/run_generative_simulation.py` (DAILY_FLOATING_AGENT_COUNT = 53)
- 매일 113명 중 53명 랜덤 샘플링 (요일 무관 고정)

### 인구 DB 근거
| 요일 | 유동인구 | 비율 | 에이전트 수 |
|------|---------|------|-----------|
| 월 | 932,211 | 13.5% | 51 |
| 화 | 944,557 | 13.6% | 51 |
| 수 | 963,659 | 13.9% | 51 |
| 목 | 954,774 | 13.8% | 51 |
| 금 | 963,558 | 13.9% | 51 |
| **토** | **1,081,679** | **15.6%** | **58** |
| **일** | **1,085,810** | **15.7%** | **58** |

### 최종 채택안
**완료** — 인구DB 비율 기반 요일별 차등 + 재방문율 10% + 전날 방문자 추적

구현 내용:
- `DAILY_FLOATING_COUNT_BY_DAY` 딕셔너리 (평일 51 / 주말 58, 인구DB 비율 1.14배)
- `REVISIT_RATE = 0.10`
- 매일 샘플링 시 전날 방문 유동 에이전트 10% 우선 포함
- 하루 종료 시 `_prev_day_visitors` 기록 (다음 날 재방문 풀)

### 변경 파일
- `scripts/run_generative_simulation.py`:
  - `DAILY_FLOATING_COUNT_BY_DAY` 딕셔너리 (평일 51 / 주말 58, 인구DB 비율 1.14배)
  - `REVISIT_RATE = 0.10`
  - 매일 샘플링 시 전날 방문 유동 에이전트 10% 우선 포함
  - 하루 종료 시 `_prev_day_visitors` 기록 (다음 날 재방문 풀)

---

## 6. 상주인구 초기위치

### 현재 방식
- **파일**: `src/simulation_layer/persona/agent.py` (RESIDENT_LOCATIONS)
- 주거유형별 고정 좌표 3개씩 (아파트1~3, 빌라1~3, 주택1~3 = 총 9개 점)
- 에이전트 생성 시 `random.choice`로 3개 중 1개 배정 → `home_location`에 저장

### 문제점
상주 47명의 주거유형별 분포:
- 아파트: 19명 → 3좌표 중 랜덤 → 좌표당 평균 6~7명
- 빌라: 10명 → 3좌표 중 랜덤 → 좌표당 평균 3~4명
- 주택: 18명 → 3좌표 중 랜덤 → 좌표당 평균 6명

같은 좌표에 배정된 6~7명이 **정확히 동일한 위도/경도**에서 시작함.
같은 위치 → 같은 반경 800m 검색 결과 → 같은 매장 후보 → 비슷한 선택 → **다양성 저하**

예) 아파트1 (37.558682, 126.898706)에 7명이 겹침
→ 7명 모두 동일한 매장 리스트를 받음 → LLM 선택도 유사해짐

### 최종 채택안
**완료** — 구역 중심점 + 반경 150m 랜덤 오프셋

```
아파트1 중심점 (37.558682, 126.898706)
  ├─ R001: +80m 북동 → (37.55940, 126.89930)
  ├─ R002: +120m 남서 → (37.55760, 126.89770)
  ├─ R003: +45m 동 → (37.55868, 126.89920)
  └─ ... (각각 다른 위치)
```

```python
offset_m = random.uniform(0, 150)
angle = random.uniform(0, 2 * math.pi)
dlat = (offset_m * math.cos(angle)) / 111320
dlng = (offset_m * math.sin(angle)) / (111320 * math.cos(math.radians(base_lat)))
home = (base_lat + dlat, base_lng + dlng)
```

### 변경 파일
- `src/simulation_layer/persona/agent.py`:
  - `import math` 추가
  - `RESIDENT_OFFSET_RADIUS_M = 150` 상수
  - `_apply_location_offset()` 헬퍼 함수
  - 아파트/빌라/주택/기본 좌표 배정 시 `_apply_location_offset()` 적용

---

## 7. 시뮬레이션 속도

### 현재 방식
- 에이전트 순차 호출 (호출당 ~1.5s)
- rate_limit_delay = 0.5s
- 7일 시뮬레이션 약 2시간 51분

### 최종 채택안
**완료** — asyncio.gather 병렬 실행 + Semaphore 동시 호출 제한 + 예상시간 병렬 반영

구현 내용:
1. **asyncio.Semaphore**: 동시 LLM 호출 수를 제한하여 API rate limit 대응 (`max_concurrent_llm_calls`)
2. **asyncio.gather**: 에이전트별 의사결정 코루틴을 동시에 실행, LLM 응답 대기 중 다른 에이전트 요청 전송
3. **병렬 호출 수 설정**:
   - `run_before_after_sim.py`: 60개
   - `run_batch_10stores.py`: 20개
   - `run_generative_simulation.py`: 기본 10개 (인자로 전달 가능)
4. **예상시간 병렬 반영**: `estimate_simulation(max_concurrent_llm_calls)`로 병렬 수에 비례해 예상 시간 계산 (`총호출 × 1.5초 / 병렬수`)
5. **tqdm 진행바**: `bar_format`에 `{elapsed}<{remaining}`, `{rate_fmt}` 명시 (경과/남은 시간, 처리 속도 표시)
6. **전략 적용 전/후 비교**: 동일 시드로 재현 가능한 비교 시뮬레이션 (`run_before_after_sim.py`)
7. **StrategyBridge Gemini API 지원**: OpenAI-compatible endpoint 활용

### 변경 파일
- `scripts/run_generative_simulation.py`: `async def run_simulation()`, `asyncio.Semaphore`, `asyncio.gather`, `estimate_simulation(max_concurrent_llm_calls)`, tqdm `bar_format` 적용
- `scripts/run_before_after_sim.py`: 전략 전/후 비교 시뮬레이션 (max_concurrent_llm_calls=60, 예상치 출력)
- `src/simulation_layer/persona/cognitive_modules/action_algorithm.py`: async 메서드 구조 유지

---

## 8. 매장/에이전트 좌표 불일치

### 현재 방식
- **파일**: `scripts/dashboard.py` (`create_map_with_routes`)
- 대시보드 지도에서 상주 에이전트가 공원 위에 표시되고, 유동 에이전트가 망원동 바깥에 표시됨

### 원인 분석
1. **개요 지도 에이전트 위치가 랜덤 생성** (핵심 원인)
   - `create_map_with_routes()`에서 에이전트 위치를 `random.uniform()`으로 생성
   - 시뮬레이션의 실제 `agent_lat`/`agent_lng` 좌표를 사용하지 않음
2. **결과**: 에이전트가 공원, 망원동 바깥 등 비현실적 위치에 표시

### 최종 채택안
**완료** — `create_map_with_routes()`에 `results_df` 파라미터 추가, 시뮬레이션 결과의 실제 좌표(`agent_lat`, `agent_lng`) 사용

### 변경 파일
- `scripts/dashboard.py`: `create_map_with_routes()` — 랜덤 fallback + 시뮬레이션 좌표 우선 적용

---

## 9. X-report to Sim 결합

### 현재 방식
- 전체 X-report를 그대로 시뮬레이션에 전달

### 최종 채택안
**완료** — StrategyBridge로 JSON 매장 속성 수정 후 시뮬 전달

구현 내용:
- `X_to_Sim.apply_x_report_strategy_async()`: X-Report MD 기반으로 `{매장}.json` 수정
- `run_before_after_sim.py`: 전략 적용 전(Before) → StrategyBridge → 전략 적용 후(After) 2회 시뮬
- 동일 시드로 재현 가능한 전·후 비교

### 변경 파일
- `X_to_Sim.py`: `apply_x_report_strategy_async()`
- `scripts/run_before_after_sim.py`: StrategyBridge 호출, `--skip-bridge` 옵션

---

## 11. 유동 매장 선택 최적화 (Step3)

### 배경

유동 에이전트의 매장 선택 흐름:

```
[Step2] LLM이 업종 선택 (예: "한식")
    ↓
[검색 랭킹] 720개 전체 매장 → 후보 N개로 축소
    ↓
[Step3] LLM이 후보 N개 중 1개 매장 선택
```

유동 에이전트는 외부인이므로 물리적 반경이 아닌 **검색 랭킹** 기반으로 매장을 필터링합니다. 상주 에이전트만 반경 제한을 적용합니다.

**핵심 문제**: 랭킹으로 720개 → 15개로 축소하는 과정에서, 별점은 높지만 리뷰가 적은 **롱테일 매장**(별점 ≥4.0, 리뷰 ≤15건)이 후보에 포함되지 못합니다. 롱테일 매장은 전체의 35~54%를 차지합니다.

---

### 기존 방식 (v1)의 문제점

**파일**: `action_algorithm.py`, `global_store.py`

기존에는 720개 전체 매장 목록을 그대로 LLM에 전달했습니다.

- 720개 × ~250자 = ~180K자 → 토큰 한계에 근접
- LLM은 긴 리스트에서 앞쪽 항목을 선호하는 경향이 있음 (positional bias)
- **결과**: 모내 11회, 장터국밥 9회 등 특정 매장에 방문이 집중

---

### 1차 개선 — Exploit-Explore 검색 랭킹 도입

스코어링 기반 랭킹으로 720개 → 15개로 축소:
- 점수 공식: `별점 × 1.5 + 에이전트평점 × 2.0 + log(리뷰수) × 0.5 + 카테고리보너스(+3.0) - 거리 × 0.3`
- **Exploit**: 점수 상위 10개 (인기매장 보장)
- **Explore**: 나머지에서 랜덤 5개 (다양성 확보)

#### 1차 후 문제: 돼지야 0방문

7일/160명 시뮬레이션에서 타겟 매장 **돼지야**(별점 4.5, 리뷰 10건)가 양쪽 모두 0방문.
돼지야는 전체 720개 중 79위 → exploit(상위 10개)에 진입 불가, explore 확률 = 5/710 = **0.7%**.

근본 원인:
1. **카테고리가 필터가 아닌 보너스(+3.0)** → 비한식 고별점 매장과도 경쟁
2. **리뷰 수 가중치 과도** → 리뷰 10건 vs 100건 점수 차이가 별점 차이의 3.7배
3. **explore_k=5로 확률 부족** → 710개 중 5개 = 0.7%

추가 발견: **상주 에이전트 반경 필터 무시 버그** — `search_ranked_stores()`가 `nearby_stores` 파라미터를 무시하고 전역 `self.stores`를 검색. 0.8km 반경 제한이 실제로 동작하지 않음.

---

### 2차 개선 — 카테고리 선필터 + candidate_stores + explore_k 증가

변경 내용:
1. 카테고리를 보너스(+3.0) → **선필터**로 변경 (한식 검색 시 한식끼리만 경쟁)
2. `candidate_stores` 파라미터 추가 (상주 반경 필터 버그 수정)
3. explore_k: 5 → 10

| 지표 | 기존 | 2차 후 |
|------|------|-------|
| 검색 대상 | 전체 720개 | 한식 186개 |
| 돼지야 순위 | 79/720 | 79/186 |
| explore 확률 | 5/710 = 0.7% | 10/176 = **5.7%** |

#### 2차 후 문제: 순위 변화 없음

한식 186개 내에서 돼지야 순위가 여전히 79위. 리뷰 수 가중치(`log1p × 0.5`)가 높아 리뷰 100건+ 매장이 상위 독점.

---

### 3차 개선 — 리뷰 수 가중치 완화 + 캡

변경: 리뷰 수 50건 캡 + 가중치 0.5 → 0.3

```python
capped_reviews = min(store.star_rating_count or 0, 50)
score += math.log1p(capped_reviews) * 0.3
```

| 매장 | 변경 전 | 변경 후 | 점수 차이 |
|------|--------|--------|---------|
| 송정 (별점 4.7, 리뷰 100건) | 9.36 | 8.23 | -1.13 |
| 돼지야 (별점 4.5, 리뷰 10건) | 7.95 | 7.47 | -0.48 |
| **두 매장 간 격차** | **1.41** | **0.76** | **46% 축소** |

돼지야 순위: 79위 → **74위** (소폭 개선)

#### 3차 후 문제: top-10 진입 불가

한식 상위 10개 점수 범위 8.20~8.51, 돼지야 7.47 → 0.73점 부족. 고정 순위 기반이면 매번 같은 top-10만 노출되어 다양성 부족.

---

### 4차 개선 — 점수 지터(노이즈) + exploit/explore 비율 조정

매 호출마다 가우시안 노이즈를 추가하여 순위를 흔들어 롱테일 매장의 top-k 진입 기회를 만듦.

```python
score += random.gauss(0, 0.8)   # σ=0.8
top_k = 7; explore_k = 13       # 10+5 → 7+13 = 20개
```

몬테카를로 시뮬레이션 결과 (10,000회):

| 설정 | 돼지야 후보 포함율 | 7일 기대 방문 |
|------|----------------|-------------|
| 기존 (top10+exp5, σ=0) | 0.7% | ~0.2회 |
| σ=0.8, top7+exp13 | **9.6%** | **~2.1회** |

#### 4차 후 문제: 인기매장 안정성 붕괴

σ=0.8이 점수 범위(상위 매장 간 차이 ~0.5점) 대비 너무 커서 exploit이 사실상 랜덤화됨.

| 매장 | σ=0 잔류율 | σ=0.8 잔류율 |
|------|----------|------------|
| 망원동 막국수 (1위) | 99%+ | **26.9%** |
| 송정 (2위) | 99%+ | **15.7%** |

**exploit-explore 구조의 근본적 한계**: "상위 N개 고정 / 나머지 랜덤" 이분법 → 노이즈를 키우면 exploit이 무너지고, 줄이면 explore 매장이 안 보임. 인기/비인기 갭 = **8.8배** (63% vs 7.2%).

---

### 5차 개선 (최종) — Softmax 가중 확률 샘플링

exploit/explore 이분법을 폐기하고, **Softmax 확률 비례 비복원 추출**로 전환.

- 기존: 상위 N개 고정 포함 + 나머지 랜덤 → 포함 여부가 0% or 100%
- 변경: 모든 매장이 점수에 비례한 확률로 추출 → 연속적 확률 분포

```python
# 점수를 Softmax 확률로 변환 (temperature = 0.5)
exp_scores = [exp((score - max_score) / temperature) for score in scores]
probs = [e / sum(exp_scores) for e in exp_scores]

# 확률 비례 비복원 추출 (20개)
selected = weighted_sample_without_replacement(stores, probs, k=20)
```

**Temperature(T)**: T↓ = 인기매장 집중, T↑ = 균등 분포

Temperature별 비교 (한식 186개 → 20개 선택, 5,000회 몬테카를로):

| 방식 | 인기매장(top5) 포함율 | 돼지야 포함율 | 인기/롱테일 갭 |
|------|---------------------|-------------|-------------|
| 4차 (exploit-explore) | 63.0% | 7.2% | 8.8배 |
| Softmax T=0.3 | 55.9% | 5.5% | 10.2배 |
| **Softmax T=0.5 (채택)** | **40.7%** | **10.8%** | **3.8배** |
| Softmax T=0.8 | 29.9% | 12.0% | 2.5배 |
| Softmax T=1.0 | 26.1% | 12.4% | 2.1배 |

**T=0.5 선택 근거**:
- 인기매장 개별 포함율 60~80% → 충분한 노출 빈도
- 롱테일 매장 포함율 0.7% → 10.8% (15배 향상)
- 인기/롱테일 갭 8.8배 → 3.8배 (57% 축소)
- 저평점 매장(별점 ≤3.0) 포함율 0.0% → 품질 하한선 유지

#### 카테고리별 일반화 검증

| 카테고리 | 매장수 | 롱테일 비율 | 기존 포함율 | 개선 후 | 개선배수 |
|---------|-------|-----------|----------|--------|---------|
| 한식 | 186 | 66개 (35%) | 0.7% | 11.2% | **16x** |
| 커피/음료 | 178 | 91개 (51%) | 0.7% | 11.2% | **16x** |
| 호프/주점 | 102 | 55개 (54%) | 0.7% | 19.2% | **28x** |
| 양식 | 69 | 33개 (48%) | 0.7% | 26.8% | **38x** |
| 일식 | 49 | 18개 (37%) | 0.7% | 37.7% | **54x** |

매장 수가 적은 카테고리일수록 효과가 큽니다 (분모가 작으니 각 매장의 확률이 더 높아짐).

---

### 최종 채택안

**완료** — 카테고리 선필터 + candidate_stores + 리뷰 캡/가중치 완화 + Softmax 가중 샘플링(T=0.5, 20개)

| 항목 | v1 (기존) | v6 (최종) |
|------|-----------|-----------|
| 카테고리 처리 | 보너스(+3.0) | 선필터 (같은 업종끼리만 경쟁) |
| 상주 반경 검색 | 전역 DB 검색 (버그) | `candidate_stores` 전달 (반경 내만) |
| 리뷰 가중치 | `log1p(reviews) × 0.5` | `log1p(min(reviews,50)) × 0.3` |
| 선택 방식 | exploit 10 + explore 5 (고정) | Softmax T=0.5 가중 샘플링 (20개) |
| 인기/롱테일 갭 | 8.8x | **3.8x** |
| 돼지야 후보 포함율 | 0.7% | **10.8%** (×15) |
| 돼지야 7일 기대 방문 | ~0.2회 | **~2.3회** (×12) |

### 변경 파일
- `src/data_layer/global_store.py`: `search_ranked_stores()` 전면 리팩토링 — 카테고리 선필터, `candidate_stores` 파라미터, 리뷰 캡/가중치 완화, Softmax 가중 샘플링
- `src/simulation_layer/persona/cognitive_modules/action_algorithm.py`: Step3에서 `candidate_stores=affordable_stores` 전달

---

## 10. 과거 경험 기반 의사결정 (Step2/Step3 memory_context)

### 현재 방식
- **Step2** (업종 선택): `recent_categories`만 전달 → 뭘 먹었는지만 알고, 좋았는지 나빴는지 모름
- **Step3** (매장 선택): `recent_stores`만 전달 → 매장별 전체 에이전트 평점은 있지만, 본인 경험은 없음
- LLM이 "이전에 여기 가서 별로였다" 같은 판단 불가능
- **결과**: 과거 경험과 무관하게 매번 새로운 판단 → 불만족했던 매장 재방문, 만족했던 매장 무시

### 최종 채택안
**완료** — memory_context 통합 (오늘 식사 + 과거 방문 + 평점 + 코멘트) + 프롬프트 유도 문구

구현 내용:

| Step | 변경 전 | 변경 후 |
|------|---------|---------|
| Step2 | `recent_categories` (카테고리명만) | `memory_context` (오늘 식사 + 과거 방문 + 평점 + 코멘트) |
| Step3 | `recent_stores` (매장명만) | `memory_context` (오늘 식사 + 과거 방문 + 평점 + 코멘트) |

**memory_context 출력 예시:**
```
오늘의 식사 기록 (현재 1끼 외식 완료):
  - 12:00 류진 (한식)

당신의 과거 경험:
  - 류진 (한식): 좋음 → "국물이 진해서 좋았음"
  - 스시야 (일식): 별로 → "가격 대비 양이 적었음"
  - 카페우아 (카페): 매우좋음 → "인테리어 예쁘고 커피도 맛있었음"
```

**프롬프트 유도 문구:**
- Step2: "과거 경험에서 만족했던 업종은 재방문, 불만족했던 업종은 피하되 새로운 업종도 시도하세요."
- Step3: "과거에 방문했던 매장이 있다면, 만족했으면 재방문하고 불만족했으면 피하세요. 단, 항상 같은 곳만 가지 말고 새로운 매장도 탐색하세요."

### 변경 파일
- `src/simulation_layer/persona/agent.py`: `VisitRecord`에 `comment` 필드, `get_memory_context(current_date)` 메서드
- `src/simulation_layer/persona/cognitive_modules/action_algorithm.py`: Step2/Step3에 `memory_context` 전달
- `src/ai_layer/prompts/step2_category.txt`: `{memory_context}` 변수 + 유도 문구
- `src/ai_layer/prompts/step3_store.txt`: `{memory_context}` 변수 + 유도 문구

---

## 12. Step5 직장인 선택지 구분

### 현재 방식
- Step5 선택지가 **상주/유동** 2가지로만 분리되어 있음
- 1인·2인·4인 생활인구 페르소나 중 직장인이 상당수 포함되어 있음
- 직장인은 상주와 유동에 걸쳐있지만, 별도 구분이 없음
- **결과**: 직장인에게 "집에서_쉬기"나 "망원동_떠나기"가 부자연스러움 → "직장" 선택지를 넣기도 애매

### 개선안: Step5 선택지 3분류 (상주/유동/직장인)

**상주 에이전트:**
- 카페_가기, 한강공원_산책, 망원시장_장보기, 배회하기, 집에서_쉬기

**유동 에이전트:**
- 카페_가기, 한강공원_산책, 배회하기, 망원동_떠나기

**직장인 에이전트:**
- 카페_가기, 한강공원_산책, 배회하기, 직장_복귀

> 직장인 판별: 페르소나 세그먼트에 "직장인" 태그가 있거나, entry_point가 망원역이면서 entry_time_slot이 점심인 경우 등으로 판별
> 직장인은 상주/유동 어디에든 속할 수 있으므로, 기존 분류 위에 오버라이드하는 방식

### 최종 채택안
미정

---

## 15. 한강 위 에이전트 노드

### 현재 방식
- **파일**: `src/data_layer/street_network.py`
- OSMnx `graph_from_point(radius=2000)` → 반경 2km 내 모든 도로 네트워크 로드
- 망원한강공원 남쪽 한강 다리(마포대교, 성산대교) 도로 노드가 포함됨

### 문제점
- 에이전트가 한강 위 다리 노드를 경유지로 사용 → 지도에서 한강 위를 걸어다니는 것처럼 표시
- 실제 보행자는 다리 위를 걷지 않음

### 최종 채택안
**완료** — `load_graph()`에서 lat < 37.550 노드 제거

```python
river_nodes = [n for n, d in self._graph.nodes(data=True) if d.get('y', 999) < 37.550]
if river_nodes:
    self._graph.remove_nodes_from(river_nodes)
```

### 변경 파일
- `src/data_layer/street_network.py`: `load_graph()` — OSM 그래프 로드 후 한강 위도 이하 노드 일괄 제거

---

## 16. 에이전트 추적 지도 UI 개선

### 현재 방식
- Folium 기반 지도, `st_folium()` 렌더링
- 색상 원(CircleMarker)만으로 에이전트/매장/랜드마크 구분
- 애니메이션 재생 시 `st.rerun()` → 전체 페이지 새로고침 발생

### 문제점
1. 색상 점이 무엇을 의미하는지 직관적으로 파악 불가
2. 에이전트 상태(식사/카페/이동 등)를 점 색상만으로 표현 → 가독성 낮음
3. `st.rerun()`이 전체 페이지를 재렌더링하여 차트/테이블까지 깜빡임

### 최종 채택안
**완료** — pydeck 전환 + 이모지 마커 + @st.fragment 부분 렌더링

1. **pydeck(deck.gl) 전환**: WebGL 기반 고성능 렌더링, `st.pydeck_chart()` 사용
2. **이모지 + 텍스트 라벨**:
   - 랜드마크: `TextLayer` — `📍망원시장`, `📍망원역` 등
   - 방문 매장: `ScatterplotLayer` + `TextLayer` — `🍽️매장명`
   - 에이전트: `TextLayer` — 상태별 이모지 (🚶이동/🍽️식사/☕카페/🌳산책/🛒장보기/🏠집/💼회사)
3. **`@st.fragment` 적용**: 애니메이션 영역만 분리하여 `st.rerun()` 시 해당 fragment만 재렌더링 → 나머지 대시보드는 깜빡이지 않음
4. **컨트롤 UI 정리**: 프로필/컨트롤 분리, 날짜+재생/정지+배속을 한 줄에 배치

### 변경 파일
- `scripts/dashboard.py`:
  - `import pydeck as pdk` 추가
  - 에이전트 추적 섹션: Folium → pydeck (TextLayer, ScatterplotLayer, PathLayer)
  - `@st.fragment`로 애니메이션 영역 분리
- `requirements.txt`: `pydeck>=0.8.0` 추가

---

## 17. 10개 매장 배치 시뮬레이션

### 배경
10개 타겟 매장에 대해 각각 전략 적용 전/후 시뮬레이션 비교가 필요.

### 최종 채택안
**완료** — `run_batch_10stores.py` 배치 스크립트

대상 매장: 돼지야, 망원부자부대찌개, 메가MGC커피 망원망리단길점, 반했닭옛날통닭 망원점, 오늘요거, 육장, 육회by유신 망원점, 전조, 정드린치킨 망원점, 크리머리

처리 흐름 (매장당):
1. 원본 JSON 백업
2. Before 시뮬레이션 (160명, 7일, seed 42)
3. StrategyBridge로 전략 적용 (X-Report MD 기반)
4. After 시뮬레이션 (동일 조건)
5. Before/After 비교 결과 출력
6. 원본 JSON 복원

기능:
- **이어서 실행**: `visit_log.csv` 존재 시 해당 매장 건너뜀
- **재시도**: 실패 시 30초 대기 후 최대 3회 재시도
- **Windows cp949 대응**: stdout/stderr UTF-8 래핑으로 이모지 인코딩 에러 방지

### 변경 파일
- `scripts/run_batch_10stores.py` (신규)
- `scripts/run_before_after_sim.py`: `--output-prefix` 인자 추가 → 매장별 폴더명 지정

---

## 18. 미사용 코드/설정 정리

### 삭제 대상

**config/settings.py**:
- `AreaSettings` 클래스 전체 (area_code, quarter, lat/lng bounds, center_lat/lng)
- `SimulationSettings`: agent_count, time_slots_per_day, simulation_days 등 9개 필드
- `PathSettings`: population_json, agents_json 등 5개 속성
- `Settings.area` 필드

**.env**:
- `AREA_*` 변수 6개, `SIM_AGENT_COUNT`, `SIM_TIME_SLOTS_PER_DAY`, `SIM_SIMULATION_DAYS`, `SIM_REPORT_RECEPTION_PROBABILITY`, `SIM_VISIT_THRESHOLD`

**requirements.txt**:
- `fastapi`, `uvicorn[standard]`, `geopandas` (어떤 .py에서도 import되지 않음)

**src/data_layer/street_network.py**:
- 미사용 `Point` import (shapely)

### 변경 파일
- `config/settings.py`, `.env`, `requirements.txt`, `src/data_layer/street_network.py`

---

## 12. 아침 "이전 식사로 충분" 버그

### 현재 방식
- **파일**: `src/simulation_layer/persona/cognitive_modules/action_algorithm.py` (Step1)
- Step1에서 외출 안 함 사유를 랜덤 선택: "배가 안 고픔", "이전 식사로 충분", "집에서 해결", "이 시간에는 안 먹음"
- 시간대/식사 횟수와 무관하게 동일한 사유 풀에서 선택

### 문제점
- **아침**인데 "이전 식사로 충분" 출력 → 아침은 그날 첫 끼니인데 이전 식사가 있을 수 없음
- **첫 끼니**(meals_today == 0)인데 "이전 식사로 충분" → 오늘 아무것도 안 먹었는데 이전 식사가 충분하다는 모순

### 최종 채택안
**완료** — 아침 또는 첫끼일 때 "이전 식사로 충분" 사유 제외

```python
if time_slot == "아침" or meals_today == 0:
    reasons_skip = [
        "배가 안 고픔",
        "집에서 해결",
        "이 시간에는 안 먹음",
    ]
else:
    reasons_skip = [
        "배가 안 고픔",
        "이전 식사로 충분",
        "집에서 해결",
        "이 시간에는 안 먹음",
    ]
```

### 변경 파일
- `src/simulation_layer/persona/cognitive_modules/action_algorithm.py`: Step1 `step1_eat_in_mangwon()` 내 사유 선택 분기 추가

---

## 13. 상주 에이전트 "적합한 매장 없음" 대량 발생

### 현재 방식
- **파일**: `src/data_layer/global_store.py`
- 상주 에이전트는 `home_location` 반경 800m 이내 매장만 후보로 검색
- 매장 데이터는 `split_by_store_id/` 디렉토리의 JSON 파일에서 로드
- 좌표(`x`, `y`)와 카테고리(`category`)가 JSON 최상위에 없으면 검색 불가

### 문제점

**A. JSON 좌표/카테고리 누락**
- JSON 파일에 `x`, `y` 좌표가 최상위가 아닌 `metadata.x`, `metadata.y`에만 존재
- `category` 필드도 최상위에 없고 `metadata.sector`에만 존재
- 반경 검색 시 좌표가 없는 매장은 거리 계산 불가 → 후보에서 제외
- **결과**: 상주(R*) 에이전트 점심에 전원 "적합한 매장 없음"으로 외출 실패

**B. 카테고리 명칭 불일치** (Step3 매장 필터링)
- Step2 LLM이 선택하는 카테고리와 JSON 실제 카테고리 명칭이 다름
- 매칭 코드가 `category.lower() in s.category.lower()` (부분 문자열)이라 식당류는 우연히 통과
- 카페/디저트/베이커리는 매칭 실패 → fallback으로 전체 매장 사용 (필터링 무의미)

| Step2 LLM 선택값 | JSON 실제 카테고리 | `in` 매칭 |
|---|---|---|
| "한식" | "한식음식점" | O (부분 문자열) |
| "양식" | "양식음식점" | O |
| "카페" | "커피-음료" | **X** |
| "디저트" | "제과점" | **X** |
| "베이커리" | "제과점" | **X** |

### 최종 채택안
**완료** — A. JSON metadata fallback + B. CATEGORY_ALIAS 매핑 테이블

**A. JSON 로딩 시 `metadata.x,y` + `metadata.sector` fallback**

```python
# 카테고리: JSON category → metadata.sector
category = data.get("category", "")
if not category:
    metadata = data.get("metadata", {})
    category = metadata.get("sector", "")

# 좌표: JSON x,y → metadata.x,y
x = data.get("x")  # longitude
y = data.get("y")  # latitude
if not x or not y:
    metadata = data.get("metadata", {})
    x = metadata.get("x")
    y = metadata.get("y")
```

**B. CATEGORY_ALIAS + match_category() 헬퍼 함수**

```python
CATEGORY_ALIAS = {
    "카페": "커피-음료", "커피": "커피-음료",
    "디저트": "제과점", "베이커리": "제과점", "브런치": "양식음식점",
    "이자카야": "호프-간이주점", "포차": "호프-간이주점",
    "와인바": "호프-간이주점", "술집": "호프-간이주점",
    "막걸리": "호프-간이주점", "칵테일바": "호프-간이주점",
}

def match_category(query: str, store_category: str) -> bool:
    """1차: 부분 문자열 매칭, 2차: CATEGORY_ALIAS 변환 후 매칭"""
    q = query.lower()
    sc = (store_category or "").lower()
    if q in sc:
        return True
    alias = CATEGORY_ALIAS.get(query, "").lower()
    if alias and alias in sc:
        return True
    return False
```

- 기존 `category.lower() in s.category.lower()` → `match_category()` 로 전부 교체
- global_store.py 4곳 + action_algorithm.py 1곳 = 총 5곳 적용

### 변경 파일
- `src/data_layer/global_store.py`:
  - `load_from_json_dir()`: metadata fallback 로직
  - `CATEGORY_ALIAS` 딕셔너리 + `match_category()` 헬퍼 함수 추가
  - `search_ranked_stores()`, `get_stores_by_category()`, `get_stores_in_budget()`, `get_top_stores_by_agent_rating()`: match_category() 적용
- `src/simulation_layer/persona/cognitive_modules/action_algorithm.py`: Step3 카테고리 필터링에 match_category() 적용

---

## 27. 카테고리 중복 방문 (Natural Constraints)

### 현재 방식
- **파일**: `src/simulation_layer/persona/cognitive_modules/action_algorithm.py`
- `process_decision` 내 루프에서 하드 가드(Hard Guard) 존재:

```python
# 기존 코드 (루프 내)
visited_cats = {v["visited_category"] for v in visits}
if loop_category in visited_cats:
    break  # 같은 카테고리면 무조건 루프 탈출
```

- 프롬프트(`step5_next_action.txt`)에서도 "같은 유형의 매장 연속 방문은 비현실적"이라고 강하게 부정

### 문제점
- **현실성 부족**: 현실에서는 "카페 1차 → 카페 2차"나 "1차 술집 → 2차 술집"이 빈번함  
- **기계적 제약**: LLM이 "카페 한 번 더 가고 싶다"고 판단해도 코드가 강제로 종료시킴

### 최종 채택안
**완료** — 하드 가드 제거 + `[고려사항]` 내부 맥락 주입 (Internal Thought)

#### 1. 하드 가드 주석 처리
```python
# 하드 가드 제거: 비현실적이지만 가능성은 열어둠 (프롬프트로 제어)
# visited_cats = {v["visited_category"] for v in visits}
# if loop_category in visited_cats:
#     break
```

#### 2. 동적 경고 (Internal Thought) 주입
```python
# step5 session_activity 구성 시
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
```

#### 3. 동작 시나리오
| 상황 | 주입 메시지 | LLM 예상 판단 |
|---|---|---|
| 카페 첫 방문 | 없음 | 자유 판단 |
| 카페 2번째 방문 | "방금 카페 다녀왔습니다... 가능합니다" | 대부분 다른 활동 선택 |
| 카페 3번째 시도 | "이미 2번... 부자연스럽습니다" | 다른 활동으로 전환 유도 |

### 변경 파일
- `src/simulation_layer/persona/cognitive_modules/action_algorithm.py`:
  - 하드 가드(`break`) 주석 처리 (루프 834~836행)
  - `session_activity` 구성 시 동적 경고 추가 (510~523행)
- `src/ai_layer/prompts/step5_next_action.txt`: "비현실적" → "일반적이진 않지만 이유가 있으면 가능" 어조 완화

---

## 14. 카페/주점 방문 불가

### 현재 방식
- **파일**: `src/simulation_layer/persona/cognitive_modules/action_algorithm.py` (720~723행)
- `destination_type`이 항상 `"식당"`으로 고정
- `DESTINATION_CATEGORIES`에 카페/주점 카테고리가 정의되어 있지만, Step2에 전달되지 않음

```python
# 현재 코드 (720~723행)
destination_type = "식당"  # 기본값
if time_slot in ["저녁", "야식"]:
    destination_type = "식당"  # 저녁/야식은 식당 위주
```

### 문제점
1. `DESTINATION_CATEGORIES`에 카페(카페, 커피, 디저트, 베이커리, 브런치)와 주점(호프, 이자카야, 포차 등)이 정의되어 있음
2. 하지만 `destination_type = "식당"` 고정이라 Step2에 식당 카테고리만 전달됨
- **결과**: 아침 커피, 저녁 술자리 등 현실적인 선택이 불가능

> 카테고리 명칭 불일치 문제(LLM 선택값 vs JSON 카테고리)는 #13에서 `CATEGORY_ALIAS` + `match_category()`로 해결 완료

### 개선안: destination_type 분기 제거 → 전체 카테고리를 Step2에 전달

현재 흐름:
```
destination_type = "식당" (고정) → DESTINATION_CATEGORIES["식당"]만 Step2에 전달
```

개선 흐름:
```
시간대별 available_categories를 직접 구성 → Step2에 전달 → LLM이 페르소나 기반으로 선택
```

```python
# 시간대별 카테고리 풀 구성
TIME_SLOT_CATEGORIES = {
    "아침": DESTINATION_CATEGORIES["식당"] + DESTINATION_CATEGORIES["카페"],
    "점심": DESTINATION_CATEGORIES["식당"] + DESTINATION_CATEGORIES["카페"],
    "저녁": DESTINATION_CATEGORIES["식당"] + DESTINATION_CATEGORIES["주점"],
    "야식": DESTINATION_CATEGORIES["식당"] + DESTINATION_CATEGORIES["주점"],
}
available_categories = TIME_SLOT_CATEGORIES.get(time_slot, DESTINATION_CATEGORIES["식당"])
```

- 확률로 destination_type을 정하지 않고, 시간대에 맞는 **전체 카테고리 풀**을 LLM에 넘김
- LLM이 페르소나(세대, 동행, 취향)를 보고 자연스럽게 선택
  - Z세대 1인 야식 → "호프" 또는 "이자카야" 선택 가능
  - 가족단위 저녁 → "한식" 또는 "양식" 선택 (주점은 자연 회피)
  - 직장인 아침 → "카페" 또는 "베이커리" 선택 가능
- Step2 프롬프트에 시간대별 힌트 추가 ("아침에는 카페/브런치도 고려하세요" 등)

### 실측 근거 (10매장 배치 시뮬 결과)

**메가MGC커피 망원망리단길점** (카테고리: `커피-음료`)
- Before: 방문 0건 / After: 방문 0건
- 전체 1,147건 방문 중 카페 카테고리 방문: **0건**
- 방문 카테고리 분포: 한식 438, 일식 243, 양식 223, 중식 146, 치킨 95, 패스트푸드 2

**돼지야** (카테고리: `한식음식점`)
- Before: 방문 0건 / After: 방문 0건
- 한식 매장이 438건 방문되었으나, 돼지야는 0건 → Softmax 랭킹에서 밀림 (별도 이슈 #23)

**원인 분석:**
- `action_algorithm.py` L720-723에서 `destination_type = "식당"` 하드코딩
- `DESTINATION_TYPES` 딕셔너리를 정의해놓고 **한 번도 사용하지 않음**
- Step2에 `DESTINATION_CATEGORIES["식당"]`만 전달 → LLM이 "카페"/"커피" 선택 불가
- 결과적으로 커피-음료, 제과점, 호프-간이주점 카테고리 매장은 **구조적으로 방문 불가능**

### 최종 채택안
**완료** — 개선안 A 채택: destination_type 분기 제거 → 전체 카테고리 풀 전달

**채택한 개선안 A: destination_type 분기 제거 → 전체 카테고리 풀 전달**
```python
TIME_SLOT_CATEGORIES = {
    "아침": DESTINATION_CATEGORIES["식당"] + DESTINATION_CATEGORIES["카페"],
    "점심": DESTINATION_CATEGORIES["식당"] + DESTINATION_CATEGORIES["카페"],
    "저녁": DESTINATION_CATEGORIES["식당"] + DESTINATION_CATEGORIES["주점"] + DESTINATION_CATEGORIES["카페"],
    "야식": DESTINATION_CATEGORIES["식당"] + DESTINATION_CATEGORIES["주점"],
}
available_categories = TIME_SLOT_CATEGORIES.get(time_slot, DESTINATION_CATEGORIES["식당"])
```
- destination_type 개념 자체를 제거, Step2에 시간대별 전체 카테고리 풀 전달
- LLM이 페르소나 기반으로 자연 선택 (Z세대 아침 → 카페, 가족 저녁 → 한식)
- Step2 프롬프트에 시간대 힌트 추가 ("아침에는 카페/브런치도 고려하세요")

**개선안 B: DESTINATION_TYPES 실제 사용 + 확률 기반 분기**
```python
dest_types = DESTINATION_TYPES.get(time_slot, ["식당"])
destination_type = random.choice(dest_types)
```
- 기존 DESTINATION_TYPES 딕셔너리 활용, 랜덤으로 목적지 유형 선택
- 장점: 변경 최소 (1줄 수정), 기존 Step2 프롬프트 그대로 사용
- 단점: 카페/주점 선택 확률이 균등 → 비현실적 (아침에 카페:식당 = 1:1)

**개선안 C: 가중 확률 기반 목적지 유형 선택**
```python
DESTINATION_WEIGHTS = {
    "아침": {"식당": 0.6, "카페": 0.4},
    "점심": {"식당": 0.75, "카페": 0.25},
    "저녁": {"식당": 0.6, "주점": 0.25, "카페": 0.15},
    "야식": {"식당": 0.5, "주점": 0.5},
}
weights = DESTINATION_WEIGHTS.get(time_slot, {"식당": 1.0})
destination_type = random.choices(list(weights.keys()), list(weights.values()))[0]
```
- 시간대별 현실적 비율 반영
- 장점: 가장 현실적인 방문 분포
- 단점: 가중치 수치를 별도 검증 필요

---

## 23. 타겟 매장 방문 0건 (Softmax 랭킹 편중)

### 현재 방식
- Step3에서 반경 내 매장을 Softmax 가중 샘플링으로 20개 추출
- LLM에 20개 후보를 보여주고 1개 선택

### 문제점
1. **돼지야**: 한식 카테고리이지만 같은 한식 438건 중 0건 → Softmax 랭킹에서 밀림
   - 별점/리뷰 수/rag_context 품질이 상위 매장 대비 낮을 가능성
   - 리뷰 수가 적으면 Softmax 점수 자체가 낮아 후보 20개에 진입 못함
2. **메가MGC커피**: 카테고리 자체가 선택 안 됨 (이슈 #14)

### 개선안 (비채택)

**A: 타겟 매장 강제 후보 포함** — 비채택 (공정성 훼손)
**B: Softmax 온도 조정** — 비채택 (부작용 우려)
**C: 카테고리 내 클러스터링** — 비채택 (구현 복잡)

### 최종 채택안
**완료** — 계층화 샘플링으로 전환 (#14 코드 수정과 병행 적용)

기존 Softmax 비복원 추출 → **3계층 샘플링**:
- 점수순 정렬 후 상위 1/3, 중위 1/3, 하위 1/3으로 분류
- 상위 60% + 중위 20% + 하위 20% 비율로 추출
- 각 계층 내에서는 Softmax 확률 비례 추출 유지
- 롱테일 매장(돼지야 등)이 하위 계층에서 15% 확률로 LLM 후보에 진입

### 실측 결과 (돼지야 7일 시뮬레이션, 160명, seed 42)

#### 수정 전 → 수정 후 비교 (코드 수정 = #14 + #23 동시 적용)

| 지표 | 수정 전 (v1) | 수정 후 (before_v2) | 변화 |
|------|-------------|-------------------|------|
| **돼지야 방문** | **0건** | **13건** | 0 → 13 |
| 전체 방문 | 1,147건 | 1,111건 | -36건 |
| 한식 | 438건 | 384건 | -54건 |
| 커피-음료 | **0건** | **35건** | 0 → 35 |
| 호프-간이주점 | 0건 | 176건 | 0 → 176 |
| 일식 | 243건 | 147건 | -96건 |
| 제과점 | 0건 | 28건 | 0 → 28 |

> **한식/커피-음료 등은 전체 시뮬레이션의 업종별 방문 분포**이며, 돼지야 개별 방문 수와는 별개.
> 커피-음료가 0→35건으로 증가한 것은 #14(destination_type 하드코딩 수정)의 효과이고,
> 돼지야가 0→13건으로 증가한 것은 #23(계층화 샘플링)의 효과.

#### before_v2 vs v2_after 비교 (StrategyBridge 전략 적용 효과)

| 지표 | before_v2 (코드 수정만) | v2_after (코드 수정 + 전략 적용) |
|------|----------------------|-------------------------------|
| **돼지야 방문** | **13건** | **1건** |
| 전체 방문 | 1,111건 | 1,079건 |
| 한식 | 384건 | 405건 |
| 커피-음료 | 35건 | 39건 |

> StrategyBridge가 `돼지야.json`의 매장 속성(키워드, 설명 등)을 변경했으나,
> 오히려 방문이 13→1건으로 감소. 전략 적용이 매장 점수에 부정적 영향을 미쳤을 가능성.
> **코드 레벨 수정(#14 + #23)이 결정적 효과를 가지며, 매장 JSON 전략 적용은 추가 검증 필요.**

#### 결론

1. **#14 + #23 코드 수정 방식 채택**: destination_type 자유 선택 + 계층화 샘플링으로 돼지야 0→13건, 카페 0→35건 달성
2. **StrategyBridge 전략 적용은 별도 검증 필요**: 매장 JSON 속성 변경만으로는 방문 증가 효과 미미 (오히려 감소)
3. 코드 수정만으로도 롱테일 매장 방문이 발생하며, 전체 업종 분포가 현실적으로 개선됨

### 변경 파일
- `src/data_layer/global_store.py`: `search_ranked_stores()` 내부 샘플링 로직
- `src/simulation_layer/persona/cognitive_modules/action_algorithm.py`: `TIME_SLOT_CATEGORIES` 도입 (#14와 연동)

---

## 24. 대시보드 에이전트 좌표 fallback

### 현재 방식
- 유동 에이전트의 `home_location`이 `[0.0, 0.0]`으로 저장됨
- 대시보드에서 `if home_location:` → `[0,0]`은 truthy → 그대로 사용 → 아프리카 앞바다에 표시

### 문제점
- P007 등 유동 에이전트가 대기/집 상태일 때 지도에서 사라지거나 엉뚱한 위치에 표시
- `entry_point` 좌표는 정상값이 있으나, 대시보드가 참조하지 않음

### 최종 채택안
**완료** — 3중 fallback 체인

1. **에이전트 로딩 시**: `home_location`이 `[0,0]`이면 `entry_point` 사용, 없으면 `FLOATING_LOCATIONS` 랜덤 할당
2. **get_agent_state 내부**: `[0,0]` 명시 체크 → `LANDMARKS["집"]` fallback
3. **animation_fragment**: 호출 전 `agent_home` 보정 (entry_point → LANDMARKS 순)

### 변경 파일
- `scripts/dashboard.py`: `_is_valid_loc()` 헬퍼, 에이전트 로딩/get_agent_state/animation_fragment 3곳

---

## 19. 대시보드 폴더 선택

### 현재 방식
- `data/output/generative_simulation_result.csv` 단일 파일만 로드
- before/after 결과 비교 불가

### 최종 채택안
**완료** — 사이드바 드롭다운으로 시뮬레이션 결과 폴더 선택

구현 내용:
1. `data/output/` 하위 폴더 중 `visit_log.csv`가 있는 폴더를 자동 탐색
2. 사이드바 "결과 폴더" 드롭다운에 목록 표시
3. 선택한 폴더의 `simulation_result.csv`, `visit_log.csv`, `agents_final.json` 로드
4. 폴더 미선택 시 기존 루트 CSV 로드 (하위 호환)

### 변경 파일
- `scripts/dashboard.py`: `pathlib.Path` import, 사이드바 폴더 선택 UI, `load_simulation_data()` 분기

---

## 20. 시뮬 시작 날짜 KST 기준

### 현재 방식
- `start_date = datetime(2025, 2, 7)` 하드코딩
- 시뮬 결과의 날짜가 항상 2025-02-07~13

### 최종 채택안
**완료** — 한국 시간(KST) 기준 가장 가까운 과거 금요일 자동 계산

```python
from datetime import datetime, timedelta, timezone
kst = timezone(timedelta(hours=9))
today_kst = datetime.now(kst).replace(tzinfo=None)
days_since_friday = (today_kst.weekday() - 4) % 7
start_date = (today_kst - timedelta(days=days_since_friday)).replace(hour=0, minute=0, second=0, microsecond=0)
```

### 변경 파일
- `scripts/run_generative_simulation.py`: `start_date` 계산 로직 변경

---

## 21. 에이전트 추적 슬라이더 수정

### 현재 방식
- `key` 기반 슬라이더 + `on_change` 콜백 → fragment 내에서 동기화 실패
- `st.rerun(scope="fragment")` → `StreamlitAPIException` 발생
- `increment = speed * 0.05` → 배속 10에서 한 틱에 30분 점프 (과도)

### 최종 채택안
**완료** — `value` 파라미터 직접 바인딩 + 속도 완화

1. **슬라이더**: `key` 대신 `value=st.session_state.current_hour` 직접 바인딩
2. **동기화**: 슬라이더 반환값을 항상 `current_hour`에 대입 (콜백 불필요)
3. **속도 완화**: `increment = 0.1 + (speed-1) * (0.4/59)` (배속1=6분/틱, 배속60=30분/틱)
4. **sleep**: 0.1s → 0.5s

### 변경 파일
- `scripts/dashboard.py`: 슬라이더 생성/동기화 로직, sleep 시간, increment 계산식

---

## 22. 지도 마커/뷰 개선

### 현재 방식
- 에이전트: 상태별 이모지만 표시 → 작고 알아보기 어려움
- 매장: 주황 원 + 🍴 → 에이전트와 구분 어려움
- 뷰: 에이전트 위치 중심 zoom=15.5 고정 → 경로가 잘려서 안 보임

### 최종 채택안
**완료** — 사람 마커 + 매장 핀 + 바운딩 박스 자동 줌

1. **에이전트 마커**: 🧑 사람 이모지 + 상태색 내부 원(r=25) + 외부 글로우 원(r=60) + 상태 라벨
2. **매장 마커**: 빨간 핀(r=30, 흰 테두리) + 🍴 아이콘 + 매장명 라벨(빨간색, 13pt)
3. **자동 줌**: 에이전트 + 경로 + 방문 매장 전체 좌표의 바운딩 박스 계산 → spread 기반 줌 레벨 자동 결정
   - spread < 0.001 → zoom 16.5 (근거리)
   - spread < 0.005 → zoom 15.5
   - spread < 0.01 → zoom 14.5
   - spread < 0.02 → zoom 13.5
   - spread >= 0.02 → zoom 12.5

### 변경 파일
- `scripts/dashboard.py`: 에이전트/매장 레이어 재구성, ViewState 바운딩 박스 계산

---

## 25. Step 5 활성화 + 타임슬롯 내 다중 방문 루프

### 현재 방식
- **파일**: `src/simulation_layer/persona/cognitive_modules/action_algorithm.py`
- `process_decision`이 Step 1→2→3→4까지만 실행
- Step 5(`step5_next_action`)는 정의만 되어 있고 **호출되지 않음** (죽은 코드)
- 에이전트가 식사 후 "카페_가기"를 결정해도 실제 매장 선택/평가가 없음
- **결과**: 한 타임슬롯에 1회만 방문 가능, 식사 후 카페 같은 현실적 다중 방문 불가

### 문제점

**A. Step 5 미호출 → 카페_가기 등이 매장 방문으로 이어지지 않음**
- Step 5 선택지(`카페_가기`, `한강공원_산책`, `배회하기` 등)는 정의되어 있으나 실제 호출되지 않음
- 기존에는 `_select_random_cafe()`로 랜덤 카페 좌표만 배정하고, 실제 매장 선택(Step 3)/평가(Step 4)를 거치지 않음
- 따라서 카페 방문이 **visit_log에 기록되지 않고**, 매장 평점/리뷰도 생성되지 않음
- 에이전트의 "다음 할 일" 결정 자체가 시뮬레이션에 반영되지 않는 구조적 문제

**B. 타임슬롯당 1회 방문만 가능 → 실제 소비 패턴과 괴리**
- 현실에서는 점심 식사 후 카페, 저녁 식사 후 2차(술자리) 같은 다중 방문이 자연스러움
- 기존 구조에서는 타임슬롯당 1개 매장 방문만 기록 → 카페/주점 방문 데이터가 구조적으로 부족
- 이는 #14(카페/주점 방문 불가) 해결 이후에도 실제 카페 방문 건수가 낮은 원인 중 하나

### 최종 채택안
**완료** — Step 4 이후 Step 5 호출 + 매장 방문 행동이면 Step 3→4→5 루프

변경 흐름:
```
Step1 → Step2 → Step3 → Step4 → Step5 ─┐
                  ↑                      │
                  └── 매장 방문 행동이면 ──┘
                      (카페_가기 등)
               비매장 행동이면 → return
               MAX_VISIT_LOOP=3 초과 → return
```

구현 내용:
1. **Step 5 호출**: Step 4(평가) 완료 후 `step5_next_action()` 호출
2. **매장 방문 행동 판별**: `STORE_VISIT_ACTIONS = {"카페_가기": "커피-음료"}` 매핑 테이블
3. **루프**: 매장 방문 행동이면 해당 카테고리로 Step 3(매장 선택) → Step 4(평가) → Step 5(다음 행동) 반복
4. **안전 장치**: `MAX_VISIT_LOOP = 3` — 최대 3회 추가 방문으로 무한루프 방지
5. **다중 방문 기록**: `visits` 리스트에 방문별 매장/카테고리/평점 저장

```python
# process_decision 반환값 확장
{
    "decision": "visit",
    "visits": [                          # 다중 방문 리스트
        {"visited_store": "돼지야", "visited_category": "한식", "ratings": {...}},
        {"visited_store": "메가커피", "visited_category": "커피-음료", "ratings": {...}},
    ],
    "visited_store": "돼지야",           # 첫 번째 방문 (하위 호환)
    "step5": {"action": "카페_가기", ...},  # 마지막 Step 5 결과
}
```

6. **agent_task 다중 레코드**: `visits` 리스트를 순회하며 방문당 1개 CSV 레코드 생성
7. **next_time_slot 전달**: `process_decision` 시그니처에 `next_time_slot: str = ""` 추가, Step 5 프롬프트에서 "다음 시간대까지 무엇을 할지" 판단에 활용

### LLM 호출 증가 분석

| 시나리오 | 기존 | 루프 후 (1회 추가 방문) |
|----------|:----:|:-----:|
| LLM 호출/에이전트/슬롯 | 3회 (S2+S3+S4) | 6회 (S2+S3+S4+S5+S3+S4) |
| 100에이전트 × 4슬롯 × 7일 | ~2,800 | ~4,200~5,600 |

### 변경 파일
- `src/simulation_layer/persona/cognitive_modules/action_algorithm.py`: `process_decision`에 Step 5 루프 추가, `STORE_VISIT_ACTIONS`, `MAX_VISIT_LOOP` 상수, `visits` 리스트 반환
- `scripts/run_generative_simulation.py`: `agent_task`에서 다중 visit 레코드 생성, `next_time_slot` 계산 및 전달, 로그에 추가 방문 `↳` 표시
- `src/ai_layer/prompts/step5_next_action.txt`: Step 5 프롬프트 (기존 죽은 코드에서 실제 호출로 전환)

---

## 26. Step 5 활동 맥락 부재 → session_activity 도입

### 현재 방식 (#25 초기 구현)
- Step 5 프롬프트에 `last_action: str` 단일 문자열만 전달
- "방금 한 일: 돼지야 방문" 정도의 정보만 제공
- LLM이 이번 타임슬롯에서 **이미 뭘 했는지** 전체 맥락을 모름
- 첫 방문 후 Step 5든, 다중 방문 루프 중이든 동일하게 맥락 없음
- **결과**: 어떤 매장을 방문했고 별점을 몇 점 줬는지 모르니 다음 행동 판단이 비현실적 (예: 카페 3곳 연속 방문)

### 개선 방향
사용자 요구: "너무 제약을 걸지 않은 채로 현실에서 사람이 선택을 할 때와 비슷하게"
→ 하드코딩 제약 대신 LLM에 충분한 맥락을 주고 자연스럽게 판단하도록 유도

### 최종 채택안
**완료** — `last_action: str` → `session_visits: List[Dict]` + `session_activity` 프롬프트

변경 내용:
1. **`step5_next_action` 시그니처 변경**: `last_action: str` → `session_visits: Optional[List[Dict]]`
2. **session_activity 생성**: 방문 리스트를 자연어로 변환하여 프롬프트에 삽입

```python
# session_visits → session_activity 변환
if session_visits:
    activity_lines = []
    for i, v in enumerate(session_visits, 1):
        store = v.get("visited_store", "?")
        cat = v.get("visited_category", "")
        rating = v.get("rating", "")
        rating_str = f" → 별점 {rating}/5" if rating else ""
        activity_lines.append(f"  {i}. {store}({cat}) 방문{rating_str}")
    session_activity = "\n".join(activity_lines)
else:
    session_activity = "  (아직 활동 없음)"
```

3. **Step 5 프롬프트 개편**:

변경 전:
```
- 방금 한 일: {last_action}
```

변경 후:
```
이번 시간대 활동:
{session_activity}

질문: 위 활동 이력을 고려하여, 다음 시간대까지 무엇을 할지 결정하세요.
```

**효과**: LLM이 "이미 한식 먹고 별점 4점 줬다" → 자연스럽게 "카페 가자" 판단.
"이미 한식 먹고 카페도 갔다" → 자연스럽게 "배회하기" 또는 "집에서 쉬기" 판단.
별도의 카테고리 중복 차단 코드 없이도 LLM이 맥락 기반으로 현실적 선택.

### 변경 파일
- `src/simulation_layer/persona/cognitive_modules/action_algorithm.py`: `step5_next_action` 시그니처 변경, `session_activity` 생성 로직
- `src/ai_layer/prompts/step5_next_action.txt`: `{last_action}` → `{session_activity}` + 활동 이력 기반 질문
