# Agent 고도화 남은 작업

## 총괄 요약표

| # | 항목 | 현재 방식 | 개선안 | 최종 채택안 |
|---|------|----------|--------|------------|
| 1 | 리뷰 멘트 다양성 | Step4 프롬프트가 "한줄 평가"만 지시 → 전원 동일한 키워드 재조합 | 페르소나 말투 지정 + 관점 분산 + 구체적 경험 유도 | **완료** (tone_instruction + focus_aspect + comment 규칙) |
| 2 | 4끼 전부 외식 | 매 타임슬롯마다 Step1 호출, memory_context에 당일 식사 이력 없음 | A. Step1 확률기반 전환 + B. memory_context에 당일 식사 이력 추가 | **A+B 완료** |
| 3 | 유동 초기위치/시작시간 | FLOATING_LOCATIONS → home_location 저장, 아침부터 전원 활동 | entry_point 분리 + entry_time_slot 도입 + "집에서_쉬기" → "망원동 떠나기" | **완료** |
| 4 | LLM 모델 선택 | GPT-4o-mini (고정) | Gemini 2.0 Flash (33% 저렴, 성능 우위, 속도 40% 빠름) | 미정 |
| 5 | 유동인구 교체 비율 | 매일 113명 중 53명 랜덤 샘플링 (고정) | 요일별 차등(평일 40/주말 70) + 재방문율 30% | 미정 |
| 6 | 상주인구 초기위치 | 주거유형별 3좌표 고정 (아파트3, 빌라3, 주택3) | 고정 좌표를 중심점으로 반경 100~200m 랜덤 오프셋 | 미정 |
| 7 | 시뮬레이션 속도 | 순차 호출 (호출당 1.5s, 7일 약 2시간 51분) | asyncio 병렬 5~10개 동시 호출 (7일 약 21~34분) | 미정 |
| 8 | 유동 매장 선택 최적화 | 720개 전체 매장을 LLM에 전달 → positional bias + 토큰 낭비 | 검색엔진 랭킹(exploit 10 + explore 5) → 15개만 LLM에 전달 | 미정 |
| 9 | 매장/에이전트 좌표 불일치 | 대시보드 지도에서 좌표가 남동쪽으로 밀림 (상주가 공원에, 유동이 망원 바깥에 표시) | 하드코딩 좌표를 실제 지도와 대조 검증 + OSM 중심점 보정 | 미정 |
| 10 | 과거 경험 기반 의사결정 | Step2는 recent_categories만, Step3는 recent_stores만 → 평점/코멘트 없음 | memory_context 통합 (오늘 식사 + 과거 평점 + 코멘트) + 유도 문구 | **완료** |

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

### 개선안
- **말투 지정**: "당신의 세대와 성격에 맞는 말투로 작성 (Z세대면 ~ㅋㅋ, X세대면 정중체)"
- **관점 분산**: 랜덤으로 관점 하나를 넘김 (맛/가격/분위기/서비스/메뉴구성 중)
- **구체적 경험 유도**: "구체적으로 무엇을 먹었고 어떤 점이 좋았/아쉬웠는지"
- **리뷰 확률 도입**: 페르소나별 리뷰 작성 확률 차등 (Z세대 높게, X세대 낮게)

### 최종 채택안
미정

---

## 2. 4끼 전부 외식

### 현재 방식
- **파일**: `src/ai_layer/prompts/step1_destination.txt`, `scripts/run_generative_simulation.py`
- 매 타임슬롯(아침/점심/저녁/야식)마다 무조건 Step1 LLM 호출
- Step1 프롬프트가 "식사를 어디서 할지"만 물음 → "안 먹는다" 선택지가 약함
- `memory_context`에 당일 식사 이력이 없어서 LLM이 "이미 몇 끼 먹었는지" 모름
- **결과**: 상주든 유동이든 4끼 모두 외식하는 비현실적 패턴

### 개선안 (2가지 병행)

**A. Step1 프롬프트 수정** (완료)
```
질문: 이 시간대에 외식을 할지 결정하세요.
- 망원동 내 매장에서 식사한다
- 이 시간대에는 먹지 않는다 (배가 안 고픔, 이전 식사로 충분, 다이어트 중 등)
- 망원동 밖에서 해결한다 (집밥, 편의점, 배달 등)

현실적으로 판단하세요. 대부분의 사람은 하루 2~3끼를 먹고, 4끼를 모두 외식하는 경우는 극히 드뭅니다.
```

**B. memory_context에 당일 식사 이력 추가** (필요)
- `run_generative_simulation.py`에서 에이전트별 당일 식사 횟수 추적
- Step1 호출 시 memory_context에 포함:
```
오늘의 식사 기록:
- 아침(07:00): 망원동 OO식당에서 식사함
- 점심(12:00): 아직 안 먹음
(현재 오늘 1끼 외식 완료)
```
- 프롬프트만으로는 불충분 — LLM이 이전 타임슬롯 결정을 모르기 때문

### 최종 채택안
미정

---

## 3. 유동 초기위치 / 시작시간

### 현재 방식
- **파일**: `src/simulation_layer/persona/agent.py` (FLOATING_LOCATIONS, home_location)
- 유동 에이전트 생성 시 FLOATING_LOCATIONS(역/정류장 6곳) 중 랜덤 → `home_location`에 저장
- 시뮬레이션은 07시(아침)부터 4타임슬롯 순차 실행 → 유동도 전원 아침부터 활동
- Step5 "집에서_쉬기" → `home_location`(=정류장) 좌표로 이동
- **결과**: 유동이 정류장을 집으로 인식, 아침부터 무조건 활동

### 개선안
1. **`home_location` → `entry_point` 분리**
   - 유동 에이전트는 `home_location` 제거, `entry_point` (진입 지점) 사용
   - 진입 지점별 페르소나 가중치: 망원역(직장인↑), 한강공원(데이트↑), 시장(생활형↑)

2. **`entry_time_slot` 속성 추가**
   - 세그먼트별 진입 시간 차등: 직장인→점심, 데이트족→저녁, 관광→아침
   - entry_time_slot 이전 타임슬롯은 스킵

3. **Step5 선택지를 유형별로 분리**

   **상주 에이전트:**
   - 집에서_쉬기: 집으로 돌아가서 휴식
   - 카페_가기: 근처 카페에서 시간 보내기
   - 한강공원_산책: 망원한강공원에서 산책
   - 망원시장_장보기: 망원시장에서 장보기
   - 배회하기: 망원동 거리를 걸으며 구경하기

   **유동 에이전트:**
   - 카페_가기: 근처 카페에서 시간 보내기
   - 한강공원_산책: 망원한강공원에서 산책
   - 망원시장_장보기: 망원시장에서 장보기
   - 배회하기: 망원동 거리를 걸으며 구경하기
   - 망원동_떠나기: 망원동에서 퇴장 (이후 타임슬롯 스킵)

   > 유동에게는 "집에서_쉬기" 선택지 자체를 제거하고, 대신 "망원동_떠나기"를 추가.
   > 떠나기 선택 시 해당 일 시뮬레이션에서 퇴장.

### 최종 채택안
**완료** — entry_point 분리 + entry_time_slot(세그먼트별) + Step5 유형별 선택지 + 망원동_떠나기

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

### 개선안: Gemini 2.0 Flash

| 항목 | GPT-4o-mini | Gemini 2.0 Flash |
|------|-------------|------------------|
| Input / 1M tokens | $0.15 | $0.10 (-33%) |
| Output / 1M tokens | $0.60 | $0.40 (-33%) |
| 컨텍스트 | 128K | 1M |
| MMLU Pro | 63.1% | 77.6% (+14.5p) |
| MATH | 70.2% | 90.9% (+20.7p) |
| TTFT | ~0.35s | ~0.34s |
| 7일 시뮬 예상시간 (순차) | ~2시간 51분 | ~1시간 43분 |
| 7일 시뮬 예상시간 (병렬5) | ~34분 | ~21분 |

### 최종 채택안
미정

---

## 5. 유동인구 교체 비율

### 현재 방식
- **파일**: `scripts/run_generative_simulation.py` (DAILY_FLOATING_AGENT_COUNT = 53)
- 매일 113명 중 53명 랜덤 샘플링 (요일 무관 고정)

### 개선안
- **요일별 차등**: 평일 40명 / 주말 70명
- **재방문율 30%**: 전날 방문한 53명 중 ~16명은 다음날도 포함, 나머지 새로운 에이전트
- 룰베이스로 `daily_floating_count` 요일 딕셔너리 + `previous_day_visitors` 추적

### 최종 채택안
미정

---

## 6. 상주인구 초기위치

### 현재 방식
- **파일**: `src/simulation_layer/persona/agent.py` (RESIDENT_LOCATIONS)
- 주거유형별 좌표 3개씩 고정 (아파트 3곳, 빌라 3곳, 주택 3곳)
- 같은 유형이면 같은 3좌표 중 하나에 몰림

### 개선안
- 기존 고정 좌표를 **클러스터 중심점**으로 사용
- 반경 100~200m 내 랜덤 오프셋 적용
```python
offset_m = random.uniform(0, 150)
angle = random.uniform(0, 2 * math.pi)
dlat = (offset_m * math.cos(angle)) / 111320
dlng = (offset_m * math.sin(angle)) / (111320 * math.cos(math.radians(base_lat)))
```
- RESIDENT_LOCATIONS 구조 그대로 유지, 코드 변경 최소

### 최종 채택안
미정

---

## 7. 시뮬레이션 속도

### 현재 방식
- 에이전트 순차 호출 (호출당 ~1.5s)
- rate_limit_delay = 0.5s
- 7일 시뮬레이션 약 2시간 51분

### 개선안
- `asyncio` 기반 병렬 호출 (동시 5~10개)
- Gemini Flash로 전환 시 호출당 ~0.9s
- **예상 소요시간**: 병렬 5개 + Gemini = **약 21분**

### 최종 채택안
미정

---

## 8. 유동 매장 선택 최적화 (Step3)

### 현재 방식
- **파일**: `scripts/run_generative_simulation.py:394-395`, `src/simulation_layer/persona/cognitive_modules/action_algorithm.py:229`
- 유동 에이전트는 `nearby_stores = list(global_store.stores.values())` → 720개 전체 매장을 LLM에 전달
- 720개 × ~250자 = ~180K자 → GPT-4o-mini 128K 토큰 한계에 근접
- LLM은 긴 리스트에서 앞쪽 항목에 편향 (positional bias)
- `random.shuffle`로 순서를 섞어도 720개를 진정으로 비교하는 건 불가능
- **결과**: 모내 11회, 장터국밥 9회 같은 특정 매장 집중 현상

### 왜 반경 제한은 안 되나
- 유동은 네이버/카카오 맵 **검색**으로 찾아가는 사람
- 물리적 거리가 아닌 검색 랭킹 기반으로 매장을 발견함
- 반경 제한은 상주에게만 적합

### 개선안: 검색엔진 랭킹 + Exploit-Explore
1. Step2에서 업종 선택 → 해당 업종만 필터 (이미 구현됨)
2. 스코어링: 평점 × 리뷰수 × 거리 × 카테고리 매칭
3. **상위 10개(랭킹) + 랜덤 5개(탐색)** = 15개만 LLM에 전달
4. LLM이 페르소나 기반으로 최종 선택

```python
def search_rank(store, query_category, agent_location):
    score = 0
    score += store.agent_taste_score * 2     # 평점
    score += store.agent_rating_count * 0.5  # 리뷰 수
    score -= distance_km * 0.3               # 거리
    if query_category in store.category:
        score += 3                           # 카테고리 일치 보너스
    return score

ranked = sort_by_score(stores)
top10 = ranked[:10]                          # 랭킹 상위 (exploit)
random5 = random.sample(ranked[10:], 5)      # 나머지에서 랜덤 (explore)
candidates = top10 + random5
```

### 부익부 빈익빈 방지
- 랜덤 5개 슬롯 → 미방문 매장에 노출 기회 제공
- 에이전트 평점이 실시간 누적 → 랭킹이 매일 변동
- LLM이 페르소나 기반 선택 → 같은 15개를 봐도 다른 매장 선택

### 최종 채택안
미정

---

## 9. 매장/에이전트 좌표 불일치

### 현재 방식
- **파일**: `src/simulation_layer/persona/agent.py` (FLOATING_LOCATIONS, RESIDENT_LOCATIONS)
- 대시보드 지도에서 전체 좌표가 **남동쪽으로 밀려있음**
  - 상주 에이전트가 공원 위에 표시됨
  - 유동 에이전트가 망원동 바깥에 표시됨
  - 매장, 집, 역, 정류장 좌표가 전체적으로 우하향

### 원인 추정
- `FLOATING_LOCATIONS`, `RESIDENT_LOCATIONS`의 하드코딩 좌표가 실제 위치와 불일치
- OSM 네트워크 중심점 계산이 매장 좌표 기반인데, 매장 좌표 자체가 밀려있을 가능성
- 좌표계 변환 오류 또는 데이터 수집 시 오프셋 발생

### 개선안
- `FLOATING_LOCATIONS`, `RESIDENT_LOCATIONS` 좌표를 구글맵/카카오맵 실제 좌표와 1:1 대조 검증
- 매장 데이터(`split_by_store_id`)의 좌표도 샘플 검증
- 전체적으로 일정한 오프셋이면 보정값 한 번에 적용

### 최종 채택안
미정

---

## 10. 과거 경험 기반 의사결정 (Step2/Step3 memory_context)

### 현재 방식
- **Step2** (업종 선택): `recent_categories`만 전달 → 뭘 먹었는지만 알고, 좋았는지 나빴는지 모름
- **Step3** (매장 선택): `recent_stores`만 전달 → 매장별 전체 에이전트 평점은 있지만, 본인 경험은 없음
- LLM이 "이전에 여기 가서 별로였다" 같은 판단 불가능
- **결과**: 과거 경험과 무관하게 매번 새로운 판단 → 불만족했던 매장 재방문, 만족했던 매장 무시

### 개선안: memory_context 통합 + 코멘트 포함 (완료)

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
- `src/simulation_layer/persona/agent.py`: `VisitRecord`에 `comment` 필드, `get_memory_context`에 코멘트 표시
- `src/simulation_layer/persona/cognitive_modules/action_algorithm.py`: Step2/Step3에 `memory_context` 전달
- `src/ai_layer/prompts/step2_category.txt`: `memory_context` + 유도 문구
- `src/ai_layer/prompts/step3_store.txt`: `memory_context` + 유도 문구

### 최종 채택안
완료
