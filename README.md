# 망원동 상권 에이전트 시뮬레이션

> 소상공인을 위한 AI 기반 비즈니스 전략 검증 시스템

망원동 실제 상권 데이터(721개 매장)를 바탕으로 160명의 생성형 AI 에이전트가 보행자 도로망 위에서 외식 의사결정을 수행합니다.
전략 적용 전/후 방문 변화를 시뮬레이션으로 비교 분석하고, Streamlit 대시보드로 결과를 시각화합니다.

---

## 주요 기능

- **생성형 에이전트**: 세대·세그먼트별 자연어 페르소나를 가진 160명의 에이전트
- **5단계 LLM 의사결정**: 외출 여부 → 카테고리 → 매장 선택 → 평가 → 다음 행동
- **실제 도로망 이동**: OSMnx 기반 보행자 도로망 위에서 이동 (한강 위 이동 제거)
- **전략 자동 반영**: X-Report(전략 문서)를 읽어 LLM이 매장 데이터를 자동 수정 (StrategyBridge)
- **비교 분석 리포트**: 전략 전/후 방문 수·평점·태그 변화를 `sim_to_y.py`로 분석
- **실시간 대시보드**: Folium 지도 + 에이전트 24시간 경로 추적 애니메이션

---

## 프로젝트 구조

```
├── .env.example                      # 환경변수 설정 예시
├── requirements.txt
├── config/
│   └── settings.py                   # Pydantic 통합 설정 (LLM / 경로 / 시뮬레이션)
├── data/
│   ├── cafe_stores.txt               # 카페 매장 목록
│   └── raw/
│       ├── stores.csv                # 전체 매장 기본 정보
│       └── 전략 md 파일들/           # 타겟 매장 X-Report (전략 문서)
├── scripts/
│   ├── run_before_after_sim.py       # ★ 메인 실행: 전략 전/후 비교 시뮬레이션
│   ├── run_generative_simulation.py  # 시뮬레이션 엔진
│   └── dashboard.py                  # Streamlit 시각화 대시보드
├── sim_to_y.py                       # 시뮬레이션 결과 분석 파이프라인
├── X_to_Sim.py                       # StrategyBridge: X-Report → 매장 JSON 반영
└── src/
    ├── ai_layer/                     # LLM 클라이언트 / Step 1~5 프롬프트
    ├── data_layer/                   # 매장 데이터 로더 / OSMnx 도로망
    └── simulation_layer/
        └── persona/
            ├── agent.py              # 에이전트 상태·기억·방문 기록
            ├── persona_definitions.py
            ├── personas_160.json     # 160명 페르소나 정의
            └── cognitive_modules/
                └── action_algorithm.py  # 5단계 의사결정 실행
```

---

## 설치

```bash
pip install -r requirements.txt
```

`.env.example`을 복사하여 `.env` 파일을 생성하고 API 키를 입력합니다:

```bash
cp .env.example .env
```

`.env` 주요 설정:

```env
# LLM Provider (openai / gemini)
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4o-mini
LLM_API_KEY=your_api_key_here

# 시뮬레이션 설정
SIM_NUM_AGENTS=160
SIM_NUM_DAYS=7
```

---

## 실행

### 1. 전략 전/후 비교 시뮬레이션 (메인)

```bash
python scripts/run_before_after_sim.py \
  --agents 160 \
  --days 7 \
  --target-store 돼지야 \
  --report "data/raw/전략 md 파일들/돼지야_report.md"
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--agents` | 160 | 에이전트 수 |
| `--days` | 7 | 시뮬레이션 기간 |
| `--target-store` | - | 분석 대상 매장명 |
| `--report` | - | X-Report `.md` 경로 |
| `--seed` | 42 | 랜덤 시드 |
| `--skip-bridge` | False | StrategyBridge 건너뜀 |
| `-y` | False | 확인 없이 바로 실행 |

실행 순서:
1. **시뮬레이션 1** — 전략 적용 전 (원본 매장 데이터)
2. **StrategyBridge** — X-Report 전략을 매장 JSON에 자동 반영
3. **시뮬레이션 2** — 전략 적용 후 (동일 에이전트·시드)
4. **비교 결과 출력** — 방문 수·평점·태그 변화

결과는 `data/output/{매장명}_before/`, `data/output/{매장명}_after/` 에 저장됩니다.

---

### 2. 단독 시뮬레이션 실행

```bash
python scripts/run_generative_simulation.py \
  --agents 160 \
  --days 1
```

결과는 `data/output/generative_simulation_result.csv` 등에 저장됩니다.

---

### 3. 결과 분석 리포트

```bash
python sim_to_y.py
```

`data/output/` 의 시뮬레이션 결과를 읽어 마크다운 리포트와 차트를 생성합니다.

---

### 4. 대시보드 실행

```bash
streamlit run scripts/dashboard.py
```

`http://localhost:8501` 에서 확인합니다.
사이드바에서 에이전트를 선택하면 24시간 이동 경로 애니메이션을 확인할 수 있습니다.

---

## 아키텍처

### 에이전트 구성 (160명)

| 유형 | 인원 | 설명 |
|---|---|---|
| 거주 에이전트 | 47명 | 매일 활성, 망원동 거주자 |
| 유동 에이전트 | 113명 | 평일 51명 / 주말 58명 일별 샘플링 |

5개 세대(Z1·Z2·Y·X·S) × 4개 그룹 유형 조합의 자연어 페르소나로 구성됩니다.

### 의사결정 파이프라인 (시간 슬롯별)

```
[시간 슬롯: 7시 / 12시 / 18시 / 22시]

Step 1 (확률)  외식 여부 결정 (아침 40% / 점심 70% / 저녁 60% / 야식 20%)
   ↓
Step 2 (LLM)  음식 카테고리 선택
   ↓
Step 3 (LLM)  반경 내 후보 매장 중 1개 선택
   ↓
Step 4 (LLM)  별점 · 태그 · 코멘트 생성 (세대별 톤 적용)
   ↓
Step 5 (LLM)  다음 행동 결정 (카페 / 산책 / 귀가 / 추가 방문 등)
```

- 비동기 병렬 실행 (`asyncio.gather`) — 최대 20개 동시 LLM 호출
- OSMnx 실제 보행자 도로망 기반 이동 (망원동 반경 2km)

### StrategyBridge (`X_to_Sim.py`)

X-Report(`.md` 전략 문서)를 읽어 LLM이 매장 JSON 필드를 자동으로 수정합니다.

- 특성 키워드(`맛` → `taste`, `가성비` → `price_value` 등) 자동 매핑
- 변화량 상한: 특성당 ±0.20
- 시간성 표현("개선되었", "업데이트") 금지 패턴 필터링

---

## 출력 파일

| 파일 | 설명 |
|---|---|
| `simulation_result.csv` | 전체 에이전트 행동 로그 (방문·재실·귀가 포함) |
| `visit_log.csv` | 방문 이벤트만 필터링 |
| `store_ratings.json` | 매장별 누적 평점 및 태그 집계 |
| `agents_final.json` | 시뮬레이션 후 에이전트 최종 상태 |
| `before_after_summary.json` | 전/후 방문 수 비교 요약 |

---

## 비용 예상 (GPT-4o-mini 기준, 7일 before/after)

| 항목 | 값 |
|---|---|
| 총 LLM 호출 | ~10,670회 |
| 입력 토큰 | ~5,335,000 |
| 출력 토큰 | ~1,067,000 |
| 예상 비용 | ~$1.44 |
| 예상 소요 시간 | 약 30~60분 |
