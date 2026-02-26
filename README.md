# 망원동 상권 에이전트 시뮬레이션

> 소상공인을 위한 AI 기반 비즈니스 전략 검증 시스템

망원동 실제 상권 데이터(721개 매장)를 바탕으로 160명의 생성형 AI 에이전트가 보행자 도로망 위에서 외식 의사결정을 시뮬레이션합니다.
전략 적용 전/후 방문 변화를 비교 분석하고, Streamlit 대시보드로 결과를 시각화합니다.

---

## 주요 기능

- **5단계 LLM 의사결정**: Step 1(외식 여부) → Step 2(업종) → Step 3(매장) → Step 4(평가) → Step 5(다음 행동)
- **다중 방문 루프**: Step 5에서 카페 등 매장 방문 행동 선택 시 Step 3→4→5 자동 반복 (최대 3회)
- **생성형 페르소나**: 세대(Z1·Z2·Y·X·S) × 세그먼트별 자연어 페르소나 160명
- **memory_context 기반 판단**: 당일 식사 기록 + 과거 방문 평점·코멘트를 LLM에 전달
- **세대별 리뷰 차별화**: Z세대 구어체부터 시니어 진중체까지 tone_instruction으로 분기
- **StrategyBridge**: X-Report 전략 문서를 읽어 매장 JSON(평점·키워드·설명 등) 자동 수정
- **Softmax 계층화 샘플링**: 인기 매장 집중 + 롱테일 진입 기회 확보 (T=0.5, 상/중/하 20개 샘플링)
- **실제 도로망 이동**: OSMnx 기반 보행자 도로망 (한강 위 노드 제거)
- **비교 분석 리포트**: 방문 수·평점·세대·재방문율 등 11개 지표 + LLM 종합 요약
- **asyncio 병렬 처리**: Semaphore 동시 호출 제한으로 시뮬레이션 속도 대폭 단축

---

## 프로젝트 구조

```
├── .env.example                      # 환경변수 설정 예시
├── requirements.txt
├── config/
│   └── settings.py                   # Pydantic 통합 설정 (LLM / 경로 / 시뮬레이션)
├── data/
│   └── cafe_stores.txt               # 카페 매장 목록
├── docs/                             # 설계 문서 (에이전트 흐름, 고도화 내역 등)
├── scripts/
│   ├── run_before_after_sim.py       # ★ 메인 실행: 전략 전/후 비교 시뮬레이션
│   ├── run_generative_simulation.py  # 시뮬레이션 엔진
│   └── dashboard.py                  # Streamlit 시각화 대시보드
├── sim_to_y.py                       # 시뮬레이션 결과 비교 분석 리포트
├── X_to_Sim.py                       # StrategyBridge: X-Report → 매장 JSON 반영
└── src/
    ├── ai_layer/                     # LLM 클라이언트 / Step 1~5 프롬프트
    ├── data_layer/                   # 매장 데이터 로더 / OSMnx 도로망
    └── simulation_layer/
        └── persona/
            ├── agent.py              # 에이전트 상태·메모리·방문 기록
            ├── persona_definitions.py
            ├── personas_160.json     # 160명 페르소나 정의
            └── cognitive_modules/
                └── action_algorithm.py  # 5단계 의사결정 + 다중 방문 루프
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
# LLM Provider (gemini / openai)
LLM_PROVIDER=gemini
LLM_MODEL_NAME=gemini-2.5-flash-lite
LLM_API_KEY=your-api-key-here
```

> **기본값은 Gemini입니다.** GPT-4o-mini 대비 비용 33% 절감, 속도 25% 향상, 리뷰 품질 우위 (벤치마크: `docs/LLM-벤치마크-요약.md`)

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
4. **비교 결과 출력** — 방문 수·평점·세대 등 변화

결과는 `data/output/{매장명}_before/`, `data/output/{매장명}_after/` 에 저장됩니다.

---

### 2. 단독 시뮬레이션 실행

```bash
python scripts/run_generative_simulation.py \
  --agents 160 \
  --days 1
```

---

### 3. 비교 분석 리포트

```bash
python sim_to_y.py --target-store 돼지야
```

`data/output/` 의 before/after 결과를 읽어 마크다운 리포트와 차트를 생성합니다.

```bash
# 경로 직접 지정 시
python sim_to_y.py \
  --target-store 돼지야 \
  --before-dir data/output/돼지야_before \
  --after-dir  data/output/돼지야_after \
  --output-dir reports
```

---

### 4. 대시보드 실행

```bash
streamlit run scripts/dashboard.py
```

`http://localhost:8501` 에서 확인합니다.
사이드바에서 before/after 결과 폴더를 선택하고 에이전트를 클릭하면 24시간 이동 경로 애니메이션을 확인할 수 있습니다.

---

## 아키텍처

### 에이전트 구성 (160명)

| 유형 | 인원 | 설명 |
|---|---|---|
| 상주 에이전트 | 47명 | 매일 활성, 망원동 거주자, 주거유형별 반경 150m 오프셋 적용 |
| 유동 에이전트 | 113명 | 평일 51명 / 주말 58명, 전날 방문자 10% 재방문 우선 포함 |

5개 세대(Z1·Z2·Y·X·S) × 4개 방문 목적(생활베이스형·사적모임형·공적모임형·가족모임형) 조합의 자연어 페르소나로 구성됩니다.

### 의사결정 파이프라인 (타임슬롯별: 아침·점심·저녁·야식)

```
Step 1 (LLM)  외식 여부 결정
              → eat_in_mangwon=false 이면 종료
   ↓
Step 2 (LLM)  업종 선택 (시간대별 카테고리 풀: 아침/점심=식당+카페, 저녁=식당+주점+카페)
   ↓
Step 3 (LLM)  반경 내 후보 매장 중 1개 선택
              (Softmax T=0.5 계층화 샘플링, random.shuffle 위치 편향 방지)
   ↓
Step 4 (LLM)  평점(1~5) + 태그(맛/가성비/분위기/서비스) + 한줄 리뷰 생성
              (세대별 tone_instruction, 평가 관점 LLM 자율 선택)
   ↓
Step 5 (LLM)  다음 행동 결정 + 걷는 속도(km/h) 결정
              → 카페_가기 등 매장 방문 행동이면 Step 3→4→5 루프 (최대 3회)
              → 비매장 행동(배회하기, 한강공원_산책 등)이면 종료
```

- **Step 1은 LLM 순수 판단** — 확률 기반 폐기. memory_context(당일 식사 이력)를 보고 결정
- **asyncio 병렬 처리** — Semaphore로 동시 LLM 호출 제한 (run_before_after_sim: 20동시)
- **OSMnx 보행자 도로망** — 망원동 반경 2km, 한강 위 노드(lat < 37.550) 제거

### StrategyBridge (`X_to_Sim.py`)

X-Report(`.md` 전략 문서)를 읽어 LLM이 매장 JSON 4개 필드를 자동 수정합니다:

| 필드 | 내용 |
|------|------|
| `feature_scores` | 맛/가성비/청결/서비스/회전율/분위기 수치 조정 (전략당 최대 ±0.10, 누적 ±0.20) |
| `rag_context` | 에이전트가 읽는 매장 설명 텍스트 (시계열 변화 표현 금지) |
| `top_keywords` | 매장 대표 키워드 (해결된 부정 키워드 제거) |
| `critical_feedback` | 에이전트가 단점으로 인식하는 태그 (해결된 항목 삭제) |

### 비교 분석 리포트 (`sim_to_y.py`)

전략 전/후 `visit_log.csv`를 읽어 11개 지표를 계산하고 SVG 차트와 마크다운 보고서를 생성합니다:

| # | 지표 | 내용 |
|---|------|------|
| 1 | 기본 방문 지표 | 방문 수, 시장 점유율, LLM 해시태그 워드클라우드 |
| 2 | 평점 분포 | KDE, 만족도(4점↑) 비율, 태그별 선택 수 |
| 3 | 시간대별 트래픽 | 아침·점심·저녁·야식 꺾은선 비교 |
| 4 | 세대별 증감 | Z1/Z2/Y/X/S/혼합 9개 버킷 비율 변화 |
| 5 | 방문 목적 분석 | 생활베이스형/사적모임형/공적모임형/가족모임형 비중·만족도 |
| 6 | 재방문율 | 유지/신규/이탈 인원 (agent_id 집합 연산) |
| 7 | 경쟁 매장 비교 | TOP 5 레이더 차트 (방문수·평점·재방문율·리뷰수) |
| 9 | 에이전트 유형 | 상주 vs 유동 비율 변화 |
| 10 | 성별 구성 | 남성만·여성만·혼성 비중 변화 |
| 11 | 세대×목적 크로스탭 | 히트맵 (행 기준 비율 정규화) |
| 8 | LLM 종합 요약 | 1~7,9~11 + 안전성 진단 → 5섹션 분석 (GPT 우선, Gemini fallback) |

**안전성 진단** — 11개 지표를 자동 스캔해 역효과를 탐지하고 위험도 점수(0~100)를 산출합니다.

---

## 출력 파일

| 파일 | 설명 |
|------|------|
| `simulation_result.csv` | 전체 에이전트 행동 로그 (방문·이동·귀가 포함) |
| `visit_log.csv` | 방문 이벤트만 필터링 |
| `store_ratings.json` | 매장별 누적 평점·태그·리뷰 버퍼 집계 |
| `agents_final.json` | 시뮬레이션 후 에이전트 최종 상태 |
| `before_after_summary.json` | 전/후 방문 수 비교 요약 |
| `reports/Comparison_Report.md` | 비교 분석 최종 보고서 |
| `reports/figures/*.svg` | 지표별 차트 (워드클라우드만 `.png`) |

---

## 비용 & 속도 (Gemini 2.5 Flash Lite 기준, 160에이전트 90일 before/after)

| 항목 | 순차 실행 | 병렬 20동시 |
|------|:---------:|:----------:|
| 소요 시간 | ~27시간 | **~81분** |
| 총 LLM 호출 | ~65,000회 | ~65,000회 |
| 예상 비용 | — | **~8,000원** |

> asyncio 병렬화로 추가 인프라 없이 약 20배 속도 향상 가능합니다.
> (7일 기준: 순차 ~2시간 → 병렬 ~9분, ~840원)
