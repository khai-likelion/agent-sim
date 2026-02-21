# 망원동 상권 에이전트 시뮬레이션

소상공인을 위한 비즈니스 전략 효과 검증 시뮬레이션 시스템.
망원동 실제 상권 데이터(721개 매장)를 기반으로 160명의 AI 에이전트가 7일간 외식 의사결정을 수행하고,
전략 적용 전/후 방문 변화를 비교 분석합니다.

---

## 프로젝트 구조

```
├── .env                          # API 키 및 시뮬레이션 설정
├── requirements.txt
├── config/
│   └── settings.py               # Pydantic 통합 설정 (LLM / 경로 / 시뮬레이션)
├── data/
│   ├── raw/
│   │   └── split_by_store_id_ver5/    # 매장별 JSON 파일 (721개)
│   └── output/                   # 시뮬레이션 결과 (CSV, JSON)
├── scripts/
│   ├── run_before_after_sim.py   # ★ 메인 실행: 전략 전/후 비교 시뮬레이션
│   ├── run_generative_simulation.py  # 시뮬레이션 엔진 모듈
│   └── dashboard.py              # Streamlit 시각화 대시보드
├── src/
│   ├── data_layer/               # Layer 1: 매장 데이터 / 공간 인덱스 / 도로망
│   ├── ai_layer/                 # Layer 2: LLM 클라이언트 / Step1~5 프롬프트
│   ├── simulation_layer/         # Layer 3: 에이전트 / 의사결정 / 평가 / 바이럴
│   └── analysis_layer/           # Layer 4: 전/후 비교 분석 리포트
├── X_to_Sim.py                       # StrategyBridge: X-Report → 매장 JSON 반영
└── 돼지야_report.md              # 타겟 매장 X-Report (전략 문서)
```

---

## 설치

```bash
pip install -r requirements.txt
```

`.env` 파일을 프로젝트 루트에 생성:

```env
# === Simulation Settings ===
SIM_H3_RESOLUTION=10
SIM_K_RING=2

# === LLM Settings ===
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512

# === OpenAI ===
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4o-mini
LLM_API_KEY=your_api_key_here

# === Gemini (대안) ===
# LLM_PROVIDER=gemini
# LLM_MODEL_NAME=gemini-2.0-flash
# LLM_API_KEY=your_api_key_here
```

---

## 실행

### 전략 전/후 비교 시뮬레이션 (메인)

```bash
python scripts/run_before_after_sim.py \
  --agents 160 \
  --days 7 \
  --target-store 돼지야 \
  --report 돼지야_report.md
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--agents` | 160 | 에이전트 수 |
| `--days` | 7 | 시뮬레이션 기간 |
| `--target-store` | 돼지야 | 분석 대상 매장 |
| `--report` | 돼지야_report.md | X-Report 경로 |
| `--seed` | 42 | 랜덤 시드 |
| `--skip-bridge` | False | StrategyBridge 건너뜀 |
| `-y` | False | 확인 없이 바로 실행 |

실행 순서:
1. **시뮬레이션 1** — 전략 적용 전 (원본 매장 데이터)
2. **StrategyBridge** — X-Report 전략을 매장 JSON에 자동 반영
3. **시뮬레이션 2** — 전략 적용 후 (동일 에이전트·시드)
4. **비교 결과 출력** — 방문 수 변화, 평점 변화

결과는 `data/output/돼지야_before/`, `data/output/돼지야_after/` 에 저장됩니다.

---

### 시뮬레이션 단독 실행

```bash
python scripts/run_generative_simulation.py \
  --agents 160 \
  --days 7 \
  --target-store 돼지야
```

---

### 대시보드 실행

```bash
streamlit run scripts/dashboard.py
```

---

## 아키텍처

### 에이전트 구성 (160명)

| 유형 | 인원 | 설명 |
|---|---|---|
| 거주 에이전트 | 47명 | 매일 활성, 망원동 거주자 |
| 유동 에이전트 | 113명 | 평일 51명 / 주말 58명 일별 샘플링 |

4개 그룹 유형 × 5개 세대(Z1·Z2·Y·X·S) 조합의 자연어 페르소나로 구성.

### 의사결정 파이프라인 (슬롯당 최대 3회 방문)

```
[시간 슬롯: 7시 / 12시 / 18시 / 22시]

Step 1 (확률)   외식 여부 결정 (아침 40% / 점심 70% / 저녁 60% / 야식 20%)
   ↓
Step 2 (LLM)   음식 카테고리 선택
   ↓
Step 3 (LLM)   반경 내 후보 매장 중 1개 선택
   ↓
Step 4 (LLM)   별점·태그·코멘트 생성 (세대별 톤 적용)
   ↓
Step 5 (LLM)   다음 행동 결정 (카페 / 산책 / 귀가 / 추가 방문 등)
```

- 비동기 병렬 실행 (`asyncio.gather`) — 동시 LLM 호출 최대 20개
- OSMnx 실제 보행자 도로망 기반 이동

### StrategyBridge

X-Report(전략 문서 `.md`)를 읽어 LLM이 매장 JSON 필드를 자동으로 수정합니다.

- 특성 키워드(`맛` → `taste`, `가성비` → `price_value` 등) 자동 매핑
- 변화량 상한: 특성당 ±0.20
- 시간성 표현("개선되었", "업데이트") 금지 패턴 필터링

---

## 출력 파일

| 파일 | 설명 |
|---|---|
| `simulation_result.csv` | 전체 에이전트 행동 로그 |
| `visit_log.csv` | 방문 이벤트만 필터링 |
| `store_ratings.json` | 시뮬레이션 후 매장별 누적 평점 |
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
