# 망원동 상권 에이전트 시뮬레이션

소상공인을 위한 비즈니스 리포트 효과 검증 시뮬레이션 시스템

## 프로젝트 구조

```
├── config/                 # 설정 (pydantic-settings)
├── data/
│   ├── raw/               # 입력 데이터 (stores.csv, 인구_DB.json)
│   └── output/            # 출력 데이터 (simulation_result.csv 등)
├── src/
│   ├── data_layer/        # Layer 1: Feature Store
│   ├── ai_layer/          # Layer 2: LLM + RAG + 프롬프트
│   ├── simulation_layer/  # Layer 3: Agent Simulation
│   └── app_layer/         # Layer 4: FastAPI
├── scripts/               # CLI 진입점
└── tests/                 # 테스트
```

## 설치

```bash
pip install -r requirements.txt
cp .env.example .env
```

## 실행

```bash
# 전체 파이프라인 (에이전트 생성 + 시뮬레이션 + 분석)
python scripts/run_simulation.py

# 분석만 실행
python scripts/run_analysis.py
```

## 아키텍처

4-Layer Architecture + Stanford Generative Agents 패턴 적용:

- **cognitive_modules/**: perceive → decide → react 사이클
- **memory_structures/**: event_memory, reflection
- **prompt_templates/**: LLM 프롬프트 템플릿 (.txt)

## 설정

모든 설정은 `.env` 파일로 관리:

```env
SIM_H3_RESOLUTION=10
SIM_AGENT_COUNT=20
SIM_SIMULATION_DAYS=7
```
