# LLM 통합 문서

> 작성일: 2024-02-04
> 목적: 에이전트 페르소나 생성 및 행동 결정에 LLM 연동

---

## 1. 변경된 파일 목록

### 1.1 신규 생성 파일

| 파일 | 설명 |
|------|------|
| `src/ai_layer/prompt_templates/persona_generation.txt` | 페르소나 생성 프롬프트 |
| `src/ai_layer/prompt_templates/decision.txt` | 행동 결정 프롬프트 |
| `docs/LLM_INTEGRATION.md` | 본 문서 |

### 1.2 수정된 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/ai_layer/llm_client.py` | Groq/OpenAI/Anthropic 클라이언트 실제 구현 |
| `src/simulation_layer/persona/census_agent_generator.py` | LLM 기반 페르소나 생성 |
| `src/simulation_layer/persona/cognitive_modules/decide.py` | LLM 기반 행동 결정 |
| `src/simulation_layer/engine.py` | time_slot 파라미터 추가 |
| `scripts/run_simulation.py` | `--use-llm` 옵션 추가 |
| `config/settings.py` | Groq 기본값, env_file 로딩 수정 |
| `.env` | Groq API 설정 추가 |
| `requirements.txt` | httpx 추가 |

---

## 2. 파일 흐름 (Architecture)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        scripts/run_simulation.py                     │
│                         (CLI Entry Point)                            │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     config/settings.py                               │
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │AreaSettings │  │SimSettings  │  │PathSettings │  │ LLMSettings │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                                      │
│  .env 파일에서 로딩:                                                  │
│  - LLM_PROVIDER=groq                                                 │
│  - LLM_MODEL_NAME=llama-3.3-70b-versatile                           │
│  - LLM_API_KEY=gsk_xxx                                               │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
┌─────────────────────────┐                 ┌─────────────────────────┐
│   Agent Generation      │                 │   Simulation Engine     │
│                         │                 │                         │
│ census_agent_generator  │                 │      engine.py          │
│         .py             │                 │                         │
└───────────┬─────────────┘                 └───────────┬─────────────┘
            │                                           │
            ▼                                           ▼
┌─────────────────────────┐                 ┌─────────────────────────┐
│   LLM Client Layer      │                 │   Cognitive Modules     │
│                         │                 │                         │
│    llm_client.py        │◄────────────────│      decide.py          │
│                         │                 │                         │
│  ┌─────────────────┐    │                 │  use_llm=True/False     │
│  │  GroqClient     │    │                 └─────────────────────────┘
│  │  OpenAIClient   │    │
│  │  AnthropicClient│    │
│  └─────────────────┘    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Prompt Templates      │
│                         │
│ prompt_templates/       │
│  ├─ persona_generation  │
│  │    .txt              │
│  └─ decision.txt        │
└─────────────────────────┘
```

---

## 3. LLM 호출 흐름

### 3.1 페르소나 생성 (census_agent_generator.py)

```
┌──────────────────────────────────────────────────────────────────┐
│                    페르소나 생성 흐름                              │
└──────────────────────────────────────────────────────────────────┘

1. Census CSV 로드
   └─ data/raw/area_summary.csv

2. 인구통계 세그먼트 생성
   └─ 나이 × 성별 × 거주유형 × 가구유형 × 소득수준

3. 캐시 키 생성
   └─ "20대_여성_아파트_1인가구_중"

4. 캐시 확인
   ├─ HIT  → 캐시된 페르소나 반환
   └─ MISS → LLM 호출

5. LLM 호출 (캐시 MISS 시)
   ├─ 프롬프트 빌드 (persona_generation.txt)
   ├─ Groq API 호출
   ├─ JSON 파싱
   ├─ 캐시 저장
   └─ Rate limit 대기 (2.5초)

6. AgentPersona 객체 생성
   └─ agents.json 저장
```

**호출 횟수 최적화:**
- 3,500명 에이전트 생성 시
- 유니크 세그먼트: ~300개
- **실제 LLM 호출: ~300회** (캐시로 중복 제거)

### 3.2 행동 결정 (decide.py)

```
┌──────────────────────────────────────────────────────────────────┐
│                    행동 결정 흐름                                  │
└──────────────────────────────────────────────────────────────────┘

매 타임스텝마다:

1. 입력 데이터 수집
   ├─ visible_stores: 주변 매장 리스트
   ├─ agent: 에이전트 페르소나
   ├─ report: 비즈니스 리포트 (optional)
   ├─ time_slot: 현재 시간대 ("11-14" 등)
   └─ memory_context: 과거 방문 기록 (TODO)

2. 모드 분기
   ├─ use_llm=True  → LLM 기반 결정
   └─ use_llm=False → 규칙 기반 결정

3. LLM 기반 결정 (use_llm=True)
   ├─ 프롬프트 빌드 (decision.txt)
   ├─ Groq API 호출
   ├─ JSON 파싱
   └─ Rate limit 대기 (2.5초)

4. 출력
   └─ {decision, decision_reason, visited_store, visited_category}
```

**호출 횟수:**
- 20명 × 6타임슬롯 × 7일 = **840회/시뮬레이션**

---

## 4. 프롬프트 템플릿

### 4.1 페르소나 생성 프롬프트

**파일:** `src/ai_layer/prompt_templates/persona_generation.txt`

**입력 변수:**
```
{age_group}      - 연령대 (10대, 20대, 30대, 40대, 50대, 60대+)
{gender}         - 성별 (남성, 여성)
{residence_type} - 거주유형 (아파트, 다세대, 단독주택, 연립주택, 영업용 건물 내 주택)
{household_type} - 가구유형 (1인가구, 1세대가구, 2세대가구, 3세대가구)
{income_level}   - 소득수준 (하, 중하, 중, 중상, 상)
```

**출력 형식 (JSON):**
```json
{
  "occupation": "직업명",
  "value_preference": "상권 이용 성향 설명 (2-3문장)",
  "store_preferences": ["카테고리1", "카테고리2", ...],
  "price_sensitivity": 0.0-1.0,
  "trend_sensitivity": 0.0-1.0,
  "quality_preference": 0.0-1.0
}
```

**예시 출력:**
```json
{
  "occupation": "프리랜서 그래픽 디자이너",
  "value_preference": "트렌디한 카페와 브런치를 선호하며, SNS에서 핫플레이스를 찾아다니는 것을 즐깁니다. 가성비보다는 분위기와 인테리어를 중시합니다.",
  "store_preferences": ["카페", "브런치", "디저트카페", "베이커리", "이탈리안"],
  "price_sensitivity": 0.4,
  "trend_sensitivity": 0.9,
  "quality_preference": 0.7
}
```

### 4.2 행동 결정 프롬프트

**파일:** `src/ai_layer/prompt_templates/decision.txt`

**입력 변수:**
```
{agent_name}       - 에이전트 이름
{age}              - 나이
{age_group}        - 연령대
{gender}           - 성별
{occupation}       - 직업
{income_level}     - 소득수준
{value_preference} - 소비성향
{store_preferences}- 선호 카테고리 (쉼표 구분)
{time_slot}        - 현재 시간대
{visible_stores}   - 주변 매장 목록 (포맷팅됨)
{report_info}      - 비즈니스 리포트 정보
{memory_context}   - 과거 방문 기록
```

**출력 형식 (JSON):**
```json
{
  "decision": "visit" | "ignore",
  "store_name": "매장명" | null,
  "reason": "결정 이유 (1문장)"
}
```

**예시 출력:**
```json
{
  "decision": "visit",
  "store_name": "스타벅스 망원점",
  "reason": "트렌디한 카페를 선호하기 때문에 스타벅스 망원점을 방문하기로 결정했습니다."
}
```

---

## 5. LLM 모델 정보

### 5.1 사용 모델

| 항목 | 값 |
|------|-----|
| Provider | **Groq** (무료) |
| Model | `llama-3.3-70b-versatile` |
| Temperature | 0.7 |
| Max Tokens | 1024 |

### 5.2 Groq API 정보

- **API Endpoint:** `https://api.groq.com/openai/v1/chat/completions`
- **Free Tier 제한:** 30 requests/minute
- **권장 딜레이:** 2.5초 (안전 마진)
- **API Key 발급:** https://console.groq.com

### 5.3 대체 모델 옵션

`.env` 파일에서 변경 가능:

```bash
# Groq (무료, 빠름) - 기본값
LLM_PROVIDER=groq
LLM_MODEL_NAME=llama-3.3-70b-versatile

# Groq 경량 모델 (더 빠름)
LLM_MODEL_NAME=llama-3.1-8b-instant

# OpenAI (유료)
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4o-mini

# Anthropic (유료)
LLM_PROVIDER=anthropic
LLM_MODEL_NAME=claude-3-haiku-20240307
```

---

## 6. 실행 방법

### 6.1 기본 실행 (규칙 기반)

```bash
python scripts/run_simulation.py
```

### 6.2 LLM 모드 실행

```bash
python scripts/run_simulation.py --use-llm
```

### 6.3 LLM 딜레이 조절

```bash
# 딜레이 2.0초로 설정 (더 빠름, rate limit 주의)
python scripts/run_simulation.py --use-llm --llm-delay 2.0

# 딜레이 3.0초로 설정 (더 안전)
python scripts/run_simulation.py --use-llm --llm-delay 3.0
```

### 6.4 CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--use-llm` | LLM 기반 행동 결정 활성화 | False |
| `--llm-delay` | LLM 호출 간 대기 시간 (초) | 2.5 |
| `--h3-legacy` | H3 그리드 모드 (deprecated) | False |

---

## 7. 예상 소요 시간

### 7.1 페르소나 생성

| 에이전트 수 | 유니크 세그먼트 | LLM 호출 | 예상 시간 |
|------------|----------------|----------|----------|
| 20명 | ~20개 | ~20회 | ~1분 |
| 100명 | ~80개 | ~80회 | ~4분 |
| 500명 | ~200개 | ~200회 | ~10분 |
| 3,500명 | ~300개 | ~300회 | ~15분 |

### 7.2 시뮬레이션 (행동 결정)

| 에이전트 | 기간 | LLM 호출 | 예상 시간 |
|---------|------|----------|----------|
| 20명 | 7일 | 840회 | ~35분 |
| 20명 | 1일 | 120회 | ~5분 |
| 5명 | 1일 | 30회 | ~1분 |

---

## 8. 코드 예시

### 8.1 LLM 클라이언트 직접 사용

```python
from src.ai_layer.llm_client import create_llm_client

client = create_llm_client()  # settings.py 기반으로 자동 선택

# 동기 호출
response = client.generate_sync("안녕하세요")

# 비동기 호출
import asyncio
response = asyncio.run(client.generate("안녕하세요"))
```

### 8.2 페르소나 생성기 사용

```python
from src.simulation_layer.persona.census_agent_generator import CensusBasedAgentGenerator

# LLM 모드
generator = CensusBasedAgentGenerator(
    target_agents=100,
    use_llm=True,
    rate_limit_delay=2.5
)
agents = generator.generate_agents()

# 규칙 기반 모드 (빠름)
generator = CensusBasedAgentGenerator(
    target_agents=100,
    use_llm=False
)
agents = generator.generate_agents()
```

### 8.3 행동 결정 모듈 사용

```python
from src.simulation_layer.persona.cognitive_modules.decide import DecideModule

# LLM 모드
decide = DecideModule(use_llm=True, rate_limit_delay=2.5)

# 규칙 기반 모드
decide = DecideModule(use_llm=False)

result = decide.process(
    visible_stores=[{"장소명": "스타벅스", "카테고리": "카페"}],
    agent=agent,
    report=None,
    time_slot="11-14",
    memory_context=""
)
```

---

## 9. 향후 개선 사항

### 9.1 Memory 연동 (TODO)

현재 `memory_context`가 빈 문자열로 전달됨. `EventMemory`와 연동 필요:

```python
# decide.py에서
memory_context = agent.event_memory.to_prompt_context(n=5)
```

### 9.2 Reflection 연동 (TODO)

페르소나 선호도를 동적으로 업데이트:

```python
# 시뮬레이션 중
if reflection.should_reflect(agent):
    preference_deltas = reflection.analyze_and_update(agent)
    agent.apply_preference_update(preference_deltas)
```

### 9.3 배치 처리 최적화

여러 에이전트의 결정을 한 번의 LLM 호출로 처리:

```python
# 현재: 에이전트당 1회 호출
# 개선: 여러 에이전트를 배치로 처리
batch_prompt = build_batch_prompt(agents[:5], visible_stores)
batch_response = client.generate_sync(batch_prompt)
```

---

## 10. 트러블슈팅

### 10.1 API Key 오류

```
ValueError: Groq API key not set. Get free key at https://console.groq.com
```

**해결:** `.env` 파일에 API 키 설정
```bash
LLM_API_KEY=gsk_your_actual_key_here
```

### 10.2 Rate Limit 오류

```
httpx.HTTPStatusError: 429 Too Many Requests
```

**해결:** `--llm-delay` 값 증가
```bash
python scripts/run_simulation.py --use-llm --llm-delay 3.0
```

### 10.3 모델 없음 오류

```
httpx.HTTPStatusError: 400 Bad Request
```

**해결:** 사용 가능한 모델 확인
```python
import httpx
response = httpx.get(
    'https://api.groq.com/openai/v1/models',
    headers={'Authorization': f'Bearer {api_key}'}
)
print(response.json())
```

### 10.4 JSON 파싱 오류

LLM이 잘못된 JSON 반환 시 자동으로 규칙 기반으로 fallback됨.
로그에서 확인 가능:
```
LLM error for 20대_여성_아파트_1인가구_중: No JSON found in response. Using fallback.
```
