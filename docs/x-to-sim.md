# X-to-Sim 전략 적용 가이드

> `X_to_Sim.py` — `StrategyBridge` 클래스
> X-Report 전략을 store JSON에 반영하여 시뮬레이션용 상태 생성

---

## 개요

X-Report(마케팅 전략 보고서)에서 전략을 추출하고, store JSON의 다음 4가지 필드에 반영한다:

| 필드 | 설명 |
|------|------|
| `feature_scores` | 매장 평가 수치 (맛/가성비/청결/서비스/회전율/분위기) |
| `rag_context` | 에이전트가 읽는 매장 설명 텍스트 |
| `top_keywords` | 매장 대표 키워드 |
| `critical_feedback` | 에이전트가 단점으로 인식하는 태그 |

---

## 처리 흐름

```
X-Report (.md)
    │
    ├─ extract_issues()       정규식으로 TOP N 이슈·키워드 추출
    ├─ extract_strategies()   전략 블록 파싱 → id/title/content/goal
    │
    └─ apply_strategies() ─── 4개 병렬 API 호출
            ├─ analyze_strategies_impact_batch()  feature delta 계산
            ├─ extract_action_mappings()           rag_context 매핑 추출
            ├─ update_critical_feedback()          해결된 단점 삭제
            └─ update_top_keywords()               부정 키워드 제거
            │
            └─ update_text_fields()  (순차)
                    ├─ 1단계: action_mappings 직접 치환
                    └─ 2단계: GPT Patch — 금지 표현 위반 문장만 수정
```

---

## LLM 프롬프트 5개

### 1. `analyze_strategies_impact_batch`
**목적:** 전략 목록 → feature별 delta 합계 계산

- 입력: 전략 리스트, 현재 feature 점수
- 출력: `{"taste": 0.05, "cleanliness": 0.10, ...}`
- 규칙: 전략당 최대 delta 0.10, 전체 feature당 최대 0.20

### 2. `extract_action_mappings`
**목적:** rag_context에서 부정 표현 → 현재 상태 팩트로 대체할 매핑 추출

- 입력: 원본 rag_context, 전략 리스트
- 출력: `[{problem_expression, solution_expression, issue_keyword, source_strategy}]`
- 핵심 규칙: 고객 경험 관점만 서술 (운영 행위 표현 금지)

### 3. `update_text_fields` (Patch 방식)
**목적:** comparison/rag_context에서 시계열 표현 위반 문장만 수정

- 입력: 매핑 적용 후 텍스트, 전략 목록
- 출력: `[{field, target_sentence, new_sentence}]`
- 금지 표현: 개선되었다, 향상됐다, 최근, 새롭게 등 시계열 변화 표현

### 4. `update_critical_feedback`
**목적:** 전략으로 해결된 단점을 `critical_feedback`에서 완전 삭제

- 입력: 현재 critical_feedback 목록, 전략 목록
- 출력: `{"items": ["남길_단점1", ...]}`
- 핵심 규칙: 삭제만 허용, 긍정/중립 팩트로 변환 금지
  - 해결된 팩트는 `rag_context`에 서술됨 (이 필드에는 치명적 단점만)

### 5. `update_top_keywords`
**목적:** 전략으로 해결된 부정 키워드 제거

- 입력: 현재 top_keywords, 전략 목록
- 출력: `{"keywords": ["유지_키워드1", ...]}`
- 중립·긍정 키워드(맛, 친절, 분위기 등)는 유지

---

## delta 정책

| 항목 | 값 |
|------|-----|
| 전략당 최대 delta | 0.10 |
| feature당 최대 delta (누적) | 0.20 |
| 수치 명시 없을 때 기본값 | 0.05 |

기대효과에 "10~20%p" 같은 수치가 있으면 평균값(0.15)을 사용한다.

---

## feature 매핑

X-Report 키워드 → feature 이름 자동 매핑:

| 키워드 | feature |
|--------|---------|
| 맛, 짠맛, 염도, 면발 | `taste` |
| 가성비, 가격, 객단가 | `price_value` |
| 청결, 위생 | `cleanliness` |
| 서비스, 친절, 응대 | `service` |
| 회전율, 웨이팅, 대기 | `turnover` |
| 분위기, 인테리어, 감성 | `atmosphere` |

---

## 금지 표현 검증 (`validate_no_time_expressions`)

적용 후 텍스트에서 아래 패턴이 감지되면 경고 출력:

**시계열 변화 표현:**
`개선되었`, `향상됐`, `도입했`, `이전에`, `과거에`, `더 나아졌`, `좋아졌`, `최근`, `새롭게` 등

**메타 설명 (괄호 안):**
`(불만)`, `(개선)`, `(해소)`, `(보완)`, `(배려)`, `(대응)`, `(위해)` 등

---

## 실행 방법

```bash
# __main__ 직접 실행 (돼지야 기본 설정)
python X_to_Sim.py

# 코드에서 호출
from X_to_Sim import apply_x_report_strategy

result = apply_x_report_strategy(
    store_json_path="data/raw/split_by_store_id_ver5/돼지야.json",
    x_report_path="data/raw/전략 md 파일들/돼지야_report.md",
    selected_strategy_ids=["S1_A", "S2_A", "S3_A"],
    api_key=os.getenv("LLM_API_KEY"),
    output_path="data/raw/돼지야_전략적용.json"
)
```

## 출력 파일

원본 store JSON 구조를 유지하면서 아래 필드만 업데이트:

```json
{
  "review_metrics": {
    "feature_scores": { "taste": {"score": 1.0}, ... },
    "overall_sentiment": { "comparison": "업데이트된 텍스트" }
  },
  "rag_context": "업데이트된 매장 설명",
  "critical_feedback": ["남은 단점들"],
  "top_keywords": ["남은 키워드들"]
}
```

---

## LLM 설정

`.env` 파일 기준으로 provider 자동 감지:

| 환경변수 | 설명 |
|---------|------|
| `LLM_PROVIDER` | `gemini` 또는 `openai` |
| `LLM_MODEL_NAME` | 기본값 `gemini-2.0-flash` |
| `LLM_API_KEY` | API 인증 키 |

Gemini 사용 시 OpenAI-compatible 엔드포인트(`v1beta/openai/`)로 자동 연결.

---

*`StrategyBridge` (`X_to_Sim.py`) — ABM Simulation Strategy Bridge*
