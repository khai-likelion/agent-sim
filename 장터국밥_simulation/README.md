# 망원동 상권 Pure LLM 시뮬레이션 🍜

**완전한 LLM 의사결정 기반 Agent-Based Model**

## 📋 개요

- **방식**: Pure LLM - 하드코딩 제거, 페르소나만 주입
- **데이터**: stores.csv의 실제 망원동 식당 (756개, 샘플링 가능)
- **에이전트**: 8개 Persona 세그먼트 × 30명
- **의사결정**: GPT-4o-mini가 "사람처럼" 모든 것을 판단
- **시간대**: 아침/점심/저녁/야식 (4블록)
- **목표**: Before/After 전략 비교

## 🎯 핵심 특징

### ✅ Pure LLM 방식

**LLM이 모든 것을 자유롭게 판단:**
- 어느 식당 갈지 (`choice`)
- 실제 갈지 말지 (`will_visit`)
- 만족도 예측 (`expected_satisfaction`)
- 리뷰 쓸지 (`will_write_review`)
- 리뷰 내용 (`review_text`)

**NO 하드코딩:**
- ❌ "위생 기준 미달은 선택 불가" 같은 규칙 없음
- ❌ "거리 가까우면 +0.3점" 같은 가이드 없음
- ❌ 만족도 계산식 없음
- ✅ 오직 페르소나 + 상황만 제공 → LLM이 판단

**유일한 제약:**
- Top 10 nearby 식당만 제시 (성능/비용 고려)

---

## 🚀 실행 방법

### 1. 설치

```bash
pip install numpy scipy pandas openai
```

### 2. OpenAI API 키 설정

**방법 1: 환경변수**
```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY=sk-your-api-key-here
```

**방법 2: 실행 시 입력**
```bash
python mangwon_pure_llm.py
# → "OpenAI API Key 입력:" 프롬프트에 입력
```

### 3. 실행

```bash
cd 장터국밥_simulation
python mangwon_pure_llm.py
```

---

## 🧑‍🤝‍🧑 8개 Persona 세그먼트

### 거주민 (R1~R4)

| 세그먼트 | 특징 | 주요 시간대 |
|---------|------|-----------:|
| **R1: 1인 가구** | 혼밥 선호, 단골집 반복, 집 근처 | 저녁 (55%), 야식 (30%) |
| **R2: 2인 가구** | 분위기 중시, 주말 외식 | 저녁 (55%), 점심 (20%) |
| **R3: 4인 가족** | 위생 최우선, 편안함, 검증된 곳 | 저녁 (65%), 점심 (40%) |
| **R4: 출퇴근 거주민** | 트렌디, 리뷰 참고, 저녁/야식 | 저녁 (65%), 야식 (40%) |

### 유동인구 (F1~F4)

| 세그먼트 | 특징 | 주요 시간대 |
|---------|------|-----------:|
| **F1: 데이트 커플** | 비주얼 압도적, 분위기 최우선 | 저녁 (70%), 점심 (30%) |
| **F2: 친구 모임** | 공유 메뉴, 핫플, 사진 찍기 | 저녁 (65%), 야식 (50%) |
| **F3: 외부 출근자** | 점심 효율, 단골 반복, 가까운 곳 | 점심 (85%), 아침 (20%) |
| **F4: 솔로 탐방객** | 숨은 맛집, 새로운 곳, 혼밥 OK | 저녁 (60%), 점심 (35%) |

**상세 설명:** [시뮬레이션_작동원리.md](시뮬레이션_작동원리.md)

---

## ⏰ 시간대 구성

```
아침 (breakfast):   7-10시  | 출근 전, 브런치
점심 (lunch):       11-14시 | 점심시간 피크 (F3 85%)
저녁 (dinner):      17-21시 | 모든 세그먼트 활발
야식 (late_night):  21-24시 | R1, R4, F2 중심
```

---

## 🧠 Pure LLM 의사결정

### 1. LLM에게 제공하는 정보

```
# 당신의 정체성
당신은 1인 가구 거주민입니다.
혼자 사는 직장인 또는 학생. 혼밥 선호, 가까운 거리, 루틴 중시

## 개인적 성향 (0~1 척도)
- 비주얼/분위기 중요도: 0.45
- 혼밥 선호도: 0.87
- 새로운 것 추구: 0.23
...

## 현재 상태
- 시간: 저녁 (17시-21시)
- 배고픔: 0.68
- 시간 여유: 0.55

## 과거 방문 경험
- 장터국밥: 3회 방문, 만족도 0.78, 2일 전

# 근처 식당 목록 (Top 10, 가까운 순)
0. **장터국밥** (한식음식점)
   - 거리: 15m
   - 위생: 0.65, 비주얼: 0.50, 편안함: 0.70
   - 혼밥 적합도: 0.80, 자극성: 0.40
   - 리뷰: 120개, 평점: 0.72
...

# 질문
위 상황에서 당신이라면 어떻게 하시겠습니까?

**중요**:
- 제약이나 규칙은 없습니다.
- 당신이 실제로 할 법한 선택을 하세요.
- 가고 싶지 않으면 will_visit: false로 해도 됩니다.
```

### 2. LLM 응답 예시

```json
{
  "choice": 0,
  "reasoning": "혼자 먹기 편하고 가까워서 자주 가는 곳이라 오늘도 장터국밥",
  "will_visit": true,
  "expected_satisfaction": 0.78,
  "will_write_review": false,
  "review_text": ""
}
```

또는

```json
{
  "choice": 2,
  "reasoning": "오늘은 새로운 카페 가보고 싶어서",
  "will_visit": true,
  "expected_satisfaction": 0.65,
  "will_write_review": true,
  "review_text": "분위기 좋고 인테리어 예뻐요! 혼자 와도 부담 없어서 좋네요."
}
```

---

## 📊 Before/After 전략 비교

### 전략 적용

```python
strategy = {
    "target_restaurant_id": 16050121,  # 장터국밥
    "improve_hygiene": True,    # hygiene_level +0.15
    "new_menu": True,           # novelty_flag = 1.0
    "improve_visual": True      # visual_score +0.12
}
```

### 예상 효과

1. **직접 효과**
   - 위생↑ → R3 (가족) 선택↑
   - 비주얼↑ → F1 (커플) 선택↑
   - 새로움↑ → F4 (탐방객) 선택↑

2. **간접 효과 (리뷰 확산)**
   - 방문↑ → 리뷰↑ → 인지도↑ → 추가 방문↑

### 측정 지표

- 타겟 식당 방문 수 변화
- 타겟 식당 리뷰 수/평점 변화
- 세그먼트별 반응 차이
- 타임블록별 변화

---

## 💰 비용 예상

### 기본 설정 (30 agents × 5 days)

```
총 LLM 호출 ≈ 30 × 0.5(출현율) × 4블록 × 5일 = 300회

GPT-4o-mini 비용:
- Input: ~500 tokens/호출 → 150k tokens → $0.02
- Output: ~50 tokens/호출 → 15k tokens → $0.01
총: $0.03 (매우 저렴!)
```

### 비용 절감 팁

1. 적은 에이전트로 테스트 (10~30명)
2. 짧은 기간 (3~5일)
3. 식당 수 제한 (sample_size=20)
4. Temperature 낮추기 (재시도 감소)

---

## ⚙️ 커스터마이징

### 에이전트 수 조정

```python
# mangwon_pure_llm.py의 main() 함수에서
N_AGENTS = 10   # 빠른 테스트
N_AGENTS = 30   # 현실적 (기본값)
N_AGENTS = 100  # 통계적 안정성
```

### 시뮬레이션 기간

```python
N_DAYS = 3   # 프로토타입
N_DAYS = 5   # 기본값
N_DAYS = 10  # 장기 관찰
```

### 식당 샘플 수

```python
restaurants = load_restaurants_from_csv(csv_path, sample_size=20)   # 20개
restaurants = load_restaurants_from_csv(csv_path, sample_size=100)  # 100개
restaurants = load_restaurants_from_csv(csv_path, sample_size=None) # 전체 756개
```

### LLM 모델 변경

```python
# llm_pure_decision() 함수에서
model="gpt-4o-mini",   # 빠르고 저렴 (권장)
model="gpt-4o",        # 더 정확
model="gpt-3.5-turbo", # 가장 저렴
```

### Temperature 조정

```python
temperature=0.5,  # 더 일관적 (결정론적)
temperature=0.8,  # 균형 (기본값)
temperature=1.2,  # 더 다양함 (랜덤)
```

---

## 📖 작동 원리 상세 설명

**반드시 읽어보세요:** [시뮬레이션_작동원리.md](시뮬레이션_작동원리.md)

내용:
- 8개 Persona 구성 방법
- Beta 분포 파라미터 의미
- 시간대별 출현 확률
- Pure LLM 프롬프트 구조
- 메모리 & 리뷰 확산 메커니즘
- Before/After 효과 분석

---

## 📂 파일 구조

```
장터국밥_simulation/
├── mangwon_pure_llm.py          ← Pure LLM 시뮬레이션 (메인)
├── agent_profiles.json           ← 30개 에이전트 프로필
├── README.md                     ← 이 파일
├── README_PURE_LLM.md            ← Pure LLM 상세 가이드
└── 시뮬레이션_작동원리.md        ← 작동 원리 설명
```

---

## 🔧 트러블슈팅

### 한글 깨짐 (Windows)

콘솔 인코딩 문제. 결과는 정상 작동함.

### ModuleNotFoundError: openai

```bash
pip install openai pandas numpy scipy
```

### API 키 에러

환경변수 확인:
```bash
echo $OPENAI_API_KEY  # Linux/Mac
echo $env:OPENAI_API_KEY  # Windows PowerShell
```

---

## 🎓 활용 예시

### 1. A/B 테스트

```python
strategies = [
    {"improve_hygiene": True},
    {"improve_visual": True},
    {"new_menu": True},
    {"improve_hygiene": True, "improve_visual": True}
]

for s in strategies:
    metrics = run_pure_llm_simulation(..., strategy=s)
    # 어느 전략이 가장 효과적?
```

### 2. 세그먼트별 반응 분석

- F1 (커플)은 visual에 가장 민감?
- R3 (가족)은 hygiene 개선에 반응?
- F3 (출근자)은 전략 무관?

### 3. 리뷰 품질 분석

- 각 세그먼트의 리뷰 스타일 차이
- 만족도와 리뷰 감성의 관계
- 리뷰가 다른 에이전트에게 미치는 영향

---

## 🆚 Pure LLM의 장단점

### 장점 ✅
1. **현실성**: 사람처럼 예측 불가능하고 다양한 선택
2. **유연성**: 프롬프트만 수정하면 새로운 요소 추가 가능
3. **자연어 출력**: 리뷰가 실제처럼 자연스러움
4. **복잡성 처리**: 다차원 의사결정을 LLM이 통합적으로 판단

### 단점 ❌
1. **비용**: API 호출 비용 (하지만 매우 저렴)
2. **속도**: 규칙 기반보다 느림 (API 대기 시간)
3. **재현성**: 같은 입력에도 다른 출력 (temperature 낮추면 개선)
4. **해석성**: 왜 그렇게 선택했는지 블랙박스 (reasoning 필드로 일부 해결)

---

## 📚 참고

- OpenAI API: https://platform.openai.com/docs
- GPT-4o-mini 가격: https://openai.com/api/pricing/
- ABM 이론: https://en.wikipedia.org/wiki/Agent-based_model

---

## ✨ 주요 특징

✅ **실제 데이터**: stores.csv 756개 망원동 식당
✅ **Pure LLM**: 하드코딩 제거, 페르소나만 주입
✅ **8개 Persona**: 다양한 소비자 유형
✅ **4개 시간대**: 아침/점심/저녁/야식
✅ **Before/After**: 전략 효과 측정
✅ **자연어 리뷰**: LLM이 생성한 진짜 같은 리뷰

---

Made with ❤️ for 망원동 상권 분석 - Pure LLM Edition
