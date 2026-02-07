# Pure LLM 시뮬레이션 🧠

**완전한 LLM 의사결정 - 하드코딩 제거, 페르소나만 주입**

## 🎯 핵심 차이점

### 기존 버전 (mangwon_sim.py, mangwon_abm_llm.py)
- ❌ 하드코딩된 규칙: "위생 기준 미달은 선택 불가", "최대 이동거리 제한"
- ❌ 의사결정 가이드: "거리 가까우면 +0.3점", "리뷰 많으면 우선"
- ❌ 만족도 계산식: 수식으로 만족도 계산
- ❌ 리뷰 생성 규칙: "만족도 0.6 이상이면 작성"

### Pure LLM 버전 (mangwon_pure_llm.py) ✅
- ✅ **제약 없음**: LLM이 자유롭게 판단
- ✅ **페르소나만 주입**: 성향, 상태, 과거 경험만 제공
- ✅ **완전한 자율성**: 갈지 말지, 어디 갈지, 리뷰 쓸지, 뭐라 쓸지 모두 LLM 결정
- ✅ **Top 10 제한만**: 성능/비용을 위해 가까운 10곳만 제시 (유일한 제약)

---

## 🚀 실행 방법

### 1. OpenAI API 키 설정

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY=sk-your-api-key-here
```

### 2. 실행

```bash
cd 장터국밥_simulation
python mangwon_pure_llm.py
```

### 3. 비용 고려

기본 설정:
- **10 agents** × **3 days** × 4 timeblocks = 최대 120회 LLM 호출
- 예상 비용: **$0.01~0.02** (매우 저렴)

에이전트/일수 조정:
```python
N_AGENTS = 30  # 더 많은 에이전트
N_DAYS = 5     # 더 긴 기간
```

---

## 🧠 LLM에게 제공하는 정보

### 1. 페르소나
```
# 당신의 정체성
당신은 1인 가구 거주민입니다.
혼자 사는 직장인 또는 학생. 혼밥 선호, 가까운 거리, 루틴 중시

## 개인적 성향 (0~1 척도)
- 비주얼/분위기 중요도: 0.45
- 혼밥 선호도: 0.87
- 새로운 것 추구: 0.23
...
```

### 2. 현재 상태
```
## 현재 상태
- 시간: 저녁 (17시-21시)
- 배고픔: 0.68
- 시간 여유: 0.55
- 현재 위치: (126.9045, 37.5563)
```

### 3. 과거 경험
```
## 과거 방문 경험
- 장터국밥: 3회 방문, 만족도 0.78, 2일 전
- 카페모던: 1회 방문, 만족도 0.65, 5일 전
```

### 4. 근처 식당 (Top 10)
```
# 근처 식당 목록 (가까운 순)

0. **장터국밥** (한식음식점)
   - 거리: 15m
   - 위생: 0.65, 비주얼: 0.50, 편안함: 0.70
   - 혼밥 적합도: 0.80, 자극성: 0.40
   - 리뷰: 120개, 평점: 0.72

1. **모던카페** (커피-음료)
   - 거리: 32m
   ...
```

### 5. 질문 (개방형)
```
# 질문
위 상황에서 당신이라면 어떻게 하시겠습니까?

식당을 선택하고, 방문 후의 느낌과 행동을 상상해서 답변해주세요.

**중요**:
- 당신의 성향과 현재 상태를 고려하여 자유롭게 판단하세요.
- 제약이나 규칙은 없습니다. 당신이 실제로 할 법한 선택을 하세요.
- 가고 싶지 않으면 will_visit: false로 해도 됩니다.
```

---

## 📤 LLM 응답 형식 (JSON)

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
  "choice": 1,
  "reasoning": "오늘은 새로운 카페 가보고 싶어서",
  "will_visit": true,
  "expected_satisfaction": 0.65,
  "will_write_review": true,
  "review_text": "분위기 좋고 인테리어 예뻐요! 혼자 와도 부담 없어서 좋네요."
}
```

또는

```json
{
  "choice": 0,
  "reasoning": "너무 피곤해서 오늘은 집에서 먹을래",
  "will_visit": false,
  "expected_satisfaction": 0,
  "will_write_review": false,
  "review_text": ""
}
```

---

## 🎭 8개 페르소나 세그먼트

### 거주민 (R1~R4)

| 코드 | 이름 | 특징 |
|------|------|------|
| **R1** | 1인 가구 | 혼밥, 단골, 루틴, 가까운 거리 |
| **R2** | 2인 가구 | 분위기, 주말 외식, 편안함 |
| **R3** | 4인 가족 | 위생, 검증됨, 편안함, 안전 |
| **R4** | 출퇴근 | 트렌디, 리뷰, 야식, 새로움 |

### 유동인구 (F1~F4)

| 코드 | 이름 | 특징 |
|------|------|------|
| **F1** | 데이트 커플 | 비주얼, 분위기, 사진, SNS |
| **F2** | 친구 모임 | 핫플, 공유 메뉴, 분위기 |
| **F3** | 외부 출근자 | 점심 효율, 가까움, 단골 |
| **F4** | 솔로 탐방객 | 숨은 맛집, 새로움, 혼밥 OK |

**상세:** [시뮬레이션_작동원리.md](시뮬레이션_작동원리.md)

---

## 📊 결과 분석

### 1. 방문 패턴
```
[방문]
  Before: 45 (의사결정 대비 75.0%)
  After:  52 (의사결정 대비 78.8%)
```
- LLM이 자율적으로 "안 가겠다" 선택 가능
- will_visit: false 비율로 에이전트 피로도/선호도 파악

### 2. 타겟 식당 효과
```
[타겟 레스토랑: 장터국밥]
  방문: 8 → 15 (+7)
```
- 위생/비주얼/신메뉴 개선 효과
- LLM이 이를 인지하고 자연스럽게 선택 증가

### 3. 생성된 리뷰
```
[생성된 리뷰 샘플]
  - 장터국밥 (R1_OnePerson)
    "혼자 먹기에 부담 없고 깔끔해서 자주 오게 되네요. 국밥 양도 푸짐하고 맛있어요!"

  - 모던카페 (F1_DateCouple)
    "인테리어가 정말 예쁘고 사진 찍기 좋아요. 데이트하기 딱!"
```
- 페르소나별로 다른 스타일의 리뷰
- 실제 사람이 쓴 것처럼 자연스러움

### 4. 의사결정 로그
```
[의사결정 로그 샘플]
  Day 0, dinner: R1_OnePerson → 장터국밥
    이유: 혼자 먹기 편하고 가까워서 자주 가는 곳이라 오늘도 장터국밥

  Day 1, lunch: F3_Commuter → 김밥천국
    이유: 점심시간 짧아서 가까운 곳으로 빠르게
```

---

## 🔬 실험 아이디어

### 1. 다양한 전략 비교
```python
strategies = [
    {"improve_hygiene": True},  # 위생만 개선
    {"improve_visual": True},   # 비주얼만 개선
    {"new_menu": True},         # 신메뉴만 출시
    {"improve_hygiene": True, "improve_visual": True, "new_menu": True}  # 전부
]

for strategy in strategies:
    metrics = run_pure_llm_simulation(..., strategy=strategy)
    # 어느 전략이 가장 효과적?
```

### 2. 세그먼트별 반응 분석
- F1 (데이트 커플): visual에 가장 민감?
- R3 (4인 가족): hygiene 개선에 큰 반응?
- F4 (솔로 탐방객): novelty에 가장 반응?

### 3. 리뷰 품질 분석
- 각 세그먼트의 리뷰 스타일 차이
- 만족도와 리뷰 길이/감성의 관계
- 리뷰가 다른 에이전트에게 미치는 영향

---

## ⚙️ 커스터마이징

### 1. 에이전트/기간 조정
```python
# main() 함수에서
N_AGENTS = 30  # 에이전트 수
N_DAYS = 5     # 시뮬레이션 기간
```

### 2. 식당 수 조정
```python
restaurants = load_restaurants_from_csv(str(csv_path), sample_size=50)  # 50개
# sample_size=None이면 전체 756개
```

### 3. Top N 조정
```python
# build_persona_prompt() 함수에서
nearby_restaurants = [r for r, _ in rest_with_dist[:20]]  # Top 20으로 확장
```
더 많은 선택지 → 더 현실적이지만 비용 증가

### 4. Temperature 조정
```python
# llm_pure_decision() 함수에서
temperature=0.5,  # 더 일관적 (결정론적)
temperature=1.0,  # 더 다양함 (랜덤)
```

### 5. LLM 모델 변경
```python
model="gpt-4o",        # 더 정확, 비쌈
model="gpt-4o-mini",   # 빠르고 저렴 (기본)
model="gpt-3.5-turbo", # 가장 저렴
```

---

## 💡 왜 Pure LLM인가?

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

### 언제 사용?
- ✅ 프로토타입 빠르게 테스트
- ✅ 자연스러운 리뷰 텍스트 필요
- ✅ 복잡한 의사결정 로직 (코딩 어려움)
- ✅ 소규모~중규모 (100~500 agents)

### 언제 규칙 기반?
- ✅ 대규모 (1000+ agents)
- ✅ 빠른 반복 실험
- ✅ 비용 제로
- ✅ 완벽한 재현성

---

## 🔍 핵심 코드

### Pure Persona Prompt
```python
decision_prompt = f"""
{persona}  # 정체성 + 성향 + 상태 + 경험
{rest_list}  # Top 10 식당 정보

# 질문
위 상황에서 당신이라면 어떻게 하시겠습니까?

**중요**:
- 제약이나 규칙은 없습니다.
- 당신이 실제로 할 법한 선택을 하세요.
- 가고 싶지 않으면 will_visit: false로 해도 됩니다.
"""
```

### LLM 호출
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "당신은 주어진 페르소나로 행동하는 시뮬레이션 에이전트입니다."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.8,
    response_format={"type": "json_object"}
)

result = json.loads(response.choices[0].message.content)
```

---

## 📁 관련 파일

| 파일 | 설명 |
|------|------|
| **mangwon_pure_llm.py** | 이 파일 - Pure LLM 시뮬레이션 |
| **mangwon_sim.py** | 공유 컴포넌트 (Agent, Restaurant, 데이터 로딩) |
| **agent_profiles.json** | 30개 에이전트 프로필 |
| **README.md** | 전체 프로젝트 가이드 |
| **시뮬레이션_작동원리.md** | 작동 원리 상세 설명 |

---

## 🎯 다음 단계

1. **API 키 설정** 후 실행해보기
2. **리뷰 샘플** 확인 - 페르소나별 스타일 차이 관찰
3. **전략 비교** - 어떤 개선이 가장 효과적?
4. **에이전트/기간 확장** - 30 agents × 5 days로 스케일업
5. **세그먼트별 분석** - 각 세그먼트의 선호 패턴 파악

---

Made with ❤️ for 망원동 상권 분석 - Pure LLM Edition
