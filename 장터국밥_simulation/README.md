# 장터국밥 시뮬레이션

## 개요

장터국밥의 개선 전략을 **전/후 비교 시뮬레이션**으로 검증하는 LLM 기반 에이전트 시스템입니다.

- **목적**: 장터국밥_X_Report.txt의 개선 제안사항이 실제 고객 방문에 미치는 영향을 측정
- **방법**: 100% LLM 기반 의사결정
  - OpenAI GPT-4o/GPT-4o-mini (권장)
  - Groq API (llama-3.3-70b, 무료)
  - Ollama (로컬 실행, 무료)
- **비교**: 개선 전략 적용 **전** vs **후**
- **실행**: 단일 실행 또는 7일 자동 실행 지원

## 주요 특징

### 1. 시간대별 시뮬레이션
4개 시간대로 고객 행동 분석:
- **아침 (06-11)**: 해장 수요
- **점심 (11-14)**: 직장인 식사
- **저녁 (17-21)**: 외식/모임
- **야식 (00-06)**: 새벽 해장

### 2. 다양한 에이전트 페르소나
- **주변 거주자** (6명): 모든 시간대 존재
  - 20대 남성/여성 1인가구
  - 40대 남성/여성 가족
  - 60대+ 남성/여성 시니어

- **외부 유동인구** (시간대별):
  - **점심**: 직장인 남성/여성 (time_sensitivity 높음)
  - **저녁**: 데이트 남성/여성, 친구 모임 남성/여성 (visual_importance 높음)
  - **야식**: 관광객 남성/여성, 혼술족 남성/여성 (social_influence 높음)

### 3. 18개 Traits 시스템
각 에이전트는 18개 성향 값(0.0-1.0)으로 정의:
- **가격/시간**: `price_sensitivity`, `waiting_tolerance`, `time_sensitivity`
- **비주얼/분위기**: `visual_importance`, `ambiance_importance`, `noise_tolerance`
- **브랜드/충성도**: `brand_loyalty`, `novelty_seeking`, `social_influence`
- **맛 선호**: `spicy_preference`, `sweet_preference`, `mild_preference`, `rich_preference`, `authentic_preference`
- **메뉴 탐험**: `new_menu_adventurous`, `cuisine_diversity`, `signature_focus`
- **기타**: `child_friendly`

### 4. 리뷰 & 별점 시스템
각 에이전트는 방문 후:
- **별점 (1-5)**: 만족도 평가
- **리뷰**: 구체적인 경험 서술 (1-2문장)
- **별점 근거**: 평가 이유 설명

### 5. 비편향 프롬프트 설계 ⭐
**핵심 원칙**: 에이전트를 "보통 사람"으로 시뮬레이션

- ❌ **잘못된 접근**: 장터국밥 특징을 명시적으로 강조하고 "고려하세요"라고 지시
- ✅ **올바른 접근**: 모든 매장을 평등하게 제시하고, 장터국밥의 개선사항은 메뉴/서비스로만 표현
- 전략 전/후 차이는 **실제로 제공되는 것**의 차이 (세트메뉴, 양념장 가이드 등)
- 프롬프트에서 특정 매장을 유도하지 않음

## 파일 구조

```
장터국밥_simulation/
├── jangter_gukbap_simulator.py    # 메인 시뮬레이터
├── run_7day_simulation.py         # 7일 자동 실행 스케줄러
├── config.py                       # API 설정 (OpenAI, Groq, HF, Ollama)
├── requirements.txt                # 필수 패키지 목록
├── README.md                       # 문서
├── data/
│   ├── stores.csv                  # 주변 매장 데이터
│   └── 장터국밥_X_Report.txt       # 개선 전략 리포트
├── results/
│   ├── simulation_results_{timestamp}.json  # 타임스탬프별 결과
│   ├── simulation_results_latest.json       # 최신 결과
│   └── 7day_summary.json                    # 7일 실행 요약
├── logs/
│   ├── simulation_{timestamp}.log           # 타임스탬프별 로그
│   └── latest_run.log                       # 최신 실행 로그
└── latest_run.log                  # 최신 실행 로그 (루트)
```

## 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. API 키 설정
`config.py`에서 사용할 API Provider를 선택하고 API 키를 입력:

```python
# OpenAI GPT API 사용 (권장)
OPENAI_API_KEY = "sk-..."  # OpenAI API 키
API_PROVIDER = "openai"

# 또는 Groq API 사용 (무료, 빠름)
GROQ_API_KEY = "gsk_..."
API_PROVIDER = "groq"

# 또는 Ollama 로컬 실행 (비용 없음, 느림)
API_PROVIDER = "ollama"
```

### 3-1. 단일 실행
```bash
python jangter_gukbap_simulator.py
```

### 3-2. 7일간 자동 실행
```bash
# 즉시 7일 연속 실행 (24시간 간격)
python run_7day_simulation.py --mode continuous

# 매일 오전 9시에 실행 (7일간)
python run_7day_simulation.py --mode scheduled --time "09:00"

# API 비용 추정
python run_7day_simulation.py --estimate-cost
```

## 개선 전략 (장터국밥_X_Report.txt 기반)

### 전략 1: 양념장 사용 가이드
- **내용**: 테이블/벽면에 "양념장으로 간 맞추면 진국 맛 완성" 안내
- **타겟**: 초회 방문자, 관광객
- **기대효과**: "맹물 맛" 불만 감소

### 전략 2: 세트 메뉴 구성
- **내용**: 순대국 + 공기밥 + 미니수육 세트 (12,000원)
- **타겟**: 가격가치 중시 세그먼트 (직장인, 가족)
- **기대효과**: 가성비 체감 개선

### 전략 3: 포장 강화
- **내용**: 포장 용기에 재가열 팁, 양념 비율 안내
- **타겟**: 1인가구, 직장인
- **기대효과**: 포장 재구매율 상승

## 시뮬레이션 결과 분석

결과는 `results/simulation_results.json`에 저장됩니다.

### 주요 지표
- **시간대별 방문율**: 각 시간대에서 장터국밥을 선택한 비율
- **전/후 비교**: 전략 적용 전과 후의 방문율 변화
- **세그먼트별 반응**: 어떤 페르소나가 개선에 더 긍정적으로 반응했는지
- **평균 별점**: 전략 적용 전/후 평균 만족도
- **리뷰 키워드 분석**: 고객들이 주로 언급하는 단어/표현

### 예시 출력
```
============================================================
[TIME] 시간대: 11-14 - 점심
============================================================
[AGENTS] 시뮬레이션 에이전트: 8명

[DATA] [전략 적용 전] 시뮬레이션 중...
  1/8 주변_거주자_20대_남성_1인가구 (남성) -> 장터국밥
  2/8 주변_거주자_20대_여성_1인가구 (여성) -> 망원동칼국수
  ...

[DATA] [전략 적용 후] 시뮬레이션 중...
  1/8 주변_거주자_20대_남성_1인가구 (남성) -> 장터국밥
  2/8 주변_거주자_20대_여성_1인가구 (여성) -> 장터국밥
  ...

[RESULT] 점심:
  전략 전: 4/8명 (50.0%)
  전략 후: 6/8명 (75.0%)
  변화: +2명 (+25.0%p)
```

## 기술 스택

- **LLM**:
  - OpenAI GPT-4o/GPT-4o-mini (권장)
  - Groq API (llama-3.3-70b-versatile, 무료)
  - Hugging Face Inference API
  - Ollama (로컬 실행)
- **언어**: Python 3.8+
- **HTTP 클라이언트**: httpx (동기)
- **데이터 처리**: pandas
- **스케줄링**: 내장 시간 관리 (run_7day_simulation.py)

## 주요 클래스

### CustomerAgent
18개 traits와 성별을 가진 고객 에이전트

### API 클라이언트
- **SimpleOpenAIClient**: OpenAI GPT API 클라이언트
- **SimpleGroqClient**: Groq API 클라이언트 (재시도 로직 포함)
- **SimpleHFClient**: Hugging Face Inference API 클라이언트
- **SimpleOllamaClient**: Ollama 로컬 API 클라이언트

### JangterGukbapSimulator
- `create_agents_by_timeslot()`: 시간대별 에이전트 생성
- `_build_decision_prompt()`: 비편향 LLM 프롬프트 생성
- `simulate_visit_decision()`: 방문 결정 시뮬레이션
- `run_simulation()`: 전체 시뮬레이션 실행

## 새로운 기능 (2024 업데이트)

### 1. 다중 API Provider 지원
- **OpenAI GPT**: 가장 정확하고 안정적 (유료)
- **Groq**: 빠르고 무료
- **Hugging Face**: 오픈소스 모델
- **Ollama**: 완전 로컬 실행

### 2. 7일 자동 실행
```bash
# 즉시 7일 연속 실행
python run_7day_simulation.py --mode continuous

# 매일 정해진 시간에 실행
python run_7day_simulation.py --mode scheduled --time "09:00"
```

### 3. API 비용 추정
```bash
python run_7day_simulation.py --estimate-cost
```

### 4. 실시간 로그
- 매 에이전트의 결정마다 실시간으로 로그 출력
- 방문 선택, 이유, 별점, 리뷰 모두 기록
- 타임스탬프별 로그 파일 자동 생성

### 5. 타임스탬프 결과 저장
- 덮어쓰기 방지: 매 실행마다 타임스탬프 포함 파일 생성
- 최신 결과: `simulation_results_latest.json` 자동 업데이트
- 7일 요약: `7day_summary.json` 생성

## 사용 예시

### OpenAI GPT로 7일 시뮬레이션 실행

1. **API 키 설정** (config.py):
```python
OPENAI_API_KEY = "sk-proj-..."  # OpenAI API 키
API_PROVIDER = "openai"
OPENAI_MODEL = "gpt-4o-mini"  # 또는 gpt-4o
```

2. **비용 추정**:
```bash
python run_7day_simulation.py --estimate-cost
# 출력: 1회 실행 약 $0.15, 7일 약 $1.05
```

3. **즉시 7일 실행**:
```bash
python run_7day_simulation.py --mode continuous
# 24시간마다 자동 실행 (총 7일)
```

4. **결과 확인**:
- `results/7day_summary.json`: 전체 요약
- `results/simulation_results_{날짜}.json`: 일별 상세 결과
- `logs/simulation_{날짜}.log`: 실행 로그

### 로그 출력 예시

```
[INFO] [1/8] 주변_거주자_20대_남성_1인가구 (남성) 처리 중...
[INFO]   ✓ 선택: 장터국밥
[INFO]     이유: 가성비가 좋고 진한 국물이 매력적이다. 세트메뉴로 더 합리적인 가격이다.
[INFO]     별점: 4/5 - 맛과 가격이 만족스러웠지만 대기 시간이 조금 길었다
[INFO]     리뷰: 순대국 세트가 푸짐하고 국물이 진해서 좋았다. 다음에도 올 것 같다.
```

## 한계점 및 개선 방향

### 현재 한계
1. **제한된 매장 정보**: 장터국밥 외 매장들은 이름/카테고리만 제공
2. **고정된 에이전트 수**: 시간대별 에이전트 수가 고정
3. **순차 API 호출**: 병렬 처리 미지원으로 시간 소요 (1회 약 15-20분)

### 개선 방향
1. **실제 리뷰 데이터 반영**: 네이버/카카오 리뷰로 매장별 특징 구체화
2. **동적 에이전트 생성**: 실제 유동인구 비율 반영
3. **비동기 처리**: asyncio로 API 호출 병렬화 (시간 단축)
4. **시각화**: matplotlib/seaborn으로 결과 그래프 생성

## 주의사항

### API 비용
- **OpenAI GPT-4o**: 1회 약 $0.30-0.50, 7일 약 $2-3
- **OpenAI GPT-4o-mini**: 1회 약 $0.10-0.15, 7일 약 $1 (권장)
- **Groq**: 무료 (Rate Limit: 분당 30회)
- **Ollama**: 완전 무료 (로컬 실행, CPU 느림)

### Rate Limit
- 각 API Provider마다 Rate Limit이 있습니다
- 5초 대기 시간이 기본 설정되어 있습니다
- Groq 무료 플랜: 분당 30회, 일일 14,400회

### 실행 시간
- 1회 실행: 약 15-20분 (API 호출 속도에 따라)
- 7일 연속 실행: 7 × 24시간 = 168시간
- 스케줄 실행 권장: 매일 새벽 시간대

## 라이센스

This project is for educational and research purposes.

## 문의

프로젝트 관련 문의는 이슈를 통해 남겨주세요.
