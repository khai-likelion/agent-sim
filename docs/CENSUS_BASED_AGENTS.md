# Census-Based Agent Generation

## 개요

실제 집계구역별 인구 통계 데이터(`area_summary.csv`)를 기반으로 3,500명의 에이전트를 생성하는 시스템입니다.

## 데이터 소스

- **파일**: `data/raw/area_summary.csv`
- **총 인구**: 35,589명 (72개 집계구역)
- **스케일**: 1/10.2 (3,500명 에이전트 생성)

## 반영된 인구 속성

### 1. 주거 유형 (Residence Type)
- 다세대 (52.5%)
- 아파트 (31.7%)
- 단독주택 (9.9%)
- 연립주택 (4.7%)
- 영업용 건물 내 주택 (1.2%)

### 2. 가구 유형 (Household Type)
- 1인가구 (44.4%)
- 2세대가구 (35.8%)
- 3세대가구 (3.2%)
- 1세대가구 (나머지)

### 3. 연령 및 성별
- CSV의 21개 연령대별 × 2개 성별 데이터를 모두 반영
- 성별: 남성 46.3%, 여성 53.7%
- 연령대:
  - 10대: 8.7%
  - 20대: 15.7%
  - 30대: 18.4%
  - 40대: 15.7%
  - 50대: 16.4%
  - 60대+: 25.0%

### 4. 추가 생성 속성
- 직업 (연령/성별 기반)
- 소득 수준 (직업/주거 유형 기반)
- 가치 선호도
- 상점 선호도
- 가격/트렌드/품질 민감도

## 사용 방법

### 1. 에이전트 생성

```python
from src.simulation_layer.persona.census_agent_generator import CensusBasedAgentGenerator

# 생성기 초기화
generator = CensusBasedAgentGenerator(target_agents=3500)

# 에이전트 생성
agents = generator.generate_agents(3500)

# 결과 저장
import pandas as pd
df = pd.DataFrame([a.to_dict() for a in agents])
df.to_csv("data/output/agents.csv", index=False, encoding='utf-8-sig')
```

### 2. 테스트 스크립트 실행

```bash
python scripts/test_census_agents.py
```

출력:
- 전체 통계 (성별, 연령, 주거 유형, 가구 유형, 소득 수준, 직업)
- 샘플 에이전트 5명
- CSV 파일 저장 (`data/output/census_agents_3500.csv`)

## 구현 세부사항

### CensusBasedAgentGenerator 클래스

**주요 메서드**:
- `__init__`: CSV 데이터 로드 및 인구 풀 구축
- `_build_population_pool`: 2197개의 (연령×성별×집계구역) 세그먼트 생성
- `generate_agents`: 비율에 맞춰 정확히 N명의 에이전트 생성

**할당 알고리즘**:
1. 각 세그먼트의 실제 인구수 × 스케일 팩터 = 목표 에이전트 수
2. 정수 부분을 먼저 할당
3. 나머지를 소수점 부분이 큰 순서로 분배 (반올림 오류 방지)

### AgentPersona 확장

새로운 필드 추가:
```python
@dataclass
class AgentPersona:
    # ... 기존 필드 ...

    # 새로운 인구 통계 속성
    residence_type: Optional[str] = None  # 주거 유형
    household_type: Optional[str] = None  # 가구 유형
    census_area_code: Optional[str] = None  # 집계구역 코드
```

## 검증 결과

✅ 정확히 3,500명 생성
✅ 모든 인구 통계 비율 일치
✅ 주거 유형 반영
✅ 1인가구/다세대가구 정확히 반영
✅ 연령대별 성별 분포 반영
✅ 72개 집계구역 모두 반영

## 기존 시스템과의 차이

### 기존 (AgentGenerator)
- `PopulationStatistics` (JSON 기반)
- 연령대/성별만 반영
- 주거 유형/가구 유형 없음
- 집계구역 정보 없음

### 신규 (CensusBasedAgentGenerator)
- `area_summary.csv` (실제 인구총조사 데이터)
- 연령/성별/주거/가구 유형 모두 반영
- 집계구역별 분포 반영
- 정확한 비율 보장

## 향후 개선 사항

1. 집계구역 위치 정보 연동 (공간 배치)
2. 시간대별 활동 패턴에 주거 유형 반영
3. 가구 유형에 따른 소비 패턴 차별화
4. 더 세밀한 연령대 구분 (5세 단위)

## 관련 파일

- `src/simulation_layer/persona/census_agent_generator.py` - 생성기 구현
- `src/simulation_layer/persona/agent_persona.py` - 데이터 모델
- `scripts/test_census_agents.py` - 테스트 스크립트
- `scripts/analyze_population_csv.py` - CSV 분석 도구
- `scripts/check_household_data.py` - 가구 데이터 검증
- `data/raw/area_summary.csv` - 원본 인구 데이터
- `data/output/census_agents_3500.csv` - 생성된 에이전트 데이터
