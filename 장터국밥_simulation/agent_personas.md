# 에이전트 페르소나 정의서

## 개요
장터국밥 시뮬레이션에 사용되는 모든 에이전트의 페르소나 정의입니다. 각 에이전트는 18개의 traits (0.0~1.0)로 구성되며, 시간대별로 다른 에이전트 조합이 활성화됩니다.

---

## 주변 거주자 (모든 시간대 존재)

### 1. 20대 남성 1인가구
- **ID**: 1
- **세그먼트**: 주변_거주자_20대_남성_1인가구
- **연령대**: 20대
- **성별**: 남성

**Traits**:
- `price_sensitivity`: 0.7 (가격에 민감)
- `waiting_tolerance`: 0.4 (웨이팅 낮음)
- `visual_importance`: 0.8 (비주얼 중요)
- `brand_loyalty`: 0.3 (브랜드 충성도 낮음)
- `novelty_seeking`: 0.8 (신규 시도 높음)
- `social_influence`: 0.7 (SNS 영향 많이 받음)
- `spicy_preference`: 0.7 (매운맛 선호)
- `sweet_preference`: 0.4
- `mild_preference`: 0.4
- `rich_preference`: 0.6 (진한맛)
- `authentic_preference`: 0.5
- `new_menu_adventurous`: 0.8 (신메뉴 모험적)
- `cuisine_diversity`: 0.8 (다양한 음식)
- `signature_focus`: 0.6
- `time_sensitivity`: 0.6
- `ambiance_importance`: 0.7 (분위기 중요)
- `noise_tolerance`: 0.8 (시끄러워도 OK)
- `child_friendly`: 0.1

**특징**: 트렌디한 맛집을 추구하는 젊은 1인가구. SNS 영향을 많이 받고, 비주얼과 분위기를 중시하지만 가격에는 민감.

---

### 2. 20대 여성 1인가구
- **ID**: 2
- **세그먼트**: 주변_거주자_20대_여성_1인가구
- **연령대**: 20대
- **성별**: 여성

**Traits**:
- `price_sensitivity`: 0.7
- `waiting_tolerance`: 0.5
- `visual_importance`: 0.9 (비주얼 매우 중요)
- `brand_loyalty`: 0.3
- `novelty_seeking`: 0.85 (신규 시도 매우 높음)
- `social_influence`: 0.9 (SNS 영향 매우 크게 받음)
- `spicy_preference`: 0.5
- `sweet_preference`: 0.7 (단맛 선호)
- `mild_preference`: 0.6 (담백함)
- `rich_preference`: 0.4
- `authentic_preference`: 0.5
- `new_menu_adventurous`: 0.9 (신메뉴 매우 모험적)
- `cuisine_diversity`: 0.9 (다양한 음식 선호)
- `signature_focus`: 0.7
- `time_sensitivity`: 0.6
- `ambiance_importance`: 0.9 (분위기 매우 중요)
- `noise_tolerance`: 0.6
- `child_friendly`: 0.1

**특징**: 인스타그래머블한 맛집을 선호하는 젊은 여성. 비주얼과 분위기가 최우선이며, 신메뉴와 트렌디한 장소에 민감.

---

### 3. 40대 남성 가족
- **ID**: 3
- **세그먼트**: 주변_거주자_40대_남성_가족
- **연령대**: 40대
- **성별**: 남성

**Traits**:
- `price_sensitivity`: 0.8 (가격 중요)
- `waiting_tolerance`: 0.6
- `visual_importance`: 0.4
- `brand_loyalty`: 0.7 (단골 선호)
- `novelty_seeking`: 0.3 (보수적)
- `social_influence`: 0.3
- `spicy_preference`: 0.6
- `sweet_preference`: 0.4
- `mild_preference`: 0.6
- `rich_preference`: 0.7 (진한맛)
- `authentic_preference`: 0.8 (정통맛 선호)
- `new_menu_adventurous`: 0.3 (익숙한 메뉴)
- `cuisine_diversity`: 0.4
- `signature_focus`: 0.8 (시그니처 메뉴 중시)
- `time_sensitivity`: 0.5
- `ambiance_importance`: 0.4
- `noise_tolerance`: 0.7
- `child_friendly`: 0.8 (아이 동반 친화)

**특징**: 가성비와 정통 맛을 중시하는 가장. 단골집을 선호하며, 아이 동반이 편한 곳을 찾음.

---

### 4. 40대 여성 가족
- **ID**: 4
- **세그먼트**: 주변_거주자_40대_여성_가족
- **연령대**: 40대
- **성별**: 여성

**Traits**:
- `price_sensitivity`: 0.8
- `waiting_tolerance`: 0.7
- `visual_importance`: 0.6
- `brand_loyalty`: 0.7
- `novelty_seeking`: 0.4
- `social_influence`: 0.5
- `spicy_preference`: 0.4
- `sweet_preference`: 0.5
- `mild_preference`: 0.7 (담백함 선호)
- `rich_preference`: 0.5
- `authentic_preference`: 0.7
- `new_menu_adventurous`: 0.4
- `cuisine_diversity`: 0.5
- `signature_focus`: 0.7
- `time_sensitivity`: 0.5
- `ambiance_importance`: 0.6
- `noise_tolerance`: 0.6
- `child_friendly`: 0.9 (아이 친화 매우 중요)

**특징**: 가족 건강과 청결을 중시하는 어머니. 담백한 맛과 아이가 먹기 좋은 곳을 선호.

---

### 5. 60대+ 남성 시니어
- **ID**: 5
- **세그먼트**: 주변_거주자_60대_남성_시니어
- **연령대**: 60대+
- **성별**: 남성

**Traits**:
- `price_sensitivity`: 0.7
- `waiting_tolerance`: 0.8 (여유로움)
- `visual_importance`: 0.3
- `brand_loyalty`: 0.9 (단골 매우 중시)
- `novelty_seeking`: 0.2 (보수적)
- `social_influence`: 0.1 (SNS 영향 낮음)
- `spicy_preference`: 0.6
- `sweet_preference`: 0.3
- `mild_preference`: 0.8 (담백함)
- `rich_preference`: 0.7 (진한맛)
- `authentic_preference`: 0.9 (정통맛 최우선)
- `new_menu_adventurous`: 0.2
- `cuisine_diversity`: 0.3
- `signature_focus`: 0.8
- `time_sensitivity`: 0.3 (시간 여유)
- `ambiance_importance`: 0.4
- `noise_tolerance`: 0.5
- `child_friendly`: 0.3

**특징**: 오랜 단골집을 선호하는 시니어. 정통적이고 진한 맛을 찾으며, 새로운 것보다 익숙한 것 선호.

---

### 6. 60대+ 여성 시니어
- **ID**: 6
- **세그먼트**: 주변_거주자_60대_여성_시니어
- **연령대**: 60대+
- **성별**: 여성

**Traits**:
- `price_sensitivity`: 0.8 (가격 중요)
- `waiting_tolerance`: 0.7
- `visual_importance`: 0.2
- `brand_loyalty`: 0.9
- `novelty_seeking`: 0.2
- `social_influence`: 0.1
- `spicy_preference`: 0.2 (맵지 않은 것)
- `sweet_preference`: 0.4
- `mild_preference`: 0.9 (담백함 최우선)
- `rich_preference`: 0.4
- `authentic_preference`: 0.9
- `new_menu_adventurous`: 0.1
- `cuisine_diversity`: 0.2
- `signature_focus`: 0.7
- `time_sensitivity`: 0.3
- `ambiance_importance`: 0.4
- `noise_tolerance`: 0.4 (조용한 곳 선호)
- `child_friendly`: 0.5

**특징**: 담백하고 자극적이지 않은 음식을 선호하는 시니어 여성. 단골집과 조용한 환경을 중시.

---

## 외부 유동인구

### 점심 시간대 (11-14)

#### 7. 직장인 남성
- **ID**: 7 (점심 시간대)
- **세그먼트**: 유동_직장인_남성_점심
- **연령대**: 30~40대
- **성별**: 남성

**Traits**:
- `price_sensitivity`: 0.6
- `waiting_tolerance`: 0.2 (웨이팅 매우 싫어함)
- `visual_importance`: 0.3
- `brand_loyalty`: 0.5
- `novelty_seeking`: 0.4
- `social_influence`: 0.3
- `spicy_preference`: 0.6
- `sweet_preference`: 0.3
- `mild_preference`: 0.5
- `rich_preference`: 0.7
- `authentic_preference`: 0.6
- `new_menu_adventurous`: 0.4
- `cuisine_diversity`: 0.5
- `signature_focus`: 0.6
- `time_sensitivity`: 0.9 (시간 매우 부족)
- `ambiance_importance`: 0.2
- `noise_tolerance`: 0.8
- `child_friendly`: 0.1

**특징**: 점심시간이 짧은 직장인. 빠른 서비스가 최우선이며, 웨이팅을 극도로 싫어함.

---

#### 8. 직장인 여성
- **ID**: 8 (점심 시간대)
- **세그먼트**: 유동_직장인_여성_점심
- **연령대**: 30~40대
- **성별**: 여성

**Traits**:
- `price_sensitivity`: 0.6
- `waiting_tolerance`: 0.3
- `visual_importance`: 0.6 (비주얼 중요)
- `brand_loyalty`: 0.5
- `novelty_seeking`: 0.5
- `social_influence`: 0.6
- `spicy_preference`: 0.4
- `sweet_preference`: 0.5
- `mild_preference`: 0.6
- `rich_preference`: 0.5
- `authentic_preference`: 0.5
- `new_menu_adventurous`: 0.5
- `cuisine_diversity`: 0.6
- `signature_focus`: 0.5
- `time_sensitivity`: 0.9 (시간 매우 부족)
- `ambiance_importance`: 0.5
- `noise_tolerance`: 0.7
- `child_friendly`: 0.1

**특징**: 점심 미팅도 고려하는 직장인 여성. 시간이 부족하지만 적당한 비주얼도 중시.

---

### 저녁 시간대 (17-21)

#### 9. 데이트 남성
- **ID**: 9 (저녁 시간대)
- **세그먼트**: 유동_데이트_남성_저녁
- **연령대**: 20~30대
- **성별**: 남성

**Traits**:
- `price_sensitivity`: 0.4 (가격 덜 민감)
- `waiting_tolerance`: 0.6
- `visual_importance`: 0.8 (데이트라 비주얼 중요)
- `brand_loyalty`: 0.3
- `novelty_seeking`: 0.7
- `social_influence`: 0.7
- `spicy_preference`: 0.6
- `sweet_preference`: 0.4
- `mild_preference`: 0.4
- `rich_preference`: 0.6
- `authentic_preference`: 0.5
- `new_menu_adventurous`: 0.7
- `cuisine_diversity`: 0.7
- `signature_focus`: 0.6
- `time_sensitivity`: 0.3 (시간 여유)
- `ambiance_importance`: 0.9 (분위기 최우선)
- `noise_tolerance`: 0.3 (조용한 곳 선호)
- `child_friendly`: 0.1

**특징**: 데이트 중인 남성. 분위기와 비주얼이 최우선이며, 상대방에게 좋은 인상 중요.

---

#### 10. 데이트 여성
- **ID**: 10 (저녁 시간대)
- **세그먼트**: 유동_데이트_여성_저녁
- **연령대**: 20~30대
- **성별**: 여성

**Traits**:
- `price_sensitivity`: 0.4
- `waiting_tolerance`: 0.7
- `visual_importance`: 0.95 (비주얼 극도로 중요)
- `brand_loyalty`: 0.3
- `novelty_seeking`: 0.8
- `social_influence`: 0.9
- `spicy_preference`: 0.5
- `sweet_preference`: 0.7
- `mild_preference`: 0.5
- `rich_preference`: 0.5
- `authentic_preference`: 0.5
- `new_menu_adventurous`: 0.8
- `cuisine_diversity`: 0.8
- `signature_focus`: 0.7
- `time_sensitivity`: 0.3
- `ambiance_importance`: 0.95 (분위기 극도로 중요)
- `noise_tolerance`: 0.2 (조용함 선호)
- `child_friendly`: 0.1

**특징**: 데이트 중인 여성. 인스타그램에 올릴 만한 비주얼과 로맨틱한 분위기가 핵심.

---

#### 11. 친구 모임 남성
- **ID**: 11 (저녁 시간대)
- **세그먼트**: 유동_친구_남성_저녁
- **연령대**: 30~40대
- **성별**: 남성

**Traits**:
- `price_sensitivity`: 0.5
- `waiting_tolerance`: 0.6
- `visual_importance`: 0.4
- `brand_loyalty`: 0.4
- `novelty_seeking`: 0.6
- `social_influence`: 0.6
- `spicy_preference`: 0.7 (매운맛)
- `sweet_preference`: 0.3
- `mild_preference`: 0.4
- `rich_preference`: 0.8 (진한맛 선호)
- `authentic_preference`: 0.6
- `new_menu_adventurous`: 0.6
- `cuisine_diversity`: 0.7
- `signature_focus`: 0.5
- `time_sensitivity`: 0.3
- `ambiance_importance`: 0.4
- `noise_tolerance`: 0.9 (시끌벅적해도 OK)
- `child_friendly`: 0.2

**특징**: 친구들과 함께 식사하는 남성. 진하고 자극적인 맛을 선호하며, 시끄러운 분위기도 괜찮음.

---

#### 12. 친구 모임 여성
- **ID**: 12 (저녁 시간대)
- **세그먼트**: 유동_친구_여성_저녁
- **연령대**: 30~40대
- **성별**: 여성

**Traits**:
- `price_sensitivity`: 0.5
- `waiting_tolerance`: 0.7
- `visual_importance`: 0.7
- `brand_loyalty`: 0.4
- `novelty_seeking`: 0.7
- `social_influence`: 0.8
- `spicy_preference`: 0.5
- `sweet_preference`: 0.5
- `mild_preference`: 0.5
- `rich_preference`: 0.6
- `authentic_preference`: 0.5
- `new_menu_adventurous`: 0.7
- `cuisine_diversity`: 0.8
- `signature_focus`: 0.5
- `time_sensitivity`: 0.3
- `ambiance_importance`: 0.6
- `noise_tolerance`: 0.8
- `child_friendly`: 0.2

**특징**: 친구들과 수다 떠는 여성. 다양한 음식을 시도하고 SNS 영향을 받음.

---

### 야식 시간대 (00-06)

#### 13. 관광객 남성
- **ID**: 13 (야식 시간대)
- **세그먼트**: 유동_관광객_남성_야식
- **연령대**: 20~40대
- **성별**: 남성

**Traits**:
- `price_sensitivity`: 0.5
- `waiting_tolerance`: 0.7
- `visual_importance`: 0.7
- `brand_loyalty`: 0.2 (처음 방문)
- `novelty_seeking`: 0.9 (새로운 경험 추구)
- `social_influence`: 0.9 (SNS/리뷰 매우 중시)
- `spicy_preference`: 0.6
- `sweet_preference`: 0.4
- `mild_preference`: 0.5
- `rich_preference`: 0.7
- `authentic_preference`: 0.8 (로컬 맛 중요)
- `new_menu_adventurous`: 0.8
- `cuisine_diversity`: 0.8
- `signature_focus`: 0.9 (시그니처 메뉴 최우선)
- `time_sensitivity`: 0.4
- `ambiance_importance`: 0.6
- `noise_tolerance`: 0.7
- `child_friendly`: 0.2

**특징**: 망원동을 찾은 관광객. 그 지역만의 특별한 맛과 시그니처 메뉴를 찾음. 리뷰 의존도 높음.

---

#### 14. 관광객 여성
- **ID**: 14 (야식 시간대)
- **세그먼트**: 유동_관광객_여성_야식
- **연령대**: 20~40대
- **성별**: 여성

**Traits**:
- `price_sensitivity`: 0.5
- `waiting_tolerance`: 0.8
- `visual_importance`: 0.9 (인스타그램 중요)
- `brand_loyalty`: 0.2
- `novelty_seeking`: 0.9
- `social_influence`: 0.95 (SNS 극도로 중시)
- `spicy_preference`: 0.5
- `sweet_preference`: 0.6
- `mild_preference`: 0.5
- `rich_preference`: 0.6
- `authentic_preference`: 0.8
- `new_menu_adventurous`: 0.8
- `cuisine_diversity`: 0.8
- `signature_focus`: 0.9
- `time_sensitivity`: 0.4
- `ambiance_importance`: 0.7
- `noise_tolerance`: 0.6
- `child_friendly`: 0.2

**특징**: 인스타그래머블한 로컬 맛집을 찾는 여성 관광객. SNS 리뷰와 비주얼 최우선.

---

#### 15. 혼술족 남성
- **ID**: 15 (야식 시간대)
- **세그먼트**: 유동_혼술_남성_야식
- **연령대**: 30~50대
- **성별**: 남성

**Traits**:
- `price_sensitivity`: 0.6
- `waiting_tolerance`: 0.6
- `visual_importance`: 0.3
- `brand_loyalty`: 0.7 (단골 선호)
- `novelty_seeking`: 0.4
- `social_influence`: 0.3
- `spicy_preference`: 0.6
- `sweet_preference`: 0.3
- `mild_preference`: 0.5
- `rich_preference`: 0.8 (진하고 깊은 맛)
- `authentic_preference`: 0.8
- `new_menu_adventurous`: 0.4
- `cuisine_diversity`: 0.5
- `signature_focus`: 0.7
- `time_sensitivity`: 0.3
- `ambiance_importance`: 0.5
- `noise_tolerance`: 0.6
- `child_friendly`: 0.1

**특징**: 새벽에 혼자 술과 국밥을 즐기는 남성. 정통 깊은 맛과 단골 분위기 선호.

---

#### 16. 혼술족 여성
- **ID**: 16 (야식 시간대)
- **세그먼트**: 유동_혼술_여성_야식
- **연령대**: 30~50대
- **성별**: 여성

**Traits**:
- `price_sensitivity`: 0.6
- `waiting_tolerance`: 0.7
- `visual_importance`: 0.5
- `brand_loyalty`: 0.7
- `novelty_seeking`: 0.5
- `social_influence`: 0.5
- `spicy_preference`: 0.4
- `sweet_preference`: 0.4
- `mild_preference`: 0.6
- `rich_preference`: 0.6
- `authentic_preference`: 0.7
- `new_menu_adventurous`: 0.4
- `cuisine_diversity`: 0.5
- `signature_focus`: 0.6
- `time_sensitivity`: 0.3
- `ambiance_importance`: 0.6
- `noise_tolerance`: 0.5
- `child_friendly`: 0.1

**특징**: 혼자만의 시간을 즐기는 여성. 편안한 분위기와 적당한 맛 선호.

---

## Traits 설명

### 가격/시간 관련
- **price_sensitivity** (0.0-1.0): 가격 민감도. 높을수록 가격을 중요하게 고려
- **waiting_tolerance** (0.0-1.0): 웨이팅 인내심. 낮을수록 줄서기 싫어함
- **time_sensitivity** (0.0-1.0): 시간 민감도. 높을수록 빠른 서비스 필요

### 비주얼/분위기 관련
- **visual_importance** (0.0-1.0): 음식 비주얼 중요도
- **ambiance_importance** (0.0-1.0): 매장 분위기 중요도
- **noise_tolerance** (0.0-1.0): 소음 인내도. 낮으면 조용한 곳 선호

### 브랜드/충성도 관련
- **brand_loyalty** (0.0-1.0): 브랜드 충성도. 높으면 단골집 선호
- **novelty_seeking** (0.0-1.0): 새로운 것 추구. 높으면 신규 매장 선호
- **social_influence** (0.0-1.0): SNS/리뷰 영향력. 높으면 리뷰 의존

### 맛 선호 관련
- **spicy_preference** (0.0-1.0): 매운맛 선호도
- **sweet_preference** (0.0-1.0): 단맛 선호도
- **mild_preference** (0.0-1.0): 담백함 선호도
- **rich_preference** (0.0-1.0): 진한맛 선호도
- **authentic_preference** (0.0-1.0): 정통맛 선호도

### 메뉴 탐험 관련
- **new_menu_adventurous** (0.0-1.0): 신메뉴 시도 성향. 높으면 모험적
- **cuisine_diversity** (0.0-1.0): 다양한 음식 선호도
- **signature_focus** (0.0-1.0): 시그니처 메뉴 중시도

### 기타
- **child_friendly** (0.0-1.0): 아이 동반 친화도. 높으면 아이와 함께 가기 좋은 곳 선호

---

## 시간대별 에이전트 조합

| 시간대 | 주변 거주자 | 외부 유동인구 | 총 에이전트 수 |
|--------|-------------|---------------|----------------|
| 아침 (06-11) | 6명 | - | 6명 |
| 점심 (11-14) | 6명 | 직장인 2명 | 8명 |
| 저녁 (17-21) | 6명 | 데이트 2명, 친구모임 2명 | 10명 |
| 야식 (00-06) | 6명 | 관광객 2명, 혼술족 2명 | 10명 |

**총 34회 방문 결정** (전략 전/후 각각)

---

## 활용 방법

1. **시뮬레이션 실행**: 각 시간대별로 해당 에이전트들이 활성화되어 식당 선택
2. **누적 리뷰**: 앞선 에이전트의 리뷰가 다음 에이전트의 결정에 영향
3. **전/후 비교**: 장터국밥 개선 전략 적용 전과 후의 방문율 변화 측정
4. **세그먼트 분석**: 어떤 페르소나가 개선 전략에 더 긍정적으로 반응하는지 파악
