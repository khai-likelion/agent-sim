# 에이전트 리뷰 관리 시스템 변경점

## 개요

에이전트 시뮬레이션의 평점/리뷰 관리 시스템을 JSON 파일 구조에 맞춰 단순화하고, 단계별 요약(Recursive Summarization)을 구현했습니다.

---

## 1. 핵심 변경 사항 요약

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| 평점 방식 | 맛/가성비/분위기 각각 1-5점 | **별점 1개 (0-5점)** + 태그 선택 |
| 태그 | 없음 | **맛/가성비/분위기/서비스 중 1-2개 선택** |
| 별점 업데이트 | agent_rate 별도 계산 | **JSON의 "별점" 필드 평균 업데이트** |
| 리뷰 요약 | - | **10개마다 요약** |

---

## 2. 변경된 파일

| 파일 | 변경 유형 |
|------|----------|
| `src/data_layer/global_store.py` | 수정 |
| `CHANGELOG_AGENT_REVIEW_SYSTEM.md` | 업데이트 |

---

## 3. 상세 변경 내용

### 3.1 `AgentRatingRecord` - 단순화된 평점 기록

```python
@dataclass
class AgentRatingRecord:
    """에이전트가 남긴 개별 평점 기록"""
    agent_id: int
    agent_name: str
    visit_datetime: str
    rating: int                              # 0~5 별점
    selected_tags: List[str] = field(default_factory=list)  # ["맛", "가성비"] 등
    comment: str = ""                        # 리뷰 코멘트
```

**변경점**:
- `taste_rating`, `value_rating`, `atmosphere_rating` → **`rating` 단일 필드**로 통합
- `selected_tags` 추가: 에이전트가 선택한 1-2개 태그

---

### 3.2 `AgentReview` - 10개마다 요약

```python
@dataclass
class AgentReview:
    comments: List[str] = field(default_factory=list)
    summary: str = ""
    total_review_count: int = 0
    summary_threshold: int = 10  # 10개 쌓이면 요약 실행
```

---

### 3.3 `StoreRating` - JSON 필드 매핑

```python
@dataclass
class StoreRating:
    # === JSON에서 로드되는 별점/태그 필드 ===
    star_rating: float = 0.0           # JSON의 "별점" 필드
    star_rating_count: int = 0         # 별점 평가 수 (review_count 기반)
    taste_count: int = 0               # JSON의 "맛" 필드
    value_count: int = 0               # JSON의 "가성비" 필드
    atmosphere_count: int = 0          # JSON의 "분위기" 필드
    service_count: int = 0             # JSON의 "서비스" 필드
```

---

### 3.4 평점 추가 로직 (`add_agent_rating`)

```python
def add_agent_rating(
    self,
    agent_id: int,
    agent_name: str,
    rating: int,
    selected_tags: List[str],
    visit_datetime: Optional[str] = None,
    comment: str = ""
):
    # === 별점 업데이트 (평균 계산) ===
    # 공식: (기존 별점 * 기존 수 + 새 별점) / (기존 수 + 1)
    old_total = self.star_rating * self.star_rating_count
    self.star_rating_count += 1
    self.star_rating = round((old_total + rating) / self.star_rating_count, 1)

    # === 태그 카운트 증가 ===
    for tag in selected_tags:
        if tag == "맛":
            self.taste_count += 1
        elif tag == "가성비":
            self.value_count += 1
        elif tag == "분위기":
            self.atmosphere_count += 1
        elif tag == "서비스":
            self.service_count += 1

    # === 리뷰 코멘트 추가 ===
    if comment:
        self.agent_review.comments.append(comment)
        self.agent_review.total_review_count += 1
```

---

### 3.5 JSON 로드 시 필드 매핑

```python
# load_from_json_files에서 추가된 로직
if "별점" in data:
    store.star_rating = float(data["별점"])
if "review_count" in data:
    store.star_rating_count = int(data["review_count"])
if "맛" in data:
    store.taste_count = int(data["맛"])
if "가성비" in data:
    store.value_count = int(data["가성비"])
if "분위기" in data:
    store.atmosphere_count = int(data["분위기"])
if "서비스" in data:
    store.service_count = int(data["서비스"])
```

---

### 3.6 JSON 업데이트 (`update_original_json_files`)

```python
def update_original_json_files(self) -> int:
    """시뮬레이션 종료 시 원본 JSON 파일에 에이전트 데이터 반영"""
    for store in self.stores.values():
        # === 별점 업데이트 (평균 계산됨) ===
        data["별점"] = store.star_rating

        # === 태그 카운트 업데이트 ===
        data["맛"] = store.taste_count
        data["가성비"] = store.value_count
        data["분위기"] = store.atmosphere_count
        data["서비스"] = store.service_count

        # === agent_review 추가 (review_count 뒤) ===
        data["agent_review"] = store.agent_review.to_dict()

        # === agent_ratings 추가 ===
        data["agent_ratings"] = [r.to_dict() for r in store.agent_ratings]
```

---

## 4. 데이터 흐름

```
[에이전트 매장 방문]
      ↓
[평점 생성: 별점(0-5) + 태그(1-2개) + 코멘트]
      ↓
[add_pending_rating() 호출]
      ↓
[타임슬롯 종료 시 flush_pending_ratings()]
      ↓
[별점 평균 계산, 태그 카운트 증가, 코멘트 저장]
      ↓
[10개 리뷰 쌓이면 Recursive Summarization]
      ↓
[시뮬레이션 종료]
      ↓
[update_original_json_files() 호출]
      ↓
[JSON 파일에 별점/태그/agent_review/agent_ratings 업데이트]
```

---

## 5. 예시: 에이전트 평점 추가

```python
# 에이전트가 별점 4점, 맛/가성비 태그 선택, 코멘트 남김
store.add_agent_rating(
    store_name="6호선닭한마리",
    agent_id=1,
    agent_name="김민준",
    rating=4,
    selected_tags=["맛", "가성비"],
    comment="국물이 진하고 맛있었어요!"
)
```

**결과**:
- `별점`: 4.2 → 4.1 (평균 재계산)
- `맛`: 9 → 10
- `가성비`: 8 → 9
- `분위기`: 2 (변경 없음)
- `서비스`: 6 (변경 없음)
- `agent_review.comments`: ["국물이 진하고 맛있었어요!"]

---

## 6. 결과 JSON 구조 (업데이트 후)

```json
{
  "store_name": "6호선닭한마리",
  "별점": 4.1,
  "맛": 10,
  "가성비": 9,
  "분위기": 2,
  "서비스": 6,
  "review_count": 28,
  "agent_review": {
    "comments": ["국물이 진하고 맛있었어요!", "정말 맛있는 닭한마리!"],
    "summary": "",
    "total_review_count": 2
  },
  "agent_ratings": [
    {
      "agent_id": 1,
      "agent_name": "김민준",
      "visit_datetime": "2026-02-15T14:30:00",
      "rating": 4,
      "selected_tags": ["맛", "가성비"],
      "comment": "국물이 진하고 맛있었어요!"
    }
  ]
}
```

---

## 7. 변경 일자

- **최초 작성**: 2026-02-15
- **v2.0.0 업데이트**: 2026-02-15 (별점/태그 시스템으로 단순화)
