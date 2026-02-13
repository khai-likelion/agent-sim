"""
망원동 상권 시뮬레이션용 페르소나 정의.

구조:
- group_type_description: 그룹 유형 자연어 (생활베이스형/사적모임형/공적모임형/가족모임형)
- generation_description: 세대 자연어 (Z1/Z2/Y/X/S)
- 생활베이스형만 망원 거주 여부를 natural_language_persona에 포함
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import json


# ============================================================================
# 1. 그룹 유형 정의 (원본 자연어 그대로)
# ============================================================================

GROUP_TYPE_DESCRIPTIONS = {
    "생활베이스형": """당신은 망원 인근에서 거주하거나 생활권이 망원에 묶여 있는 사람이다. 외식은 특별한 이벤트가 아니라 일상의 일부이며, 주로 평일 점심이나 저녁 시간에 자연스럽게 식당을 찾는다. 굳이 멀리 이동하지 않으며, 집이나 직장 동선 안에서 해결하려는 경향이 강하다. 가격 대비 만족이 중요하고, 대기가 길거나 복잡한 동선은 피하려 한다. 새로운 곳을 일부러 찾아다니기보다는 이미 경험해본 곳이나 익숙한 가게를 선호한다. 한 번 만족하면 반복 방문하는 경향이 있고, 불편한 경험이 누적되면 조용히 다른 선택지로 이동한다. 집밥이나 배달은 항상 강력한 대안이다.""",

    "사적모임형": """당신은 친구, 연인, 지인과 함께 시간을 보내기 위해 망원을 방문한다. 목적은 단순한 식사가 아니라 분위기와 경험이다. 방문 전 리뷰나 검색을 참고하는 경우가 많고, 공간의 감성, 인테리어, 메뉴의 개성, 사진이 잘 나오는 요소 등을 중요하게 여긴다. 다소 거리가 있더라도 매력적인 곳이라면 이동할 수 있다. 맛도 중요하지만 분위기가 더 큰 비중을 차지할 수 있다. 재방문보다는 새로운 공간을 탐색하는 경향이 있으며, 경험이 좋다면 공유하거나 이야기 소재로 삼는다. 일정이 마음에 들지 않으면 다른 활동으로 전환하기도 한다. 당신은 평일 점심이나 저녁, 또는 주말에 자연스럽게 식당을 찾는 경향이 강하다.""",

    "공적모임형": """당신은 회식, 업무 미팅, 공식적인 만남과 같은 목적을 가지고 방문한다. 실패하면 곤란한 자리이기 때문에 안정성과 신뢰를 우선한다. 단체 수용이 가능한지, 좌석 배치가 적절한지, 서비스가 일정 수준 이상 유지되는지가 중요하다. 과도하게 시끄럽거나 혼잡한 공간은 선호하지 않는다. 굳이 멀리 이동하지 않으며, 직장 동선 안에서 해결하려는 경향이 강하다. 검증되지 않은 선택은 피하는 경향이 있으며, 무난하고 실패 가능성이 낮은 선택을 한다. 거리 자체는 절대적 요소는 아니지만, 장소의 적합성이 더 중요하다. 방문이 만족스러우면 이후에도 비슷한 목적의 모임에서 다시 고려한다.""",

    "가족모임형": """당신은 가족 단위로 방문하며, 아이 또는 노약자가 포함될 수 있다. 외식은 모두가 불편하지 않아야 성공이다. 위생과 안전, 좌석의 여유, 메뉴 선택 폭을 중요하게 여긴다. 대기가 길거나 공간이 좁으면 스트레스를 받는다. 새로운 곳보다는 이미 검증된 곳을 선호하는 경향이 있다. 한 명이라도 불편하면 전체 선택이 바뀔 수 있다. 구성원 모두가 가능한 시간대에만 만난다. 집에서 해결하는 것도 항상 현실적인 선택지다. 만족스러운 경험은 반복 방문으로 이어지며, 안정적인 소비 패턴을 형성한다.""",
}


# ============================================================================
# 2. 세대 유형 정의 (원본 자연어 그대로)
# ============================================================================

GENERATION_DESCRIPTIONS = {
    "Z1": """당신은 아직 소비 결정권을 완전히 가지지 못한 나이이며, 보호자나 동행인의 영향을 크게 받습니다. 예산을 직접 통제하는 경우는 드물지만, 메뉴 선택이나 장소 선호에는 강한 감정 반응을 보입니다. 불쾌한 맛, 불편한 공간, 긴 대기 시간에 대한 인내심이 매우 낮으며, 즉각적인 만족을 선호합니다.

혼자일 경우에는 건강을 고려하기보다는 자극적이고 익숙한 메뉴를 선택하는 경향이 있습니다. 단 음식, 매운 음식, 시각적으로 강한 메뉴에 쉽게 반응합니다. 다만 혼자 이동 가능한 거리에는 한계가 있습니다.

집단 의사결정 상황에서는 경제적 결정권은 약하지만, 불만 표출이나 강한 선호 표현이 동행인의 최종 선택에 큰 영향을 줄 수 있습니다. 친구 집단에서는 유행과 또래 의견에 매우 민감하며, 인기 있는 장소를 따라가려는 경향이 강합니다.

재방문은 감정적 경험에 크게 좌우됩니다. "재밌었다" 또는 "별로였다"라는 기억이 강하게 남습니다.""",

    "Z2": """당신은 새로운 경험을 적극적으로 탐색하는 성향이 강하며, 트렌드와 SNS, 리뷰에 매우 민감합니다. 화제가 되는 신메뉴나 감성적인 공간, 사진으로 남기기 좋은 장소에 자연스럽게 끌립니다. 소비는 단순한 식사가 아니라 하나의 경험이며, '가볼 만하다'는 느낌이 들면 일정 수준의 실패 가능성은 크게 두려워하지 않습니다.

당신은 또래와의 모임에서 술을 동반한 소비를 자주 하며, 오래 머무를 수 있는 분위기나 대화가 가능한 공간을 선호합니다. 웨이팅이 있더라도 그 장소가 트렌디하거나 상징성이 있다면 감수할 가능성이 높습니다.

거리 자체는 큰 제약이 되지 않으며, 독특함이나 화제성이 충분하다면 평소 이동 범위를 벗어나는 것도 마다하지 않습니다. 가격은 고려 요소이지만, 절대적 저가보다는 '이 경험이 그 값어치를 하는가'를 더 중요하게 판단합니다.

재방문 성향은 높지 않은 편으로, 한 번 경험한 곳은 다시 가기보다는 새로운 장소를 탐색하려는 경향이 있습니다. 다만, 그 장소가 또래 집단 내에서 상징적인 공간이 되거나 정체성의 일부가 되면 반복 방문할 수 있습니다.

집단 의사결정 상황에서는 분위기나 트렌드 요소에 대한 의견을 강하게 제시하는 편이며, 선택에 영향을 줄 가능성이 높습니다.""",

    "Y": """당신은 식사를 감정적인 충동이 아니라 현실적인 판단의 문제로 접근합니다. 유행에 완전히 무관심하지는 않지만, 단순히 "핫하다"는 이유만으로 선택하지는 않습니다. 시간, 동선 효율, 대기 시간, 가격 대비 만족도 같은 요소를 종합적으로 고려합니다. 30분 이상 기다려야 하거나 이동 동선이 비효율적이라면, 아무리 유명해도 다른 선택지를 찾을 가능성이 높습니다.

가격은 절대적인 저렴함보다 "이 가격이면 이 정도는 나와야 한다"는 기준이 중요합니다. 음식의 품질, 서비스, 공간의 쾌적함이 가격과 균형을 이룬다고 느껴야 만족합니다. 기대에 미치지 못하면 "한 번으로 충분하다"라고 판단합니다.

새로운 경험에 대한 거부감은 없지만, 어느 정도 정보가 축적된 후 움직이는 편입니다. 리뷰와 평점을 참고하되 감정적으로 휘둘리기보다는 비교·분석합니다. Z₂보다 위험 감수 성향은 낮습니다.

집단 의사결정 상황에서는 분위기를 따라가기보다는 현실적인 질문을 던지는 역할을 합니다. "웨이팅 얼마나 돼?", "가격대는?", "거리 멀지 않아?" 같은 질문을 통해 극단적인 선택을 조정합니다. 당신의 의견은 집단을 균형점으로 이동시키는 경향이 있습니다.

만족한 경험은 루틴으로 이어질 가능성이 높습니다. 반복 방문 확률이 비교적 높으며, 구조적인 불편이 누적되면 서서히 배제합니다.""",

    "X": """당신은 "실패하지 않는 선택"을 선호합니다. 트렌드보다 신뢰, 화제성보다 기본기를 중요하게 여깁니다. 음식의 맛뿐 아니라 직원의 태도, 공간의 정돈 상태, 소음 수준, 전체적인 서비스 흐름까지 종합적으로 평가합니다.

과도하게 자극적인 메뉴나 실험적인 콘셉트는 선호하지 않습니다. 익숙하고 안정적인 메뉴, 부담 없는 맛, 예측 가능한 품질을 선호합니다. 지나치게 혼잡하거나 시끄러운 공간은 기피 대상이 됩니다.

거리 자체가 절대적인 제약은 아니지만, "굳이 멀리 갈 이유가 있는가?"를 먼저 따집니다. SNS 화제성만으로 이동하지는 않으며, 과거 경험이나 신뢰할 수 있는 추천이 더 중요합니다.

집단 의사결정에서는 강하게 주장하지 않더라도, 당신이 불편함을 느끼면 분위기가 바뀔 수 있습니다. 특히 서비스 불만이나 위생 문제에는 민감하게 반응하며, 이는 집단 선택을 보수적으로 만드는 요인이 됩니다.

한 번 신뢰가 형성되면 장기적으로 방문합니다. 그러나 서비스 질이 눈에 띄게 떨어지면 재방문 의사가 급격히 낮아집니다.""",

    "S": """당신은 새로운 경험보다 안정성과 편안함을 우선합니다. 위생과 청결은 가장 중요한 판단 기준이며, 공간이 어수선하거나 소음이 크면 즉각적인 거부감을 느낍니다.

메뉴는 복잡하지 않고 이해하기 쉬워야 하며, 자극적이거나 실험적인 음식은 부담스럽게 느껴질 수 있습니다. 음식은 맛있어야 하지만 동시에 속이 편안해야 합니다.

이동 거리에는 현실적인 제한이 있으며, 접근성이 좋고 동선이 단순한 장소를 선호합니다. 긴 대기 시간이나 복잡한 주문 시스템은 강한 불편 요소로 작용합니다.

집단 상황에서 발언이 많지 않더라도, 불편함을 느끼는 순간 그 분위기가 집단에 전달됩니다. "시끄럽다", "복잡하다", "불편하다"는 반응은 집단의 방향을 바꿀 수 있습니다.

한 번 편안함과 신뢰를 느낀 장소는 반복 방문할 가능성이 매우 높습니다. 반대로 위생이나 서비스 문제로 불쾌한 경험을 하면 재방문 확률은 거의 사라집니다.""",
}


# ============================================================================
# 3. 거주지 수식어 (생활베이스형 전용)
# ============================================================================

RESIDENCE_MODIFIER = {
    "망원동_거주": """당신은 망원동에 거주하며, 이 동네가 생활의 중심입니다. 집 근처 식당들은 이미 익숙하고, 동네 분위기와 가게들의 변화를 자연스럽게 인지합니다. 외식은 '멀리 나가는 것'이 아니라 '집 앞에서 해결하는 것'에 가깝습니다.""",

    "망원동_외부": """당신은 망원동 외부에 거주하지만, 직장이나 일상적인 동선이 망원 인근을 지나갑니다. 출퇴근길이나 업무 중 자연스럽게 망원 상권을 이용하게 되며, 점심이나 저녁 식사를 이 동네에서 해결하는 경우가 많습니다. 동네 주민만큼 깊이 알지는 못하지만, 자주 다니는 루트의 가게들은 파악하고 있습니다.""",
}


# ============================================================================
# 4. 페르소나 조합 정의
# ============================================================================

@dataclass
class PersonaCombination:
    """페르소나 조합 정의"""
    id: str
    group_type: str
    group_size: int
    generation: str
    gender_composition: str
    is_mangwon_resident: Optional[bool] = None  # 생활베이스형만 해당 (True: 거주, False: 외부)
    special_condition: Optional[str] = None


def generate_all_combinations() -> List[PersonaCombination]:
    """모든 페르소나 조합 생성"""
    combinations = []
    combo_id = 1

    # ========================================
    # 1인
    # ========================================

    # 생활베이스형 1인: 5세대 × 2성별 × 2거주지 = 20가지
    for is_resident in [True, False]:
        for gen in ["Z1", "Z2", "Y", "X", "S"]:
            for gender in ["남", "여"]:
                combinations.append(PersonaCombination(
                    id=f"P{combo_id:03d}",
                    group_type="생활베이스형",
                    group_size=1,
                    generation=gen,
                    gender_composition=gender,
                    is_mangwon_resident=is_resident,
                ))
                combo_id += 1

    # 사적모임형 1인: 5세대 × 2성별 = 10가지
    for gen in ["Z1", "Z2", "Y", "X", "S"]:
        for gender in ["남", "여"]:
            combinations.append(PersonaCombination(
                id=f"P{combo_id:03d}",
                group_type="사적모임형",
                group_size=1,
                generation=gen,
                gender_composition=gender,
            ))
            combo_id += 1

    # ========================================
    # 2인
    # ========================================

    # 생활베이스형 2인: 3세대(Z1,Z2,Y) × 3성별조합 × 2거주지 = 18가지
    for is_resident in [True, False]:
        for gen in ["Z1", "Z2", "Y"]:
            for gender in ["남", "여", "혼성"]:
                combinations.append(PersonaCombination(
                    id=f"P{combo_id:03d}",
                    group_type="생활베이스형",
                    group_size=2,
                    generation=gen,
                    gender_composition=gender,
                    is_mangwon_resident=is_resident,
                ))
                combo_id += 1

    # 사적모임형 2인: Z1(3) + Z2(3) + Y(3) + X(3) + S(1) = 13가지
    for gen in ["Z1", "Z2", "Y", "X"]:
        for gender in ["남", "여", "혼성"]:
            combinations.append(PersonaCombination(
                id=f"P{combo_id:03d}",
                group_type="사적모임형",
                group_size=2,
                generation=gen,
                gender_composition=gender,
            ))
            combo_id += 1
    combinations.append(PersonaCombination(
        id=f"P{combo_id:03d}",
        group_type="사적모임형",
        group_size=2,
        generation="S",
        gender_composition="혼성",
    ))
    combo_id += 1

    # 가족모임형 2인: 18가지
    for z1_gender in ["남", "여"]:
        combinations.append(PersonaCombination(
            id=f"P{combo_id:03d}",
            group_type="가족모임형",
            group_size=2,
            generation="Z1",
            gender_composition=z1_gender,
            special_condition="S포함_Z1포함",
        ))
        combo_id += 1

    combinations.append(PersonaCombination(
        id=f"P{combo_id:03d}",
        group_type="가족모임형",
        group_size=2,
        generation="S",
        gender_composition="혼성",
        special_condition="S포함_Z1미포함",
    ))
    combo_id += 1

    for z1_gender in ["남", "여", "혼성"]:
        combinations.append(PersonaCombination(
            id=f"P{combo_id:03d}",
            group_type="가족모임형",
            group_size=2,
            generation="Z1",
            gender_composition=z1_gender,
            special_condition="S미포함_Z1포함",
        ))
        combo_id += 1

    for gen in ["Z2", "Y", "X"]:
        for gender in ["남", "여", "혼성"]:
            combinations.append(PersonaCombination(
                id=f"P{combo_id:03d}",
                group_type="가족모임형",
                group_size=2,
                generation=gen,
                gender_composition=gender,
                special_condition="S미포함_Z1미포함",
            ))
            combo_id += 1
    for gender in ["남", "여", "혼성"]:
        combinations.append(PersonaCombination(
            id=f"P{combo_id:03d}",
            group_type="가족모임형",
            group_size=2,
            generation="혼합",
            gender_composition=gender,
            special_condition="S미포함_Z1미포함_혼합세대",
        ))
        combo_id += 1

    # ========================================
    # 4인
    # ========================================

    # 생활베이스형 4인: 3세대(Z1,Z2,Y) × 3성별조합 × 2거주지 = 18가지
    for is_resident in [True, False]:
        for gen in ["Z1", "Z2", "Y"]:
            for gender in ["남", "여", "혼성"]:
                combinations.append(PersonaCombination(
                    id=f"P{combo_id:03d}",
                    group_type="생활베이스형",
                    group_size=4,
                    generation=gen,
                    gender_composition=gender,
                    is_mangwon_resident=is_resident,
                ))
                combo_id += 1

    # 사적모임형 4인
    for gen in ["Z1", "Z2", "Y", "X"]:
        for gender in ["남", "여", "혼성"]:
            combinations.append(PersonaCombination(
                id=f"P{combo_id:03d}",
                group_type="사적모임형",
                group_size=4,
                generation=gen,
                gender_composition=gender,
            ))
            combo_id += 1
    combinations.append(PersonaCombination(
        id=f"P{combo_id:03d}",
        group_type="사적모임형",
        group_size=4,
        generation="S",
        gender_composition="혼성",
    ))
    combo_id += 1

    # 공적모임형 4인
    for gen in ["Z2", "Y", "X"]:
        for gender in ["남", "여", "혼성"]:
            combinations.append(PersonaCombination(
                id=f"P{combo_id:03d}",
                group_type="공적모임형",
                group_size=4,
                generation=gen,
                gender_composition=gender,
            ))
            combo_id += 1
    for mix_type in ["Z2+Y", "Y+X", "Z2+X"]:
        combinations.append(PersonaCombination(
            id=f"P{combo_id:03d}",
            group_type="공적모임형",
            group_size=4,
            generation=f"혼합({mix_type})",
            gender_composition="혼성",
        ))
        combo_id += 1

    # 가족모임형 4인
    for z1_gender in ["남", "여", "혼성"]:
        combinations.append(PersonaCombination(
            id=f"P{combo_id:03d}",
            group_type="가족모임형",
            group_size=4,
            generation="Z1",
            gender_composition=z1_gender,
            special_condition="S포함_Z1포함",
        ))
        combo_id += 1

    combinations.append(PersonaCombination(
        id=f"P{combo_id:03d}",
        group_type="가족모임형",
        group_size=4,
        generation="S",
        gender_composition="혼성",
        special_condition="S포함_Z1미포함",
    ))
    combo_id += 1

    for z1_gender in ["남", "여", "혼성"]:
        combinations.append(PersonaCombination(
            id=f"P{combo_id:03d}",
            group_type="가족모임형",
            group_size=4,
            generation="Z1",
            gender_composition=z1_gender,
            special_condition="S미포함_Z1포함",
        ))
        combo_id += 1

    for gen in ["Z2", "Y", "X"]:
        for gender in ["남", "여", "혼성"]:
            combinations.append(PersonaCombination(
                id=f"P{combo_id:03d}",
                group_type="가족모임형",
                group_size=4,
                generation=gen,
                gender_composition=gender,
                special_condition="S미포함_Z1미포함",
            ))
            combo_id += 1
    for gender in ["남", "여", "혼성"]:
        combinations.append(PersonaCombination(
            id=f"P{combo_id:03d}",
            group_type="가족모임형",
            group_size=4,
            generation="혼합",
            gender_composition=gender,
            special_condition="S미포함_Z1미포함_혼합세대",
        ))
        combo_id += 1

    return combinations


# ============================================================================
# 5. 자연어 페르소나 생성
# ============================================================================

def get_generations_from_combo(combo: PersonaCombination) -> List[str]:
    """
    조합에서 포함된 모든 세대를 추출.

    2인/4인 그룹의 세대 구성 규칙:
    - 생활베이스형/사적모임형: 동일 세대로만 구성 (단일 세대)
    - 공적모임형 4인: 단일 세대 또는 혼합(Z2+Y, Y+X, Z2+X)
    - 가족모임형:
      - S(o)Z₁(o): S + Z1 두 세대
      - S(o)Z₁(x): S + 중간 세대(Y 또는 X)
      - S(x)Z₁(o): Z1 + 부모 세대(Y 또는 X)
      - S(x)Z₁(x) 단일: 해당 세대만 (Z2, Y, X)
      - S(x)Z₁(x) 혼합: Z2, Y, X 모든 세대
    """

    # 1인은 단일 세대
    if combo.group_size == 1:
        return [combo.generation]

    # 가족모임형의 special_condition 처리
    if combo.group_type == "가족모임형" and combo.special_condition:
        if combo.special_condition == "S포함_Z1포함":
            # S세대(조부모)와 Z1세대(손자녀) 모두 포함
            # 중간 세대(Y 또는 X)도 존재하지만, 의사결정은 S와 Z1에 맞춰짐
            return ["S", "Z1"]
        elif combo.special_condition == "S포함_Z1미포함":
            # S세대(조부모) 포함, Z1 없음
            # S + 성인 자녀(Y 또는 X) 구성
            return ["S", "Y"]
        elif combo.special_condition == "S미포함_Z1포함":
            # Z1세대(어린 자녀) 포함, S 없음
            # Z1 + 부모 세대(Y 또는 X) 구성
            return ["Z1", "Y"]
        elif combo.special_condition == "S미포함_Z1미포함":
            # S와 Z1 모두 없음 - 단일 세대 (Z2, Y, X 중 하나)
            return [combo.generation]
        elif combo.special_condition == "S미포함_Z1미포함_혼합세대":
            # S와 Z1 없이 Z2, Y, X 혼합 - 조합이 특정되지 않아 대표 세대(Y) 사용
            return ["Y"]

    # 공적모임형의 혼합 세대 처리
    if combo.generation.startswith("혼합("):
        # "혼합(Z2+Y)" -> ["Z2", "Y"]
        inner = combo.generation[3:-1]  # "Z2+Y"
        return inner.split("+")

    # 2인/4인 단일 세대 (생활베이스형, 사적모임형 등)
    # 동일 세대로만 구성됨
    return [combo.generation]


def generate_natural_language_persona(combo: PersonaCombination) -> str:
    """조합을 기반으로 자연어 페르소나 생성 (group_type + residence + generation)"""

    parts = []

    # 1. 그룹 유형 설명
    parts.append(GROUP_TYPE_DESCRIPTIONS[combo.group_type])

    # 2. 거주지 설명 (생활베이스형만)
    if combo.group_type == "생활베이스형" and combo.is_mangwon_resident is not None:
        if combo.is_mangwon_resident:
            parts.append(RESIDENCE_MODIFIER["망원동_거주"])
        else:
            parts.append(RESIDENCE_MODIFIER["망원동_외부"])

    # 3. 세대 설명 (다중 세대일 경우 모든 세대의 설명 포함)
    generations = get_generations_from_combo(combo)

    # 모든 세대의 설명을 추가 (원본 자연어만 사용)
    for gen in generations:
        if gen in GENERATION_DESCRIPTIONS:
            if len(generations) > 1:
                # 다중 세대인 경우 세대 구분 헤더 추가
                parts.append(f"[{gen} 세대 구성원]\n{GENERATION_DESCRIPTIONS[gen]}")
            else:
                # 단일 세대인 경우 헤더 없이 설명만
                parts.append(GENERATION_DESCRIPTIONS[gen])

    return "\n\n".join(parts)


def export_personas_to_json(filepath: str):
    """페르소나를 JSON 파일로 내보내기"""
    combinations = generate_all_combinations()

    data = {
        "total_count": len(combinations),
        "group_type_descriptions": GROUP_TYPE_DESCRIPTIONS,
        "generation_descriptions": GENERATION_DESCRIPTIONS,
        "residence_modifier": RESIDENCE_MODIFIER,
        "personas": []
    }

    for combo in combinations:
        persona_data = {
            "id": combo.id,
            "group_type": combo.group_type,
            "group_size": combo.group_size,
            "generation": combo.generation,
            "gender_composition": combo.gender_composition,
            "natural_language_persona": generate_natural_language_persona(combo),
        }

        if combo.special_condition:
            persona_data["special_condition"] = combo.special_condition

        data["personas"].append(persona_data)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def export_personas_to_markdown(filepath: str):
    """페르소나를 Markdown 파일로 내보내기"""
    combinations = generate_all_combinations()

    lines = [
        "# 망원동 상권 시뮬레이션 - 페르소나 목록",
        "",
        f"총 {len(combinations)}가지 페르소나",
        "",
        "---",
        "",
    ]

    current_group = None
    current_size = None

    for combo in combinations:
        if combo.group_type != current_group:
            current_group = combo.group_type
            current_size = None
            lines.append(f"# {current_group}")
            lines.append("")

        if combo.group_size != current_size:
            current_size = combo.group_size
            lines.append(f"## {current_size}인")
            lines.append("")

        # 제목
        title_parts = [combo.id, combo.group_type, f"{combo.group_size}인", combo.generation, combo.gender_composition]
        if combo.group_type == "생활베이스형" and combo.is_mangwon_resident is not None:
            residence_label = "망원동 거주" if combo.is_mangwon_resident else "망원동 외부"
            title_parts.insert(2, residence_label)
        lines.append(f"### {' / '.join(title_parts)}")
        lines.append("")

        # 자연어 페르소나
        lines.append(generate_natural_language_persona(combo))
        lines.append("")
        lines.append("---")
        lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_distribution():
    """페르소나 분포 출력"""
    combinations = generate_all_combinations()
    print(f"\n총 페르소나 수: {len(combinations)}")

    size_counts = {}
    for c in combinations:
        size_counts[c.group_size] = size_counts.get(c.group_size, 0) + 1
    print("\n인원별 분포:")
    for size, count in sorted(size_counts.items()):
        print(f"  {size}인: {count}가지")

    group_counts = {}
    for c in combinations:
        group_counts[c.group_type] = group_counts.get(c.group_type, 0) + 1
    print("\n그룹 유형별 분포:")
    for group, count in sorted(group_counts.items()):
        print(f"  {group}: {count}가지")


if __name__ == "__main__":
    print_distribution()
