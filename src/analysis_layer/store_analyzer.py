"""
Store Analyzer - 단일 매장 고객 행동 변화 심층 분석

전략 적용 전후 시뮬레이션 결과를 비교하여:
1. 에이전트 유입 경로 분석 (신규/이탈/전환)
2. 세그먼트 변화 추적 (연령대, 성별, 라이프스타일)
3. 의사결정 이유 심층 비교 (reason, comment 대조)
4. 종합 리포트 생성
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path


@dataclass
class AgentVisitRecord:
    """에이전트 방문 기록"""
    agent_id: int
    agent_name: str
    visit_datetime: str
    rating: int
    selected_tags: List[str]
    comment: str
    reason: str = ""  # 방문 이유


@dataclass
class AgentPersona:
    """에이전트 페르소나 정보"""
    agent_id: int
    name: str
    age: int
    gender: str
    generation: str  # Z세대, Y세대, X세대, S세대
    lifestyle: str   # 1인생활베이스형, 2~4인사적모임형 등
    occupation: str
    income_level: str
    residence_type: str  # Resident(상주), Visitor(유동)
    preference_tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentPersona":
        return cls(
            agent_id=data.get("agent_id", 0),
            name=data.get("name", "Unknown"),
            age=data.get("age", 30),
            gender=data.get("gender", "Unknown"),
            generation=data.get("generation", "Unknown"),
            lifestyle=data.get("lifestyle", "Unknown"),
            occupation=data.get("occupation", "Unknown"),
            income_level=data.get("income_level", "중간"),
            residence_type=data.get("residence_type", "Visitor"),
            preference_tags=data.get("preference_tags", []),
        )


@dataclass
class SimulationResult:
    """시뮬레이션 결과 데이터"""
    store_name: str
    visits: List[AgentVisitRecord]
    star_rating: float
    total_visits: int
    taste_count: int
    value_count: int
    atmosphere_count: int
    service_count: int
    agent_review_summary: str = ""


@dataclass
class InflowAnalysis:
    """에이전트 유입 분석 결과"""
    new_visitors: List[AgentPersona]           # 신규 유입
    churned_visitors: List[AgentPersona]       # 이탈 고객
    retained_visitors: List[AgentPersona]      # 유지 고객
    converted_visitors: List[Tuple[AgentPersona, str]]  # 전환 고객 (페르소나, 이전 방문 매장)


@dataclass
class SegmentAnalysis:
    """세그먼트 분석 결과"""
    generation_before: Dict[str, int]
    generation_after: Dict[str, int]
    gender_before: Dict[str, int]
    gender_after: Dict[str, int]
    lifestyle_before: Dict[str, int]
    lifestyle_after: Dict[str, int]
    residence_before: Dict[str, int]
    residence_after: Dict[str, int]
    age_group_before: Dict[str, int]
    age_group_after: Dict[str, int]


@dataclass
class ReasoningAnalysis:
    """의사결정 이유 분석 결과"""
    keywords_before: Dict[str, int]
    keywords_after: Dict[str, int]
    positive_reasons_before: List[str]
    positive_reasons_after: List[str]
    negative_reasons_before: List[str]
    negative_reasons_after: List[str]
    sentiment_shift: str  # "positive", "negative", "neutral"


class StoreAnalyzer:
    """
    단일 매장 고객 행동 변화 심층 분석기.

    사용법:
    ```python
    analyzer = StoreAnalyzer(
        target_store="스몰굿커피 망원역점",
        baseline_result=baseline_sim_result,
        strategy_result=strategy_sim_result,
        agent_personas=agent_personas_dict,
        all_store_visits_before=all_visits_before,
        all_store_visits_after=all_visits_after,
    )
    report = analyzer.generate_report()
    analyzer.save_report("target_store_analysis.md")
    ```
    """

    # 세대 분류 기준
    GENERATION_MAP = {
        (10, 27): "Z세대",
        (28, 43): "Y세대(밀레니얼)",
        (44, 59): "X세대",
        (60, 100): "S세대(시니어)",
    }

    # 연령대 분류
    AGE_GROUP_MAP = {
        (10, 19): "10대",
        (20, 29): "20대",
        (30, 39): "30대",
        (40, 49): "40대",
        (50, 59): "50대",
        (60, 100): "60대+",
    }

    def __init__(
        self,
        target_store: str,
        baseline_result: SimulationResult,
        strategy_result: SimulationResult,
        agent_personas: Dict[int, AgentPersona],
        all_store_visits_before: Dict[str, List[int]],  # store_name -> [agent_ids]
        all_store_visits_after: Dict[str, List[int]],
        applied_strategies: Optional[List[Dict[str, Any]]] = None,
    ):
        self.target_store = target_store
        self.baseline = baseline_result
        self.strategy = strategy_result
        self.personas = agent_personas
        self.visits_before = all_store_visits_before
        self.visits_after = all_store_visits_after
        self.applied_strategies = applied_strategies or []

        # 분석 결과 캐시
        self._inflow_analysis: Optional[InflowAnalysis] = None
        self._segment_analysis: Optional[SegmentAnalysis] = None
        self._reasoning_analysis: Optional[ReasoningAnalysis] = None

    def _get_generation(self, age: int) -> str:
        """연령에서 세대 추출"""
        for (min_age, max_age), gen in self.GENERATION_MAP.items():
            if min_age <= age <= max_age:
                return gen
        return "Unknown"

    def _get_age_group(self, age: int) -> str:
        """연령에서 연령대 추출"""
        for (min_age, max_age), group in self.AGE_GROUP_MAP.items():
            if min_age <= age <= max_age:
                return group
        return "Unknown"

    def _get_visitor_ids(self, visits: List[AgentVisitRecord]) -> Set[int]:
        """방문 기록에서 에이전트 ID 집합 추출"""
        return {v.agent_id for v in visits}

    # ==================== 1. 에이전트 유입 분석 ====================

    def analyze_inflow(self) -> InflowAnalysis:
        """
        에이전트 유입 경로 분석.

        - 신규 유입: Baseline에서 방문 안 함 → 전략 후 방문
        - 이탈: Baseline에서 방문 → 전략 후 방문 안 함
        - 유지: 전후 모두 방문
        - 전환: 경쟁점 방문 → 타겟 매장으로 전환
        """
        if self._inflow_analysis:
            return self._inflow_analysis

        before_ids = self._get_visitor_ids(self.baseline.visits)
        after_ids = self._get_visitor_ids(self.strategy.visits)

        # 신규 유입
        new_ids = after_ids - before_ids
        new_visitors = [self.personas[aid] for aid in new_ids if aid in self.personas]

        # 이탈 고객
        churned_ids = before_ids - after_ids
        churned_visitors = [self.personas[aid] for aid in churned_ids if aid in self.personas]

        # 유지 고객
        retained_ids = before_ids & after_ids
        retained_visitors = [self.personas[aid] for aid in retained_ids if aid in self.personas]

        # 전환 고객 분석 (경쟁점에서 넘어온 고객)
        converted_visitors = []
        for aid in new_ids:
            if aid not in self.personas:
                continue
            persona = self.personas[aid]
            # 이전에 방문했던 다른 매장 찾기
            previous_stores = []
            for store_name, visitor_ids in self.visits_before.items():
                if store_name != self.target_store and aid in visitor_ids:
                    previous_stores.append(store_name)
            if previous_stores:
                # 가장 자주 방문한 매장을 경쟁점으로 간주
                converted_visitors.append((persona, previous_stores[0]))

        self._inflow_analysis = InflowAnalysis(
            new_visitors=new_visitors,
            churned_visitors=churned_visitors,
            retained_visitors=retained_visitors,
            converted_visitors=converted_visitors,
        )
        return self._inflow_analysis

    # ==================== 2. 세그먼트 변화 분석 ====================

    def analyze_segments(self) -> SegmentAnalysis:
        """세그먼트(인구통계학적) 변화 분석"""
        if self._segment_analysis:
            return self._segment_analysis

        before_ids = self._get_visitor_ids(self.baseline.visits)
        after_ids = self._get_visitor_ids(self.strategy.visits)

        def count_segments(agent_ids: Set[int]) -> Dict[str, Dict[str, int]]:
            generation = Counter()
            gender = Counter()
            lifestyle = Counter()
            residence = Counter()
            age_group = Counter()

            for aid in agent_ids:
                if aid not in self.personas:
                    continue
                p = self.personas[aid]
                generation[p.generation] += 1
                gender[p.gender] += 1
                lifestyle[p.lifestyle] += 1
                residence[p.residence_type] += 1
                age_group[self._get_age_group(p.age)] += 1

            return {
                "generation": dict(generation),
                "gender": dict(gender),
                "lifestyle": dict(lifestyle),
                "residence": dict(residence),
                "age_group": dict(age_group),
            }

        before_segments = count_segments(before_ids)
        after_segments = count_segments(after_ids)

        self._segment_analysis = SegmentAnalysis(
            generation_before=before_segments["generation"],
            generation_after=after_segments["generation"],
            gender_before=before_segments["gender"],
            gender_after=after_segments["gender"],
            lifestyle_before=before_segments["lifestyle"],
            lifestyle_after=after_segments["lifestyle"],
            residence_before=before_segments["residence"],
            residence_after=after_segments["residence"],
            age_group_before=before_segments["age_group"],
            age_group_after=after_segments["age_group"],
        )
        return self._segment_analysis

    # ==================== 3. 의사결정 이유 분석 ====================

    def analyze_reasoning(self) -> ReasoningAnalysis:
        """의사결정 이유(reason, comment) 심층 비교"""
        if self._reasoning_analysis:
            return self._reasoning_analysis

        def extract_keywords(texts: List[str]) -> Dict[str, int]:
            """텍스트에서 주요 키워드 추출"""
            # 간단한 키워드 추출 (실제로는 형태소 분석기 사용 권장)
            keywords = Counter()
            stopwords = {"이", "가", "을", "를", "에", "의", "와", "과", "도", "로", "으로", "는", "은", "다", "함", "임", "됨", "것", "수", "좀", "너무", "정말", "아주", "매우", "그냥", "진짜"}

            for text in texts:
                # 간단한 토큰화
                words = text.replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ").split()
                for word in words:
                    word = word.strip()
                    if len(word) >= 2 and word not in stopwords:
                        keywords[word] += 1
            return dict(keywords.most_common(20))

        def classify_sentiment(texts: List[str]) -> Tuple[List[str], List[str]]:
            """텍스트를 긍정/부정으로 분류"""
            positive_keywords = ["좋", "맛있", "친절", "깨끗", "편", "저렴", "가성비", "분위기", "추천", "만족", "최고", "훌륭"]
            negative_keywords = ["별로", "비싸", "불친절", "더러", "좁", "시끄럽", "오래", "늦", "실망", "후회", "짜증"]

            positive = []
            negative = []

            for text in texts:
                is_positive = any(kw in text for kw in positive_keywords)
                is_negative = any(kw in text for kw in negative_keywords)

                if is_positive and not is_negative:
                    positive.append(text)
                elif is_negative:
                    negative.append(text)

            return positive, negative

        # Before 데이터
        before_comments = [v.comment for v in self.baseline.visits if v.comment]
        before_reasons = [v.reason for v in self.baseline.visits if v.reason]
        before_texts = before_comments + before_reasons

        # After 데이터
        after_comments = [v.comment for v in self.strategy.visits if v.comment]
        after_reasons = [v.reason for v in self.strategy.visits if v.reason]
        after_texts = after_comments + after_reasons

        # 키워드 추출
        keywords_before = extract_keywords(before_texts)
        keywords_after = extract_keywords(after_texts)

        # 감성 분류
        positive_before, negative_before = classify_sentiment(before_texts)
        positive_after, negative_after = classify_sentiment(after_texts)

        # 감성 변화 판단
        before_sentiment_ratio = len(positive_before) / max(len(before_texts), 1)
        after_sentiment_ratio = len(positive_after) / max(len(after_texts), 1)

        if after_sentiment_ratio > before_sentiment_ratio + 0.1:
            sentiment_shift = "positive"
        elif after_sentiment_ratio < before_sentiment_ratio - 0.1:
            sentiment_shift = "negative"
        else:
            sentiment_shift = "neutral"

        self._reasoning_analysis = ReasoningAnalysis(
            keywords_before=keywords_before,
            keywords_after=keywords_after,
            positive_reasons_before=positive_before[:10],
            positive_reasons_after=positive_after[:10],
            negative_reasons_before=negative_before[:10],
            negative_reasons_after=negative_after[:10],
            sentiment_shift=sentiment_shift,
        )
        return self._reasoning_analysis

    # ==================== 4. 리포트 생성 ====================

    def generate_report(self) -> str:
        """종합 분석 리포트 생성 (Markdown)"""
        inflow = self.analyze_inflow()
        segments = self.analyze_segments()
        reasoning = self.analyze_reasoning()

        report = []
        report.append(f"# {self.target_store} 고객 행동 변화 심층 분석 리포트")
        report.append(f"\n> 분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n---\n")

        # 1. 핵심 지표 변화
        report.append("## 1. 핵심 지표 변화\n")
        report.append("| 지표 | Before | After | 변화 |")
        report.append("|------|--------|-------|------|")

        rating_change = self.strategy.star_rating - self.baseline.star_rating
        rating_sign = "+" if rating_change >= 0 else ""
        report.append(f"| 별점 | {self.baseline.star_rating:.1f} | {self.strategy.star_rating:.1f} | {rating_sign}{rating_change:.1f} |")

        visit_change = self.strategy.total_visits - self.baseline.total_visits
        visit_sign = "+" if visit_change >= 0 else ""
        report.append(f"| 방문객 수 | {self.baseline.total_visits} | {self.strategy.total_visits} | {visit_sign}{visit_change} |")

        # 태그 변화
        for tag_name, before, after in [
            ("맛", self.baseline.taste_count, self.strategy.taste_count),
            ("가성비", self.baseline.value_count, self.strategy.value_count),
            ("분위기", self.baseline.atmosphere_count, self.strategy.atmosphere_count),
            ("서비스", self.baseline.service_count, self.strategy.service_count),
        ]:
            change = after - before
            sign = "+" if change >= 0 else ""
            report.append(f"| {tag_name} 태그 | {before} | {after} | {sign}{change} |")

        report.append("\n---\n")

        # 2. 에이전트 유입 분석
        report.append("## 2. 에이전트 유입 분석 (Inflow Analysis)\n")

        report.append(f"### 2.1 신규 유입 고객 ({len(inflow.new_visitors)}명)\n")
        if inflow.new_visitors:
            report.append("**Baseline에서 방문하지 않았으나, 전략 적용 후 새로 방문하게 된 고객:**\n")
            report.append("| 이름 | 연령 | 성별 | 세대 | 라이프스타일 | 거주유형 |")
            report.append("|------|------|------|------|--------------|----------|")
            for p in inflow.new_visitors[:15]:  # 상위 15명
                report.append(f"| {p.name} | {p.age} | {p.gender} | {p.generation} | {p.lifestyle} | {p.residence_type} |")
            if len(inflow.new_visitors) > 15:
                report.append(f"\n*... 외 {len(inflow.new_visitors) - 15}명*\n")
        else:
            report.append("*신규 유입 고객 없음*\n")

        report.append(f"\n### 2.2 이탈 고객 ({len(inflow.churned_visitors)}명)\n")
        if inflow.churned_visitors:
            report.append("**전략 적용 후 방문을 중단한 고객:**\n")
            report.append("| 이름 | 연령 | 성별 | 세대 | 라이프스타일 |")
            report.append("|------|------|------|------|--------------|")
            for p in inflow.churned_visitors[:10]:
                report.append(f"| {p.name} | {p.age} | {p.gender} | {p.generation} | {p.lifestyle} |")
        else:
            report.append("*이탈 고객 없음*\n")

        report.append(f"\n### 2.3 전환 고객 ({len(inflow.converted_visitors)}명)\n")
        if inflow.converted_visitors:
            report.append("**경쟁점에서 이 매장으로 마음을 돌린 고객:**\n")
            report.append("| 이름 | 연령 | 세대 | 이전 방문 매장 |")
            report.append("|------|------|------|----------------|")
            for p, prev_store in inflow.converted_visitors[:10]:
                report.append(f"| {p.name} | {p.age} | {p.generation} | {prev_store} |")
        else:
            report.append("*전환 고객 없음*\n")

        report.append("\n---\n")

        # 3. 세그먼트 변화 추적
        report.append("## 3. 세그먼트 변화 추적\n")

        def format_segment_table(before: Dict, after: Dict, title: str) -> List[str]:
            lines = [f"### {title}\n"]
            all_keys = sorted(set(before.keys()) | set(after.keys()))
            if not all_keys:
                lines.append("*데이터 없음*\n")
                return lines

            total_before = sum(before.values()) or 1
            total_after = sum(after.values()) or 1

            lines.append("| 구분 | Before | 비율 | After | 비율 | 변화 |")
            lines.append("|------|--------|------|-------|------|------|")

            for key in all_keys:
                b = before.get(key, 0)
                a = after.get(key, 0)
                b_pct = b / total_before * 100
                a_pct = a / total_after * 100
                change = a_pct - b_pct
                sign = "+" if change >= 0 else ""
                lines.append(f"| {key} | {b} | {b_pct:.1f}% | {a} | {a_pct:.1f}% | {sign}{change:.1f}%p |")

            return lines

        report.extend(format_segment_table(segments.generation_before, segments.generation_after, "3.1 세대별 분포"))
        report.append("")
        report.extend(format_segment_table(segments.gender_before, segments.gender_after, "3.2 성별 분포"))
        report.append("")
        report.extend(format_segment_table(segments.lifestyle_before, segments.lifestyle_after, "3.3 라이프스타일별 분포"))
        report.append("")
        report.extend(format_segment_table(segments.residence_before, segments.residence_after, "3.4 거주유형별 분포 (상주/유동)"))
        report.append("")
        report.extend(format_segment_table(segments.age_group_before, segments.age_group_after, "3.5 연령대별 분포"))

        report.append("\n---\n")

        # 4. 의사결정 이유 심층 비교
        report.append("## 4. 의사결정 이유 심층 비교\n")

        report.append("### 4.1 주요 키워드 변화\n")
        report.append("**Before 키워드 TOP 10:**")
        before_kw = list(reasoning.keywords_before.items())[:10]
        if before_kw:
            report.append(" | ".join([f"`{kw}` ({cnt})" for kw, cnt in before_kw]))
        else:
            report.append("*데이터 없음*")

        report.append("\n**After 키워드 TOP 10:**")
        after_kw = list(reasoning.keywords_after.items())[:10]
        if after_kw:
            report.append(" | ".join([f"`{kw}` ({cnt})" for kw, cnt in after_kw]))
        else:
            report.append("*데이터 없음*")

        # 새로 등장한 키워드
        new_keywords = set(reasoning.keywords_after.keys()) - set(reasoning.keywords_before.keys())
        if new_keywords:
            report.append(f"\n**새로 등장한 키워드:** {', '.join([f'`{kw}`' for kw in list(new_keywords)[:10]])}")

        # 사라진 키워드
        lost_keywords = set(reasoning.keywords_before.keys()) - set(reasoning.keywords_after.keys())
        if lost_keywords:
            report.append(f"\n**사라진 키워드:** {', '.join([f'`{kw}`' for kw in list(lost_keywords)[:10]])}")

        report.append("\n### 4.2 고객 리뷰 감성 변화\n")
        sentiment_emoji = {"positive": "📈 긍정적", "negative": "📉 부정적", "neutral": "➡️ 중립"}
        report.append(f"**전반적 감성 변화:** {sentiment_emoji.get(reasoning.sentiment_shift, '알 수 없음')}\n")

        report.append("#### Before 주요 긍정 의견:\n")
        for reason in reasoning.positive_reasons_before[:5]:
            report.append(f"- \"{reason[:100]}...\"" if len(reason) > 100 else f"- \"{reason}\"")

        report.append("\n#### After 주요 긍정 의견:\n")
        for reason in reasoning.positive_reasons_after[:5]:
            report.append(f"- \"{reason[:100]}...\"" if len(reason) > 100 else f"- \"{reason}\"")

        report.append("\n#### Before 주요 부정 의견:\n")
        if reasoning.negative_reasons_before:
            for reason in reasoning.negative_reasons_before[:5]:
                report.append(f"- \"{reason[:100]}...\"" if len(reason) > 100 else f"- \"{reason}\"")
        else:
            report.append("*없음*")

        report.append("\n#### After 주요 부정 의견:\n")
        if reasoning.negative_reasons_after:
            for reason in reasoning.negative_reasons_after[:5]:
                report.append(f"- \"{reason[:100]}...\"" if len(reason) > 100 else f"- \"{reason}\"")
        else:
            report.append("*없음*")

        report.append("\n---\n")

        # 5. 전략 유효성 검증
        report.append("## 5. 전략 유효성 검증\n")

        if self.applied_strategies:
            report.append("### 5.1 적용된 전략\n")
            for i, strategy in enumerate(self.applied_strategies, 1):
                report.append(f"{i}. **{strategy.get('title', 'Unknown')}**")
                if strategy.get('goal'):
                    report.append(f"   - 목표: {strategy.get('goal')}")

        report.append("\n### 5.2 종합 의견\n")

        # 자동 분석 결과 요약
        insights = []

        # 방문객 변화 분석
        if len(inflow.new_visitors) > len(inflow.churned_visitors):
            insights.append(f"전략 적용 후 신규 고객 {len(inflow.new_visitors)}명이 유입되었으며, 이탈 고객({len(inflow.churned_visitors)}명)보다 많아 **순유입 증가** 효과가 있었습니다.")

        # 세그먼트 변화 분석
        gen_changes = []
        for gen in ["Z세대", "Y세대(밀레니얼)", "X세대", "S세대(시니어)"]:
            before = segments.generation_before.get(gen, 0)
            after = segments.generation_after.get(gen, 0)
            if after > before:
                gen_changes.append(f"{gen} +{after - before}명")
        if gen_changes:
            insights.append(f"세대별로는 {', '.join(gen_changes)}의 유입 증가가 관찰되었습니다.")

        # 감성 변화 분석
        if reasoning.sentiment_shift == "positive":
            insights.append("고객 리뷰의 전반적인 감성이 **긍정적으로 개선**되었습니다.")
        elif reasoning.sentiment_shift == "negative":
            insights.append("주의: 고객 리뷰의 감성이 **부정적으로 변화**되었습니다. 전략 재검토가 필요합니다.")

        # 전환 고객 분석
        if inflow.converted_visitors:
            report.append(f"경쟁점에서 {len(inflow.converted_visitors)}명의 고객이 이 매장으로 전환하였습니다.")

        if insights:
            for insight in insights:
                report.append(f"- {insight}")
        else:
            report.append("- 충분한 데이터가 수집되면 더 상세한 분석이 가능합니다.")

        report.append("\n---\n")
        report.append(f"\n*이 리포트는 StoreAnalyzer v1.0에 의해 자동 생성되었습니다.*")

        return "\n".join(report)

    def save_report(self, output_path: str):
        """리포트를 파일로 저장"""
        report = self.generate_report()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"리포트 저장 완료: {output_path}")


# ============================================================================
# 헬퍼 함수들
# ============================================================================

def load_simulation_result(json_path: str, store_name: str) -> SimulationResult:
    """JSON 파일에서 시뮬레이션 결과 로드"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    visits = []
    for rating in data.get("agent_ratings", []):
        visits.append(AgentVisitRecord(
            agent_id=rating.get("agent_id", 0),
            agent_name=rating.get("agent_name", "Unknown"),
            visit_datetime=rating.get("visit_datetime", ""),
            rating=rating.get("rating", 0),
            selected_tags=rating.get("selected_tags", []),
            comment=rating.get("comment", ""),
            reason=rating.get("reason", ""),
        ))

    return SimulationResult(
        store_name=store_name,
        visits=visits,
        star_rating=data.get("별점", 0.0),
        total_visits=len(visits),
        taste_count=data.get("맛", 0),
        value_count=data.get("가성비", 0),
        atmosphere_count=data.get("분위기", 0),
        service_count=data.get("서비스", 0),
        agent_review_summary=data.get("agent_review", {}).get("summary", ""),
    )


def load_agent_personas(personas_json_path: str) -> Dict[int, AgentPersona]:
    """에이전트 페르소나 데이터 로드"""
    with open(personas_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    personas = {}
    for agent_data in data:
        persona = AgentPersona.from_dict(agent_data)
        personas[persona.agent_id] = persona

    return personas


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    # 예시 데이터로 테스트
    print("StoreAnalyzer 모듈 로드 완료")
    print("사용법:")
    print("""
    from store_analyzer import StoreAnalyzer, SimulationResult, AgentPersona

    # 시뮬레이션 결과 준비
    baseline = SimulationResult(...)
    strategy = SimulationResult(...)

    # 분석기 생성
    analyzer = StoreAnalyzer(
        target_store="스몰굿커피 망원역점",
        baseline_result=baseline,
        strategy_result=strategy,
        agent_personas=personas_dict,
        all_store_visits_before=visits_before,
        all_store_visits_after=visits_after,
    )

    # 리포트 생성 및 저장
    analyzer.save_report("target_store_analysis.md")
    """)
