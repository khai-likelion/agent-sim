"""
A/B Comparison module for X-Report analysis.

Compares simulation results with and without business reports (X-Reports)
to measure the impact of promotions on consumer behavior.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class ComparisonMetrics:
    """Metrics comparing two simulation runs (A=with reports, B=without reports)."""

    # Overall conversion
    conversion_rate_a: float = 0.0
    conversion_rate_b: float = 0.0
    conversion_delta: float = 0.0
    conversion_pct_change: float = 0.0

    # Visit counts
    total_visits_a: int = 0
    total_visits_b: int = 0
    visit_delta: int = 0

    # Active counts
    total_active_a: int = 0
    total_active_b: int = 0

    # Category distribution
    category_distribution_a: Dict[str, int] = field(default_factory=dict)
    category_distribution_b: Dict[str, int] = field(default_factory=dict)
    category_delta: Dict[str, int] = field(default_factory=dict)

    # Report target store visits
    report_store_visits_a: Dict[str, int] = field(default_factory=dict)
    report_store_visits_b: Dict[str, int] = field(default_factory=dict)
    report_store_delta: Dict[str, int] = field(default_factory=dict)

    # Segment-level conversion
    segment_conversion_a: Dict[str, float] = field(default_factory=dict)
    segment_conversion_b: Dict[str, float] = field(default_factory=dict)
    segment_conversion_delta: Dict[str, float] = field(default_factory=dict)

    # Information diffusion metrics
    report_reception_rate: float = 0.0
    report_influenced_rate: float = 0.0

    # Time-slot breakdown
    timeslot_visits_a: Dict[str, int] = field(default_factory=dict)
    timeslot_visits_b: Dict[str, int] = field(default_factory=dict)

    # Discovery channel breakdown (A only, since reports may shift channels)
    discovery_channel_a: Dict[str, int] = field(default_factory=dict)
    discovery_channel_b: Dict[str, int] = field(default_factory=dict)


class ABComparator:
    """Compares two simulation DataFrames (A=with reports, B=without reports)."""

    def __init__(self, report_store_names: Optional[List[str]] = None):
        self.report_store_names = report_store_names or []

    def compute(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> ComparisonMetrics:
        """Compute comparison metrics between two simulation runs."""
        metrics = ComparisonMetrics()

        # Filter active events
        active_a = df_a[df_a["is_active"] == True]
        active_b = df_b[df_b["is_active"] == True]
        visits_a = df_a[df_a["decision"] == "visit"]
        visits_b = df_b[df_b["decision"] == "visit"]

        # Overall conversion
        metrics.total_active_a = len(active_a)
        metrics.total_active_b = len(active_b)
        metrics.total_visits_a = len(visits_a)
        metrics.total_visits_b = len(visits_b)
        metrics.visit_delta = metrics.total_visits_a - metrics.total_visits_b

        metrics.conversion_rate_a = (
            metrics.total_visits_a / metrics.total_active_a
            if metrics.total_active_a > 0 else 0.0
        )
        metrics.conversion_rate_b = (
            metrics.total_visits_b / metrics.total_active_b
            if metrics.total_active_b > 0 else 0.0
        )
        metrics.conversion_delta = metrics.conversion_rate_a - metrics.conversion_rate_b
        metrics.conversion_pct_change = (
            (metrics.conversion_delta / metrics.conversion_rate_b * 100)
            if metrics.conversion_rate_b > 0 else 0.0
        )

        # Category distribution
        metrics.category_distribution_a = (
            visits_a["visited_category"].value_counts().to_dict()
            if len(visits_a) > 0 else {}
        )
        metrics.category_distribution_b = (
            visits_b["visited_category"].value_counts().to_dict()
            if len(visits_b) > 0 else {}
        )
        all_cats = set(list(metrics.category_distribution_a.keys()) + list(metrics.category_distribution_b.keys()))
        metrics.category_delta = {
            cat: metrics.category_distribution_a.get(cat, 0) - metrics.category_distribution_b.get(cat, 0)
            for cat in all_cats
        }

        # Report target store visits
        if self.report_store_names:
            for store_name in self.report_store_names:
                count_a = len(visits_a[visits_a["visited_store"] == store_name])
                count_b = len(visits_b[visits_b["visited_store"] == store_name])
                metrics.report_store_visits_a[store_name] = count_a
                metrics.report_store_visits_b[store_name] = count_b
                metrics.report_store_delta[store_name] = count_a - count_b

        # Segment-level conversion
        self._compute_segment_metrics(active_a, visits_a, active_b, visits_b, metrics)

        # Information diffusion (A only - has reports)
        if "report_received" in df_a.columns:
            report_received = active_a["report_received"].notna().sum()
            metrics.report_reception_rate = (
                report_received / len(active_a)
                if len(active_a) > 0 else 0.0
            )
            # Influenced = received report AND visited a report target store
            if self.report_store_names:
                received_and_visited = visits_a[
                    (visits_a["report_received"].notna()) &
                    (visits_a["visited_store"].isin(self.report_store_names))
                ]
                metrics.report_influenced_rate = (
                    len(received_and_visited) / report_received
                    if report_received > 0 else 0.0
                )

        # Time-slot breakdown
        metrics.timeslot_visits_a = visits_a["time_slot"].value_counts().to_dict() if len(visits_a) > 0 else {}
        metrics.timeslot_visits_b = visits_b["time_slot"].value_counts().to_dict() if len(visits_b) > 0 else {}

        # Discovery channel breakdown
        if "discovery_channel" in visits_a.columns:
            metrics.discovery_channel_a = (
                visits_a["discovery_channel"].dropna().value_counts().to_dict()
            )
        if "discovery_channel" in visits_b.columns:
            metrics.discovery_channel_b = (
                visits_b["discovery_channel"].dropna().value_counts().to_dict()
            )

        return metrics

    def _compute_segment_metrics(
        self,
        active_a: pd.DataFrame,
        visits_a: pd.DataFrame,
        active_b: pd.DataFrame,
        visits_b: pd.DataFrame,
        metrics: ComparisonMetrics,
    ) -> None:
        """Compute segment-level conversion rates."""
        # Try to get segment from age_group as proxy if segment column not present
        segment_col = "age_group"
        for col in ["segment", "age_group"]:
            if col in active_a.columns:
                segment_col = col
                break

        for segment in active_a[segment_col].unique():
            seg_active_a = len(active_a[active_a[segment_col] == segment])
            seg_visits_a = len(visits_a[visits_a[segment_col] == segment])
            conv_a = seg_visits_a / seg_active_a if seg_active_a > 0 else 0.0
            metrics.segment_conversion_a[segment] = round(conv_a, 4)

            seg_active_b = len(active_b[active_b[segment_col] == segment])
            seg_visits_b = len(visits_b[visits_b[segment_col] == segment])
            conv_b = seg_visits_b / seg_active_b if seg_active_b > 0 else 0.0
            metrics.segment_conversion_b[segment] = round(conv_b, 4)

            metrics.segment_conversion_delta[segment] = round(conv_a - conv_b, 4)

    def generate_summary_text(self, metrics: ComparisonMetrics) -> str:
        """Generate markdown summary of A/B comparison."""
        lines = []
        lines.append("# X-Report A/B 비교 분석 결과\n")

        lines.append("## 전체 전환율")
        lines.append(f"- **A (리포트 有)**: {metrics.conversion_rate_a:.2%} ({metrics.total_visits_a}회 방문 / {metrics.total_active_a}회 활동)")
        lines.append(f"- **B (리포트 無)**: {metrics.conversion_rate_b:.2%} ({metrics.total_visits_b}회 방문 / {metrics.total_active_b}회 활동)")
        lines.append(f"- **변화**: {metrics.conversion_delta:+.2%} ({metrics.conversion_pct_change:+.1f}%)")
        lines.append(f"- **방문수 차이**: {metrics.visit_delta:+d}\n")

        if metrics.report_store_visits_a:
            lines.append("## 리포트 대상 매장 방문")
            for store_name in metrics.report_store_visits_a:
                a = metrics.report_store_visits_a[store_name]
                b = metrics.report_store_visits_b.get(store_name, 0)
                delta = metrics.report_store_delta.get(store_name, 0)
                lines.append(f"- **{store_name}**: A={a}회, B={b}회 ({delta:+d})")
            lines.append("")

        lines.append("## 정보 확산")
        lines.append(f"- 리포트 수신율: {metrics.report_reception_rate:.1%}")
        lines.append(f"- 리포트 영향 전환율: {metrics.report_influenced_rate:.1%}\n")

        if metrics.segment_conversion_delta:
            lines.append("## 세그먼트별 전환율 변화")
            sorted_segments = sorted(
                metrics.segment_conversion_delta.items(),
                key=lambda x: abs(x[1]), reverse=True
            )
            for seg, delta in sorted_segments:
                a = metrics.segment_conversion_a.get(seg, 0)
                b = metrics.segment_conversion_b.get(seg, 0)
                lines.append(f"- **{seg}**: A={a:.2%}, B={b:.2%} ({delta:+.2%})")
            lines.append("")

        if metrics.category_delta:
            lines.append("## 카테고리별 방문 변화")
            sorted_cats = sorted(
                metrics.category_delta.items(),
                key=lambda x: abs(x[1]), reverse=True
            )
            for cat, delta in sorted_cats[:10]:
                a = metrics.category_distribution_a.get(cat, 0)
                b = metrics.category_distribution_b.get(cat, 0)
                lines.append(f"- **{cat}**: A={a}회, B={b}회 ({delta:+d})")

        return "\n".join(lines)

    def to_dict(self, metrics: ComparisonMetrics) -> dict:
        """Convert metrics to JSON-serializable dict."""
        return asdict(metrics)
