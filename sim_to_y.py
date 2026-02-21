"""
전략 전(Sim 1) vs 후(Sim 2) 비교 분석 보고서 생성기

ComparisonReportGenerator 클래스 기반 11가지 핵심 지표 분석:
  1. 기본 방문 지표 (Overview) + 키워드 워드클라우드
  2. 평점 분포 분석 (Rating Spread)
  3. 시간대별 손님 변화 (Hourly Traffic)
  4. 세대별 증감 분석 (Generation Impact) — 혼합 세대 포함 6버킷
  5. 방문 목적별 상세 분석 (Purpose Analysis) ⭐
  6. 재방문율 분석 (Retention)
  7. 경쟁 매장 비교 (Radar Chart)
  8. 종합 평가 (LLM Summary)
  9. 에이전트 유형별 분석 (Agent Type: 상주 vs 유동)
 10. 성별 구성 분석 (Gender Composition)
 11. 세대 × 방문목적 크로스탭 히트맵 (Crosstab Heatmap)

사용법:
    python sim_to_y.py --target-store 돼지야
    python sim_to_y.py --target-store 돼지야 \\
        --before-dir data/output/돼지야_before \\
        --after-dir  data/output/돼지야_after \\
        --output-dir reports
"""

import os
import sys
import json
import re
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# 프로젝트 루트 설정
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────
# 한글 폰트 설정 (Windows / macOS / Linux 자동 감지)
# ──────────────────────────────────────────────────────────────
def setup_korean_font() -> str:
    """시스템에 맞는 한글 폰트를 설정하고 font_path를 반환."""
    candidates = [
        # Windows
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/malgunbd.ttf",
        # macOS
        "/Library/Fonts/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        # Linux (NanumGothic)
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/nanum/NanumGothic.ttf",
    ]

    font_path = None
    for path in candidates:
        if os.path.exists(path):
            font_path = path
            break

    if font_path:
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        rcParams["font.family"] = prop.get_name()
    else:
        for f in fm.fontManager.ttflist:
            if any(kw in f.name for kw in ["Gothic", "Nanum", "Malgun", "Apple", "CJK"]):
                rcParams["font.family"] = f.name
                font_path = f.fname
                break

    rcParams["axes.unicode_minus"] = False
    return font_path or ""


FONT_PATH = setup_korean_font()


# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────
PURPOSE_KEYWORDS = ["생활베이스형", "사적모임형", "공적모임형", "가족모임형"]
PURPOSE_LABELS   = ["생활베이스", "사적모임", "공적모임", "가족모임"]
PURPOSE_COLORS   = ["#76B7B2", "#59A14F", "#EDC948", "#B07AA1"]

# 혼합 세대: special_condition의 S/Z1 포함 유무로 4가지 구분
GENERATION_ORDER = [
    "Z1", "Z2", "Y", "X", "S",
    "혼합(S, Z1)",
    "혼합(S, Z1미포함)",
    "혼합(S미포함, Z1)",
    "혼합(S미포함, Z1미포함)",
]
GEN_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#B07AA1", "#9C755F", "#BAB0AC", "#D37295",
]

TIME_SLOT_MAP = {"아침": 7, "점심": 12, "저녁": 18, "야식": 22}

COLOR_SIM1 = "#4E79A7"   # 파랑 (전략 전)
COLOR_SIM2 = "#F28E2B"   # 주황 (전략 후)


# ──────────────────────────────────────────────────────────────
# ComparisonReportGenerator
# ──────────────────────────────────────────────────────────────
class ComparisonReportGenerator:
    """
    전략 전(Sim 1) vs 후(Sim 2) 비교 분석 보고서 생성기.

    데이터 소스:
        {before_dir}/visit_log.csv        — 전략 전 방문 로그
        {after_dir}/visit_log.csv         — 전략 후 방문 로그
        {before_dir}/simulation_result.csv — (visit_log 없을 때 fallback)
        {before_dir}/store_ratings.json   — 전략 전 매장 평점
        {after_dir}/store_ratings.json    — 전략 후 매장 평점
        {before_dir}/agents_final.json    — 전략 전 에이전트 최종 상태
        {after_dir}/agents_final.json     — 전략 후 에이전트 최종 상태
    """

    def __init__(
        self,
        target_store: str,
        before_dir: Path,
        after_dir: Path,
        output_dir: Path,
        openai_api_key: Optional[str] = None,
    ):
        self.target_store   = target_store
        self.before_dir     = Path(before_dir)
        self.after_dir      = Path(after_dir)
        self.output_dir     = Path(output_dir)
        self.figures_dir    = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")

        # 데이터 홀더
        self.df1: pd.DataFrame = pd.DataFrame()
        self.df2: pd.DataFrame = pd.DataFrame()
        self.target1: pd.DataFrame = pd.DataFrame()
        self.target2: pd.DataFrame = pd.DataFrame()
        self.agents1: List[Dict] = []
        self.agents2: List[Dict] = []
        self.store_ratings1: Dict = {}
        self.store_ratings2: Dict = {}

        # 분석 결과 캐시 (LLM 요약용)
        self.metrics_summary: Dict[str, Any] = {}

        self._load_data()

    # ──────────────────────────────────────────────────────────────
    # 데이터 로드 및 전처리
    # ──────────────────────────────────────────────────────────────

    def _load_data(self):
        """모든 데이터 소스 로드."""
        self.df1, self.target1 = self._load_sim_data(self.before_dir, "Sim1")
        self.df2, self.target2 = self._load_sim_data(self.after_dir,  "Sim2")
        self.agents1 = self._load_json_list(self.before_dir / "agents_final.json")
        self.agents2 = self._load_json_list(self.after_dir  / "agents_final.json")
        self.store_ratings1 = self._load_json_dict(self.before_dir / "store_ratings.json")
        self.store_ratings2 = self._load_json_dict(self.after_dir  / "store_ratings.json")

        print(f"[데이터 로드 완료]")
        print(f"  Sim1: 전체 방문 {len(self.df1):,}건 / '{self.target_store}' {len(self.target1):,}건")
        print(f"  Sim2: 전체 방문 {len(self.df2):,}건 / '{self.target_store}' {len(self.target2):,}건")

    def _load_sim_data(
        self, data_dir: Path, label: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """visit_log.csv (또는 simulation_result.csv) 로드 후 전처리."""
        _EMPTY_COLS = [
            "timestamp", "agent_id", "persona_id", "generation",
            "gender_composition", "segment", "special_condition",
            "weekday", "time_slot",
            "decision", "visited_store", "visited_category",
            "taste_rating", "value_rating", "atmosphere_rating",
            "reason", "comment",
        ]

        visit_path = data_dir / "visit_log.csv"
        sim_path   = data_dir / "simulation_result.csv"

        if visit_path.exists():
            df = pd.read_csv(visit_path, encoding="utf-8-sig")
        elif sim_path.exists():
            raw = pd.read_csv(sim_path, encoding="utf-8-sig")
            df = raw[raw["decision"] == "visit"].copy()
        else:
            print(f"  [경고] {label}: 데이터 파일 없음 ({data_dir})")
            df = pd.DataFrame(columns=_EMPTY_COLS)

        # generation 혼합 세분화:
        # special_condition의 S/Z1 포함 유무로 4가지 레이블 변환
        # 예) generation="혼합(Z2+Y)", special_condition="S포함_Z1미포함" → "혼합(S, Z1미포함)"
        if "generation" in df.columns:
            sc_series = df["special_condition"] if "special_condition" in df.columns else pd.Series("", index=df.index)
            df["generation"] = [
                self._resolve_generation_label(g, sc)
                for g, sc in zip(df["generation"], sc_series)
            ]

        # persona_type: segment 문자열에서 4대 방문 목적 추출
        df["persona_type"] = df.get("segment", pd.Series(dtype=str)).apply(
            self._extract_persona_type
        )

        # agent_type: segment 문자열에서 "유동" / "상주" 추출
        df["agent_type"] = df.get("segment", pd.Series(dtype=str)).apply(
            self._extract_agent_type
        )

        # 평점 컬럼 숫자 변환
        for col in ["taste_rating", "value_rating", "atmosphere_rating"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 타임스탬프 파싱
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["hour"] = df["timestamp"].dt.hour
        else:
            df["hour"] = np.nan

        # 타겟 매장 필터
        target_df = (
            df[df["visited_store"] == self.target_store].copy()
            if "visited_store" in df.columns
            else pd.DataFrame(columns=df.columns)
        )

        return df, target_df

    @staticmethod
    def _extract_persona_type(segment: str) -> str:
        """'유동_생활베이스형_1인' → '생활베이스형'"""
        if not isinstance(segment, str):
            return "기타"
        for kw in PURPOSE_KEYWORDS:
            if kw in segment:
                return kw
        return "기타"

    @staticmethod
    def _extract_agent_type(segment: str) -> str:
        """'유동_사적모임형_2인' → '유동' / '상주_생활베이스형_1인' → '상주'"""
        if not isinstance(segment, str) or "_" not in segment:
            return "기타"
        first = segment.split("_")[0]
        return first if first in ("유동", "상주") else "기타"

    @staticmethod
    def _resolve_generation_label(generation: str, special_condition: str) -> str:
        """
        혼합 세대를 special_condition 기반으로 4가지 레이블로 세분화.
        non-혼합 세대(Z1/Z2/Y/X/S)는 그대로 반환.

        예)
          generation="혼합(Z2+Y)", special_condition="S포함_Z1미포함" → "혼합(S, Z1미포함)"
          generation="혼합",       special_condition="S미포함_Z1포함" → "혼합(S미포함, Z1)"
          generation="Z1"                                            → "Z1"
        """
        if not isinstance(generation, str) or not generation.startswith("혼합"):
            return generation
        sc = "" if pd.isna(special_condition) else str(special_condition)
        s_label  = "S"       if "S포함"  in sc else "S미포함"
        z1_label = "Z1"      if "Z1포함" in sc else "Z1미포함"
        return f"혼합({s_label}, {z1_label})"

    @staticmethod
    def _load_json_list(path: Path) -> List[Dict]:
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_json_dict(path: Path) -> Dict:
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # ──────────────────────────────────────────────────────────────
    # 공통 유틸
    # ──────────────────────────────────────────────────────────────

    def _savefig(self, name: str) -> Path:
        path = self.figures_dir / f"{name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        return path

    @staticmethod
    def _composite_rating(df: pd.DataFrame) -> pd.Series:
        """맛/가성비/분위기 3개 평점의 행별 평균."""
        cols = [c for c in ["taste_rating", "value_rating", "atmosphere_rating"] if c in df.columns]
        if not cols:
            return pd.Series(dtype=float)
        return df[cols].mean(axis=1).dropna()

    # ──────────────────────────────────────────────────────────────
    # 1. 기본 방문 지표 (Overview) + 워드클라우드
    # ──────────────────────────────────────────────────────────────

    def analysis_1_overview(self) -> Dict:
        """방문 수, 시장 점유율, 키워드 워드클라우드 비교."""
        visits1 = len(self.target1)
        visits2 = len(self.target2)
        total1  = max(len(self.df1), 1)
        total2  = max(len(self.df2), 1)
        share1  = visits1 / total1 * 100
        share2  = visits2 / total2 * 100

        self._plot_wordcloud(self.target1, self.target2)

        result = {
            "visits_before":       visits1,
            "visits_after":        visits2,
            "visit_change_pct":    (visits2 - visits1) / max(visits1, 1) * 100,
            "market_share_before": share1,
            "market_share_after":  share2,
        }
        self.metrics_summary["overview"] = result
        print(f"  [1] Overview: {visits1} → {visits2}건 ({result['visit_change_pct']:+.1f}%)")
        return result

    def _plot_wordcloud(self, t1: pd.DataFrame, t2: pd.DataFrame):
        """comment(우선) → reason(fallback) + visited_category 텍스트로 워드클라우드 생성."""
        try:
            from wordcloud import WordCloud
        except ImportError:
            print("  [경고] wordcloud 미설치 → 워드클라우드 스킵 (pip install wordcloud)")
            return

        def build_text(df: pd.DataFrame) -> str:
            parts: List[str] = []
            # comment 우선 (Step4 실제 리뷰), 없으면 reason(방문 결정 이유) fallback
            if "comment" in df.columns and df["comment"].dropna().astype(str).str.strip().any():
                parts.extend(df["comment"].dropna().astype(str).tolist())
            elif "reason" in df.columns:
                parts.extend(df["reason"].dropna().astype(str).tolist())
            # visited_category는 항상 추가 (업종 키워드 보강)
            if "visited_category" in df.columns:
                parts.extend(df["visited_category"].dropna().astype(str).tolist())
            return " ".join(parts)

        texts  = [build_text(t1), build_text(t2)]
        labels = ["Sim 1 (전략 전)", "Sim 2 (전략 후)"]
        cmaps  = ["Blues", "Oranges"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"[{self.target_store}] 방문 키워드 워드클라우드 비교",
            fontsize=14, fontweight="bold"
        )

        for ax, text, label, cmap in zip(axes, texts, labels, cmaps):
            ax.set_title(label, fontsize=12, fontweight="bold")
            ax.axis("off")
            if not text.strip():
                ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center",
                        transform=ax.transAxes, fontsize=14)
                continue
            wc = WordCloud(
                font_path=FONT_PATH if FONT_PATH else None,
                background_color="white",
                width=600, height=380,
                colormap=cmap,
                max_words=80,
                min_font_size=10,
                collocations=False,
            ).generate(text)
            ax.imshow(wc, interpolation="bilinear")

        plt.tight_layout()
        self._savefig("01_wordcloud")

    # ──────────────────────────────────────────────────────────────
    # 2. 평점 분포 분석 (Rating Spread)
    # ──────────────────────────────────────────────────────────────

    def analysis_2_rating_spread(self) -> Dict:
        """맛/가성비/분위기 KDE 분포 + 만족도(4점 이상) 비율."""
        r1 = self._composite_rating(self.target1)
        r2 = self._composite_rating(self.target2)
        sat1 = (r1 >= 4).mean() * 100 if len(r1) > 0 else 0.0
        sat2 = (r2 >= 4).mean() * 100 if len(r2) > 0 else 0.0

        # ── KDE 서브플롯 (맛 / 가성비 / 분위기) ──
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            f"[{self.target_store}] 평점 분포 분석 (전략 전 vs 후)",
            fontsize=14, fontweight="bold"
        )

        rating_cols = [
            ("taste_rating",      "맛 평점"),
            ("value_rating",      "가성비 평점"),
            ("atmosphere_rating", "분위기 평점"),
        ]
        x_range = np.linspace(0.5, 5.5, 200)

        for ax, (col, col_label) in zip(axes, rating_cols):
            for df, color, label in [
                (self.target1, COLOR_SIM1, "Sim1 전략 전"),
                (self.target2, COLOR_SIM2, "Sim2 전략 후"),
            ]:
                s = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
                if len(s) >= 2:
                    kde = gaussian_kde(s, bw_method=0.5)
                    y   = kde(x_range)
                    ax.plot(x_range, y, color=color, lw=2.5, label=label)
                    ax.fill_between(x_range, y, alpha=0.15, color=color)
                elif len(s) == 1:
                    ax.axvline(s.iloc[0], color=color, lw=2.5, label=label)

            ax.set_xlim(1, 5)
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xlabel("점수", fontsize=10)
            ax.set_ylabel("밀도", fontsize=10)
            ax.set_title(col_label, fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        self._savefig("02_rating_spread")

        # ── 만족도 비율 막대 ──
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        bars = ax2.bar(
            ["Sim1\n(전략 전)", "Sim2\n(전략 후)"],
            [sat1, sat2],
            color=[COLOR_SIM1, COLOR_SIM2],
            width=0.5, edgecolor="white", linewidth=1.5,
        )
        for bar, val in zip(bars, [sat1, sat2]):
            ax2.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold",
            )
        ax2.set_title(
            f"[{self.target_store}] 만족도(4점 이상) 비율",
            fontsize=13, fontweight="bold",
        )
        ax2.set_ylabel("비율 (%)")
        ax2.set_ylim(0, 110)
        ax2.axhline(50, color="gray", linestyle="--", alpha=0.4, label="50% 기준선")
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self._savefig("02b_satisfaction_rate")

        result = {
            "avg_rating_before":       float(r1.mean()) if len(r1) > 0 else 0.0,
            "avg_rating_after":        float(r2.mean()) if len(r2) > 0 else 0.0,
            "satisfaction_rate_before": sat1,
            "satisfaction_rate_after":  sat2,
        }
        self.metrics_summary["rating"] = result
        print(f"  [2] Rating: 평균 {result['avg_rating_before']:.2f} → {result['avg_rating_after']:.2f}"
              f" | 만족도 {sat1:.1f}% → {sat2:.1f}%")
        return result

    # ──────────────────────────────────────────────────────────────
    # 3. 시간대별 손님 변화 (Hourly Traffic)
    # ──────────────────────────────────────────────────────────────

    def analysis_3_hourly_traffic(self) -> Dict:
        """타임슬롯(아침/점심/저녁/야식)별 방문 꺾은선 그래프 비교."""

        def get_traffic(df: pd.DataFrame) -> Dict[int, int]:
            if "time_slot" in df.columns:
                counts = df["time_slot"].value_counts()
                return {TIME_SLOT_MAP[k]: int(counts.get(k, 0)) for k in TIME_SLOT_MAP}
            if "hour" in df.columns:
                counts = df["hour"].value_counts().to_dict()
                return {int(k): int(v) for k, v in counts.items()}
            return {}

        traffic1 = get_traffic(self.target1)
        traffic2 = get_traffic(self.target2)
        hours    = sorted(set(traffic1) | set(traffic2))
        slot_labels = {7: "아침(07)", 12: "점심(12)", 18: "저녁(18)", 22: "야식(22)"}

        fig, ax = plt.subplots(figsize=(10, 5))
        v1 = [traffic1.get(h, 0) for h in hours]
        v2 = [traffic2.get(h, 0) for h in hours]

        ax.plot(hours, v1, marker="o", ms=9, lw=2.5, color=COLOR_SIM1, label="Sim1 (전략 전)")
        ax.plot(hours, v2, marker="s", ms=9, lw=2.5, color=COLOR_SIM2, label="Sim2 (전략 후)")
        ax.fill_between(hours, v1, v2, alpha=0.1, color="purple")

        for h, y1, y2 in zip(hours, v1, v2):
            if y1 > 0:
                ax.annotate(str(y1), (h, y1), textcoords="offset points",
                            xytext=(-12, 6), fontsize=8, color=COLOR_SIM1)
            if y2 > 0:
                ax.annotate(str(y2), (h, y2), textcoords="offset points",
                            xytext=(4, 6), fontsize=8, color=COLOR_SIM2)

        ax.set_xticks(list(slot_labels.keys()))
        ax.set_xticklabels(list(slot_labels.values()), fontsize=11)
        ax.set_xlabel("시간대", fontsize=11)
        ax.set_ylabel("방문 횟수", fontsize=11)
        ax.set_title(f"[{self.target_store}] 시간대별 방문 트래픽 변화", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        self._savefig("03_hourly_traffic")

        peak1 = max(traffic1, key=traffic1.get) if traffic1 else None
        peak2 = max(traffic2, key=traffic2.get) if traffic2 else None
        result = {
            "peak_slot_before": slot_labels.get(peak1, "?"),
            "peak_slot_after":  slot_labels.get(peak2, "?"),
            "traffic_before":   traffic1,
            "traffic_after":    traffic2,
        }
        self.metrics_summary["traffic"] = result
        print(f"  [3] Hourly Traffic: 피크 {result['peak_slot_before']} → {result['peak_slot_after']}")
        return result

    # ──────────────────────────────────────────────────────────────
    # 4. 세대별 증감 분석 (Generation Impact) — 혼합 세대 포함 6버킷
    # ──────────────────────────────────────────────────────────────

    def analysis_4_generation_impact(self) -> Dict:
        """Z1, Z2, Y, X, S, 혼합 세대별 방문 비율 변화 바 차트."""

        def gen_ratio(df: pd.DataFrame) -> Dict[str, float]:
            total = max(len(df), 1)
            if "generation" not in df.columns:
                return {g: 0.0 for g in GENERATION_ORDER}
            counts = df["generation"].value_counts()
            return {g: counts.get(g, 0) / total * 100 for g in GENERATION_ORDER}

        r1 = gen_ratio(self.target1)
        r2 = gen_ratio(self.target2)

        x     = np.arange(len(GENERATION_ORDER))
        width = 0.35

        fig, ax = plt.subplots(figsize=(13, 6))
        bars1 = ax.bar(x - width / 2, [r1[g] for g in GENERATION_ORDER], width,
                       label="Sim1 (전략 전)", color=COLOR_SIM1, alpha=0.85)
        bars2 = ax.bar(x + width / 2, [r2[g] for g in GENERATION_ORDER], width,
                       label="Sim2 (전략 후)", color=COLOR_SIM2, alpha=0.85)

        for bar in bars1:
            v = bar.get_height()
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=8, color=COLOR_SIM1)
        for bar in bars2:
            v = bar.get_height()
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=8, color=COLOR_SIM2)

        for i, g in enumerate(GENERATION_ORDER):
            diff   = r2[g] - r1[g]
            y_pos  = max(r1[g], r2[g]) + 2.5
            color  = "#27AE60" if diff > 0 else ("#E74C3C" if diff < 0 else "gray")
            sign   = "+" if diff > 0 else ""
            ax.text(x[i], y_pos, f"{sign}{diff:.1f}%p",
                    ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(GENERATION_ORDER, fontsize=13)
        ax.set_ylabel("방문 비율 (%)", fontsize=11)
        ax.set_title(f"[{self.target_store}] 세대별 방문 비율 변화", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self._savefig("04_generation_impact")

        result = {"before": r1, "after": r2}
        self.metrics_summary["generation"] = result
        deltas = {g: r2[g] - r1[g] for g in GENERATION_ORDER}
        max_g  = max(deltas, key=lambda g: deltas[g])
        min_g  = min(deltas, key=lambda g: deltas[g])
        print(f"  [4] Generation: 최대 증가={max_g}({deltas[max_g]:+.1f}%p) / 최대 감소={min_g}({deltas[min_g]:+.1f}%p)")
        return result

    # ──────────────────────────────────────────────────────────────
    # 5. 방문 목적별 상세 분석 (Purpose Analysis) ⭐
    # ──────────────────────────────────────────────────────────────

    def analysis_5_purpose_analysis(self) -> Dict:
        """
        5-1. 방문 목적별 비중 변화 (100% Stacked Bar Chart)
        5-2. 유형별 평균 만족도 비교 (그룹 바 차트)
        """

        def purpose_ratio(df: pd.DataFrame) -> Dict[str, float]:
            total = max(len(df), 1)
            if "persona_type" not in df.columns:
                return {p: 0.0 for p in PURPOSE_KEYWORDS}
            counts = df["persona_type"].value_counts()
            return {p: counts.get(p, 0) / total * 100 for p in PURPOSE_KEYWORDS}

        def purpose_avg_rating(df: pd.DataFrame) -> Dict[str, float]:
            result = {}
            rating_cols = [c for c in ["taste_rating", "value_rating", "atmosphere_rating"] if c in df.columns]
            for p in PURPOSE_KEYWORDS:
                sub = df[df["persona_type"] == p] if "persona_type" in df.columns else pd.DataFrame()
                result[p] = float(sub[rating_cols].mean().mean()) if len(sub) > 0 and rating_cols else 0.0
            return result

        ratio1  = purpose_ratio(self.target1)
        ratio2  = purpose_ratio(self.target2)
        rating1 = purpose_avg_rating(self.target1)
        rating2 = purpose_avg_rating(self.target2)

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        fig.suptitle(f"[{self.target_store}] 방문 목적별 상세 분석", fontsize=14, fontweight="bold")

        # ── 5-1: 100% Stacked Bar ──
        ax = axes[0]
        x_labels = ["Sim1\n(전략 전)", "Sim2\n(전략 후)"]
        bottoms  = [0.0, 0.0]

        for purpose, color, label in zip(PURPOSE_KEYWORDS, PURPOSE_COLORS, PURPOSE_LABELS):
            vals = [ratio1[purpose], ratio2[purpose]]
            bars = ax.bar(x_labels, vals, bottom=bottoms,
                          color=color, label=label, alpha=0.9,
                          edgecolor="white", linewidth=1.2)
            for j, (bar, val) in enumerate(zip(bars, vals)):
                if val > 3:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottoms[j] + val / 2,
                        f"{val:.1f}%",
                        ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white",
                    )
            bottoms = [bottoms[j] + vals[j] for j in range(2)]

        ax.set_ylim(0, 108)
        ax.set_ylabel("비율 (%)", fontsize=11)
        ax.set_title("방문 목적 유형 비중 변화\n(100% Stacked Bar)", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
        ax.grid(axis="y", alpha=0.2)

        # ── 5-2: 유형별 만족도 비교 ──
        ax2   = axes[1]
        x     = np.arange(len(PURPOSE_KEYWORDS))
        width = 0.35

        bars1 = ax2.bar(x - width / 2, [rating1[p] for p in PURPOSE_KEYWORDS], width,
                        label="Sim1 (전략 전)", color=COLOR_SIM1, alpha=0.85)
        bars2 = ax2.bar(x + width / 2, [rating2[p] for p in PURPOSE_KEYWORDS], width,
                        label="Sim2 (전략 후)", color=COLOR_SIM2, alpha=0.85)

        for bar in bars1:
            v = bar.get_height()
            if v > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                         f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            v = bar.get_height()
            if v > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                         f"{v:.2f}", ha="center", va="bottom", fontsize=8)

        for i, p in enumerate(PURPOSE_KEYWORDS):
            diff  = rating2[p] - rating1[p]
            if diff != 0:
                y_pos = max(rating1[p], rating2[p]) + 0.15
                color = "#27AE60" if diff > 0 else "#E74C3C"
                sign  = "+" if diff > 0 else ""
                ax2.text(x[i], y_pos, f"{sign}{diff:.2f}",
                         ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")

        ax2.set_xticks(x)
        ax2.set_xticklabels(PURPOSE_LABELS, fontsize=10)
        ax2.set_ylabel("평균 평점 (1~5점)", fontsize=11)
        ax2.set_ylim(0, 5.8)
        ax2.axhline(4.0, color="green", linestyle="--", alpha=0.45, linewidth=1.2, label="만족 기준(4.0)")
        ax2.set_title("방문 목적 유형별 평균 만족도\n(전략 전 vs 후)", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        self._savefig("05_purpose_analysis")

        result = {
            "purpose_ratio_before":        ratio1,
            "purpose_ratio_after":         ratio2,
            "purpose_satisfaction_before": rating1,
            "purpose_satisfaction_after":  rating2,
        }
        self.metrics_summary["purpose"] = result

        p_deltas    = {p: ratio2[p] - ratio1[p] for p in PURPOSE_KEYWORDS}
        top_purpose = max(p_deltas, key=lambda p: abs(p_deltas[p]))
        print(f"  [5] Purpose: 가장 변화 큰 유형={top_purpose} ({p_deltas[top_purpose]:+.1f}%p)")
        return result

    # ──────────────────────────────────────────────────────────────
    # 6. 재방문율 분석 (Retention)
    # ──────────────────────────────────────────────────────────────

    def analysis_6_retention(self) -> Dict:
        """기존 고객 유지율(Retention) vs 신규 유입(New User) vs 이탈(Churn)."""
        def visitor_ids(df: pd.DataFrame) -> set:
            if "agent_id" not in df.columns:
                return set()
            return set(df["agent_id"].dropna().unique())

        visitors1 = visitor_ids(self.target1)
        visitors2 = visitor_ids(self.target2)

        retained  = visitors1 & visitors2
        new_users = visitors2 - visitors1
        churned   = visitors1 - visitors2

        retention_rate = len(retained) / max(len(visitors1), 1) * 100
        new_user_rate  = len(new_users) / max(len(visitors2), 1) * 100

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle(
            f"[{self.target_store}] 재방문율 분석 (Retention Analysis)",
            fontsize=14, fontweight="bold",
        )

        # ── 파이 차트: Sim2 방문자 구성 ──
        ax1 = axes[0]
        sizes = [len(retained), len(new_users)]
        if sum(sizes) > 0:
            wedges, texts, autotexts = ax1.pie(
                sizes,
                labels=["재방문\n(기존 고객)", "신규 방문\n(새 고객)"],
                colors=[COLOR_SIM1, COLOR_SIM2],
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2},
            )
            for at in autotexts:
                at.set_fontsize(11)
                at.set_fontweight("bold")
        else:
            ax1.text(0.5, 0.5, "데이터 없음", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title(f"Sim2 방문자 구성\n(총 {len(visitors2)}명)", fontsize=12, fontweight="bold")

        # ── 수직 바 차트: 이탈/유지/신규 ──
        ax2 = axes[1]
        categories = ["이탈\n(Churn)", "재방문\n(Retained)", "신규\n(New)"]
        values     = [len(churned), len(retained), len(new_users)]
        colors_bar = ["#E74C3C", "#2ECC71", "#3498DB"]

        bars = ax2.bar(categories, values, color=colors_bar, alpha=0.88,
                       edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=13, fontweight="bold",
            )
        ax2.set_ylabel("에이전트 수", fontsize=11)
        ax2.set_title("고객 이탈 / 유지 / 신규 현황", fontsize=12, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        self._savefig("06_retention")

        result = {
            "visitors_before":  len(visitors1),
            "visitors_after":   len(visitors2),
            "retained":         len(retained),
            "new_users":        len(new_users),
            "churned":          len(churned),
            "retention_rate":   retention_rate,
            "new_user_rate":    new_user_rate,
        }
        self.metrics_summary["retention"] = result
        print(f"  [6] Retention: 유지율 {retention_rate:.1f}% | 신규 {len(new_users)}명 | 이탈 {len(churned)}명")
        return result

    # ──────────────────────────────────────────────────────────────
    # 7. 경쟁 매장 비교 (Radar Chart)
    # ──────────────────────────────────────────────────────────────

    def analysis_7_radar_chart(self) -> Dict:
        """동일 카테고리 TOP 5 매장과 5각형 레이더 차트 비교."""

        def compute_metrics(df: pd.DataFrame, ratings_json: Dict) -> Dict[str, Dict]:
            if df.empty or "visited_store" not in df.columns:
                return {}
            metrics: Dict[str, Dict] = {}
            for store, grp in df.groupby("visited_store"):
                r_cols = [c for c in ["taste_rating", "value_rating", "atmosphere_rating"] if c in grp.columns]
                avg_rt = float(grp[r_cols].mean().mean()) if r_cols else 0.0

                if "agent_id" in grp.columns:
                    agent_vc = grp["agent_id"].value_counts()
                    revisit  = (agent_vc >= 2).sum() / max(grp["agent_id"].nunique(), 1) * 100
                else:
                    revisit = 0.0

                store_data = ratings_json.get(str(store), {})
                avg_price  = store_data.get("average_price", 0) or 0

                metrics[store] = {
                    "visits":       len(grp),
                    "avg_rating":   avg_rt,
                    "revisit_rate": revisit,
                    "avg_price":    avg_price,
                    "review_count": len(grp),
                }
            return metrics

        metrics2 = compute_metrics(self.df2, self.store_ratings2)
        metrics1 = compute_metrics(self.df1, self.store_ratings1)

        if not metrics2:
            print("  [7] Radar: 데이터 부족으로 레이더 차트 스킵")
            return {}

        sorted_by_visits = sorted(metrics2.items(), key=lambda x: x[1]["visits"], reverse=True)
        top5 = [s for s, _ in sorted_by_visits[:5]]
        if self.target_store not in top5 and self.target_store in metrics2:
            top5 = [self.target_store] + top5[:4]

        radar_metrics = ["방문수", "평균평점", "재방문율", "리뷰수"]
        raw_data = {
            "방문수":   [metrics2.get(s, {}).get("visits",       0) for s in top5],
            "평균평점": [metrics2.get(s, {}).get("avg_rating",   0) for s in top5],
            "재방문율": [metrics2.get(s, {}).get("revisit_rate", 0) for s in top5],
            "리뷰수":   [metrics2.get(s, {}).get("review_count", 0) for s in top5],
        }

        def normalize(vals: List[float]) -> List[float]:
            arr = np.array(vals, dtype=float)
            mn, mx = arr.min(), arr.max()
            if mx == mn:
                return [50.0] * len(arr)
            return ((arr - mn) / (mx - mn) * 100).tolist()

        norm_data = {k: normalize(v) for k, v in raw_data.items()}

        N      = len(radar_metrics)
        angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
        colors  = plt.cm.Set2(np.linspace(0, 1, len(top5)))

        for idx, store in enumerate(top5):
            values = [norm_data[m][idx] for m in radar_metrics] + [norm_data[radar_metrics[0]][idx]]
            lw         = 3.0 if store == self.target_store else 1.5
            alpha_fill = 0.25 if store == self.target_store else 0.07
            label = (f"★ {store}" if store == self.target_store else store)[:16]

            ax.plot(angles, values, "o-", lw=lw, color=colors[idx], label=label)
            ax.fill(angles, values, alpha=alpha_fill, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_metrics, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color="gray")
        ax.grid(color="gray", linestyle="--", alpha=0.3)
        ax.set_title(
            f"[{self.target_store}] 경쟁 매장 비교\n(Sim2 기준, 정규화 점수 0~100)",
            fontsize=13, fontweight="bold", pad=20,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.15), fontsize=9)
        plt.tight_layout()
        self._savefig("07_radar_chart")

        tm1 = metrics1.get(self.target_store, {})
        tm2 = metrics2.get(self.target_store, {})
        result = {
            "top_stores":            top5,
            "target_metrics_before": tm1,
            "target_metrics_after":  tm2,
        }
        self.metrics_summary["radar"] = result
        print(f"  [7] Radar: TOP5 = {', '.join(top5)}")
        return result

    # ──────────────────────────────────────────────────────────────
    # 9. 에이전트 유형별 분석 (Agent Type: 상주 vs 유동)
    # ──────────────────────────────────────────────────────────────

    def analysis_9_agent_type(self) -> Dict:
        """상주 vs 유동 에이전트 비율 변화 분석."""

        def type_ratio(df: pd.DataFrame) -> Dict[str, float]:
            total = max(len(df), 1)
            if "agent_type" not in df.columns:
                return {}
            counts = df["agent_type"].value_counts()
            return {k: v / total * 100 for k, v in counts.items()}

        ratio1 = type_ratio(self.target1)
        ratio2 = type_ratio(self.target2)
        all_types = sorted(
            set(list(ratio1.keys()) + list(ratio2.keys())),
            key=lambda t: (t == "기타", t)
        )

        if not all_types:
            print("  [9] Agent Type: 데이터 없음")
            return {}

        x     = np.arange(len(all_types))
        width = 0.35
        vals1 = [ratio1.get(t, 0) for t in all_types]
        vals2 = [ratio2.get(t, 0) for t in all_types]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars1 = ax.bar(x - width / 2, vals1, width,
                       label="Sim1 (전략 전)", color=COLOR_SIM1, alpha=0.85)
        bars2 = ax.bar(x + width / 2, vals2, width,
                       label="Sim2 (전략 후)", color=COLOR_SIM2, alpha=0.85)

        for bar, val in zip(bars1, vals1):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color=COLOR_SIM1)
        for bar, val in zip(bars2, vals2):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color=COLOR_SIM2)

        for i, t in enumerate(all_types):
            diff  = vals2[i] - vals1[i]
            y_pos = max(vals1[i], vals2[i]) + 2.5
            color = "#27AE60" if diff > 0 else ("#E74C3C" if diff < 0 else "gray")
            sign  = "+" if diff > 0 else ""
            ax.text(x[i], y_pos, f"{sign}{diff:.1f}%p",
                    ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(all_types, fontsize=13)
        ax.set_ylabel("방문 비율 (%)", fontsize=11)
        ax.set_title(
            f"[{self.target_store}] 에이전트 유형별 방문 비율 변화\n(상주 vs 유동)",
            fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self._savefig("09_agent_type")

        result = {"before": ratio1, "after": ratio2, "types": all_types}
        self.metrics_summary["agent_type"] = result
        print(f"  [9] Agent Type: {', '.join(f'{t}={ratio2.get(t,0):.1f}%' for t in all_types)}")
        return result

    # ──────────────────────────────────────────────────────────────
    # 10. 성별 구성 분석 (Gender Composition)
    # ──────────────────────────────────────────────────────────────

    def analysis_10_gender_composition(self) -> Dict:
        """성별 구성 비율 변화 분석 (남성만 / 여성만 / 혼성)."""
        GENDER_COLORS = ["#4E79A7", "#E15759", "#76B7B2", "#B07AA1"]

        def gender_ratio(df: pd.DataFrame) -> Dict[str, float]:
            total = max(len(df), 1)
            if "gender_composition" not in df.columns:
                return {}
            counts = df["gender_composition"].value_counts()
            return {k: v / total * 100 for k, v in counts.items()}

        ratio1 = gender_ratio(self.target1)
        ratio2 = gender_ratio(self.target2)
        all_genders = sorted(set(list(ratio1.keys()) + list(ratio2.keys())))

        if not all_genders:
            print("  [10] Gender: 데이터 없음")
            return {}

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f"[{self.target_store}] 성별 구성 비율 변화",
            fontsize=14, fontweight="bold"
        )

        # ── 100% Stacked Bar ──
        ax = axes[0]
        x_labels = ["Sim1\n(전략 전)", "Sim2\n(전략 후)"]
        bottoms  = [0.0, 0.0]

        for gender, color in zip(all_genders, GENDER_COLORS):
            vals = [ratio1.get(gender, 0), ratio2.get(gender, 0)]
            bars = ax.bar(x_labels, vals, bottom=bottoms, color=color,
                          label=gender, alpha=0.9, edgecolor="white", linewidth=1.2)
            for j, (bar, val) in enumerate(zip(bars, vals)):
                if val > 4:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottoms[j] + val / 2,
                        f"{val:.1f}%",
                        ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white",
                    )
            bottoms = [bottoms[j] + vals[j] for j in range(2)]

        ax.set_ylim(0, 108)
        ax.set_ylabel("비율 (%)", fontsize=11)
        ax.set_title("성별 구성 비중\n(100% Stacked Bar)", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.2)

        # ── 그룹 바 차트 (변화량 표기) ──
        ax2   = axes[1]
        x     = np.arange(len(all_genders))
        width = 0.35
        vals1 = [ratio1.get(g, 0) for g in all_genders]
        vals2 = [ratio2.get(g, 0) for g in all_genders]

        bars1 = ax2.bar(x - width / 2, vals1, width,
                        label="Sim1 (전략 전)", color=COLOR_SIM1, alpha=0.85)
        bars2 = ax2.bar(x + width / 2, vals2, width,
                        label="Sim2 (전략 후)", color=COLOR_SIM2, alpha=0.85)

        for bar, val in zip(bars1, vals1):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                         f"{val:.1f}%", ha="center", va="bottom", fontsize=8, color=COLOR_SIM1)
        for bar, val in zip(bars2, vals2):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                         f"{val:.1f}%", ha="center", va="bottom", fontsize=8, color=COLOR_SIM2)

        for i, g in enumerate(all_genders):
            diff  = vals2[i] - vals1[i]
            y_pos = max(vals1[i], vals2[i]) + 2.5
            color = "#27AE60" if diff > 0 else ("#E74C3C" if diff < 0 else "gray")
            sign  = "+" if diff > 0 else ""
            ax2.text(x[i], y_pos, f"{sign}{diff:.1f}%p",
                     ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")

        ax2.set_xticks(x)
        ax2.set_xticklabels(all_genders, fontsize=11)
        ax2.set_ylabel("방문 비율 (%)", fontsize=11)
        ax2.set_title("성별 구성 비율 변화\n(Sim1 vs Sim2)", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        self._savefig("10_gender_composition")

        result = {"before": ratio1, "after": ratio2, "genders": all_genders}
        self.metrics_summary["gender"] = result
        print(f"  [10] Gender: {', '.join(f'{g}={ratio2.get(g,0):.1f}%' for g in all_genders)}")
        return result

    # ──────────────────────────────────────────────────────────────
    # 11. 세대 × 방문목적 크로스탭 히트맵
    # ──────────────────────────────────────────────────────────────

    def analysis_11_crosstab_heatmap(self) -> Dict:
        """세대 × 방문목적 크로스탭 히트맵 — 행 기준 비율(%) Sim1 vs Sim2."""

        def make_crosstab(df: pd.DataFrame) -> pd.DataFrame:
            if "generation" not in df.columns or "persona_type" not in df.columns:
                return pd.DataFrame()
            ct = pd.crosstab(df["generation"], df["persona_type"])
            gen_idx  = [g for g in GENERATION_ORDER if g in ct.index]
            pur_cols = [p for p in PURPOSE_KEYWORDS if p in ct.columns]
            return ct.reindex(index=gen_idx, columns=pur_cols, fill_value=0)

        ct1 = make_crosstab(self.target1)
        ct2 = make_crosstab(self.target2)

        if ct1.empty and ct2.empty:
            print("  [11] Crosstab: 데이터 없음")
            return {}

        # 통일된 인덱스/컬럼
        all_gen = [g for g in GENERATION_ORDER if g in (set(ct1.index) | set(ct2.index))]
        all_pur = [p for p in PURPOSE_KEYWORDS  if p in (set(ct1.columns) | set(ct2.columns))]

        ct1 = ct1.reindex(index=all_gen, columns=all_pur, fill_value=0)
        ct2 = ct2.reindex(index=all_gen, columns=all_pur, fill_value=0)

        # 행 기준 비율(%) 정규화
        ct1_pct = ct1.div(ct1.sum(axis=1).replace(0, 1), axis=0) * 100
        ct2_pct = ct2.div(ct2.sum(axis=1).replace(0, 1), axis=0) * 100

        vmax = max(
            ct1_pct.values.max() if not ct1_pct.empty else 0,
            ct2_pct.values.max() if not ct2_pct.empty else 0,
            1,
        )

        pur_labels = [p.replace("형", "") for p in all_pur]

        fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(all_gen) * 0.9 + 2)))
        fig.suptitle(
            f"[{self.target_store}] 세대 × 방문목적 크로스탭 히트맵\n(행 기준 비율 %)",
            fontsize=14, fontweight="bold"
        )

        for ax, ct_pct, label in zip(
            axes,
            [ct1_pct, ct2_pct],
            ["Sim 1 (전략 전)", "Sim 2 (전략 후)"],
        ):
            im = ax.imshow(ct_pct.values, aspect="auto",
                           cmap="YlOrRd", vmin=0, vmax=vmax)
            ax.set_xticks(np.arange(len(all_pur)))
            ax.set_xticklabels(pur_labels, fontsize=10, rotation=15, ha="right")
            ax.set_yticks(np.arange(len(all_gen)))
            ax.set_yticklabels(all_gen, fontsize=11)
            ax.set_title(label, fontsize=12, fontweight="bold")

            # 셀 값 표기
            for i in range(len(all_gen)):
                for j in range(len(all_pur)):
                    val = ct_pct.values[i, j]
                    text_color = "white" if val > vmax * 0.6 else "black"
                    ax.text(j, i, f"{val:.1f}%",
                            ha="center", va="center",
                            fontsize=9, color=text_color, fontweight="bold")

            fig.colorbar(im, ax=ax, shrink=0.8, label="비율 (%)")

        plt.tight_layout()
        self._savefig("11_crosstab_heatmap")

        result = {
            "crosstab_before":     ct1.to_dict(),
            "crosstab_after":      ct2.to_dict(),
            "crosstab_pct_before": ct1_pct.to_dict(),
            "crosstab_pct_after":  ct2_pct.to_dict(),
            "generations": all_gen,
            "purposes":    all_pur,
        }
        self.metrics_summary["crosstab"] = result
        print(f"  [11] Crosstab: {len(all_gen)}세대 × {len(all_pur)}방문목적")
        return result

    # ──────────────────────────────────────────────────────────────
    # 8. 종합 평가 (LLM Summary)
    # ──────────────────────────────────────────────────────────────

    def analysis_8_llm_summary(self) -> str:
        """전략 효과와 주 고객층 변화를 OpenAI API로 서술."""
        if not self.openai_api_key:
            print("  [8] LLM Summary: OPENAI_API_KEY 없음 → 규칙 기반 요약 사용")
            return self._rule_based_summary()

        try:
            from openai import OpenAI
        except ImportError:
            print("  [8] LLM Summary: openai 패키지 없음 → pip install openai")
            return self._rule_based_summary()

        prompt = self._build_llm_prompt()

        try:
            client   = OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "당신은 마케팅 전략 분석 전문가입니다. "
                            "ABM 시뮬레이션 데이터를 기반으로 전략의 효과와 "
                            "고객층 변화를 한국어 3~5 문단으로 서술합니다."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1400,
            )
            summary = response.choices[0].message.content.strip()
            print(f"  [8] LLM Summary: OpenAI 완료 ({len(summary)}자)")
            return summary
        except Exception as e:
            print(f"  [8] LLM Summary: OpenAI 오류 ({e}) → 규칙 기반 요약 사용")
            return self._rule_based_summary()

    def _build_llm_prompt(self) -> str:
        m        = self.metrics_summary
        overview = m.get("overview", {})
        rating   = m.get("rating",   {})
        gen      = m.get("generation", {})
        purpose  = m.get("purpose",   {})
        retention= m.get("retention", {})
        at       = m.get("agent_type", {})
        gender   = m.get("gender", {})

        gen_before = gen.get("before", {})
        gen_after  = gen.get("after",  {})
        gen_lines  = "\n".join(
            f"- {g}: {gen_before.get(g,0):.1f}% → {gen_after.get(g,0):.1f}%"
            f" ({gen_after.get(g,0) - gen_before.get(g,0):+.1f}%p)"
            for g in GENERATION_ORDER
        )

        p_rb  = purpose.get("purpose_ratio_before", {})
        p_ra  = purpose.get("purpose_ratio_after",  {})
        p_sb  = purpose.get("purpose_satisfaction_before", {})
        p_sa  = purpose.get("purpose_satisfaction_after",  {})
        p_lines = "\n".join(
            f"- {p}: 비중 {p_rb.get(p,0):.1f}%→{p_ra.get(p,0):.1f}% "
            f"({p_ra.get(p,0)-p_rb.get(p,0):+.1f}%p) | "
            f"만족도 {p_sb.get(p,0):.2f}→{p_sa.get(p,0):.2f}"
            for p in PURPOSE_KEYWORDS
        )

        at_b = at.get("before", {})
        at_a = at.get("after",  {})
        at_types = at.get("types", [])
        at_lines = "\n".join(
            f"- {t}: {at_b.get(t,0):.1f}% → {at_a.get(t,0):.1f}% ({at_a.get(t,0)-at_b.get(t,0):+.1f}%p)"
            for t in at_types
        ) if at_types else "- 데이터 없음"

        g_b = gender.get("before", {})
        g_a = gender.get("after",  {})
        g_gs = gender.get("genders", [])
        g_lines = "\n".join(
            f"- {g}: {g_b.get(g,0):.1f}% → {g_a.get(g,0):.1f}% ({g_a.get(g,0)-g_b.get(g,0):+.1f}%p)"
            for g in g_gs
        ) if g_gs else "- 데이터 없음"

        return f"""다음은 '{self.target_store}' 매장의 전략 적용 전(Sim 1) vs 후(Sim 2) ABM 시뮬레이션 결과입니다.

## 핵심 지표 요약

### 방문 지표
- 방문 수: {overview.get('visits_before',0)}건 → {overview.get('visits_after',0)}건 ({overview.get('visit_change_pct',0):+.1f}%)
- 시장 점유율: {overview.get('market_share_before',0):.1f}% → {overview.get('market_share_after',0):.1f}%

### 평점 및 만족도
- 평균 평점: {rating.get('avg_rating_before',0):.2f} → {rating.get('avg_rating_after',0):.2f}
- 만족도(4점 이상 비율): {rating.get('satisfaction_rate_before',0):.1f}% → {rating.get('satisfaction_rate_after',0):.1f}%

### 세대별 방문 비율 변화
{gen_lines}

### 방문 목적 유형별 비중·만족도 변화
{p_lines}

### 에이전트 유형 (상주 vs 유동) 변화
{at_lines}

### 성별 구성 변화
{g_lines}

### 재방문율
- 기존 고객 유지율: {retention.get('retention_rate',0):.1f}%
- 신규 유입: {retention.get('new_users',0)}명
- 이탈: {retention.get('churned',0)}명

위 데이터를 바탕으로 **3~5 문단 한국어**로 아래 네 가지를 분석해 주세요:
1. **전략의 효과**: 방문·만족도 변화와 그 원인
2. **바뀐 주 고객층의 특성**: 증가한 세대·방문 목적·상주/유동·성별과 시사점
3. **재방문율 및 고객 충성도** 평가
4. **향후 권장 사항**: 데이터에서 발견된 기회 요소와 리스크"""

    def _rule_based_summary(self) -> str:
        """OpenAI 없을 때의 규칙 기반 한국어 요약."""
        m        = self.metrics_summary
        overview = m.get("overview",   {})
        rating   = m.get("rating",     {})
        gen      = m.get("generation", {})
        purpose  = m.get("purpose",    {})
        retention= m.get("retention",  {})
        at       = m.get("agent_type", {})

        v_chg   = overview.get("visit_change_pct",          0)
        r_bef   = rating.get("avg_rating_before",           0)
        r_aft   = rating.get("avg_rating_after",            0)
        sat_bef = rating.get("satisfaction_rate_before",    0)
        sat_aft = rating.get("satisfaction_rate_after",     0)

        gen_before = gen.get("before", {})
        gen_after  = gen.get("after",  {})
        gen_deltas = {g: gen_after.get(g, 0) - gen_before.get(g, 0) for g in GENERATION_ORDER}
        top_gen_up   = max(gen_deltas, key=lambda g: gen_deltas[g])
        top_gen_down = min(gen_deltas, key=lambda g: gen_deltas[g])

        p_before = purpose.get("purpose_ratio_before", {})
        p_after  = purpose.get("purpose_ratio_after",  {})
        p_deltas = {p: p_after.get(p, 0) - p_before.get(p, 0) for p in PURPOSE_KEYWORDS}
        top_p_up   = max(p_deltas, key=lambda p: p_deltas[p])
        top_p_down = min(p_deltas, key=lambda p: p_deltas[p])

        ret_rate  = retention.get("retention_rate", 0)
        new_users = retention.get("new_users",      0)
        churned   = retention.get("churned",        0)

        at_b     = at.get("before", {})
        at_a     = at.get("after",  {})
        at_types = at.get("types",  [])
        at_summary = ""
        if at_types:
            at_lines = ", ".join(
                f"{t} {at_a.get(t,0):.1f}%({at_a.get(t,0)-at_b.get(t,0):+.1f}%p)"
                for t in at_types
            )
            at_summary = f"\n\n**에이전트 유형 변화**\n\n에이전트 유형별 비율은 {at_lines}로 변화하였습니다."

        trend    = "증가" if v_chg > 0 else "감소"
        r_sign   = "+" if r_aft - r_bef >= 0 else ""
        sat_sign = "+" if sat_aft - sat_bef >= 0 else ""

        return f"""**전략의 효과 분석**

전략 적용 후 '{self.target_store}'의 방문 수는 {v_chg:+.1f}%로 {trend}하였습니다. 평균 평점은 {r_bef:.2f}점에서 {r_aft:.2f}점으로({r_sign}{r_aft - r_bef:.2f}) 변화했으며, 4점 이상 만족도 비율은 {sat_bef:.1f}%에서 {sat_aft:.1f}%로 {sat_sign}{sat_aft - sat_bef:.1f}%p {"개선" if sat_aft >= sat_bef else "하락"}되었습니다. {"전략이 방문 지표에 전반적으로 긍정적인 영향을 미쳤습니다." if v_chg > 0 else "방문 수 측면에서는 아직 전략의 효과가 충분히 가시화되지 않았으며, 품질 지표 개선에 집중할 필요가 있습니다."}

**바뀐 주 고객층의 특성**

세대 분포에서 {top_gen_up} 세대의 비율이 {gen_deltas[top_gen_up]:+.1f}%p로 가장 크게 증가하였고, {top_gen_down} 세대는 {gen_deltas[top_gen_down]:+.1f}%p 감소하였습니다. 방문 목적 유형에서는 '{top_p_up}'이 {p_deltas[top_p_up]:+.1f}%p 증가하여 핵심 고객층으로 부상한 반면, '{top_p_down}'은 {p_deltas[top_p_down]:+.1f}%p 감소하였습니다. 이는 전략이 특정 라이프스타일 그룹에 더 효과적으로 작용했음을 의미합니다.{at_summary}

**재방문율 및 고객 충성도**

전략 적용 전 고객 중 {ret_rate:.1f}%가 전략 후에도 재방문하여 기존 고객 유지 측면에서 {"양호한 성과를 보였습니다" if ret_rate >= 50 else "개선의 여지가 있습니다"}. 신규 유입 고객 {new_users}명과 이탈 고객 {churned}명을 고려할 때, {"신규 유입이 이탈을 상회하여 순 고객 증가가 이루어졌습니다" if new_users > churned else "이탈을 줄이기 위한 추가 고객 경험 개선이 필요합니다"}.

**향후 권장 사항**

'{top_p_up}' 유형 고객의 증가세를 유지하기 위해 해당 고객층이 선호하는 분위기·메뉴·서비스 요소를 강화하는 것을 권장합니다. 또한 {top_gen_up} 세대를 주 타겟으로 한 맞춤형 마케팅 채널(SNS, 리뷰 플랫폼)을 활용하고, {top_gen_down} 세대의 이탈 원인을 세부 분석하여 개선 방안을 수립하는 것이 중요합니다."""

    # ──────────────────────────────────────────────────────────────
    # 보고서 생성 오케스트레이터
    # ──────────────────────────────────────────────────────────────

    def generate_all(self) -> Path:
        """모든 분석 실행 후 Comparison_Report.md 생성."""
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"비교 보고서 생성: '{self.target_store}'")
        print(f"  Sim1 경로: {self.before_dir}")
        print(f"  Sim2 경로: {self.after_dir}")
        print(f"  출력 경로: {self.output_dir}")
        print(f"{sep}\n")

        print("[분석 시작]")
        r1  = self.analysis_1_overview()
        r2  = self.analysis_2_rating_spread()
        r3  = self.analysis_3_hourly_traffic()
        r4  = self.analysis_4_generation_impact()
        r5  = self.analysis_5_purpose_analysis()
        r6  = self.analysis_6_retention()
        r7  = self.analysis_7_radar_chart()
        r9  = self.analysis_9_agent_type()
        r10 = self.analysis_10_gender_composition()
        r11 = self.analysis_11_crosstab_heatmap()

        print("\n[LLM 종합 평가 생성 중...]")
        summary_text = self.analysis_8_llm_summary()

        report_path = self._write_markdown_report(
            r1, r2, r3, r4, r5, r6, r7, r9, r10, r11, summary_text
        )

        print(f"\n{sep}")
        print(f"보고서 생성 완료!")
        print(f"  보고서: {report_path}")
        print(f"  그래프: {self.figures_dir}")
        print(f"{sep}")
        return report_path

    # ──────────────────────────────────────────────────────────────
    # Markdown 보고서 작성
    # ──────────────────────────────────────────────────────────────

    def _write_markdown_report(
        self,
        r1: Dict, r2: Dict, r3: Dict, r4: Dict,
        r5: Dict, r6: Dict, r7: Dict,
        r9: Dict, r10: Dict, r11: Dict,
        summary_text: str,
    ) -> Path:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 세대별 테이블 행
        gen_b = r4.get("before", {})
        gen_a = r4.get("after",  {})
        gen_rows = "\n".join(
            f"| {g} | {gen_b.get(g, 0):.1f}% | {gen_a.get(g, 0):.1f}% "
            f"| {gen_a.get(g, 0) - gen_b.get(g, 0):+.1f}%p |"
            for g in GENERATION_ORDER
        )

        # 방문 목적별 테이블 행
        p_rb = r5.get("purpose_ratio_before",        {})
        p_ra = r5.get("purpose_ratio_after",         {})
        p_sb = r5.get("purpose_satisfaction_before", {})
        p_sa = r5.get("purpose_satisfaction_after",  {})
        purpose_rows = "\n".join(
            f"| {p} | {p_rb.get(p,0):.1f}% | {p_ra.get(p,0):.1f}% "
            f"| {p_ra.get(p,0)-p_rb.get(p,0):+.1f}%p "
            f"| {p_sb.get(p,0):.2f} | {p_sa.get(p,0):.2f} |"
            for p in PURPOSE_KEYWORDS
        )

        radar_stores = r7.get("top_stores", []) if r7 else []
        tm1 = r7.get("target_metrics_before", {}) if r7 else {}
        tm2 = r7.get("target_metrics_after",  {}) if r7 else {}

        # 에이전트 유형 테이블 행
        at_types = r9.get("types", []) if r9 else []
        at_b = r9.get("before", {}) if r9 else {}
        at_a = r9.get("after",  {}) if r9 else {}
        agent_type_rows = "\n".join(
            f"| {t} | {at_b.get(t,0):.1f}% | {at_a.get(t,0):.1f}% "
            f"| {at_a.get(t,0)-at_b.get(t,0):+.1f}%p |"
            for t in at_types
        ) if at_types else "| (데이터 없음) | - | - | - |"

        # 성별 구성 테이블 행
        g_gs = r10.get("genders", []) if r10 else []
        g_b  = r10.get("before",  {}) if r10 else {}
        g_a  = r10.get("after",   {}) if r10 else {}
        gender_rows = "\n".join(
            f"| {g} | {g_b.get(g,0):.1f}% | {g_a.get(g,0):.1f}% "
            f"| {g_a.get(g,0)-g_b.get(g,0):+.1f}%p |"
            for g in g_gs
        ) if g_gs else "| (데이터 없음) | - | - | - |"

        report = f"""# 전략 전/후 비교 분석 보고서 (ABM Simulation)

> **대상 매장:** {self.target_store}
> **생성 일시:** {now}
> **Sim 1:** 전략 적용 전 (Baseline)
> **Sim 2:** 전략 적용 후 (After Strategy)

---

## 1. 기본 방문 지표 (Overview)

| 지표 | Sim 1 (전략 전) | Sim 2 (전략 후) | 변화 |
|------|:--------------:|:--------------:|:----:|
| 총 방문 수 | {r1.get('visits_before', 0):,}건 | {r1.get('visits_after', 0):,}건 | **{r1.get('visit_change_pct', 0):+.1f}%** |
| 시장 점유율 | {r1.get('market_share_before', 0):.2f}% | {r1.get('market_share_after', 0):.2f}% | {r1.get('market_share_after', 0) - r1.get('market_share_before', 0):+.2f}%p |

### 방문 키워드 워드클라우드

![wordcloud](figures/01_wordcloud.png)

---

## 2. 평점 분포 분석 (Rating Spread)

| 지표 | Sim 1 (전략 전) | Sim 2 (전략 후) | 변화 |
|------|:--------------:|:--------------:|:----:|
| 평균 종합 평점 | {r2.get('avg_rating_before', 0):.2f}점 | {r2.get('avg_rating_after', 0):.2f}점 | **{r2.get('avg_rating_after', 0) - r2.get('avg_rating_before', 0):+.2f}** |
| 만족도 (4점 이상) | {r2.get('satisfaction_rate_before', 0):.1f}% | {r2.get('satisfaction_rate_after', 0):.1f}% | {r2.get('satisfaction_rate_after', 0) - r2.get('satisfaction_rate_before', 0):+.1f}%p |

![rating_spread](figures/02_rating_spread.png)

![satisfaction_rate](figures/02b_satisfaction_rate.png)

---

## 3. 시간대별 손님 변화 (Hourly Traffic)

| 지표 | Sim 1 | Sim 2 |
|------|:-----:|:-----:|
| 피크 타임슬롯 | {r3.get('peak_slot_before', '?')} | {r3.get('peak_slot_after', '?')} |

![hourly_traffic](figures/03_hourly_traffic.png)

---

## 4. 세대별 증감 분석 (Generation Impact)

> 혼합 세대("혼합(Z2+Y)" 등)는 "혼합" 단일 버킷으로 통합됩니다.

| 세대 | Sim 1 (전략 전) | Sim 2 (전략 후) | 변화 |
|:----:|:--------------:|:--------------:|:----:|
{gen_rows}

![generation_impact](figures/04_generation_impact.png)

---

## 5. 방문 목적별 상세 분석 (Purpose Analysis) ⭐

| 방문 목적 | 비중(전) | 비중(후) | 비중 변화 | 만족도(전) | 만족도(후) |
|:--------:|:-------:|:-------:|:--------:|:---------:|:---------:|
{purpose_rows}

![purpose_analysis](figures/05_purpose_analysis.png)

---

## 6. 재방문율 분석 (Retention)

| 지표 | 수치 |
|------|:----:|
| Sim 1 방문 에이전트 수 | {r6.get('visitors_before', 0)}명 |
| Sim 2 방문 에이전트 수 | {r6.get('visitors_after', 0)}명 |
| 재방문 (기존 고객 유지) | {r6.get('retained', 0)}명 |
| 신규 유입 | {r6.get('new_users', 0)}명 |
| 이탈 (Churn) | {r6.get('churned', 0)}명 |
| **기존 고객 유지율** | **{r6.get('retention_rate', 0):.1f}%** |
| Sim2 신규 고객 비율 | {r6.get('new_user_rate', 0):.1f}% |

![retention](figures/06_retention.png)

---

## 7. 경쟁 매장 비교 (Radar Chart)

> **비교 대상 (Sim2 방문수 TOP5):** {', '.join(radar_stores) if radar_stores else 'N/A'}
> **비교 지표:** 방문수, 평균평점, 재방문율, 리뷰수 (정규화 0~100)

| 지표 | Sim 1 (타겟) | Sim 2 (타겟) |
|------|:-----------:|:-----------:|
| 방문수 | {tm1.get('visits', 0)}건 | {tm2.get('visits', 0)}건 |
| 평균 평점 | {tm1.get('avg_rating', 0):.2f} | {tm2.get('avg_rating', 0):.2f} |
| 재방문율 | {tm1.get('revisit_rate', 0):.1f}% | {tm2.get('revisit_rate', 0):.1f}% |

![radar_chart](figures/07_radar_chart.png)

---

## 9. 에이전트 유형별 분석 (Agent Type)

| 유형 | Sim 1 (전략 전) | Sim 2 (전략 후) | 변화 |
|:----:|:--------------:|:--------------:|:----:|
{agent_type_rows}

![agent_type](figures/09_agent_type.png)

---

## 10. 성별 구성 분석 (Gender Composition)

| 성별 구성 | Sim 1 (전략 전) | Sim 2 (전략 후) | 변화 |
|:--------:|:--------------:|:--------------:|:----:|
{gender_rows}

![gender_composition](figures/10_gender_composition.png)

---

## 11. 세대 × 방문목적 크로스탭 히트맵 (Crosstab Heatmap)

> 각 셀: 해당 세대의 방문 건수 중 그 방문목적이 차지하는 비율(%)

![crosstab_heatmap](figures/11_crosstab_heatmap.png)

---

## 8. 종합 평가 (LLM Summary)

{summary_text}

---

*Report generated by `sim_to_y.py` — ABM Simulation Analysis*
*{now}*
"""

        report_path = self.output_dir / "Comparison_Report.md"
        report_path.write_text(report, encoding="utf-8")
        return report_path


# ──────────────────────────────────────────────────────────────
# CLI 진입점
# ──────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="전략 전/후 비교 보고서 생성기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예:
  # 기본 경로 자동 추론 (data/output/{store}_before, _after)
  python sim_to_y.py --target-store 돼지야

  # 경로 직접 지정
  python sim_to_y.py \\
    --target-store 돼지야 \\
    --before-dir data/output/돼지야_before \\
    --after-dir  data/output/돼지야_after \\
    --output-dir reports

  # OpenAI API 키 지정 (없으면 규칙 기반 요약)
  python sim_to_y.py --target-store 돼지야 --openai-key sk-...
""",
    )
    parser.add_argument("--target-store", type=str, default="돼지야",
                        help="분석 대상 매장명 (기본: 돼지야)")
    parser.add_argument("--before-dir",   type=str, default=None,
                        help="전략 전 시뮬레이션 결과 폴더 (기본: data/output/{store}_before)")
    parser.add_argument("--after-dir",    type=str, default=None,
                        help="전략 후 시뮬레이션 결과 폴더 (기본: data/output/{store}_after)")
    parser.add_argument("--output-dir",   type=str, default="reports",
                        help="보고서 출력 폴더 (기본: reports/)")
    parser.add_argument("--openai-key",   type=str, default=None,
                        help="OpenAI API 키 (없으면 규칙 기반 요약 사용)")
    args = parser.parse_args()

    store = args.target_store

    before_dir = (
        Path(args.before_dir)
        if args.before_dir
        else PROJECT_ROOT / "data" / "output" / f"{store}_before"
    )
    after_dir = (
        Path(args.after_dir)
        if args.after_dir
        else PROJECT_ROOT / "data" / "output" / f"{store}_after"
    )
    output_dir = PROJECT_ROOT / args.output_dir

    openai_key = args.openai_key or os.getenv("OPENAI_API_KEY", "")

    generator = ComparisonReportGenerator(
        target_store=store,
        before_dir=before_dir,
        after_dir=after_dir,
        output_dir=output_dir,
        openai_api_key=openai_key,
    )
    generator.generate_all()


if __name__ == "__main__":
    main()
