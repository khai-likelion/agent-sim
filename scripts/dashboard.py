"""
Real-time Simulation Dashboard using Streamlit.
Run with: streamlit run scripts/dashboard.py
"""

import json
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="망원동 에이전트 시뮬레이션",
    page_icon="🗺️",
    layout="wide",
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
SIMULATION_CSV = OUTPUT_DIR / "simulation_result.csv"
VISIT_LOG_CSV = OUTPUT_DIR / "visit_log.csv"
AGENTS_JSON = OUTPUT_DIR / "agents.json"
AB_COMPARISON_JSON = OUTPUT_DIR / "ab_comparison.json"
SIM_A_CSV = OUTPUT_DIR / "simulation_result_A.csv"
SIM_B_CSV = OUTPUT_DIR / "simulation_result_B.csv"

# Mangwon-dong bounds
LAT_MIN, LAT_MAX = 37.550, 37.560
LNG_MIN, LNG_MAX = 126.900, 126.915


@st.cache_data(ttl=5)  # Refresh every 5 seconds
def load_simulation_data():
    """Load simulation results."""
    if not SIMULATION_CSV.exists():
        return None
    try:
        df = pd.read_csv(SIMULATION_CSV, encoding="utf-8-sig")
        return df
    except Exception:
        return None


@st.cache_data(ttl=5)
def load_agents():
    """Load agent data."""
    if not AGENTS_JSON.exists():
        return None
    try:
        with open(AGENTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=10)
def load_ab_comparison():
    """Load A/B comparison results."""
    if not AB_COMPARISON_JSON.exists():
        return None
    try:
        with open(AB_COMPARISON_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=10)
def load_ab_dataframes():
    """Load A/B simulation DataFrames."""
    df_a, df_b = None, None
    try:
        if SIM_A_CSV.exists():
            df_a = pd.read_csv(SIM_A_CSV, encoding="utf-8-sig")
        if SIM_B_CSV.exists():
            df_b = pd.read_csv(SIM_B_CSV, encoding="utf-8-sig")
    except Exception:
        pass
    return df_a, df_b


def main():
    st.title("🗺️ 망원동 에이전트 시뮬레이션 대시보드")

    # Auto-refresh toggle
    st.sidebar.checkbox("자동 새로고침 (5초)", value=True)

    # Load data
    df = load_simulation_data()
    agents = load_agents()

    if df is None or len(df) == 0:
        st.warning("⏳ 시뮬레이션 데이터 대기 중... `python scripts/run_simulation.py --archetype --chained` 실행하세요.")
        st.info("데이터가 생성되면 자동으로 업데이트됩니다.")
        return

    # Sidebar - Filters
    st.sidebar.header("필터")

    # Get unique values
    days = sorted(df["timestamp"].str[:10].unique())
    selected_day = st.sidebar.selectbox("날짜", days, index=len(days) - 1 if days else 0)

    time_slots = df["time_slot"].unique().tolist()
    selected_slots = st.sidebar.multiselect("시간대", time_slots, default=time_slots)

    # Filter data
    df_filtered = df[
        (df["timestamp"].str[:10] == selected_day) &
        (df["time_slot"].isin(selected_slots))
    ]

    # ============ Main Metrics ============
    st.header(f"📊 {selected_day} 현황")

    col1, col2, col3, col4 = st.columns(4)

    total_events = len(df_filtered)
    active_events = len(df_filtered[df_filtered["is_active"] == True])
    visit_events = len(df_filtered[df_filtered["decision"] == "visit"])
    conversion = (visit_events / active_events * 100) if active_events > 0 else 0

    col1.metric("총 이벤트", f"{total_events:,}")
    col2.metric("활성 이벤트", f"{active_events:,}")
    col3.metric("방문 이벤트", f"{visit_events:,}")
    col4.metric("전환율", f"{conversion:.1f}%")

    # ============ Two Column Layout ============
    left_col, right_col = st.columns([2, 1])

    with left_col:
        # Map visualization
        st.subheader("🗺️ 에이전트 위치 & 방문 현황")

        visits = df_filtered[df_filtered["decision"] == "visit"]
        if len(visits) > 0:
            fig = px.scatter_mapbox(
                visits,
                lat="current_lat",
                lon="current_lng",
                color="visited_category",
                hover_name="agent_name",
                hover_data=["visited_store", "time_slot", "decision_reason"],
                zoom=14,
                height=500,
                title=f"방문 위치 ({len(visits)}건)",
            )
            fig.update_layout(
                mapbox_style="carto-positron",
                mapbox=dict(
                    center=dict(lat=(LAT_MIN + LAT_MAX) / 2, lon=(LNG_MIN + LNG_MAX) / 2),
                ),
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("선택한 기간에 방문 데이터가 없습니다.")

    with right_col:
        # Time slot breakdown
        st.subheader("⏰ 시간대별 방문")

        slot_stats = df_filtered.groupby("time_slot").agg({
            "agent_id": "count",
            "decision": lambda x: (x == "visit").sum()
        }).rename(columns={"agent_id": "총", "decision": "방문"})

        if len(slot_stats) > 0:
            fig = px.bar(
                slot_stats.reset_index(),
                x="time_slot",
                y=["총", "방문"],
                barmode="group",
                height=250,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Top stores
        st.subheader("🏪 인기 매장 TOP 10")

        if len(visits) > 0:
            top_stores = visits["visited_store"].value_counts().head(10)
            st.bar_chart(top_stores)
        else:
            st.info("방문 데이터 없음")

    # ============ Category Analysis ============
    st.header("📈 카테고리 분석")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("카테고리별 방문 분포")
        if len(visits) > 0:
            cat_counts = visits["visited_category"].value_counts()
            fig = px.pie(
                values=cat_counts.values,
                names=cat_counts.index,
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("시간대 × 카테고리 히트맵")
        if len(visits) > 0:
            heatmap_data = pd.crosstab(visits["time_slot"], visits["visited_category"])
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="카테고리", y="시간대", color="방문수"),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ============ Discovery Channel & Visit Purpose ============
    all_visits = df[df["decision"] == "visit"]

    has_discovery = "discovery_channel" in all_visits.columns and all_visits["discovery_channel"].notna().any()
    has_purpose = "visit_purpose" in all_visits.columns and all_visits["visit_purpose"].notna().any()

    if has_discovery or has_purpose:
        st.header("🔍 발견 경로 & 방문 목적 분석")

        col1, col2 = st.columns(2)

        with col1:
            if has_discovery:
                st.subheader("발견 경로별 방문 분포")
                channel_counts = all_visits["discovery_channel"].dropna().value_counts()
                fig = px.pie(
                    values=channel_counts.values,
                    names=channel_counts.index,
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if has_purpose:
                st.subheader("방문 목적별 분포")
                purpose_counts = all_visits["visit_purpose"].dropna().value_counts()
                fig = px.pie(
                    values=purpose_counts.values,
                    names=purpose_counts.index,
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Cross-analysis: channel × purpose
        if has_discovery and has_purpose:
            st.subheader("발견 경로 × 방문 목적 히트맵")
            cross = pd.crosstab(
                all_visits["discovery_channel"].dropna(),
                all_visits["visit_purpose"].dropna(),
            )
            if len(cross) > 0:
                fig = px.imshow(
                    cross,
                    labels=dict(x="방문 목적", y="발견 경로", color="방문수"),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

    # ============ Agent Archetype Analysis ============
    if agents and len(visits) > 0:
        st.header("👥 에이전트 프로필 분석")

        # Merge agent info with visits
        agent_df = pd.DataFrame(agents)
        if "agent_name" in visits.columns and "name" in agent_df.columns:
            visits_with_arch = visits.merge(
                agent_df[["name", "age_group", "occupation", "income_level"]],
                left_on="agent_name",
                right_on="name",
                how="left"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("👤 연령대별")
                if "age_group" in visits_with_arch.columns:
                    age_counts = visits_with_arch["age_group"].value_counts()
                    st.bar_chart(age_counts)

            with col2:
                st.subheader("💼 직업별")
                if "occupation" in visits_with_arch.columns:
                    occ_counts = visits_with_arch["occupation"].value_counts().head(10)
                    st.bar_chart(occ_counts)

            with col3:
                st.subheader("💰 소득수준별")
                if "income_level" in visits_with_arch.columns:
                    income_counts = visits_with_arch["income_level"].value_counts()
                    st.bar_chart(income_counts)

    # ============ A/B Comparison Section ============
    ab_data = load_ab_comparison()
    if ab_data:
        _render_ab_comparison(ab_data)

    # ============ Recent Events Log ============
    st.header("📋 최근 방문 로그")

    recent_visits = visits.tail(20).sort_values("timestamp", ascending=False)
    if len(recent_visits) > 0:
        display_cols = ["timestamp", "agent_name", "time_slot", "visited_store", "visited_category", "decision_reason"]
        if has_discovery:
            display_cols.append("discovery_channel")
        if has_purpose:
            display_cols.append("visit_purpose")
        display_cols = [c for c in display_cols if c in recent_visits.columns]
        st.dataframe(recent_visits[display_cols], use_container_width=True)
    else:
        st.info("방문 기록이 없습니다.")

    # ============ Progress ============
    st.sidebar.header("📊 전체 진행 상황")

    all_days = sorted(df["timestamp"].str[:10].unique())
    total_days = len(all_days)
    current_days = len(all_days)

    st.sidebar.progress(1.0)  # Complete
    st.sidebar.write(f"진행: {current_days}일 완료 ✅")

    total_records = len(df)
    st.sidebar.write(f"레코드: {total_records:,}개")


def _render_ab_comparison(ab_data: dict):
    """Render A/B comparison section on the dashboard."""
    st.header("📊 X-Report A/B 비교 분석")

    # Main metrics row
    col1, col2, col3 = st.columns(3)

    conv_a = ab_data.get("conversion_rate_a", 0)
    conv_b = ab_data.get("conversion_rate_b", 0)
    conv_delta = ab_data.get("conversion_delta", 0)
    pct_change = ab_data.get("conversion_pct_change", 0)

    col1.metric(
        "전환율 A (리포트 有)",
        f"{conv_a:.2%}",
        f"{ab_data.get('total_visits_a', 0)}회 방문",
    )
    col2.metric(
        "전환율 B (리포트 無)",
        f"{conv_b:.2%}",
        f"{ab_data.get('total_visits_b', 0)}회 방문",
    )
    col3.metric(
        "변화율",
        f"{pct_change:+.1f}%",
        f"전환율 차이: {conv_delta:+.4f}",
    )

    # Information diffusion
    st.subheader("📡 정보 확산")
    info_col1, info_col2 = st.columns(2)
    info_col1.metric("리포트 수신율", f"{ab_data.get('report_reception_rate', 0):.1%}")
    info_col2.metric("리포트 영향 전환율", f"{ab_data.get('report_influenced_rate', 0):.1%}")

    # Report target store comparison
    report_visits_a = ab_data.get("report_store_visits_a", {})
    report_visits_b = ab_data.get("report_store_visits_b", {})

    if report_visits_a:
        st.subheader("🎯 리포트 대상 매장 방문수 비교")

        store_names = list(report_visits_a.keys())
        visits_a_vals = [report_visits_a.get(s, 0) for s in store_names]
        visits_b_vals = [report_visits_b.get(s, 0) for s in store_names]

        fig = go.Figure(data=[
            go.Bar(name="A (리포트 有)", x=store_names, y=visits_a_vals, marker_color="#636EFA"),
            go.Bar(name="B (리포트 無)", x=store_names, y=visits_b_vals, marker_color="#EF553B"),
        ])
        fig.update_layout(barmode="group", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Segment conversion comparison
    seg_conv_a = ab_data.get("segment_conversion_a", {})
    seg_conv_b = ab_data.get("segment_conversion_b", {})

    if seg_conv_a:
        st.subheader("👥 세그먼트별 전환율 비교")

        segments = list(seg_conv_a.keys())
        conv_a_vals = [seg_conv_a.get(s, 0) for s in segments]
        conv_b_vals = [seg_conv_b.get(s, 0) for s in segments]

        fig = go.Figure(data=[
            go.Bar(name="A (리포트 有)", x=segments, y=conv_a_vals, marker_color="#636EFA"),
            go.Bar(name="B (리포트 無)", x=segments, y=conv_b_vals, marker_color="#EF553B"),
        ])
        fig.update_layout(barmode="group", height=350, yaxis_title="전환율")
        st.plotly_chart(fig, use_container_width=True)

    # Category distribution comparison
    cat_dist_a = ab_data.get("category_distribution_a", {})
    cat_dist_b = ab_data.get("category_distribution_b", {})

    if cat_dist_a or cat_dist_b:
        st.subheader("📂 카테고리 분포 비교")

        all_cats = sorted(set(list(cat_dist_a.keys()) + list(cat_dist_b.keys())))
        cat_a_vals = [cat_dist_a.get(c, 0) for c in all_cats]
        cat_b_vals = [cat_dist_b.get(c, 0) for c in all_cats]

        fig = go.Figure(data=[
            go.Bar(name="A (리포트 有)", x=all_cats, y=cat_a_vals, marker_color="#636EFA"),
            go.Bar(name="B (리포트 無)", x=all_cats, y=cat_b_vals, marker_color="#EF553B"),
        ])
        fig.update_layout(barmode="group", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Discovery channel comparison (if available)
    disc_a = ab_data.get("discovery_channel_a", {})
    disc_b = ab_data.get("discovery_channel_b", {})

    if disc_a or disc_b:
        st.subheader("🔍 발견 경로 비교 (A vs B)")

        all_channels = sorted(set(list(disc_a.keys()) + list(disc_b.keys())))
        ch_a_vals = [disc_a.get(c, 0) for c in all_channels]
        ch_b_vals = [disc_b.get(c, 0) for c in all_channels]

        fig = go.Figure(data=[
            go.Bar(name="A (리포트 有)", x=all_channels, y=ch_a_vals, marker_color="#636EFA"),
            go.Bar(name="B (리포트 無)", x=all_channels, y=ch_b_vals, marker_color="#EF553B"),
        ])
        fig.update_layout(barmode="group", height=350)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
