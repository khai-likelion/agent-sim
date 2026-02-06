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

    # ============ Recent Events Log ============
    st.header("📋 최근 방문 로그")

    recent_visits = visits.tail(20).sort_values("timestamp", ascending=False)
    if len(recent_visits) > 0:
        display_cols = ["timestamp", "agent_name", "time_slot", "visited_store", "visited_category", "decision_reason"]
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


if __name__ == "__main__":
    main()
