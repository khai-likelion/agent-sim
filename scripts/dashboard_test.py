"""
테스트용 대시보드 - 단일 에이전트 시뮬레이션 결과 확인

실행 방법:
    streamlit run scripts/dashboard_test.py
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import folium
from streamlit_folium import st_folium
import time as time_module

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"

# 페이지 설정
st.set_page_config(
    page_title="에이전트 테스트 대시보드",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 단일 에이전트 테스트 대시보드")


def load_data():
    """시뮬레이션 데이터 로드 (캐시 없음 - 항상 최신 데이터)"""
    # 전체 결과
    result_path = OUTPUT_DIR / "generative_simulation_result.csv"
    if result_path.exists():
        results_df = pd.read_csv(result_path)
        results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
    else:
        results_df = pd.DataFrame()

    # 방문 로그
    visit_path = OUTPUT_DIR / "generative_visit_log.csv"
    if visit_path.exists():
        visits_df = pd.read_csv(visit_path)
        visits_df['timestamp'] = pd.to_datetime(visits_df['timestamp'])
    else:
        visits_df = pd.DataFrame()

    # 매장 데이터 (좌표 정보용)
    stores_dict = {}
    json_dir = DATA_DIR / "raw" / "split_by_store_id"
    if json_dir.exists():
        for json_file in json_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data and len(data) > 0:
                        store = data[0]
                        stores_dict[store.get('store_name', '')] = {
                            'lat': store.get('y', 0),
                            'lng': store.get('x', 0),
                            'category': store.get('category', '')
                        }
            except Exception:
                continue

    # 방문 로그에 매장 좌표 추가
    if not visits_df.empty and stores_dict:
        visits_df['store_lat'] = visits_df['visited_store'].map(lambda x: stores_dict.get(x, {}).get('lat', 0))
        visits_df['store_lng'] = visits_df['visited_store'].map(lambda x: stores_dict.get(x, {}).get('lng', 0))

    # 전체 결과에도 매장 좌표 추가
    if not results_df.empty and stores_dict:
        results_df['store_lat'] = results_df['visited_store'].map(lambda x: stores_dict.get(x, {}).get('lat', 0) if pd.notna(x) else None)
        results_df['store_lng'] = results_df['visited_store'].map(lambda x: stores_dict.get(x, {}).get('lng', 0) if pd.notna(x) else None)

    return results_df, visits_df, stores_dict


# 데이터 로드
results_df, visits_df, stores_dict = load_data()

if results_df.empty:
    st.error("시뮬레이션 결과가 없습니다. 먼저 시뮬레이션을 실행하세요.")
    st.stop()

# 사이드바
st.sidebar.header("에이전트 정보")

# 에이전트 정보
agent_name = results_df['agent_name'].iloc[0]
agent_segment = results_df['segment'].iloc[0]
agent_health = results_df['health_preference'].iloc[0]
agent_change = results_df['change_preference'].iloc[0]

st.sidebar.markdown(f"""
- **이름**: {agent_name}
- **세그먼트**: {agent_segment}
- **건강성향**: {agent_health}
- **변화성향**: {agent_change}
""")

# 통계
st.sidebar.markdown("---")
st.sidebar.markdown("### 시뮬레이션 통계")
total_slots = len(results_df)
visit_count = len(visits_df)
stay_home_count = total_slots - visit_count

col1, col2 = st.sidebar.columns(2)
col1.metric("총 타임슬롯", f"{total_slots}개")
col2.metric("망원동 방문", f"{visit_count}회")

# 방문 매장 목록
if not visits_df.empty:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 방문 매장 TOP")
    store_counts = visits_df['visited_store'].value_counts()
    for store, count in store_counts.head(5).items():
        st.sidebar.write(f"• {store}: {count}회")

# 메인 영역
tab1, tab2, tab3 = st.tabs(["🗺️ 지도", "📋 방문 로그", "📊 통계"])

# ========== TAB 1: 지도 ==========
with tab1:
    st.markdown("## 에이전트 이동 경로")

    # 지도 옵션
    opt_col1, opt_col2, opt_col3 = st.columns([1, 1, 2])
    show_path = opt_col1.checkbox("이동경로 표시", value=True)
    show_all_stores = opt_col2.checkbox("방문 매장만 표시", value=True)

    # 타임라인 슬라이더
    results_sorted = results_df.sort_values('timestamp')
    timestamps = results_sorted['timestamp'].tolist()

    if len(timestamps) > 1:
        st.markdown("### 타임라인")

        # 애니메이션 컨트롤
        anim_col1, anim_col2, anim_col3 = st.columns([1, 1, 4])

        if 'current_step' not in st.session_state:
            st.session_state.current_step = len(timestamps) - 1

        if anim_col1.button("⏮️ 처음"):
            st.session_state.current_step = 0

        if anim_col2.button("▶️ 재생"):
            for i in range(st.session_state.current_step, len(timestamps)):
                st.session_state.current_step = i
                time_module.sleep(0.5)
                st.rerun()

        current_step = st.slider(
            "시간대 선택",
            0, len(timestamps) - 1,
            st.session_state.current_step,
            format=f"Step %d"
        )
        st.session_state.current_step = current_step

        current_time = timestamps[current_step]
        current_row = results_sorted.iloc[current_step]

        # 현재 상태 표시
        st.markdown(f"""
        **현재 시간**: {current_time.strftime('%Y-%m-%d %H:%M')} ({current_row['weekday']}요일 {current_row['time_slot']})
        """)

        if current_row['decision'] == 'visit':
            st.success(f"🍽️ **{current_row['visited_store']}** 방문 중")
        else:
            st.info(f"🏠 망원동 외부 식사")

    # 방문 기록이 있는 경우 지도 표시
    if not visits_df.empty:
        # 지도 중심점
        center_lat = visits_df['store_lat'].mean()
        center_lng = visits_df['store_lng'].mean()

        # Folium 지도 생성
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=15,
            tiles='cartodbpositron'
        )

        # 시간순 정렬
        visits_sorted = visits_df.sort_values('timestamp')

        # 현재 시점까지의 방문 기록
        if len(timestamps) > 1:
            visits_until_now = visits_sorted[visits_sorted['timestamp'] <= current_time]
        else:
            visits_until_now = visits_sorted

        # 이동경로 표시
        if show_path and len(visits_until_now) > 1:
            path_coords = []
            for _, row in visits_until_now.iterrows():
                path_coords.append([row['store_lat'], row['store_lng']])

            folium.PolyLine(
                path_coords,
                color='#3388ff',
                weight=4,
                opacity=0.8,
                dash_array='10'
            ).add_to(m)

        # 방문 매장 마커
        for idx, (_, row) in enumerate(visits_until_now.iterrows(), 1):
            store_name = row['visited_store']
            timestamp = row['timestamp'].strftime('%m/%d %H:%M')
            category = row.get('visited_category', '')
            taste = row.get('taste_rating', '-')
            value = row.get('value_rating', '-')
            atmosphere = row.get('atmosphere_rating', '-')
            reason = str(row.get('reason', ''))[:150]

            popup_html = f"""
            <div style="width: 280px; font-family: sans-serif;">
                <h4 style="margin: 0 0 8px 0; color: #333;">#{idx} {store_name}</h4>
                <p style="margin: 0 0 5px 0; color: #666; font-size: 12px;">{category}</p>
                <hr style="margin: 8px 0; border: none; border-top: 1px solid #eee;">
                <p style="margin: 4px 0;"><b>방문시간:</b> {timestamp}</p>
                <p style="margin: 4px 0;"><b>평점:</b> 맛 {taste} / 가성비 {value} / 분위기 {atmosphere}</p>
                <hr style="margin: 8px 0; border: none; border-top: 1px solid #eee;">
                <p style="margin: 4px 0; font-size: 11px; color: #555;"><b>방문 이유:</b><br>{reason}...</p>
            </div>
            """

            # 마커 색상 (최근 방문은 빨강, 이전은 파랑)
            is_current = (len(timestamps) > 1 and row['timestamp'] == visits_until_now['timestamp'].max())
            marker_color = 'red' if is_current else 'blue'

            folium.Marker(
                location=[row['store_lat'], row['store_lng']],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"#{idx} {store_name}",
                icon=folium.Icon(color=marker_color, icon='cutlery', prefix='fa')
            ).add_to(m)

            # 순서 번호 원
            folium.CircleMarker(
                location=[row['store_lat'], row['store_lng']],
                radius=12,
                color='white',
                fill=True,
                fillColor=marker_color,
                fillOpacity=0.9,
                weight=2
            ).add_to(m)

            folium.Marker(
                location=[row['store_lat'], row['store_lng']],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 11px; color: white; font-weight: bold; text-align: center; line-height: 24px;">{idx}</div>',
                    icon_size=(24, 24),
                    icon_anchor=(12, 12)
                )
            ).add_to(m)

        # 에이전트 마커 (현재 위치)
        if len(timestamps) > 1 and current_row['decision'] == 'visit' and pd.notna(current_row.get('store_lat')):
            folium.Marker(
                location=[current_row['store_lat'], current_row['store_lng']],
                icon=folium.Icon(color='green', icon='user', prefix='fa'),
                tooltip=f"🧑 {agent_name} (현재 위치)"
            ).add_to(m)

        # 지도 표시
        st_folium(m, width=1100, height=550)

    else:
        st.warning("방문 기록이 없습니다.")

# ========== TAB 2: 방문 로그 ==========
with tab2:
    st.markdown("## 전체 방문 로그")

    # 날짜별 그룹핑
    results_df['date'] = results_df['timestamp'].dt.date
    dates = sorted(results_df['date'].unique())

    for date in dates:
        day_data = results_df[results_df['date'] == date].sort_values('timestamp')
        weekday = day_data['weekday'].iloc[0]

        with st.expander(f"📅 {date} ({weekday}요일)", expanded=(date == dates[0])):
            for _, row in day_data.iterrows():
                time_slot = row['time_slot']
                decision = row['decision']
                reason = str(row.get('reason', ''))

                if decision == 'visit':
                    store = row['visited_store']
                    category = row['visited_category']
                    taste = row.get('taste_rating', '-')
                    value = row.get('value_rating', '-')
                    atm = row.get('atmosphere_rating', '-')

                    st.markdown(f"""
                    <div style="background: #d4edda; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                        <b>[{time_slot}]</b> 🍽️ <b>{store}</b> 방문<br>
                        <small style="color: #666;">카테고리: {category}</small><br>
                        <small>평점: 맛 {taste} / 가성비 {value} / 분위기 {atm}</small><br>
                        <hr style="margin: 8px 0; border: none; border-top: 1px solid #c3e6cb;">
                        <small><b>방문 이유:</b> {reason}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #cce5ff; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                        <b>[{time_slot}]</b> 🏠 망원동 외부 식사<br>
                        <hr style="margin: 8px 0; border: none; border-top: 1px solid #b8daff;">
                        <small><b>이유:</b> {reason}</small>
                    </div>
                    """, unsafe_allow_html=True)

# ========== TAB 3: 통계 ==========
with tab3:
    st.markdown("## 시뮬레이션 통계")

    if not visits_df.empty:
        # 평점 요약
        st.markdown("### 평균 평점")
        col1, col2, col3 = st.columns(3)

        avg_taste = visits_df['taste_rating'].mean()
        avg_value = visits_df['value_rating'].mean()
        avg_atm = visits_df['atmosphere_rating'].mean()

        col1.metric("맛", f"{avg_taste:.1f} / 5")
        col2.metric("가성비", f"{avg_value:.1f} / 5")
        col3.metric("분위기", f"{avg_atm:.1f} / 5")

        # 카테고리별 방문
        st.markdown("### 카테고리별 방문 횟수")
        category_counts = visits_df['visited_category'].value_counts()
        st.bar_chart(category_counts)

        # 방문 매장 테이블
        st.markdown("### 방문 매장 상세")
        visit_summary = visits_df.groupby('visited_store').agg({
            'timestamp': 'count',
            'taste_rating': 'mean',
            'value_rating': 'mean',
            'atmosphere_rating': 'mean'
        }).rename(columns={
            'timestamp': '방문횟수',
            'taste_rating': '평균맛',
            'value_rating': '평균가성비',
            'atmosphere_rating': '평균분위기'
        }).round(1).sort_values('방문횟수', ascending=False)

        st.dataframe(visit_summary, use_container_width=True)

    else:
        st.info("방문 기록이 없어 통계를 표시할 수 없습니다.")
