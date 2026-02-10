"""
망원동 에이전트 시뮬레이션 대시보드

Streamlit 기반 대시보드로 Generative Agents 시뮬레이션 결과를 시각화합니다.

실행 방법:
    streamlit run scripts/dashboard.py
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import random
import networkx as nx
import osmnx as ox
import time as time_module

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"


def load_target_store() -> str:
    """타겟 매장 설정 로드 (시뮬레이션과 공유)"""
    config_path = OUTPUT_DIR / "target_store.json"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get("target_store", "류진")
        except Exception:
            return "류진"
    return "류진"


# 타겟 매장 로드
TARGET_STORE = load_target_store()

# 페이지 설정
st.set_page_config(
    page_title="망원동 에이전트 시뮬레이션 대시보드",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS 스타일
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_simulation_data():
    """시뮬레이션 데이터 로드"""
    # 전체 결과
    result_path = OUTPUT_DIR / "generative_simulation_result.csv"
    if result_path.exists():
        results_df = pd.read_csv(result_path)
        results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
        results_df['date'] = results_df['timestamp'].dt.date
    else:
        results_df = pd.DataFrame()

    # 방문 로그
    visit_path = OUTPUT_DIR / "generative_visit_log.csv"
    if visit_path.exists():
        visits_df = pd.read_csv(visit_path)
        visits_df['timestamp'] = pd.to_datetime(visits_df['timestamp'])
        visits_df['date'] = visits_df['timestamp'].dt.date
    else:
        visits_df = pd.DataFrame()

    # 에이전트 상태
    agents_path = OUTPUT_DIR / "agents_final_state.json"
    if agents_path.exists():
        with open(agents_path, 'r', encoding='utf-8') as f:
            agents = json.load(f)
    else:
        agents = []

    # 매장 데이터
    stores_path = DATA_DIR / "raw" / "stores.csv"
    if stores_path.exists():
        stores_df = pd.read_csv(stores_path)
    else:
        stores_df = pd.DataFrame()

    return results_df, visits_df, agents, stores_df


@st.cache_data
def load_street_network():
    """OSM 도로망 로드"""
    try:
        # 망원동 중심으로 도로망 로드
        G = ox.graph_from_point((37.5565, 126.9029), dist=800, network_type='walk')
        return G
    except Exception as e:
        st.warning(f"도로망 로드 실패: {e}")
        return None


def get_route_on_network(G, start_coords, end_coords):
    """OSM 네트워크 위의 경로 계산"""
    if G is None:
        return None
    try:
        # 가장 가까운 노드 찾기
        start_node = ox.nearest_nodes(G, start_coords[1], start_coords[0])
        end_node = ox.nearest_nodes(G, end_coords[1], end_coords[0])

        # 최단 경로 계산
        route = nx.shortest_path(G, start_node, end_node, weight='length')

        # 노드 좌표 추출
        route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
        return route_coords
    except Exception:
        return None


def create_animated_agent_map(results_df, visits_df, stores_df, agent_name, agent_info,
                               current_step=None, show_trail=True, G=None):
    """
    특정 에이전트의 이동을 애니메이션으로 보여주는 지도 생성.

    Args:
        current_step: 현재 표시할 단계 (0부터 시작). None이면 전체 경로 표시.
        show_trail: 이전 경로를 표시할지 여부
    """
    agent_results = results_df[results_df['agent_name'] == agent_name].copy()
    agent_visits = visits_df[visits_df['agent_name'] == agent_name].copy() if not visits_df.empty else pd.DataFrame()

    if agent_results.empty:
        return None, []

    has_location = 'agent_lat' in agent_results.columns and 'agent_lng' in agent_results.columns

    # 지도 중심 계산
    if has_location:
        valid_locs = agent_results.dropna(subset=['agent_lat', 'agent_lng'])
        if not valid_locs.empty:
            center_lat = valid_locs['agent_lat'].mean()
            center_lng = valid_locs['agent_lng'].mean()
        else:
            center_lat, center_lng = 37.5565, 126.9029
    else:
        center_lat, center_lng = 37.5565, 126.9029

    m = folium.Map(location=[center_lat, center_lng], zoom_start=15)

    # 색상 설정
    SEGMENT_COLORS = {
        '상주_1인가구': '#2ecc71', '상주_외부출퇴근': '#27ae60',
        '상주_2인가구': '#58d68d', '상주_4인가구': '#1abc9c',
        '유동_망원유입직장인': '#e67e22', '유동_나홀로방문': '#e91e63',
        '유동_데이트': '#e74c3c', '유동_약속모임': '#9b59b6',
    }
    agent_color = SEGMENT_COLORS.get(agent_info.get('segment', ''), '#3498db')

    TIME_COLORS = {
        '아침': '#FFA726', '점심': '#66BB6A', '저녁': '#42A5F5', '야식': '#AB47BC',
    }

    # 타임라인 데이터 수집
    timeline_data = []

    if has_location:
        agent_results = agent_results.sort_values('timestamp')

        for idx, row in agent_results.iterrows():
            if pd.notna(row.get('agent_lat')) and pd.notna(row.get('agent_lng')):
                timeline_data.append({
                    'lat': row['agent_lat'],
                    'lng': row['agent_lng'],
                    'time_slot': row.get('time_slot', ''),
                    'timestamp': row['timestamp'],
                    'decision': row.get('decision', ''),
                    'visited_store': row.get('visited_store', ''),
                    'date': row['timestamp'].strftime('%Y-%m-%d') if pd.notna(row['timestamp']) else '',
                    'time': row['timestamp'].strftime('%H:%M') if pd.notna(row['timestamp']) else '',
                })

    if not timeline_data:
        return m, timeline_data

    # 표시할 범위 결정
    if current_step is None:
        display_data = timeline_data
        current_idx = len(timeline_data) - 1
    else:
        current_idx = min(current_step, len(timeline_data) - 1)
        display_data = timeline_data[:current_idx + 1]

    # 이전 경로 표시 (trail)
    if show_trail and len(display_data) > 1:
        trail_coords = [(p['lat'], p['lng']) for p in display_data]
        folium.PolyLine(
            trail_coords,
            weight=3,
            color=agent_color,
            opacity=0.5,
            dash_array='5',
        ).add_to(m)

    # 과거 위치 마커 (작은 점)
    for i, point in enumerate(display_data[:-1]):
        time_color = TIME_COLORS.get(point['time_slot'], '#999')
        is_visit = point['decision'] == 'visit'

        folium.CircleMarker(
            location=[point['lat'], point['lng']],
            radius=4 if is_visit else 3,
            color=time_color,
            fill=True,
            fill_color=time_color,
            fill_opacity=0.4,
            tooltip=f"{point['date']} {point['time_slot']} ({point['time']})"
        ).add_to(m)

    # 현재 위치 마커 (크게 강조)
    if display_data:
        current = display_data[-1]
        time_color = TIME_COLORS.get(current['time_slot'], '#999')
        is_visit = current['decision'] == 'visit'

        # 현재 위치 펄스 효과 (외부 원)
        folium.CircleMarker(
            location=[current['lat'], current['lng']],
            radius=20,
            color=time_color,
            fill=True,
            fill_color=time_color,
            fill_opacity=0.2,
            weight=2,
        ).add_to(m)

        # 현재 위치 마커
        popup_html = f"""
        <div style="min-width: 180px;">
        <b>📍 현재 위치</b><br>
        <hr style="margin: 5px 0;">
        <b>날짜:</b> {current['date']}<br>
        <b>시간:</b> {current['time_slot']} ({current['time']})<br>
        <b>결정:</b> {'🍽️ 방문' if is_visit else '🏠 외출안함'}<br>
        """
        if is_visit and current['visited_store']:
            popup_html += f"<b>매장:</b> {current['visited_store']}"
        popup_html += "</div>"

        folium.CircleMarker(
            location=[current['lat'], current['lng']],
            radius=12,
            color='white',
            fill=True,
            fill_color=time_color,
            fill_opacity=1.0,
            weight=3,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"📍 현재: {current['time_slot']} - {'방문' if is_visit else '외출안함'}"
        ).add_to(m)

        # 방문한 매장 표시
        if is_visit and current['visited_store'] and not stores_df.empty:
            store_row = stores_df[stores_df['장소명'] == current['visited_store']]
            if not store_row.empty:
                store = store_row.iloc[0]
                folium.Marker(
                    location=[float(store['y']), float(store['x'])],
                    icon=folium.Icon(color='red', icon='cutlery', prefix='fa'),
                    tooltip=f"🍽️ {current['visited_store']}"
                ).add_to(m)

    # 범례
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.3);">
    <b>🏃 {agent_name}</b><br>
    <small>Step {current_idx + 1} / {len(timeline_data)}</small><br>
    <hr style="margin: 5px 0;">
    <i style="background:#FFA726; width:12px; height:12px; display:inline-block; border-radius:50%;"></i> 아침<br>
    <i style="background:#66BB6A; width:12px; height:12px; display:inline-block; border-radius:50%;"></i> 점심<br>
    <i style="background:#42A5F5; width:12px; height:12px; display:inline-block; border-radius:50%;"></i> 저녁<br>
    <i style="background:#AB47BC; width:12px; height:12px; display:inline-block; border-radius:50%;"></i> 야식<br>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m, timeline_data


def create_agent_trajectory_map(results_df, visits_df, stores_df, agent_name, agent_info, G=None):
    """특정 에이전트의 이동 경로를 시각화하는 지도 생성"""
    # 에이전트 데이터 필터링
    agent_results = results_df[results_df['agent_name'] == agent_name].copy()
    agent_visits = visits_df[visits_df['agent_name'] == agent_name].copy() if not visits_df.empty else pd.DataFrame()

    if agent_results.empty:
        return None

    # 위치 정보가 있는지 확인
    has_location = 'agent_lat' in agent_results.columns and 'agent_lng' in agent_results.columns

    # 지도 중심 계산
    if has_location:
        valid_locs = agent_results.dropna(subset=['agent_lat', 'agent_lng'])
        if not valid_locs.empty:
            center_lat = valid_locs['agent_lat'].mean()
            center_lng = valid_locs['agent_lng'].mean()
        else:
            center_lat, center_lng = 37.5565, 126.9029
    else:
        center_lat, center_lng = 37.5565, 126.9029

    m = folium.Map(location=[center_lat, center_lng], zoom_start=15)

    # 세그먼트별 색상
    SEGMENT_COLORS = {
        '상주_1인가구': '#2ecc71',
        '상주_외부출퇴근': '#27ae60',
        '상주_2인가구': '#58d68d',
        '상주_4인가구': '#1abc9c',
        '유동_망원유입직장인': '#e67e22',
        '유동_나홀로방문': '#e91e63',
        '유동_데이트': '#e74c3c',
        '유동_약속모임': '#9b59b6',
    }
    agent_color = SEGMENT_COLORS.get(agent_info.get('segment', ''), '#3498db')

    # 타임슬롯별 색상
    TIME_COLORS = {
        '아침': '#FFA726',  # 주황
        '점심': '#66BB6A',  # 초록
        '저녁': '#42A5F5',  # 파랑
        '야식': '#AB47BC',  # 보라
    }

    # 이동 경로 그리기 (위치 정보가 있을 때)
    if has_location:
        agent_results = agent_results.sort_values('timestamp')
        trajectory_points = []

        for idx, row in agent_results.iterrows():
            if pd.notna(row.get('agent_lat')) and pd.notna(row.get('agent_lng')):
                lat, lng = row['agent_lat'], row['agent_lng']
                time_slot = row.get('time_slot', '')
                decision = row.get('decision', '')
                timestamp = row['timestamp']

                trajectory_points.append({
                    'lat': lat,
                    'lng': lng,
                    'time_slot': time_slot,
                    'timestamp': timestamp,
                    'decision': decision,
                    'visited_store': row.get('visited_store', ''),
                })

        # 이동 경로 라인 그리기
        if len(trajectory_points) > 1:
            coords = [(p['lat'], p['lng']) for p in trajectory_points]
            folium.PolyLine(
                coords,
                weight=3,
                color=agent_color,
                opacity=0.7,
                dash_array='5',
                tooltip=f"{agent_name} 이동 경로"
            ).add_to(m)

        # 각 타임슬롯 위치에 마커 추가
        for i, point in enumerate(trajectory_points):
            time_color = TIME_COLORS.get(point['time_slot'], '#999')
            is_visit = point['decision'] == 'visit'

            popup_html = f"""
            <div style="min-width: 150px;">
            <b>📍 {point['time_slot']}</b><br>
            <small>{point['timestamp']}</small><br>
            <hr style="margin: 5px 0;">
            <b>결정:</b> {'🍽️ 방문' if is_visit else '🏠 외출안함'}<br>
            """
            if is_visit and point['visited_store']:
                popup_html += f"<b>매장:</b> {point['visited_store']}"
            popup_html += "</div>"

            # 방문 시에는 더 큰 마커
            radius = 10 if is_visit else 6
            fill_opacity = 0.9 if is_visit else 0.5

            folium.CircleMarker(
                location=[point['lat'], point['lng']],
                radius=radius,
                color=time_color,
                fill=True,
                fill_color=time_color,
                fill_opacity=fill_opacity,
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=f"{point['time_slot']} - {'방문' if is_visit else '외출안함'}"
            ).add_to(m)

            # 순서 번호 표시
            folium.Marker(
                location=[point['lat'], point['lng']],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 10px; font-weight: bold; color: white; background: {time_color}; border-radius: 50%; width: 16px; height: 16px; text-align: center; line-height: 16px;">{i+1}</div>',
                    icon_size=(16, 16),
                    icon_anchor=(8, 8)
                )
            ).add_to(m)

    # 방문한 매장 마커 추가
    if not stores_df.empty and not agent_visits.empty:
        for _, visit in agent_visits.iterrows():
            store_name = visit['visited_store']
            store_row = stores_df[stores_df['장소명'] == store_name]
            if not store_row.empty:
                store = store_row.iloc[0]
                lat, lng = float(store['y']), float(store['x'])

                popup_html = f"""
                <div style="min-width: 180px;">
                <b>🍽️ {store_name}</b><br>
                <hr style="margin: 5px 0;">
                <b>시간:</b> {visit['time_slot']}<br>
                <b>카테고리:</b> {visit.get('visited_category', '')}<br>
                <b>맛:</b> {visit.get('taste_rating', '-')}점<br>
                <b>가성비:</b> {visit.get('value_rating', '-')}점<br>
                </div>
                """

                folium.Marker(
                    location=[lat, lng],
                    icon=folium.Icon(color='red', icon='cutlery', prefix='fa'),
                    popup=folium.Popup(popup_html, max_width=200),
                    tooltip=f"🍽️ {store_name}"
                ).add_to(m)

    # 범례 추가
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.3);">
    <b>타임슬롯</b><br>
    <i style="background:#FFA726; width:12px; height:12px; display:inline-block; border-radius:50%;"></i> 아침 07:00<br>
    <i style="background:#66BB6A; width:12px; height:12px; display:inline-block; border-radius:50%;"></i> 점심 12:00<br>
    <i style="background:#42A5F5; width:12px; height:12px; display:inline-block; border-radius:50%;"></i> 저녁 18:00<br>
    <i style="background:#AB47BC; width:12px; height:12px; display:inline-block; border-radius:50%;"></i> 야식 22:00<br>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def create_map_with_routes(visits_df, stores_df, agents, selected_date=None,
                           store_filter=None, show_routes=False, G=None,
                           target_store=None):
    """Folium 지도 생성 (타겟 매장 방문 에이전트 중심)"""
    # 타겟 매장 설정
    if target_store is None:
        target_store = TARGET_STORE
    # 지도 중심 (망원동)
    center_lat, center_lon = 37.5565, 126.9029

    m = folium.Map(location=[center_lat, center_lon], zoom_start=16)

    # 날짜 필터링
    if selected_date is not None:
        visits_filtered = visits_df[visits_df['date'] == selected_date]
    else:
        visits_filtered = visits_df

    # 매장 필터링
    if store_filter and store_filter != "전체":
        visits_filtered = visits_filtered[visits_filtered['visited_store'] == store_filter]

    # 방문 횟수 계산
    visit_counts = visits_filtered['visited_store'].value_counts().to_dict()

    # 세그먼트별 색상
    SEGMENT_COLORS = {
        '상주_1인가구': '#2ecc71',
        '상주_외부출퇴근': '#27ae60',
        '상주_2인가구': '#58d68d',
        '상주_4인가구': '#1abc9c',
        '유동_망원유입직장인': '#e67e22',
        '유동_나홀로방문': '#e91e63',
        '유동_데이트': '#e74c3c',
        '유동_약속모임': '#9b59b6',
    }

    # 매장 위치 딕셔너리
    store_locations = {}
    if not stores_df.empty:
        for _, store in stores_df.iterrows():
            store_locations[store['장소명']] = (float(store['y']), float(store['x']))

    # 에이전트 위치 생성 (일관된 시드)
    random.seed(42)
    lat_min, lat_max = 37.552, 37.562
    lon_min, lon_max = 126.895, 126.911

    agent_locations = {}
    for agent in agents:
        segment = agent['segment']

        # 세그먼트별 위치 클러스터링
        if '상주' in segment:
            lat = random.uniform(lat_min + 0.003, lat_max - 0.002)
            lon = random.uniform(lon_min + 0.003, lon_max - 0.005)
        else:
            lat = random.uniform(lat_min + 0.001, lat_max - 0.001)
            lon = random.uniform(lon_min + 0.005, lon_max - 0.002)

        agent_locations[agent['name']] = (lat, lon)

    # 타겟 매장 방문 에이전트 목록
    target_visitors = set()
    if not visits_df.empty:
        target_visits = visits_df[visits_df['visited_store'] == target_store]
        target_visitors = set(target_visits['agent_name'].unique())

    # 타겟 매장 위치
    target_loc = store_locations.get(target_store, None)

    # 타겟 매장 방문 에이전트의 경로를 FeatureGroup으로 관리 (마우스 호버 시 표시)
    route_groups = {}

    for agent_name in target_visitors:
        if agent_name in agent_locations and target_loc:
            agent_loc = agent_locations[agent_name]
            color = '#3498db'  # 기본 파란색

            # 에이전트 정보 찾기
            for agent in agents:
                if agent['name'] == agent_name:
                    color = SEGMENT_COLORS.get(agent['segment'], '#3498db')
                    break

            # 경로 그룹 생성
            fg = folium.FeatureGroup(name=f"route_{agent_name}", show=False)

            # OSM 네트워크 경로 계산
            if G is not None:
                route_coords = get_route_on_network(G, agent_loc, target_loc)
                if route_coords and len(route_coords) > 1:
                    folium.PolyLine(
                        route_coords,
                        weight=4,
                        color=color,
                        opacity=0.8,
                        tooltip=f"{agent_name} → {target_store}"
                    ).add_to(fg)
                else:
                    folium.PolyLine(
                        [agent_loc, target_loc],
                        weight=4,
                        color=color,
                        opacity=0.8,
                        dash_array='10',
                        tooltip=f"{agent_name} → {target_store}"
                    ).add_to(fg)
            else:
                folium.PolyLine(
                    [agent_loc, target_loc],
                    weight=4,
                    color=color,
                    opacity=0.8,
                    dash_array='10',
                    tooltip=f"{agent_name} → {target_store}"
                ).add_to(fg)

            fg.add_to(m)
            route_groups[agent_name] = fg._name

    # 타겟 매장 방문 에이전트만 마커 추가
    marker_idx = 0
    for agent in agents:
        agent_name = agent['name']

        # 타겟 매장 방문자만 표시
        if agent_name not in target_visitors:
            continue

        if agent_name not in agent_locations:
            continue

        segment = agent['segment']
        color = SEGMENT_COLORS.get(segment, '#95a5a6')
        lat, lon = agent_locations[agent_name]

        # 방문 기록에서 타겟 매장 방문 정보 찾기
        target_visit_info = ""
        if not visits_df.empty:
            agent_target = visits_df[(visits_df['visited_store'] == target_store) &
                                      (visits_df['agent_name'] == agent_name)]
            if not agent_target.empty:
                for _, visit in agent_target.iterrows():
                    target_visit_info += f"<br>• {visit['time_slot']} - 맛:{visit['taste_rating']}점, 가성비:{visit['value_rating']}점"

        popup_html = f"""
        <div style="min-width: 200px;">
        <b style="font-size: 14px;">📍 {agent_name}</b><br>
        <hr style="margin: 5px 0;">
        <b>세대:</b> {agent['generation']}<br>
        <b>세그먼트:</b> {segment}<br>
        <b>건강성향:</b> {agent['health_preference']}<br>
        <b>변화성향:</b> {agent['change_preference']}<br>
        <b>예산:</b> {agent['budget_per_meal']:,}원<br>
        <hr style="margin: 5px 0;">
        <b>{target_store} 방문 기록:</b>{target_visit_info}
        </div>
        """

        # 마커 ID 생성
        marker_id = f"marker_{marker_idx}"
        marker_idx += 1

        # 에이전트 마커 추가 (마우스 호버 이벤트 포함)
        marker = folium.CircleMarker(
            location=[lat, lon],
            radius=10,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"📍 {agent_name} ({agent['generation']}세대, {segment})"
        )
        marker.add_to(m)

        # 마우스 호버 시 경로 표시를 위한 JavaScript 이벤트 추가
        if agent_name in route_groups:
            route_name = route_groups[agent_name]
            # JavaScript로 마우스 이벤트 처리
            hover_js = f"""
            <script>
            (function() {{
                var marker = document.querySelector('[data-marker-id="{marker_id}"]');
                if (marker) {{
                    marker.addEventListener('mouseover', function() {{
                        var route = document.querySelector('[data-route="{agent_name}"]');
                        if (route) route.style.display = 'block';
                    }});
                    marker.addEventListener('mouseout', function() {{
                        var route = document.querySelector('[data-route="{agent_name}"]');
                        if (route) route.style.display = 'none';
                    }});
                }}
            }})();
            </script>
            """

    # 이동경로 표시 옵션이 켜져있으면 모든 경로 표시
    if show_routes:
        for agent_name in target_visitors:
            if agent_name in agent_locations and target_loc:
                agent_loc = agent_locations[agent_name]
                color = '#3498db'

                for agent in agents:
                    if agent['name'] == agent_name:
                        color = SEGMENT_COLORS.get(agent['segment'], '#3498db')
                        break

                if G is not None:
                    route_coords = get_route_on_network(G, agent_loc, target_loc)
                    if route_coords and len(route_coords) > 1:
                        folium.PolyLine(
                            route_coords,
                            weight=3,
                            color=color,
                            opacity=0.7,
                            tooltip=f"{agent_name} → {target_store}"
                        ).add_to(m)
                    else:
                        folium.PolyLine(
                            [agent_loc, target_loc],
                            weight=3,
                            color=color,
                            opacity=0.7,
                            dash_array='8',
                            tooltip=f"{agent_name} → {target_store}"
                        ).add_to(m)
                else:
                    folium.PolyLine(
                        [agent_loc, target_loc],
                        weight=3,
                        color=color,
                        opacity=0.7,
                        dash_array='8',
                        tooltip=f"{agent_name} → {target_store}"
                    ).add_to(m)

    # 전체 방문 에이전트 표시 (store_filter가 전체일 때)
    if store_filter == "전체" or store_filter is None:
        visited_agents = set(visits_filtered['agent_name'].unique()) if not visits_filtered.empty else set()
        for agent in agents:
            agent_name = agent['name']
            # 타겟 매장 방문자는 이미 표시됨
            if agent_name in target_visitors:
                continue
            if agent_name not in visited_agents:
                continue
            if agent_name not in agent_locations:
                continue

            segment = agent['segment']
            color = SEGMENT_COLORS.get(segment, '#95a5a6')
            lat, lon = agent_locations[agent_name]

            # 작은 마커로 표시
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.4,
                tooltip=f"{agent_name} ({agent['generation']}세대)"
            ).add_to(m)

    # 매장 마커 추가
    if not stores_df.empty:
        for _, store in stores_df.iterrows():
            store_name = store['장소명']
            lat = float(store['y'])
            lon = float(store['x'])
            count = visit_counts.get(store_name, 0)

            # 타겟 매장 특별 표시
            if store_name == target_store:
                folium.Marker(
                    location=[lat, lon],
                    icon=folium.Icon(color='red', icon='star', prefix='fa'),
                    popup=f"⭐ {store_name}<br>방문: {count}회<br>카테고리: {store['카테고리']}",
                    tooltip=f"⭐ {store_name}: {count}회"
                ).add_to(m)
            elif count > 0:
                # 방문된 매장 (일반 마커)
                if count >= 3:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=6 + count,
                        color='blue',
                        fill=True,
                        fill_opacity=0.7,
                        popup=f"{store_name}<br>방문: {count}회",
                        tooltip=f"{store_name}: {count}회"
                    ).add_to(m)
                else:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=5,
                        color='lightblue',
                        fill=True,
                        fill_opacity=0.5,
                        tooltip=f"{store_name}: {count}회"
                    ).add_to(m)

    return m


def main():
    # 데이터 로드
    results_df, visits_df, agents, stores_df = load_simulation_data()

    # 사이드바 - 필터
    st.sidebar.title("필터")
    st.sidebar.markdown("---")

    # 날짜 선택
    if not visits_df.empty:
        available_dates = sorted(visits_df['date'].unique())
        date_options = ["전체"] + [str(d) for d in available_dates]
        selected_date_str = st.sidebar.selectbox("날짜", date_options, index=0)

        if selected_date_str == "전체":
            selected_date = None
            filtered_visits = visits_df
            filtered_results = results_df
        else:
            selected_date = pd.to_datetime(selected_date_str).date()
            filtered_visits = visits_df[visits_df['date'] == selected_date]
            filtered_results = results_df[results_df['date'] == selected_date]
    else:
        selected_date = None
        filtered_visits = visits_df
        filtered_results = results_df

    # 시간대 선택
    time_slots = ["전체", "아침", "점심", "저녁", "야식"]
    selected_time = st.sidebar.selectbox("시간대", time_slots, index=0)

    if selected_time != "전체" and not filtered_visits.empty:
        filtered_visits = filtered_visits[filtered_visits['time_slot'] == selected_time]
        filtered_results = filtered_results[filtered_results['time_slot'] == selected_time]

    # 매장 필터 (타겟 매장 포함)
    st.sidebar.markdown("---")
    st.sidebar.subheader("매장 필터")

    if not visits_df.empty:
        all_stores = ["전체", TARGET_STORE] + sorted([s for s in visits_df['visited_store'].unique() if s != TARGET_STORE])
        store_filter = st.sidebar.selectbox("특정 매장만 보기", all_stores, index=0)

        if store_filter != "전체":
            filtered_visits = filtered_visits[filtered_visits['visited_store'] == store_filter]
    else:
        store_filter = "전체"

    # 이동경로 표시 옵션
    st.sidebar.markdown("---")
    show_routes = st.sidebar.checkbox("이동경로 표시", value=False)

    # 에이전트 선택 (개별 추적용)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 에이전트 추적")

    if agents and not results_df.empty:
        agent_names = ["전체"] + sorted([a['name'] for a in agents])
        selected_agent = st.sidebar.selectbox(
            "에이전트 선택",
            agent_names,
            index=0,
            help="특정 에이전트의 이동 경로와 방문 기록을 확인합니다"
        )
    else:
        selected_agent = "전체"

    # 도로망 로드 (이동경로 표시 시)
    G = None
    if show_routes:
        with st.sidebar:
            with st.spinner("도로망 로드 중..."):
                G = load_street_network()

    # 전체 진행 상황
    st.sidebar.markdown("---")
    st.sidebar.subheader("전체 진행 상황")

    if not results_df.empty:
        total_events = len(results_df)
        total_visits = len(visits_df)
        st.sidebar.markdown(f"진행: **{total_events}** 건")
        st.sidebar.markdown(f"방문: **{total_visits}** 건")

    # ⭐ 타겟 매장 방문 현황 (범례)
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"⭐ {TARGET_STORE} 방문 현황")

    if not visits_df.empty:
        target_visits = visits_df[visits_df['visited_store'] == TARGET_STORE]
        target_count = len(target_visits)
        st.sidebar.metric(f"{TARGET_STORE} 방문 횟수", f"{target_count}회")

        if target_count > 0:
            st.sidebar.markdown("**방문 에이전트:**")
            for _, visit in target_visits.iterrows():
                agent_name = visit['agent_name']
                generation = visit['generation']
                segment = visit['segment']
                time_slot = visit['time_slot']
                st.sidebar.markdown(f"- {agent_name} ({generation}세대)")
                st.sidebar.caption(f"  {segment} / {time_slot}")
        else:
            st.sidebar.info(f"{TARGET_STORE} 방문 기록 없음")
    else:
        st.sidebar.info("방문 데이터 없음")

    # 메인 콘텐츠
    st.markdown("## 🗺️ 망원동 에이전트 시뮬레이션 대시보드")

    # 현황 카드
    if selected_date:
        st.markdown(f"### 📅 {selected_date} 현황")
    else:
        st.markdown("### 📅 전체 기간 현황")

    if store_filter != "전체":
        st.markdown(f"**🔍 필터: {store_filter}**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if store_filter != "전체" and not filtered_visits.empty:
            active_agents = filtered_visits['agent_name'].nunique()
        else:
            active_agents = len(agents)
        st.metric("활동에이전트", f"{active_agents}")

    with col2:
        total_visits = len(filtered_visits)
        st.metric("총방문횟수", f"{total_visits}")

    with col3:
        if not filtered_visits.empty:
            unique_stores = filtered_visits['visited_store'].nunique()
        else:
            unique_stores = 0
        st.metric("방문 업체수", f"{unique_stores}")

    with col4:
        if not filtered_results.empty and len(filtered_results) > 0:
            conversion_rate = len(filtered_visits) / len(filtered_results) * 100
        else:
            conversion_rate = 0
        st.metric("전환율", f"{conversion_rate:.1f}%")

    st.markdown("---")

    # 지도와 차트
    col_map, col_charts = st.columns([2, 1])

    with col_map:
        st.markdown("### 🗺️ 에이전트 위치 & 방문 현황")
        if show_routes:
            st.caption("🛤️ 파란선: OSM 도로망 경로 / 점선: 직선 경로")
        if agents and not stores_df.empty:
            m = create_map_with_routes(
                visits_df if store_filter == "전체" else filtered_visits,
                stores_df, agents, selected_date, store_filter, show_routes, G,
                target_store=TARGET_STORE
            )
            st_folium(m, width=700, height=500)
        else:
            st.warning("지도 데이터를 불러올 수 없습니다.")

    with col_charts:
        # 시간대별 방문
        st.markdown("### 📊 시간대별 방문")
        if not filtered_visits.empty:
            time_visits = filtered_visits.groupby('time_slot').size().reset_index(name='count')
            time_order = ['아침', '점심', '저녁', '야식']
            time_visits['time_slot'] = pd.Categorical(time_visits['time_slot'], categories=time_order, ordered=True)
            time_visits = time_visits.sort_values('time_slot')

            fig_time = px.bar(
                time_visits,
                x='time_slot',
                y='count',
                color_discrete_sequence=['#1f77b4']
            )
            fig_time.update_layout(
                xaxis_title="",
                yaxis_title="방문 수",
                height=200,
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("방문 데이터 없음")

        # 인기 매장 TOP 10
        st.markdown("### 🏆 인기 매장 TOP 10")
        if not filtered_visits.empty:
            top_stores = filtered_visits['visited_store'].value_counts().head(10).reset_index()
            top_stores.columns = ['store', 'count']

            fig_stores = px.bar(
                top_stores,
                x='count',
                y='store',
                orientation='h',
                color_discrete_sequence=['#ff7f0e']
            )
            fig_stores.update_layout(
                xaxis_title="방문 수",
                yaxis_title="",
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_stores, use_container_width=True)
        else:
            st.info("방문 데이터 없음")

    # ==================== 에이전트 추적 섹션 ====================
    if selected_agent != "전체":
        st.markdown("---")
        st.markdown(f"## 🔍 에이전트 추적: {selected_agent}")

        # 에이전트 정보 가져오기
        agent_info = next((a for a in agents if a['name'] == selected_agent), None)

        if agent_info:
            # 에이전트 프로필
            col_profile, col_trajectory = st.columns([1, 2])

            with col_profile:
                st.markdown("### 👤 에이전트 프로필")
                st.markdown(f"**이름:** {agent_info['name']}")
                st.markdown(f"**세대:** {agent_info['generation']}")
                st.markdown(f"**나이:** {agent_info.get('age', '-')}세")
                st.markdown(f"**성별:** {agent_info.get('gender', '-')}")
                st.markdown(f"**세그먼트:** {agent_info['segment']}")
                st.markdown(f"**건강 성향:** {agent_info['health_preference']}")
                st.markdown(f"**변화 성향:** {agent_info['change_preference']}")
                st.markdown(f"**예산:** {agent_info['budget_per_meal']:,}원")

                # 최근 방문 기록 (에이전트 메모리)
                if 'recent_history' in agent_info and agent_info['recent_history']:
                    st.markdown("#### 📝 메모리 (최근 방문)")
                    for i, visit in enumerate(agent_info['recent_history'][-5:], 1):
                        st.caption(f"{i}. {visit.get('store_name', '?')} ({visit.get('category', '')})")

            with col_trajectory:
                st.markdown("### 🗺️ 이동 경로 애니메이션")

                # 타임라인 데이터 미리 로드
                _, timeline_data = create_animated_agent_map(
                    results_df, visits_df, stores_df,
                    selected_agent, agent_info, current_step=0
                )

                if timeline_data:
                    total_steps = len(timeline_data)

                    # 애니메이션 컨트롤
                    st.markdown("#### ▶️ 재생 컨트롤")

                    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 2, 1])

                    with col_ctrl1:
                        play_speed = st.selectbox(
                            "속도",
                            options=[0.5, 1.0, 2.0, 4.0],
                            index=1,
                            format_func=lambda x: f"{x}x"
                        )

                    with col_ctrl2:
                        # 세션 상태 초기화
                        if 'animation_step' not in st.session_state:
                            st.session_state.animation_step = 0
                        if 'is_playing' not in st.session_state:
                            st.session_state.is_playing = False

                        # 타임라인 슬라이더
                        current_step = st.slider(
                            "타임라인",
                            min_value=0,
                            max_value=total_steps - 1,
                            value=st.session_state.animation_step,
                            key="timeline_slider",
                            help="드래그하여 특정 시점으로 이동"
                        )
                        st.session_state.animation_step = current_step

                    with col_ctrl3:
                        # 현재 시점 정보
                        if current_step < len(timeline_data):
                            curr = timeline_data[current_step]
                            st.caption(f"📅 {curr['date']}")
                            st.caption(f"⏰ {curr['time_slot']} {curr['time']}")

                    # 재생/정지/리셋 버튼
                    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

                    with col_btn1:
                        if st.button("⏮️ 처음", use_container_width=True):
                            st.session_state.animation_step = 0
                            st.rerun()

                    with col_btn2:
                        if st.button("◀️ 이전", use_container_width=True):
                            if st.session_state.animation_step > 0:
                                st.session_state.animation_step -= 1
                                st.rerun()

                    with col_btn3:
                        if st.button("▶️ 다음", use_container_width=True):
                            if st.session_state.animation_step < total_steps - 1:
                                st.session_state.animation_step += 1
                                st.rerun()

                    with col_btn4:
                        if st.button("⏭️ 끝", use_container_width=True):
                            st.session_state.animation_step = total_steps - 1
                            st.rerun()

                    # 자동 재생 버튼
                    st.markdown("---")
                    col_play1, col_play2 = st.columns(2)

                    with col_play1:
                        if st.button("▶️ 자동 재생", use_container_width=True, type="primary"):
                            # 자동 재생 모드
                            map_placeholder = st.empty()
                            info_placeholder = st.empty()
                            progress_bar = st.progress(0)

                            for step in range(st.session_state.animation_step, total_steps):
                                st.session_state.animation_step = step

                                # 지도 업데이트
                                anim_map, _ = create_animated_agent_map(
                                    results_df, visits_df, stores_df,
                                    selected_agent, agent_info,
                                    current_step=step, show_trail=True
                                )

                                with map_placeholder.container():
                                    st_folium(anim_map, width=600, height=350, key=f"anim_map_{step}")

                                # 정보 업데이트
                                curr = timeline_data[step]
                                with info_placeholder.container():
                                    decision_text = "🍽️ 방문" if curr['decision'] == 'visit' else "🏠 외출안함"
                                    store_text = f" → {curr['visited_store']}" if curr['visited_store'] else ""
                                    st.info(f"Step {step + 1}/{total_steps}: {curr['date']} {curr['time_slot']} ({curr['time']}) - {decision_text}{store_text}")

                                # 진행률
                                progress_bar.progress((step + 1) / total_steps)

                                # 속도 조절
                                time_module.sleep(1.0 / play_speed)

                            st.success("✅ 재생 완료!")

                    with col_play2:
                        show_trail = st.checkbox("경로 표시", value=True)

                    # 현재 시점 지도 표시
                    st.markdown("---")
                    current_map, _ = create_animated_agent_map(
                        results_df, visits_df, stores_df,
                        selected_agent, agent_info,
                        current_step=st.session_state.animation_step,
                        show_trail=show_trail
                    )

                    if current_map:
                        st_folium(current_map, width=600, height=400, key="current_map")

                        # 현재 시점 상세 정보
                        if st.session_state.animation_step < len(timeline_data):
                            curr = timeline_data[st.session_state.animation_step]
                            col_info1, col_info2 = st.columns(2)
                            with col_info1:
                                st.markdown(f"**날짜:** {curr['date']}")
                                st.markdown(f"**시간:** {curr['time_slot']} ({curr['time']})")
                            with col_info2:
                                decision_emoji = "🍽️" if curr['decision'] == 'visit' else "🏠"
                                decision_text = "매장 방문" if curr['decision'] == 'visit' else "외출 안 함"
                                st.markdown(f"**결정:** {decision_emoji} {decision_text}")
                                if curr['visited_store']:
                                    st.markdown(f"**방문 매장:** {curr['visited_store']}")
                else:
                    st.info("이동 경로 데이터가 없습니다.")

            # 에이전트 방문 로그 상세
            st.markdown("### 📋 방문 기록 상세")

            agent_results = results_df[results_df['agent_name'] == selected_agent].copy()
            agent_visits = visits_df[visits_df['agent_name'] == selected_agent].copy() if not visits_df.empty else pd.DataFrame()

            if not agent_results.empty:
                # 날짜별로 그룹화
                agent_results['date'] = agent_results['timestamp'].dt.date
                dates = sorted(agent_results['date'].unique())

                for date in dates:
                    st.markdown(f"#### 📅 {date}")
                    day_results = agent_results[agent_results['date'] == date].sort_values('timestamp')

                    for _, row in day_results.iterrows():
                        time_slot = row.get('time_slot', '')
                        decision = row.get('decision', '')
                        timestamp = row['timestamp'].strftime('%H:%M') if pd.notna(row['timestamp']) else ''

                        if decision == 'visit':
                            store_name = row.get('visited_store', '')
                            category = row.get('visited_category', '')
                            taste = row.get('taste_rating', '-')
                            value = row.get('value_rating', '-')
                            atmosphere = row.get('atmosphere_rating', '-')
                            reason = row.get('reason', '')

                            with st.expander(f"🍽️ {time_slot} ({timestamp}) → {store_name}", expanded=False):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**매장:** {store_name}")
                                    st.markdown(f"**카테고리:** {category}")
                                    st.markdown(f"**평점:** 맛 {taste}점 / 가성비 {value}점 / 분위기 {atmosphere}점")
                                with col2:
                                    st.markdown("**방문 이유:**")
                                    if reason and '→' in str(reason):
                                        steps = str(reason).split('→')
                                        for i, step in enumerate(steps, 1):
                                            st.caption(f"Step {i}: {step.strip()}")
                                    else:
                                        st.caption(str(reason) if reason else "기록 없음")
                        elif decision == 'stay_home':
                            reason = row.get('reason', '외출 안 함')
                            st.markdown(f"🏠 **{time_slot} ({timestamp})** - 외출 안 함: _{reason}_")
                        elif decision == 'llm_failed':
                            st.markdown(f"⚠️ **{time_slot} ({timestamp})** - LLM 호출 실패")
            else:
                st.info("이 에이전트의 활동 기록이 없습니다.")

            # 에이전트 방문 통계
            if not agent_visits.empty:
                st.markdown("### 📊 방문 통계")
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

                with col_stat1:
                    st.metric("총 방문 횟수", f"{len(agent_visits)}회")
                with col_stat2:
                    unique_stores = agent_visits['visited_store'].nunique()
                    st.metric("방문 매장 수", f"{unique_stores}개")
                with col_stat3:
                    avg_taste = agent_visits['taste_rating'].mean()
                    st.metric("평균 맛 평점", f"{avg_taste:.1f}점")
                with col_stat4:
                    avg_value = agent_visits['value_rating'].mean()
                    st.metric("평균 가성비 평점", f"{avg_value:.1f}점")

                # 방문 매장 목록
                st.markdown("#### 🍽️ 방문한 매장")
                store_summary = agent_visits.groupby('visited_store').agg({
                    'taste_rating': 'mean',
                    'value_rating': 'mean',
                    'timestamp': 'count'
                }).reset_index()
                store_summary.columns = ['매장', '평균 맛', '평균 가성비', '방문횟수']
                store_summary = store_summary.sort_values('방문횟수', ascending=False)
                st.dataframe(store_summary, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 세부 분석
    st.markdown("### 📈 세부 분석")

    tab1, tab2, tab3 = st.tabs(["세그먼트별 방문", "세대별 분석", "평점 분포"])

    with tab1:
        if not filtered_visits.empty:
            segment_visits = filtered_visits.groupby('segment').size().reset_index(name='count')
            fig_segment = px.pie(
                segment_visits,
                values='count',
                names='segment',
                title="세그먼트별 방문 비율"
            )
            st.plotly_chart(fig_segment, use_container_width=True)
        else:
            st.info("방문 데이터 없음")

    with tab2:
        if not filtered_visits.empty:
            gen_visits = filtered_visits.groupby('generation').size().reset_index(name='count')
            gen_order = ['Alpha', 'Z', 'Y', 'X', 'BB', 'S']
            gen_visits['generation'] = pd.Categorical(gen_visits['generation'], categories=gen_order, ordered=True)
            gen_visits = gen_visits.sort_values('generation')

            fig_gen = px.bar(
                gen_visits,
                x='generation',
                y='count',
                title="세대별 방문 수",
                color='generation',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_gen, use_container_width=True)
        else:
            st.info("방문 데이터 없음")

    with tab3:
        if not filtered_visits.empty and 'taste_rating' in filtered_visits.columns:
            col_t, col_v, col_a = st.columns(3)

            # 1~5점 스케일 라벨
            rating_labels = {1: '매우별로(1)', 2: '별로(2)', 3: '보통(3)', 4: '좋음(4)', 5: '매우좋음(5)'}
            rating_colors = ['#ff6b6b', '#ffa06b', '#ffd93d', '#a8e063', '#6bcb77']

            with col_t:
                taste_dist = filtered_visits['taste_rating'].value_counts().reset_index()
                taste_dist.columns = ['rating', 'count']
                taste_dist['rating_label'] = taste_dist['rating'].map(rating_labels)
                taste_dist = taste_dist.dropna(subset=['rating_label'])

                fig_taste = px.pie(
                    taste_dist,
                    values='count',
                    names='rating_label',
                    title="맛 평점 분포",
                    color_discrete_sequence=rating_colors
                )
                st.plotly_chart(fig_taste, use_container_width=True)

            with col_v:
                value_dist = filtered_visits['value_rating'].value_counts().reset_index()
                value_dist.columns = ['rating', 'count']
                value_dist['rating_label'] = value_dist['rating'].map(rating_labels)
                value_dist = value_dist.dropna(subset=['rating_label'])

                fig_value = px.pie(
                    value_dist,
                    values='count',
                    names='rating_label',
                    title="가성비 평점 분포",
                    color_discrete_sequence=rating_colors
                )
                st.plotly_chart(fig_value, use_container_width=True)

            with col_a:
                if 'atmosphere_rating' in filtered_visits.columns:
                    atmos_dist = filtered_visits['atmosphere_rating'].value_counts().reset_index()
                    atmos_dist.columns = ['rating', 'count']
                    atmos_dist['rating_label'] = atmos_dist['rating'].map(rating_labels)
                    atmos_dist = atmos_dist.dropna(subset=['rating_label'])

                    fig_atmos = px.pie(
                        atmos_dist,
                        values='count',
                        names='rating_label',
                        title="분위기 평점 분포",
                        color_discrete_sequence=rating_colors
                    )
                    st.plotly_chart(fig_atmos, use_container_width=True)
                else:
                    st.info("분위기 평점 데이터 없음")
        else:
            st.info("평점 데이터 없음")

    # 방문 로그 테이블
    st.markdown("---")
    st.markdown("### 📋 방문 로그")

    if not filtered_visits.empty:
        # 사용 가능한 컬럼 확인
        available_cols = filtered_visits.columns.tolist()
        base_cols = ['timestamp', 'agent_name', 'generation', 'segment',
                    'visited_store', 'visited_category', 'taste_rating', 'value_rating']
        base_names = ['시간', '에이전트', '세대', '세그먼트',
                     '방문매장', '카테고리', '맛', '가성비']

        # 분위기 평점 추가
        if 'atmosphere_rating' in available_cols:
            base_cols.append('atmosphere_rating')
            base_names.append('분위기')

        # 방문 이유 추가
        if 'reason' in available_cols:
            base_cols.append('reason')
            base_names.append('방문이유')

        display_df = filtered_visits[base_cols].copy()
        display_df.columns = base_names

        # 방문이유 열 너비 조정을 위한 설정
        st.dataframe(
            display_df.head(50),
            use_container_width=True,
            column_config={
                "방문이유": st.column_config.TextColumn(
                    "방문이유",
                    width="large",
                    help="에이전트의 4단계 의사결정 근거"
                )
            }
        )

        # 선택한 행의 상세 보기
        st.markdown("#### 🔍 방문 상세 보기")
        selected_idx = st.selectbox(
            "상세 보기할 방문 선택",
            range(min(50, len(display_df))),
            format_func=lambda i: f"{display_df.iloc[i]['에이전트']} → {display_df.iloc[i]['방문매장']}"
        )

        if selected_idx is not None:
            selected_row = display_df.iloc[selected_idx]
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**에이전트:** {selected_row['에이전트']}")
                st.markdown(f"**세대:** {selected_row['세대']} / **세그먼트:** {selected_row['세그먼트']}")
                st.markdown(f"**방문매장:** {selected_row['방문매장']}")

                # 평점 표시 (1~5점 스케일)
                rating_text = f"맛 {int(selected_row['맛'])}점 / 가성비 {int(selected_row['가성비'])}점"
                if '분위기' in selected_row.index:
                    rating_text += f" / 분위기 {int(selected_row['분위기'])}점"
                st.markdown(f"**평점:** {rating_text}")

            with col2:
                if '방문이유' in selected_row.index:
                    st.markdown("**방문 이유:**")
                    # 화살표로 구분된 단계별 표시
                    reason_text = selected_row['방문이유']
                    if '→' in str(reason_text):
                        steps = str(reason_text).split('→')
                        for i, step in enumerate(steps, 1):
                            st.markdown(f"**Step {i}:** {step.strip()}")
                    else:
                        st.markdown(str(reason_text) if pd.notna(reason_text) else "기록 없음")
    else:
        st.info("방문 데이터가 없습니다.")


if __name__ == "__main__":
    main()
