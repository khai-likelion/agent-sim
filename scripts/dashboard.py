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
import pydeck as pdk
import random
import networkx as nx
import osmnx as ox
import time as time_module

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"

# ── 24시간 애니메이션 상수 ────────────────────────────────

# 타임슬롯 시간 매핑
TIMESLOT_HOURS = {
    "아침": 7,
    "점심": 12,
    "저녁": 18,
    "야식": 22
}

# Step 5 행동 목적지 좌표
LANDMARKS = {
    "한강공원": {"lat": 37.5530, "lng": 126.8950, "name": "망원한강공원"},
    "망원시장": {"lat": 37.5560, "lng": 126.9050, "name": "망원시장"},
    "집": {"lat": 37.5565, "lng": 126.9029, "name": "집"},
    "회사": {"lat": 37.5550, "lng": 126.9100, "name": "회사"},
}

# 유동 에이전트 초기 위치 후보 (망원동 주요 정류장/거점)
FLOATING_LOCATIONS = {
    "망원역": (37.556069, 126.910108),
    "역 정류장": (37.556097, 126.910283),
    "시장 정류장": (37.557637, 126.905902),
    "입구 정류장": (37.557944, 126.907324),
    "한강 진입 정류장": (37.550704, 126.912613),
    "망원 한강공원 입구": (37.551025, 126.898877),
}

# 상주 에이전트 주거지 좌표
RESIDENT_LOCATIONS = {
    "아파트1": {"lat": 37.558682, "lng": 126.898706, "type": "아파트", "color": "#e74c3c"},
    "아파트2": {"lat": 37.553427, "lng": 126.904841, "type": "아파트", "color": "#e74c3c"},
    "아파트3": {"lat": 37.559734, "lng": 126.901044, "type": "아파트", "color": "#e74c3c"},
    "빌라1": {"lat": 37.553972, "lng": 126.903356, "type": "빌라", "color": "#3498db"},
    "빌라2": {"lat": 37.555740, "lng": 126.904030, "type": "빌라", "color": "#3498db"},
    "빌라3": {"lat": 37.554726, "lng": 126.908740, "type": "빌라", "color": "#3498db"},
    "주택1": {"lat": 37.555097, "lng": 126.907753, "type": "주택", "color": "#2ecc71"},
    "주택2": {"lat": 37.554986, "lng": 126.902714, "type": "주택", "color": "#2ecc71"},
    "주택3": {"lat": 37.552770, "lng": 126.905787, "type": "주택", "color": "#2ecc71"},
}

# Step 5 행동별 지속 시간 (시간 단위)
ACTION_DURATION = {
    "카페_가기": None,
    "배회하기": 0.5,
    "한강공원_산책": None,
    "망원시장_장보기": None,
    "집에서_쉬기": None,
    "회사_가기": None,
}


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
    .time-display {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        font-family: 'SF Mono', 'Fira Code', 'Courier New', monospace;
        color: #1a1a2e;
        padding: 12px 20px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        margin-bottom: 16px;
        border: 1px solid #dee2e6;
        letter-spacing: 2px;
    }
    .status-box {
        padding: 16px 20px;
        border-radius: 12px;
        margin: 8px 0;
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .status-box h4 { margin: 0 0 6px 0; font-size: 1.1rem; }
    .status-box p { margin: 2px 0; font-size: 0.9rem; color: #444; }
    .status-eating {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 4px solid #28a745;
    }
    .status-cafe {
        background: linear-gradient(135deg, #fff3cd, #ffeeba);
        border-left: 4px solid #ffc107;
    }
    .status-idle {
        background: linear-gradient(135deg, #e9ecef, #dee2e6);
        border-left: 4px solid #6c757d;
    }
    .status-moving {
        background: linear-gradient(135deg, #cce5ff, #b8daff);
        border-left: 4px solid #007bff;
    }
    .status-wander {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-left: 4px solid #dc3545;
    }
    .status-park {
        background: linear-gradient(135deg, #d1e7dd, #c3dfd1);
        border-left: 4px solid #198754;
    }
    .status-market {
        background: linear-gradient(135deg, #e2d5f1, #d6c5e8);
        border-left: 4px solid #6f42c1;
    }
    .status-work {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_simulation_data(sim_folder: str = ""):
    """시뮬레이션 데이터 로드. sim_folder가 주어지면 해당 하위 폴더에서 로드."""
    base = OUTPUT_DIR / sim_folder if sim_folder else OUTPUT_DIR

    # 전체 결과 — 하위 폴더에서는 simulation_result.csv 사용
    result_path = base / "simulation_result.csv" if sim_folder else base / "generative_simulation_result.csv"
    if result_path.exists():
        results_df = pd.read_csv(result_path)
        results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
        results_df['date'] = results_df['timestamp'].dt.date
    else:
        results_df = pd.DataFrame()

    # 방문 로그
    visit_path = base / "visit_log.csv" if sim_folder else base / "generative_visit_log.csv"
    if visit_path.exists():
        visits_df = pd.read_csv(visit_path)
        visits_df['timestamp'] = pd.to_datetime(visits_df['timestamp'])
        visits_df['date'] = visits_df['timestamp'].dt.date
    else:
        visits_df = pd.DataFrame()

    # 에이전트 상태
    agents_path = base / "agents_final.json" if sim_folder else base / "agents_final_state.json"
    if agents_path.exists():
        with open(agents_path, 'r', encoding='utf-8') as f:
            agents = json.load(f)
    else:
        agents = []

    # home_location이 없는 에이전트에 유형별 초기 위치 할당
    for agent in agents:
        if not agent.get('home_location'):
            if agent.get('agent_type') == '유동':
                loc = random.choice(list(FLOATING_LOCATIONS.values()))
                agent['home_location'] = list(loc)
            elif (agent.get('agent_type') == '상주'
                  and agent.get('group_type') == '가족모임형'
                  and agent.get('group_size') == 4):
                apts = [v for v in RESIDENT_LOCATIONS.values() if v["type"] == "아파트"]
                apt = random.choice(apts)
                agent['home_location'] = [apt["lat"], apt["lng"]]
            elif (agent.get('agent_type') == '상주'
                  and agent.get('housing_type') == '단독·연립(주택)'):
                houses = [v for v in RESIDENT_LOCATIONS.values() if v["type"] == "주택"]
                house = random.choice(houses)
                agent['home_location'] = [house["lat"], house["lng"]]
            elif (agent.get('agent_type') == '상주'
                  and agent.get('housing_type') == '다세대(빌라)'):
                villas = [v for v in RESIDENT_LOCATIONS.values() if v["type"] == "빌라"]
                villa = random.choice(villas)
                agent['home_location'] = [villa["lat"], villa["lng"]]

    # 매장 데이터 (JSON 파일에서 로드)
    json_dir = DATA_DIR / "raw" / "split_by_store_id"
    if json_dir.exists():
        stores_list = []
        for json_file in json_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # dict 형식 (신규) 또는 list 형식 (구형) 모두 지원
                    if isinstance(data, dict):
                        store = data
                    elif isinstance(data, list) and len(data) > 0:
                        store = data[0]
                    else:
                        continue
                    # 좌표: 최상위 x/y 또는 metadata.x/y
                    meta = store.get('metadata', {}) or {}
                    x = store.get('x') or meta.get('x') or 0
                    y = store.get('y') or meta.get('y') or 0
                    stores_list.append({
                        '장소명': store.get('store_name', ''),
                        'x': float(x) if x else 0,
                        'y': float(y) if y else 0,
                        '카테고리': store.get('category', '') or meta.get('sector', ''),
                        'address': store.get('address', '') or meta.get('area', ''),
                        'store_id': store.get('store_id', '')
                    })
            except Exception:
                continue
        stores_df = pd.DataFrame(stores_list) if stores_list else pd.DataFrame()
    else:
        stores_df = pd.DataFrame()

    # stores_dict: 매장명 → {lat, lng, category, store_id} (애니메이션용)
    stores_dict = {}
    if not stores_df.empty:
        for _, row in stores_df.iterrows():
            stores_dict[row['장소명']] = {
                'lat': float(row['y']),
                'lng': float(row['x']),
                'category': row.get('카테고리', ''),
                'store_id': row.get('store_id', ''),
            }

    return results_df, visits_df, agents, stores_df, stores_dict


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


@st.cache_data
def load_cafe_stores():
    """카페 매장 목록 로드"""
    cafe_path = DATA_DIR / "cafe_stores.txt"
    if cafe_path.exists():
        with open(cafe_path, 'r', encoding='utf-8') as f:
            return [line.strip().replace('.json', '') for line in f if line.strip()]
    return []


def get_walking_speed(segment, seed=None):
    """페르소나 세그먼트 기반 걷는 속도 계산 (km/h)"""
    if seed is not None:
        random.seed(seed)

    # 세그먼트에서 특징 추출
    group_size = 1
    generation = ""
    if segment:
        parts = segment.split("_")
        for p in parts:
            if "인" in p:
                try:
                    group_size = int(p.replace("인", ""))
                except ValueError:
                    pass

    # 그룹 크기에 따른 기본 속도 범위
    if group_size >= 4:
        base_min, base_max = 3.0, 3.8
    elif group_size == 2:
        base_min, base_max = 3.5, 4.5
    else:
        base_min, base_max = 3.8, 5.0

    speed = random.uniform(base_min, base_max)
    return max(1.5, min(6.0, round(speed, 1)))


def calculate_route_distance(route_coords):
    """경로 좌표에서 총 거리 계산 (km)"""
    if not route_coords or len(route_coords) < 2:
        return 0.0
    total_dist = 0.0
    for i in range(len(route_coords) - 1):
        lat1, lng1 = route_coords[i]
        lat2, lng2 = route_coords[i + 1]
        dlat = (lat2 - lat1) * 111
        dlng = (lng2 - lng1) * 88
        total_dist += (dlat ** 2 + dlng ** 2) ** 0.5
    return total_dist


def calculate_travel_time(route_coords, walking_speed):
    """경로 거리와 걷는 속도로 이동 시간 계산 (시간 단위)"""
    distance = calculate_route_distance(route_coords)
    if walking_speed <= 0:
        return 0.5
    travel_time = distance / walking_speed
    return max(5/60, min(1.0, travel_time))


@st.cache_data
def get_route_coords(_G, start_lat, start_lng, end_lat, end_lng):
    """OSM 네트워크 위의 경로 좌표 계산"""
    if _G is None:
        return [(start_lat, start_lng), (end_lat, end_lng)]
    try:
        start_node = ox.nearest_nodes(_G, start_lng, start_lat)
        end_node = ox.nearest_nodes(_G, end_lng, end_lat)
        route = nx.shortest_path(_G, start_node, end_node, weight='length')
        return [(_G.nodes[node]['y'], _G.nodes[node]['x']) for node in route]
    except Exception:
        return [(start_lat, start_lng), (end_lat, end_lng)]


def generate_wander_path(_G, start_lat, start_lng, num_nodes=10, seed=None):
    """배회 경로 생성 - OSM 네트워크에서 랜덤 노드 선택"""
    if _G is None:
        return [(start_lat, start_lng)]
    if seed is not None:
        random.seed(seed)
    try:
        current_node = ox.nearest_nodes(_G, start_lng, start_lat)
        path = [current_node]
        for _ in range(num_nodes):
            neighbors = list(_G.neighbors(current_node))
            if neighbors:
                if len(path) > 1:
                    neighbors = [n for n in neighbors if n != path[-2]]
                if not neighbors:
                    neighbors = list(_G.neighbors(current_node))
                current_node = random.choice(neighbors)
                path.append(current_node)
            else:
                break
        return [(_G.nodes[node]['y'], _G.nodes[node]['x']) for node in path]
    except Exception:
        return [(start_lat, start_lng)]


def interpolate_on_route(route_coords, progress):
    """경로 위에서 progress (0~1)에 해당하는 위치 계산"""
    if not route_coords or len(route_coords) < 2:
        return route_coords[0] if route_coords else (37.5565, 126.9029)

    total_length = 0
    segment_lengths = []
    for i in range(len(route_coords) - 1):
        lat1, lng1 = route_coords[i]
        lat2, lng2 = route_coords[i + 1]
        seg_len = ((lat2 - lat1) ** 2 + (lng2 - lng1) ** 2) ** 0.5
        segment_lengths.append(seg_len)
        total_length += seg_len

    if total_length == 0:
        return route_coords[0]

    target_length = total_length * progress
    accumulated = 0

    for i, seg_len in enumerate(segment_lengths):
        if accumulated + seg_len >= target_length:
            seg_progress = (target_length - accumulated) / seg_len if seg_len > 0 else 0
            lat1, lng1 = route_coords[i]
            lat2, lng2 = route_coords[i + 1]
            return (
                lat1 + (lat2 - lat1) * seg_progress,
                lng1 + (lng2 - lng1) * seg_progress
            )
        accumulated += seg_len

    return route_coords[-1]


def get_step5_action(current_slot, next_slot, seed=None, segment=None):
    """Step 5 행동 결정 (간단 버전 - 실제로는 LLM 호출)"""
    if seed is not None:
        random.seed(seed)

    actions = ["카페_가기", "배회하기", "한강공원_산책", "망원시장_장보기", "집에서_쉬기"]
    weights = [0.3, 0.25, 0.2, 0.15, 0.1]

    return random.choices(actions, weights=weights)[0]


def _get_step5_end_position(step5_action, store_lat, store_lng, home_lat, home_lng, seed, G, cafe_stores, stores_dict):
    """Step5 행동의 최종 위치를 반환"""
    if step5_action == "집에서_쉬기":
        return home_lat, home_lng
    elif step5_action == "회사_가기":
        return LANDMARKS["회사"]["lat"], LANDMARKS["회사"]["lng"]
    elif step5_action == "배회하기":
        wander_path = generate_wander_path(G, store_lat, store_lng, num_nodes=15, seed=seed)
        return wander_path[-1] if wander_path else (store_lat, store_lng)
    elif step5_action == "카페_가기":
        # 카페에 머무름
        if cafe_stores and stores_dict:
            random.seed(seed)
            for sn, info in stores_dict.items():
                if info.get('store_id') in cafe_stores:
                    return info.get('lat', store_lat), info.get('lng', store_lng)
        return store_lat, store_lng
    elif step5_action == "한강공원_산책":
        # 한강공원에 머무름
        return LANDMARKS["한강공원"]["lat"], LANDMARKS["한강공원"]["lng"]
    elif step5_action == "망원시장_장보기":
        # 망원시장에 머무름
        return LANDMARKS["망원시장"]["lat"], LANDMARKS["망원시장"]["lng"]
    return store_lat, store_lng


def _get_slot_end_position(slot_name, slot_idx, slot_data, stores_dict, home_lat, home_lng, seed_base, segment, G, cafe_stores):
    """특정 타임슬롯이 끝났을 때 에이전트의 최종 위치를 반환"""
    if slot_name not in slot_data:
        return home_lat, home_lng

    row = slot_data[slot_name]
    if row['decision'] != 'visit':
        return home_lat, home_lng

    store_info = stores_dict.get(row['visited_store'], {})
    store_lat = store_info.get('lat', home_lat)
    store_lng = store_info.get('lng', home_lng)

    seed = seed_base + slot_idx
    step5_action = get_step5_action(slot_name, None, seed, segment)
    return _get_step5_end_position(step5_action, store_lat, store_lng, home_lat, home_lng, seed, G, cafe_stores, stores_dict)


def get_agent_state(results_df, stores_dict, G, cafe_stores, current_date, current_hour, segment, persona_id=None, home_location=None):
    """현재 시간에 에이전트 상태 및 위치 계산 (OSM 경로 기반 + Step 5)"""
    day_data = results_df[results_df['timestamp'].dt.date == current_date]
    if day_data.empty:
        return None, None, "idle", None, [], None

    speed_seed = hash(persona_id) if persona_id else hash(segment)
    walking_speed = get_walking_speed(segment, seed=speed_seed)

    if home_location:
        home_lat, home_lng = home_location[0], home_location[1]
    else:
        home_lat, home_lng = LANDMARKS["집"]["lat"], LANDMARKS["집"]["lng"]

    slot_data = {}
    for slot in TIMESLOT_HOURS.keys():
        slot_rows = day_data[day_data['time_slot'] == slot]
        if not slot_rows.empty:
            slot_data[slot] = slot_rows.iloc[0]

    sorted_slots = sorted(TIMESLOT_HOURS.items(), key=lambda x: x[1])
    seed_base = int(current_date.toordinal())

    prev_slot = None
    next_slot = None
    prev_slot_idx = -1

    for i, (slot, hour) in enumerate(sorted_slots):
        if hour <= current_hour:
            prev_slot = slot
            prev_slot_idx = i
        if hour > current_hour and next_slot is None:
            next_slot = slot

    route_coords = []
    step5_action = None

    if prev_slot and prev_slot in slot_data:
        prev_row = slot_data[prev_slot]
        prev_hour = TIMESLOT_HOURS[prev_slot]

        if prev_row['decision'] == 'visit':
            store_name = prev_row['visited_store']
            store_info = stores_dict.get(store_name, {})
            store_lat = store_info.get('lat', home_lat)
            store_lng = store_info.get('lng', home_lng)

            # 이전 타임슬롯의 Step5 최종 위치에서 출발 (첫 타임슬롯은 집)
            if prev_slot_idx == 0:
                start_lat, start_lng = home_lat, home_lng
            else:
                prev_prev_slot = sorted_slots[prev_slot_idx - 1][0]
                start_lat, start_lng = _get_slot_end_position(
                    prev_prev_slot, prev_slot_idx - 1, slot_data, stores_dict,
                    home_lat, home_lng, seed_base, segment, G, cafe_stores
                )

            route_coords = get_route_coords(G, start_lat, start_lng, store_lat, store_lng)
            travel_time = calculate_travel_time(route_coords, walking_speed)

            arrival_time = prev_hour + travel_time
            eating_end = arrival_time + 1.5

            if current_hour < arrival_time:
                progress = (current_hour - prev_hour) / travel_time if travel_time > 0 else 1.0
                progress = min(1.0, max(0.0, progress))
                lat, lng = interpolate_on_route(route_coords, progress)
                return lat, lng, "moving", prev_row, route_coords, None

            elif current_hour < eating_end:
                return store_lat, store_lng, "eating", prev_row, [], None

            else:
                next_meal_hour = 24.0
                if next_slot and next_slot in slot_data:
                    next_meal_hour = TIMESLOT_HOURS[next_slot]

                free_time_end = next_meal_hour

                if current_hour < free_time_end:
                    seed = seed_base + prev_slot_idx
                    step5_action = get_step5_action(prev_slot, next_slot, seed, segment)

                    time_in_action = current_hour - eating_end
                    action_duration = ACTION_DURATION.get(step5_action)
                    move_time = 0.25

                    if action_duration is None:
                        total_action_time = free_time_end - eating_end
                    else:
                        total_action_time = move_time + action_duration + move_time

                    if time_in_action < total_action_time:
                        progress = time_in_action / total_action_time if total_action_time > 0 else 0
                        progress = min(1.0, max(0.0, progress))

                        if step5_action == "카페_가기":
                            cafe_lat, cafe_lng = store_lat, store_lng
                            cafe_name = None
                            if cafe_stores and stores_dict:
                                random.seed(seed)
                                for sn, info in stores_dict.items():
                                    if info.get('store_id') in cafe_stores:
                                        cafe_name = sn
                                        cafe_lat = info.get('lat', store_lat)
                                        cafe_lng = info.get('lng', store_lng)
                                        break
                            move_ratio = move_time / total_action_time if total_action_time > 0 else 0.1
                            if progress < move_ratio:
                                route_coords = get_route_coords(G, store_lat, store_lng, cafe_lat, cafe_lng)
                                lat, lng = interpolate_on_route(route_coords, progress / move_ratio)
                                return lat, lng, "moving_cafe", {"name": cafe_name}, route_coords, step5_action
                            else:
                                return cafe_lat, cafe_lng, "cafe", {"name": cafe_name}, [], step5_action

                        elif step5_action == "배회하기":
                            wander_path = generate_wander_path(G, store_lat, store_lng, num_nodes=15, seed=seed)
                            lat, lng = interpolate_on_route(wander_path, progress)
                            return lat, lng, "wander", None, wander_path, step5_action

                        elif step5_action == "한강공원_산책":
                            park = LANDMARKS["한강공원"]
                            move_ratio = move_time / total_action_time if total_action_time > 0 else 0.1
                            if progress < move_ratio:
                                route_coords = get_route_coords(G, store_lat, store_lng, park["lat"], park["lng"])
                                lat, lng = interpolate_on_route(route_coords, progress / move_ratio)
                                return lat, lng, "moving_park", None, route_coords, step5_action
                            else:
                                return park["lat"], park["lng"], "park", None, [], step5_action

                        elif step5_action == "망원시장_장보기":
                            market = LANDMARKS["망원시장"]
                            move_ratio = move_time / total_action_time if total_action_time > 0 else 0.1
                            if progress < move_ratio:
                                route_coords = get_route_coords(G, store_lat, store_lng, market["lat"], market["lng"])
                                lat, lng = interpolate_on_route(route_coords, progress / move_ratio)
                                return lat, lng, "moving_market", None, route_coords, step5_action
                            else:
                                return market["lat"], market["lng"], "market", None, [], step5_action

                        elif step5_action == "집에서_쉬기":
                            move_ratio = move_time / total_action_time if total_action_time > 0 else 0.1
                            if progress < move_ratio:
                                route_coords = get_route_coords(G, store_lat, store_lng, home_lat, home_lng)
                                lat, lng = interpolate_on_route(route_coords, progress / move_ratio)
                                return lat, lng, "moving_home", None, route_coords, step5_action
                            else:
                                return home_lat, home_lng, "home", None, [], step5_action

                        elif step5_action == "회사_가기":
                            work = LANDMARKS["회사"]
                            move_ratio = move_time / total_action_time if total_action_time > 0 else 0.1
                            if progress < move_ratio:
                                route_coords = get_route_coords(G, store_lat, store_lng, work["lat"], work["lng"])
                                lat, lng = interpolate_on_route(route_coords, progress / move_ratio)
                                return lat, lng, "moving_work", None, route_coords, step5_action
                            else:
                                return work["lat"], work["lng"], "work", None, [], step5_action

                    else:
                        # Step5 행동 시간 종료 → Step5 최종 위치에 머무름
                        end_lat, end_lng = _get_step5_end_position(
                            step5_action, store_lat, store_lng, home_lat, home_lng,
                            seed, G, cafe_stores, stores_dict
                        )
                        return end_lat, end_lng, "idle", None, [], None

                else:
                    # 다음 타임슬롯 시간대 → Step5 최종 위치에 대기
                    end_lat, end_lng = _get_step5_end_position(
                        get_step5_action(prev_slot, next_slot, seed_base + prev_slot_idx, segment),
                        store_lat, store_lng, home_lat, home_lng,
                        seed_base + prev_slot_idx, G, cafe_stores, stores_dict
                    )
                    return end_lat, end_lng, "idle", None, [], None

    if next_slot and next_slot in slot_data:
        next_row = slot_data[next_slot]
        next_hour = TIMESLOT_HOURS[next_slot]
        if next_row['decision'] == 'visit':
            next_store = stores_dict.get(next_row['visited_store'], {})
            next_lat = next_store.get('lat', home_lat)
            next_lng = next_store.get('lng', home_lng)
            move_start = next_hour
            if current_hour >= move_start:
                route_coords = get_route_coords(G, home_lat, home_lng, next_lat, next_lng)
                next_travel_time = calculate_travel_time(route_coords, walking_speed)
                progress = (current_hour - move_start) / next_travel_time if next_travel_time > 0 else 1.0
                progress = min(1.0, max(0.0, progress))
                lat, lng = interpolate_on_route(route_coords, progress)
                return lat, lng, "moving", next_row, route_coords, None

    return home_lat, home_lng, "idle", None, [], None


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
    agent_results = results_df[results_df['persona_id'] == agent_name].copy()
    agent_visits = visits_df[visits_df['persona_id'] == agent_name].copy() if not visits_df.empty else pd.DataFrame()

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
        '상주_생활베이스형_1인': '#2ecc71', '상주_가족모임형_2인': '#27ae60',
        '상주_가족모임형_4인': '#1abc9c',
        '유동_생활베이스형_1인': '#e67e22', '유동_생활베이스형_2인': '#f39c12',
        '유동_생활베이스형_4인': '#d35400',
        '유동_사적모임형_1인': '#e91e63', '유동_사적모임형_2인': '#e74c3c',
        '유동_사적모임형_4인': '#c0392b',
        '유동_공적모임형_4인': '#9b59b6',
        '유동_가족모임형_2인': '#3498db', '유동_가족모임형_4인': '#2980b9',
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
    agent_results = results_df[results_df['persona_id'] == agent_name].copy()
    agent_visits = visits_df[visits_df['persona_id'] == agent_name].copy() if not visits_df.empty else pd.DataFrame()

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
                           results_df=None):
    """Folium 지도 생성 (에이전트 위치 & 방문 현황)"""
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
        '상주_생활베이스형_1인': '#2ecc71', '상주_가족모임형_2인': '#27ae60',
        '상주_가족모임형_4인': '#1abc9c',
        '유동_생활베이스형_1인': '#e67e22', '유동_생활베이스형_2인': '#f39c12',
        '유동_생활베이스형_4인': '#d35400',
        '유동_사적모임형_1인': '#e91e63', '유동_사적모임형_2인': '#e74c3c',
        '유동_사적모임형_4인': '#c0392b',
        '유동_공적모임형_4인': '#9b59b6',
        '유동_가족모임형_2인': '#3498db', '유동_가족모임형_4인': '#2980b9',
    }

    # 에이전트 위치: 시뮬레이션 좌표 우선, 없으면 랜덤 fallback
    random.seed(42)
    lat_min, lat_max = 37.552, 37.562
    lon_min, lon_max = 126.895, 126.911

    agent_locations = {}
    for agent in agents:
        agent_name = agent['persona_id']
        segment = agent['segment']
        # 기본값: 랜덤 좌표
        if '상주' in segment:
            lat = random.uniform(lat_min + 0.003, lat_max - 0.002)
            lon = random.uniform(lon_min + 0.003, lon_max - 0.005)
        else:
            lat = random.uniform(lat_min + 0.001, lat_max - 0.001)
            lon = random.uniform(lon_min + 0.005, lon_max - 0.002)
        agent_locations[agent_name] = (lat, lon)

    # 시뮬레이션 결과가 있으면 실제 좌표로 덮어쓰기
    if results_df is not None and not results_df.empty:
        for agent in agents:
            agent_name = agent['persona_id']
            agent_rows = results_df[results_df['persona_id'] == agent_name]
            if agent_rows.empty:
                continue
            valid = agent_rows.dropna(subset=['agent_lat', 'agent_lng'])
            if not valid.empty:
                last = valid.iloc[-1]
                agent_locations[agent_name] = (last['agent_lat'], last['agent_lng'])

    # 방문한 에이전트 마커 표시
    visited_agents = set(visits_filtered['persona_id'].unique()) if not visits_filtered.empty else set()
    for agent in agents:
        agent_name = agent['persona_id']
        if agent_name not in visited_agents:
            continue
        if agent_name not in agent_locations:
            continue

        segment = agent['segment']
        color = SEGMENT_COLORS.get(segment, '#95a5a6')
        lat, lon = agent_locations[agent_name]

        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            tooltip=f"📍 {agent_name} ({agent['generation']}세대, {segment})"
        ).add_to(m)

    # 매장 마커 추가
    if not stores_df.empty:
        for _, store in stores_df.iterrows():
            store_name = store['장소명']
            lat = float(store['y'])
            lon = float(store['x'])
            count = visit_counts.get(store_name, 0)

            if count > 0:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    color='blue',
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"{store_name}<br>방문: {count}회",
                    tooltip=f"{store_name}: {count}회"
                ).add_to(m)

    return m


def main():
    # ── 시뮬레이션 폴더 선택 ──
    sim_folders = ["(기본)"]
    for d in sorted(OUTPUT_DIR.iterdir()):
        if d.is_dir() and (d / "visit_log.csv").exists():
            sim_folders.append(d.name)

    st.sidebar.title("시뮬레이션 결과")
    selected_sim = st.sidebar.selectbox(
        "결과 폴더", sim_folders, index=0,
        help="before/after 비교 시뮬레이션 결과를 선택하세요"
    )
    sim_folder = "" if selected_sim == "(기본)" else selected_sim

    # 데이터 로드
    results_df, visits_df, agents, stores_df, stores_dict = load_simulation_data(sim_folder)
    cafe_stores = load_cafe_stores()

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

    # 매장 필터
    st.sidebar.markdown("---")
    st.sidebar.subheader("매장 필터")

    if not visits_df.empty:
        all_stores = ["전체"] + sorted(visits_df['visited_store'].unique())
        store_filter = st.sidebar.selectbox("특정 매장만 보기", all_stores, index=0)

        if store_filter != "전체":
            filtered_visits = filtered_visits[filtered_visits['visited_store'] == store_filter]
    else:
        store_filter = "전체"

    # 에이전트 선택 (개별 추적용)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 에이전트 추적")

    if agents and not results_df.empty:
        agent_names = ["전체"] + sorted([a['persona_id'] for a in agents])
        selected_agent = st.sidebar.selectbox(
            "에이전트 선택",
            agent_names,
            index=0,
            help="특정 에이전트의 이동 경로와 방문 기록을 확인합니다"
        )
    else:
        selected_agent = "전체"

    # 전체 진행 상황
    st.sidebar.markdown("---")
    st.sidebar.subheader("전체 진행 상황")

    if not results_df.empty:
        total_events = len(results_df)
        total_visits = len(visits_df)
        st.sidebar.markdown(f"진행: **{total_events}** 건")
        st.sidebar.markdown(f"방문: **{total_visits}** 건")

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
            active_agents = filtered_visits['persona_id'].nunique()
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

    # ==================== 전체 보기: 방문 현황 지도 + 매장 평점 ====================
    if selected_agent == "전체":
        col_map, col_ratings = st.columns([2, 1])

        with col_map:
            st.markdown("### 🗺️ 방문 현황 지도")
            if not filtered_visits.empty and not stores_df.empty:
                m = create_map_with_routes(filtered_visits, stores_df, agents,
                                           selected_date=selected_date,
                                           store_filter=store_filter,
                                           results_df=filtered_results)
                st_folium(m, width=700, height=500, key="overview_map")
            else:
                st.info("지도를 표시할 데이터가 없습니다.")

        with col_ratings:
            st.markdown("### 🏅 매장 평점 현황")
            ratings_path = OUTPUT_DIR / "store_ratings.json"
            if ratings_path.exists():
                with open(ratings_path, 'r', encoding='utf-8') as f:
                    ratings_data = json.load(f)

                stats = ratings_data.get('statistics', {})
                st.markdown(f"**평점 보유 매장:** {stats.get('stores_with_agent_ratings', 0)}개 / {stats.get('total_stores', 0)}개")
                st.markdown(f"**총 평점 수:** {stats.get('total_agent_ratings', 0)}건")
                st.markdown(f"**평균 맛:** {stats.get('avg_agent_taste_score', 0):.2f}점")
                st.markdown(f"**평균 가성비:** {stats.get('avg_agent_value_score', 0):.2f}점")
                st.markdown(f"**평균 분위기:** {stats.get('avg_agent_atmosphere_score', 0):.2f}점")

                st.markdown("---")
                st.markdown("#### 매장별 평점 TOP 10")
                stores_rated = ratings_data.get('stores', [])
                stores_rated_sorted = sorted(stores_rated, key=lambda x: x.get('agent_rating_count', 0), reverse=True)

                for i, store in enumerate(stores_rated_sorted[:10], 1):
                    with st.expander(f"{i}. {store['store_name']} ({store['agent_rating_count']}건)", expanded=(i <= 3)):
                        st.markdown(f"**카테고리:** {store.get('category', '-')}")
                        st.markdown(f"**평균가격:** {store.get('average_price', 0):,.0f}원")
                        st.markdown(f"**맛:** {store.get('agent_taste_score', 0):.1f} / **가성비:** {store.get('agent_value_score', 0):.1f} / **분위기:** {store.get('agent_atmosphere_score', 0):.1f}")
                        st.markdown(f"**종합:** {store.get('agent_overall_score', 0):.2f}점")
            else:
                st.info("매장 평점 데이터가 없습니다.")

        st.markdown("---")

    # ==================== 에이전트 추적 (24시간 애니메이션) ====================
    if selected_agent != "전체":
        st.markdown(f"## 🔍 에이전트 추적: {selected_agent}")

        # 에이전트 정보 가져오기
        agent_info = next((a for a in agents if a['persona_id'] == selected_agent), None)

        if agent_info:
            agent_segment = agent_info.get('segment', '')

            # OSM 도로망 로드
            with st.spinner("도로망 로드 중..."):
                G_anim = load_street_network()

            # 에이전트 프로필
            st.markdown("### 👤 에이전트 프로필")
            prof_cols = st.columns(4)
            prof_cols[0].markdown(f"**ID:** {agent_info['persona_id']}")
            prof_cols[1].markdown(f"**세대:** {agent_info['generation']}")
            prof_cols[2].markdown(f"**세그먼트:** {agent_segment}")
            walking_speed_display = get_walking_speed(agent_segment, seed=hash(selected_agent))
            prof_cols[3].markdown(f"**걷기:** {walking_speed_display:.1f} km/h")

            # ── 애니메이션 영역 (fragment로 부분 렌더링) ──
            @st.fragment
            def animation_fragment():
                # 날짜 선택
                agent_results_anim = results_df[results_df['persona_id'] == selected_agent].copy()
                if agent_results_anim.empty:
                    st.info("이 에이전트의 활동 기록이 없습니다.")
                    return

                agent_results_anim['date'] = agent_results_anim['timestamp'].dt.date
                anim_dates = sorted(agent_results_anim['date'].unique())

                # 애니메이션 세션 상태
                if 'current_hour' not in st.session_state:
                    st.session_state.current_hour = 6.0
                if 'anim_playing' not in st.session_state:
                    st.session_state.anim_playing = False

                # 날짜 선택
                anim_selected_date = st.selectbox(
                    "날짜", anim_dates, key="anim_date_select", label_visibility="collapsed"
                )

                # 컨트롤 바: ⏮ ▶/⏸ ⏭ | 배속
                btn_cols = st.columns([1, 1, 1, 3])
                if btn_cols[0].button("⏮ 처음", key="anim_start", use_container_width=True):
                    st.session_state.current_hour = 6.0
                    st.session_state.anim_playing = False
                play_label = "⏸ 정지" if st.session_state.anim_playing else "▶ 재생"
                if btn_cols[1].button(play_label, key="anim_play", use_container_width=True):
                    st.session_state.anim_playing = not st.session_state.anim_playing
                if btn_cols[2].button("⏭ 끝", key="anim_end", use_container_width=True):
                    st.session_state.current_hour = 24.0
                    st.session_state.anim_playing = False
                speed = btn_cols[3].slider("배속", 1, 60, 10, 1, key="anim_speed", label_visibility="collapsed")

                # 자동 재생: 슬라이더 값을 직접 업데이트
                if st.session_state.anim_playing:
                    # 배속1=0.1시간/틱, 배속10=0.2시간/틱, 배속60=0.5시간/틱
                    increment = 0.1 + (speed - 1) * (0.4 / 59)
                    new_hour = st.session_state.current_hour + increment
                    if new_hour >= 24.0:
                        new_hour = 6.0
                        st.session_state.anim_playing = False
                    st.session_state.current_hour = new_hour

                # 슬라이더: 항상 current_hour를 기본값으로
                current_hour = st.slider(
                    "시간", 6.0, 24.0,
                    value=st.session_state.current_hour,
                    step=0.1, format="%.1f",
                    label_visibility="collapsed",
                )
                # 슬라이더 값을 current_hour에 항상 반영 (사용자 드래그 포함)
                st.session_state.current_hour = current_hour

                # 시간 표시
                hours = int(current_hour)
                remaining = (current_hour - hours) * 60
                minutes = int(remaining)
                seconds = int((remaining - minutes) * 60)
                time_period = "오전" if hours < 12 else "오후"
                st.markdown(
                    f'<div class="time-display">{hours:02d}:{minutes:02d}:{seconds:02d}'
                    f'<span style="font-size:1.2rem; color:#888; margin-left:10px;">{time_period}</span></div>',
                    unsafe_allow_html=True
                )

                # 에이전트 상태 계산
                agent_home = agent_info.get('home_location', None)
                agent_lat, agent_lng, status, current_activity, route_coords, step5_action = get_agent_state(
                    results_df[results_df['persona_id'] == selected_agent],
                    stores_dict, G_anim, cafe_stores, anim_selected_date, current_hour,
                    agent_segment, selected_agent, home_location=agent_home
                )

                # 지도 + 상태 표시
                map_col, status_col = st.columns([2, 1])

                with map_col:
                    if agent_lat and agent_lng:
                        layers = []

                        # 랜드마크 (아이콘 + 라벨)
                        lm_data = []
                        lm_icons = {"한강공원": "🌊", "망원시장": "🏪", "집": None, "회사": "🏢"}
                        for k, v in LANDMARKS.items():
                            icon = lm_icons.get(k)
                            if icon is None:
                                continue
                            lm_data.append({
                                "lat": v["lat"], "lng": v["lng"],
                                "icon": icon, "name": v['name'],
                            })
                        if lm_data:
                            layers.append(pdk.Layer(
                                "TextLayer", data=lm_data,
                                get_position='[lng, lat]', get_text='icon',
                                get_size=28, get_color=[0, 0, 0],
                                get_text_anchor='"middle"',
                                get_alignment_baseline='"center"',
                            ))
                            layers.append(pdk.Layer(
                                "TextLayer", data=lm_data,
                                get_position='[lng, lat]', get_text='name',
                                get_size=11, get_color=[80, 80, 80],
                                get_pixel_offset='[0, 22]',
                                get_text_anchor='"middle"',
                            ))

                        # 이동 경로
                        if route_coords and len(route_coords) > 1:
                            if status == "wander":
                                path_color = [231, 76, 60]
                            elif "park" in status:
                                path_color = [46, 204, 113]
                            elif "market" in status:
                                path_color = [142, 68, 173]
                            else:
                                path_color = [52, 152, 219]

                            # 전체 예정 경로 (점선 느낌, 연하게)
                            full_path = [{"path": [[c[1], c[0]] for c in route_coords]}]
                            layers.append(pdk.Layer(
                                "PathLayer", data=full_path,
                                get_path="path", get_width=3,
                                get_color=path_color + [60],
                                width_min_pixels=2,
                                get_dash_array=[4, 4],
                            ))

                            # 이동 완료 구간 (진하게)
                            traveled = [route_coords[0]]
                            for i in range(1, len(route_coords)):
                                coord = route_coords[i]
                                dist_to_agent = ((coord[0] - agent_lat) ** 2 + (coord[1] - agent_lng) ** 2) ** 0.5
                                if dist_to_agent < 0.0001:
                                    traveled.append(coord)
                                    break
                                traveled.append(coord)
                                if i < len(route_coords) - 1:
                                    next_coord = route_coords[i + 1]
                                    seg_len = ((next_coord[0] - coord[0]) ** 2 + (next_coord[1] - coord[1]) ** 2) ** 0.5
                                    agent_dist = ((agent_lat - coord[0]) ** 2 + (agent_lng - coord[1]) ** 2) ** 0.5
                                    if agent_dist < seg_len:
                                        traveled.append((agent_lat, agent_lng))
                                        break
                            if traveled and traveled[-1] != (agent_lat, agent_lng):
                                last = traveled[-1]
                                if ((last[0] - agent_lat) ** 2 + (last[1] - agent_lng) ** 2) ** 0.5 > 0.00001:
                                    traveled.append((agent_lat, agent_lng))

                            if len(traveled) > 1:
                                path_data = [{"path": [[c[1], c[0]] for c in traveled]}]
                                layers.append(pdk.Layer(
                                    "PathLayer", data=path_data,
                                    get_path="path", get_width=5,
                                    get_color=path_color + [220],
                                    width_min_pixels=3,
                                ))

                            # 출발지/도착지 마커
                            start_pt = route_coords[0]
                            end_pt = route_coords[-1]
                            endpoint_data = [
                                {"lat": start_pt[0], "lng": start_pt[1], "label": "출발", "color": [100, 100, 100]},
                                {"lat": end_pt[0], "lng": end_pt[1], "label": "도착", "color": path_color},
                            ]
                            layers.append(pdk.Layer(
                                "ScatterplotLayer", data=endpoint_data,
                                get_position='[lng, lat]', get_radius=15,
                                get_fill_color='color', get_line_color=[255, 255, 255],
                                line_width_min_pixels=2, stroked=True,
                            ))
                            layers.append(pdk.Layer(
                                "TextLayer", data=endpoint_data,
                                get_position='[lng, lat]', get_text='label',
                                get_size=11, get_color=[60, 60, 60],
                                get_pixel_offset='[0, -18]',
                            ))

                        # 방문 매장 라벨
                        agent_visits_anim = visits_df[
                            (visits_df['persona_id'] == selected_agent) &
                            (visits_df['timestamp'].dt.date == anim_selected_date)
                        ] if not visits_df.empty else pd.DataFrame()

                        visited_labels = []
                        if not agent_visits_anim.empty:
                            for _, row in agent_visits_anim.iterrows():
                                visit_hour = TIMESLOT_HOURS.get(row['time_slot'], 0)
                                if visit_hour + 0.5 <= current_hour:
                                    s_info = stores_dict.get(row['visited_store'], {})
                                    if s_info:
                                        slot_label = row['time_slot']
                                        visited_labels.append({
                                            "lat": s_info['lat'], "lng": s_info['lng'],
                                            "icon": "🍴",
                                            "name": row['visited_store'],
                                            "detail": f"{slot_label} 방문",
                                        })
                        if visited_labels:
                            # 매장 핀 마커 (빨간 원 + 흰 테두리)
                            layers.append(pdk.Layer(
                                "ScatterplotLayer", data=visited_labels,
                                get_position='[lng, lat]', get_radius=30,
                                get_fill_color=[220, 50, 50, 200],
                                get_line_color=[255, 255, 255, 255],
                                line_width_min_pixels=3, stroked=True, pickable=True,
                            ))
                            # 매장 아이콘 (📍 핀)
                            layers.append(pdk.Layer(
                                "TextLayer", data=visited_labels,
                                get_position='[lng, lat]', get_text='icon',
                                get_size=24, get_color=[255, 255, 255],
                                get_text_anchor='"middle"',
                                get_alignment_baseline='"center"',
                            ))
                            # 매장명 라벨 (위쪽, 배경 느낌)
                            layers.append(pdk.Layer(
                                "TextLayer", data=visited_labels,
                                get_position='[lng, lat]', get_text='name',
                                get_size=13, get_color=[220, 50, 50],
                                get_pixel_offset='[0, -28]',
                                get_text_anchor='"middle"',
                                font_family='"Noto Sans KR", sans-serif',
                            ))

                        # 에이전트 마커 (사람 아이콘 + 상태)
                        status_info = {
                            "eating":       {"label": "식사 중",  "color": [231, 76, 60]},
                            "cafe":         {"label": "카페",    "color": [155, 89, 182]},
                            "wander":       {"label": "배회",    "color": [230, 126, 34]},
                            "park":         {"label": "공원",    "color": [46, 204, 113]},
                            "market":       {"label": "시장",    "color": [142, 68, 173]},
                            "home":         {"label": "집",      "color": [52, 152, 219]},
                            "work":         {"label": "출근",    "color": [44, 62, 80]},
                            "idle":         {"label": "대기",    "color": [149, 165, 166]},
                        }
                        matched = {"label": "이동 중", "color": [52, 152, 219]}
                        for key, info in status_info.items():
                            if key in status:
                                matched = info
                                break
                        if "moving" in status:
                            matched = {"label": "이동 중", "color": [52, 152, 219]}

                        agent_color = matched["color"]
                        agent_data = [{"lat": agent_lat, "lng": agent_lng,
                                       "emoji": "🧑",
                                       "status_label": matched["label"]}]

                        # 배경 원 (위치 강조, 펄스 느낌)
                        layers.append(pdk.Layer(
                            "ScatterplotLayer", data=agent_data,
                            get_position='[lng, lat]', get_radius=60,
                            get_fill_color=agent_color + [30],
                            get_line_color=agent_color + [100],
                            line_width_min_pixels=2, stroked=True,
                        ))
                        # 사람 내부 원 (진한 색)
                        layers.append(pdk.Layer(
                            "ScatterplotLayer", data=agent_data,
                            get_position='[lng, lat]', get_radius=25,
                            get_fill_color=agent_color + [220],
                            get_line_color=[255, 255, 255, 255],
                            line_width_min_pixels=3, stroked=True,
                        ))
                        # 사람 이모지 (크게)
                        layers.append(pdk.Layer(
                            "TextLayer", data=agent_data,
                            get_position='[lng, lat]', get_text='emoji',
                            get_size=32, get_color=[255, 255, 255],
                            get_text_anchor='"middle"',
                            get_alignment_baseline='"center"',
                        ))
                        # 상태 라벨 (아래쪽)
                        layers.append(pdk.Layer(
                            "TextLayer", data=agent_data,
                            get_position='[lng, lat]', get_text='status_label',
                            get_size=13, get_color=agent_color,
                            get_pixel_offset='[0, 30]',
                            get_text_anchor='"middle"',
                            font_family='"Noto Sans KR", sans-serif',
                        ))

                        # 모든 포인트를 수집해서 바운딩 박스 계산
                        all_lats = [agent_lat]
                        all_lngs = [agent_lng]
                        if route_coords:
                            for c in route_coords:
                                all_lats.append(c[0])
                                all_lngs.append(c[1])
                        for vl in visited_labels:
                            all_lats.append(vl["lat"])
                            all_lngs.append(vl["lng"])
                        min_lat, max_lat = min(all_lats), max(all_lats)
                        min_lng, max_lng = min(all_lngs), max(all_lngs)
                        center_lat = (min_lat + max_lat) / 2
                        center_lng = (min_lng + max_lng) / 2
                        # 경로 범위에 따라 줌 레벨 결정
                        lat_range = max_lat - min_lat
                        lng_range = max_lng - min_lng
                        spread = max(lat_range, lng_range)
                        if spread < 0.001:
                            zoom = 16.5
                        elif spread < 0.005:
                            zoom = 15.5
                        elif spread < 0.01:
                            zoom = 14.5
                        elif spread < 0.02:
                            zoom = 13.5
                        else:
                            zoom = 12.5
                        view_state = pdk.ViewState(
                            latitude=center_lat, longitude=center_lng,
                            zoom=zoom, pitch=0,
                        )
                        deck = pdk.Deck(
                            layers=layers,
                            initial_view_state=view_state,
                            map_style="light",
                            tooltip={"text": "{name}"},
                        )
                        st.pydeck_chart(deck, height=500)
                    else:
                        st.info("이 시간에 에이전트 위치 데이터가 없습니다.")

                with status_col:
                    st.markdown("### 현재 상태")

                    if status == "eating" and current_activity is not None:
                        st.markdown(f'<div class="status-box status-eating"><h4>🍽️ 식사 중</h4><p><b>매장:</b> {current_activity["visited_store"]}</p><p><b>카테고리:</b> {current_activity["visited_category"]}</p></div>', unsafe_allow_html=True)
                    elif status == "cafe":
                        cafe_name = current_activity.get('name', '카페') if (current_activity and isinstance(current_activity, dict)) else '카페'
                        st.markdown(f'<div class="status-box status-cafe"><h4>☕ 카페에서 휴식</h4><p><b>장소:</b> {cafe_name}</p></div>', unsafe_allow_html=True)
                    elif status == "wander":
                        st.markdown('<div class="status-box status-wander"><h4>🚶 배회 중</h4><p>망원동 거리를 걸으며 구경</p></div>', unsafe_allow_html=True)
                    elif status == "park":
                        st.markdown('<div class="status-box status-park"><h4>🌳 한강공원 산책</h4><p>망원한강공원에서 산책 중</p></div>', unsafe_allow_html=True)
                    elif status == "market":
                        st.markdown('<div class="status-box status-market"><h4>🛒 망원시장 장보기</h4><p>망원시장에서 장보기 중</p></div>', unsafe_allow_html=True)
                    elif status == "home":
                        st.markdown('<div class="status-box status-idle"><h4>🏠 집에서 휴식</h4><p>집에서 쉬는 중</p></div>', unsafe_allow_html=True)
                    elif status == "work":
                        st.markdown('<div class="status-box status-work"><h4>💼 회사에서 근무</h4><p>회사에서 일하는 중</p></div>', unsafe_allow_html=True)
                    elif "moving" in status and current_activity is not None:
                        if isinstance(current_activity, dict):
                            dest = current_activity.get('visited_store') or current_activity.get('name', '?')
                        elif hasattr(current_activity, 'get'):
                            dest = current_activity.get('visited_store', '?')
                        else:
                            dest = "?"
                        st.markdown(f'<div class="status-box status-moving"><h4>🚶 이동 중</h4><p><b>목적지:</b> {dest}</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status-box status-idle"><h4>🏠 대기 중</h4><p>집 또는 망원동 외부</p></div>', unsafe_allow_html=True)

                    if step5_action:
                        action_names = {
                            "카페_가기": "☕ 카페 가기", "배회하기": "🚶 배회하기",
                            "한강공원_산책": "🌳 한강공원 산책", "망원시장_장보기": "🛒 망원시장 장보기",
                            "집에서_쉬기": "🏠 집에서 쉬기", "회사_가기": "💼 회사 가기"
                        }
                        st.info(f"**현재 행동:** {action_names.get(step5_action, step5_action)}")

                    # 오늘 스케줄
                    st.markdown("### 📅 오늘의 스케줄")
                    day_data_anim = results_df[
                        (results_df['persona_id'] == selected_agent) &
                        (results_df['timestamp'].dt.date == anim_selected_date)
                    ].sort_values('timestamp')

                    for _, row in day_data_anim.iterrows():
                        slot = row['time_slot']
                        slot_hour = TIMESLOT_HOURS.get(slot, 0)
                        is_past = slot_hour + 2 <= current_hour
                        is_current = slot_hour <= current_hour < slot_hour + 2

                        if row['decision'] == 'visit':
                            icon = "▶️" if is_current else ("✅" if is_past else "⏳")
                            st.markdown(f"**{icon} {slot} ({slot_hour}:00)** - {row['visited_store']}")
                        else:
                            icon = "⬜" if is_past else "⏳"
                            st.markdown(f"**{icon} {slot} ({slot_hour}:00)** - 외부 식사")

                # 자동 재생: sleep 후 전체 rerun (시간 증가는 상단에서 처리)
                if st.session_state.anim_playing:
                    time_module.sleep(0.5)
                    st.rerun()

            animation_fragment()

            # 에이전트 방문 로그 상세
            st.markdown("### 📋 방문 기록 상세")

            agent_results = results_df[results_df['persona_id'] == selected_agent].copy()
            agent_visits = visits_df[visits_df['persona_id'] == selected_agent].copy() if not visits_df.empty else pd.DataFrame()

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

    # 차트
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("### 📊 시간대별 방문")
        if not filtered_visits.empty:
            time_visits = filtered_visits.groupby('time_slot').size().reset_index(name='count')
            time_order = ['아침', '점심', '저녁', '야식']
            time_visits['time_slot'] = pd.Categorical(time_visits['time_slot'], categories=time_order, ordered=True)
            time_visits = time_visits.sort_values('time_slot')

            fig_time = px.bar(
                time_visits, x='time_slot', y='count',
                color_discrete_sequence=['#1f77b4']
            )
            fig_time.update_layout(
                xaxis_title="", yaxis_title="방문 수",
                height=250, margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("방문 데이터 없음")

    with col_chart2:
        st.markdown("### 🏆 인기 매장 TOP 10")
        if not filtered_visits.empty:
            top_stores = filtered_visits['visited_store'].value_counts().head(10).reset_index()
            top_stores.columns = ['store', 'count']

            fig_stores = px.bar(
                top_stores, x='count', y='store',
                orientation='h', color_discrete_sequence=['#ff7f0e']
            )
            fig_stores.update_layout(
                xaxis_title="방문 수", yaxis_title="",
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_stores, use_container_width=True)
        else:
            st.info("방문 데이터 없음")

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
            gen_order = ['Z1', 'Z2', 'Y', 'X', 'S', '혼합', '혼합(Y+X)']
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
        base_cols = ['timestamp', 'persona_id', 'generation', 'segment',
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
