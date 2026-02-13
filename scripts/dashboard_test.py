"""
테스트용 대시보드 - 24시간 에이전트 시뮬레이션 애니메이션

OSM 도로망 기반 에이전트 이동 시각화 + Step 5 행동 (카페, 배회, 한강공원, 망원시장)

실행 방법:
    streamlit run scripts/dashboard_test.py
"""

import streamlit as st
import pandas as pd
import json
import random
from pathlib import Path
import folium
from streamlit_folium import st_folium
import time as time_module
from datetime import datetime, timedelta
import osmnx as ox
import networkx as nx

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"

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
}

# Step 5 행동별 지속 시간 (시간 단위)
ACTION_DURATION = {
    "카페_가기": 1.0,        # 1시간
    "배회하기": 0.5,         # 30분
    "한강공원_산책": 1.0,    # 1시간
    "망원시장_장보기": 0.75, # 45분
    "집에서_쉬기": None,     # 다음 timeslot까지 계속
    "회사_가기": None,       # 다음 timeslot까지 회사에서 근무
}

# 회사 위치 (기본값)
LANDMARKS["회사"] = {"lat": 37.5550, "lng": 126.9100, "name": "회사"}

# 페이지 설정
st.set_page_config(
    page_title="에이전트 24시간 시뮬레이션",
    page_icon="🕐",
    layout="wide",
)

# CSS 스타일
st.markdown("""
<style>
    .time-display {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        font-family: 'Courier New', monospace;
        color: #1f77b4;
        padding: 10px;
        background: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .status-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .status-eating {
        background: #d4edda;
        border-left: 5px solid #28a745;
    }
    .status-cafe {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .status-idle {
        background: #e2e3e5;
        border-left: 5px solid #6c757d;
    }
    .status-moving {
        background: #cce5ff;
        border-left: 5px solid #007bff;
    }
    .status-wander {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .status-park {
        background: #d1e7dd;
        border-left: 5px solid #198754;
    }
    .status-market {
        background: #e2d5f1;
        border-left: 5px solid #6f42c1;
    }
    .status-work {
        background: #d1ecf1;
        border-left: 5px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)


def get_walking_speed(segment, health_preference, seed=None):
    """페르소나 기반 걷는 속도 계산 (km/h) - LLM 스타일 시뮬레이션

    실제 시뮬레이션에서는 LLM이 페르소나를 보고 직접 판단.
    대시보드에서는 세그먼트/건강성향 기반으로 자연스러운 속도 범위 내에서 결정.
    """
    if seed is not None:
        random.seed(seed)

    # 세그먼트에서 특징 추출
    is_elderly = "고령자" in segment
    is_worker = "직장인" in segment
    has_young_child = "영유아" in segment or "초등생" in segment
    is_single = "1인가구" in segment

    # 기본 속도 범위 설정 (LLM이 판단하듯이)
    if is_elderly:
        # 고령자: 2.5 ~ 3.5 km/h
        base_min, base_max = 2.5, 3.5
    elif has_young_child:
        # 어린 자녀와 함께: 3.0 ~ 3.8 km/h
        base_min, base_max = 3.0, 3.8
    elif is_worker and is_single:
        # 1인가구 직장인: 4.0 ~ 5.5 km/h (바쁜 편)
        base_min, base_max = 4.0, 5.5
    elif is_worker:
        # 가정있는 직장인: 3.8 ~ 4.8 km/h
        base_min, base_max = 3.8, 4.8
    else:
        # 일반: 3.5 ~ 4.5 km/h
        base_min, base_max = 3.5, 4.5

    # 건강성향에 따른 조정
    health_adjust = 0.0
    if health_preference == "매우 중요":
        health_adjust = 0.5  # 더 활발하게 걸음
    elif health_preference == "중요함":
        health_adjust = 0.2
    elif health_preference == "중요하지 않음":
        health_adjust = -0.2
    elif health_preference == "전혀 중요하지 않음":
        health_adjust = -0.4

    # 최종 속도 계산 (범위 내에서 랜덤 + 건강성향 조정)
    speed = random.uniform(base_min, base_max) + health_adjust

    # 유효 범위 내로 제한 (1.5 ~ 6.0 km/h)
    return max(1.5, min(6.0, round(speed, 1)))


def calculate_route_distance(route_coords):
    """경로 좌표에서 총 거리 계산 (km)"""
    if not route_coords or len(route_coords) < 2:
        return 0.0

    total_dist = 0.0
    for i in range(len(route_coords) - 1):
        lat1, lng1 = route_coords[i]
        lat2, lng2 = route_coords[i + 1]
        # Haversine 공식 간소화 (작은 거리에서는 직선 거리 근사)
        # 1도 위도 ≈ 111km, 1도 경도 ≈ 88km (서울 위도에서)
        dlat = (lat2 - lat1) * 111
        dlng = (lng2 - lng1) * 88
        total_dist += (dlat ** 2 + dlng ** 2) ** 0.5

    return total_dist


def calculate_travel_time(route_coords, walking_speed):
    """경로 거리와 걷는 속도로 이동 시간 계산 (시간 단위)"""
    distance = calculate_route_distance(route_coords)
    if walking_speed <= 0:
        return 0.5  # 기본값 30분

    travel_time = distance / walking_speed
    # 최소 5분, 최대 1시간
    return max(5/60, min(1.0, travel_time))


@st.cache_data
def load_street_network():
    """OSM 도로망 로드"""
    try:
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


@st.cache_data
def get_route_coords(_G, start_lat, start_lng, end_lat, end_lng):
    """OSM 네트워크 위의 경로 좌표 계산"""
    if _G is None:
        return [(start_lat, start_lng), (end_lat, end_lng)]
    try:
        start_node = ox.nearest_nodes(_G, start_lng, start_lat)
        end_node = ox.nearest_nodes(_G, end_lng, end_lat)
        route = nx.shortest_path(_G, start_node, end_node, weight='length')
        route_coords = [(_G.nodes[node]['y'], _G.nodes[node]['x']) for node in route]
        return route_coords
    except Exception:
        return [(start_lat, start_lng), (end_lat, end_lng)]


def generate_wander_path(_G, start_lat, start_lng, num_nodes=10, seed=None):
    """배회 경로 생성 - OSM 네트워크에서 랜덤하게 노드 선택"""
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
                # 랜덤하게 인접 노드 선택 (이전 노드 제외)
                if len(path) > 1:
                    neighbors = [n for n in neighbors if n != path[-2]]
                if not neighbors:
                    neighbors = list(_G.neighbors(current_node))

                current_node = random.choice(neighbors)
                path.append(current_node)
            else:
                break

        # 노드 좌표로 변환
        coords = [(_G.nodes[node]['y'], _G.nodes[node]['x']) for node in path]
        return coords
    except Exception:
        return [(start_lat, start_lng)]


def interpolate_on_route(route_coords, progress):
    """경로 위에서 progress (0~1)에 해당하는 위치 계산"""
    if not route_coords or len(route_coords) < 2:
        return route_coords[0] if route_coords else (37.5565, 126.9029)

    # 전체 경로 길이 계산
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

    # progress에 해당하는 위치 찾기
    target_length = total_length * progress
    accumulated = 0

    for i, seg_len in enumerate(segment_lengths):
        if accumulated + seg_len >= target_length:
            if seg_len > 0:
                seg_progress = (target_length - accumulated) / seg_len
            else:
                seg_progress = 0
            lat1, lng1 = route_coords[i]
            lat2, lng2 = route_coords[i + 1]
            return (
                lat1 + (lat2 - lat1) * seg_progress,
                lng1 + (lng2 - lng1) * seg_progress
            )
        accumulated += seg_len


def get_traveled_path(route_coords, progress):
    """지나온 경로만 반환 (발자취)"""
    if not route_coords or len(route_coords) < 2:
        return route_coords if route_coords else []

    if progress <= 0:
        return [route_coords[0]]
    if progress >= 1:
        return route_coords

    # 전체 경로 길이 계산
    total_length = 0
    segment_lengths = []
    for i in range(len(route_coords) - 1):
        lat1, lng1 = route_coords[i]
        lat2, lng2 = route_coords[i + 1]
        seg_len = ((lat2 - lat1) ** 2 + (lng2 - lng1) ** 2) ** 0.5
        segment_lengths.append(seg_len)
        total_length += seg_len

    if total_length == 0:
        return [route_coords[0]]

    # progress에 해당하는 위치까지의 경로
    target_length = total_length * progress
    accumulated = 0
    traveled = [route_coords[0]]

    for i, seg_len in enumerate(segment_lengths):
        if accumulated + seg_len >= target_length:
            # 현재 세그먼트 내에서 중간 지점
            if seg_len > 0:
                seg_progress = (target_length - accumulated) / seg_len
            else:
                seg_progress = 0
            lat1, lng1 = route_coords[i]
            lat2, lng2 = route_coords[i + 1]
            current_pos = (
                lat1 + (lat2 - lat1) * seg_progress,
                lng1 + (lng2 - lng1) * seg_progress
            )
            traveled.append(current_pos)
            break
        else:
            traveled.append(route_coords[i + 1])
        accumulated += seg_len

    return traveled

    return route_coords[-1]


def load_data():
    """시뮬레이션 데이터 로드"""
    result_path = OUTPUT_DIR / "generative_simulation_result.csv"
    if result_path.exists():
        results_df = pd.read_csv(result_path)
        results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
    else:
        results_df = pd.DataFrame()

    visit_path = OUTPUT_DIR / "generative_visit_log.csv"
    if visit_path.exists():
        visits_df = pd.read_csv(visit_path)
        visits_df['timestamp'] = pd.to_datetime(visits_df['timestamp'])
    else:
        visits_df = pd.DataFrame()

    # 매장 데이터
    stores_dict = {}
    json_dir = DATA_DIR / "raw" / "split_by_store_id"
    if json_dir.exists():
        for json_file in json_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data and len(data) > 0:
                        store = data[0]
                        store_id = json_file.stem
                        stores_dict[store.get('store_name', '')] = {
                            'store_id': store_id,
                            'lat': store.get('y', 0),
                            'lng': store.get('x', 0),
                            'category': store.get('category', '')
                        }
            except Exception:
                continue

    # 좌표 추가
    if not visits_df.empty and stores_dict:
        visits_df['store_lat'] = visits_df['visited_store'].map(lambda x: stores_dict.get(x, {}).get('lat', 0))
        visits_df['store_lng'] = visits_df['visited_store'].map(lambda x: stores_dict.get(x, {}).get('lng', 0))

    if not results_df.empty and stores_dict:
        results_df['store_lat'] = results_df['visited_store'].map(lambda x: stores_dict.get(x, {}).get('lat', 0) if pd.notna(x) else None)
        results_df['store_lng'] = results_df['visited_store'].map(lambda x: stores_dict.get(x, {}).get('lng', 0) if pd.notna(x) else None)

    return results_df, visits_df, stores_dict


def get_step5_action(current_slot, next_slot, seed=None, segment=None):
    """Step 5 행동 결정 (간단 버전 - 실제로는 LLM 호출)

    seed는 날짜 + timeslot 기반으로 고정되어 같은 timeslot에서는 항상 같은 행동 선택
    직장인은 회사 가기 옵션이 추가됨
    """
    if seed is not None:
        random.seed(seed)  # current_hour 제거 - timeslot 내 고정

    is_worker = segment and "직장인" in segment

    if is_worker:
        # 직장인은 회사 가기 옵션 포함
        actions = ["카페_가기", "배회하기", "한강공원_산책", "망원시장_장보기", "집에서_쉬기", "회사_가기"]
        weights = [0.2, 0.15, 0.15, 0.1, 0.1, 0.3]  # 직장인은 회사 가기 비중 높음
    else:
        actions = ["카페_가기", "배회하기", "한강공원_산책", "망원시장_장보기", "집에서_쉬기"]
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]

    return random.choices(actions, weights=weights)[0]


def get_agent_state(results_df, stores_dict, G, cafe_stores, current_date, current_hour, segment, health_preference, agent_name=None):
    """현재 시간에 에이전트 상태 및 위치 계산 (OSM 경로 기반 + Step 5)

    페르소나 기반 걷는 속도를 적용하여 동적 이동 시간 계산
    걷는 속도는 LLM이 페르소나 특징을 보고 직접 판단 (시뮬레이션)
    """
    day_data = results_df[results_df['timestamp'].dt.date == current_date]
    if day_data.empty:
        return None, None, "idle", None, [], None

    # 페르소나 기반 걷는 속도 계산 (LLM 스타일 - 에이전트별 일관된 속도)
    speed_seed = hash(agent_name) if agent_name else hash(segment + str(health_preference))
    walking_speed = get_walking_speed(segment, health_preference, seed=speed_seed)

    # 기본 위치 (망원동 중심 - 집)
    home_lat, home_lng = LANDMARKS["집"]["lat"], LANDMARKS["집"]["lng"]

    # 타임슬롯별 데이터 정리
    slot_data = {}
    for slot in TIMESLOT_HOURS.keys():
        slot_rows = day_data[day_data['time_slot'] == slot]
        if not slot_rows.empty:
            slot_data[slot] = slot_rows.iloc[0]

    # 현재 시간 기준 상태 결정
    sorted_slots = sorted(TIMESLOT_HOURS.items(), key=lambda x: x[1])

    prev_slot = None
    next_slot = None
    prev_slot_idx = -1

    for i, (slot, hour) in enumerate(sorted_slots):
        if hour <= current_hour:
            prev_slot = slot
            prev_slot_idx = i
        if hour > current_hour and next_slot is None:
            next_slot = slot

    # 상태 및 경로 결정
    route_coords = []
    step5_action = None

    # 이전 슬롯에서 방문한 경우
    if prev_slot and prev_slot in slot_data:
        prev_row = slot_data[prev_slot]
        prev_hour = TIMESLOT_HOURS[prev_slot]

        if prev_row['decision'] == 'visit':
            store_name = prev_row['visited_store']
            store_info = stores_dict.get(store_name, {})
            store_lat = store_info.get('lat', home_lat)
            store_lng = store_info.get('lng', home_lng)

            # 출발 위치 결정 (이전 슬롯 매장 or 집)
            start_lat, start_lng = home_lat, home_lng
            prev_prev_slot = None
            for s, h in sorted_slots:
                if h < prev_hour:
                    prev_prev_slot = s

            if prev_prev_slot and prev_prev_slot in slot_data:
                pp_row = slot_data[prev_prev_slot]
                if pp_row['decision'] == 'visit':
                    pp_store = stores_dict.get(pp_row['visited_store'], {})
                    start_lat = pp_store.get('lat', home_lat)
                    start_lng = pp_store.get('lng', home_lng)

            # 경로 계산 및 동적 이동 시간 계산
            route_coords = get_route_coords(G, start_lat, start_lng, store_lat, store_lng)
            travel_time = calculate_travel_time(route_coords, walking_speed)

            # 타임슬롯 시간에 출발 (prev_hour에 출발)
            arrival_time = prev_hour + travel_time
            eating_end = arrival_time + 1.5

            if current_hour < arrival_time:
                # 매장으로 이동 중
                progress = (current_hour - prev_hour) / travel_time if travel_time > 0 else 1.0
                progress = min(1.0, max(0.0, progress))
                lat, lng = interpolate_on_route(route_coords, progress)
                return lat, lng, "moving", prev_row, route_coords, None

            elif current_hour < eating_end:
                # 식사 중
                return store_lat, store_lng, "eating", prev_row, [], None

            else:
                # 식사 끝남 - Step 5 행동 결정
                next_meal_hour = 24.0  # 기본값
                if next_slot and next_slot in slot_data:
                    next_meal_hour = TIMESLOT_HOURS[next_slot]

                # 다음 식사 시간까지 Step 5 행동 (timeslot 정시에 출발하므로)
                free_time_end = next_meal_hour

                if current_hour < free_time_end:
                    # Step 5 행동 수행 (seed는 날짜+timeslot으로 고정)
                    seed = int(current_date.toordinal()) + prev_slot_idx
                    step5_action = get_step5_action(prev_slot, next_slot, seed, segment)

                    time_in_action = current_hour - eating_end
                    action_duration = ACTION_DURATION.get(step5_action)
                    move_time = 0.25  # 이동 시간 15분

                    # 행동별 총 소요 시간 (이동 + 행동 + 복귀)
                    if action_duration is None:
                        # 집에서 쉬기: 다음 timeslot까지
                        total_action_time = free_time_end - eating_end
                    else:
                        total_action_time = move_time + action_duration + move_time

                    # 행동이 아직 진행 중인지 확인
                    if time_in_action < total_action_time:
                        progress = time_in_action / total_action_time if total_action_time > 0 else 0
                        progress = min(1.0, max(0.0, progress))

                        if step5_action == "카페_가기":
                            # 카페로 이동 후 머무르기
                            if cafe_stores and stores_dict:
                                random.seed(seed)
                                cafe_name = None
                                for store_name, info in stores_dict.items():
                                    if info.get('store_id') in cafe_stores:
                                        cafe_name = store_name
                                        break

                                if cafe_name:
                                    cafe_info = stores_dict[cafe_name]
                                    cafe_lat = cafe_info.get('lat', home_lat)
                                    cafe_lng = cafe_info.get('lng', home_lng)

                                    move_ratio = move_time / total_action_time
                                    stay_ratio = action_duration / total_action_time

                                    if progress < move_ratio:
                                        # 카페로 이동 중
                                        route_coords = get_route_coords(G, store_lat, store_lng, cafe_lat, cafe_lng)
                                        move_progress = progress / move_ratio
                                        lat, lng = interpolate_on_route(route_coords, move_progress)
                                        return lat, lng, "moving_cafe", {"name": cafe_name}, route_coords, step5_action
                                    elif progress < move_ratio + stay_ratio:
                                        # 카페에서 휴식
                                        return cafe_lat, cafe_lng, "cafe", {"name": cafe_name}, [], step5_action
                                    else:
                                        # 원래 위치로 복귀
                                        route_coords = get_route_coords(G, cafe_lat, cafe_lng, store_lat, store_lng)
                                        return_progress = (progress - move_ratio - stay_ratio) / move_ratio
                                        lat, lng = interpolate_on_route(route_coords, return_progress)
                                        return lat, lng, "moving", None, route_coords, step5_action

                            return store_lat, store_lng, "cafe", None, [], step5_action

                        elif step5_action == "배회하기":
                            # OSM 네트워크에서 배회
                            wander_seed = seed
                            wander_path = generate_wander_path(G, store_lat, store_lng, num_nodes=15, seed=wander_seed)
                            lat, lng = interpolate_on_route(wander_path, progress)
                            return lat, lng, "wander", None, wander_path, step5_action

                        elif step5_action == "한강공원_산책":
                            park = LANDMARKS["한강공원"]
                            move_ratio = move_time / total_action_time
                            stay_ratio = action_duration / total_action_time

                            if progress < move_ratio:
                                route_coords = get_route_coords(G, store_lat, store_lng, park["lat"], park["lng"])
                                move_progress = progress / move_ratio
                                lat, lng = interpolate_on_route(route_coords, move_progress)
                                return lat, lng, "moving_park", None, route_coords, step5_action
                            elif progress < move_ratio + stay_ratio:
                                wander_path = generate_wander_path(G, park["lat"], park["lng"], num_nodes=10, seed=seed)
                                park_progress = (progress - move_ratio) / stay_ratio
                                lat, lng = interpolate_on_route(wander_path, park_progress)
                                return lat, lng, "park", None, wander_path, step5_action
                            else:
                                route_coords = get_route_coords(G, park["lat"], park["lng"], store_lat, store_lng)
                                return_progress = (progress - move_ratio - stay_ratio) / move_ratio
                                lat, lng = interpolate_on_route(route_coords, return_progress)
                                return lat, lng, "moving", None, route_coords, step5_action

                        elif step5_action == "망원시장_장보기":
                            market = LANDMARKS["망원시장"]
                            move_ratio = move_time / total_action_time
                            stay_ratio = action_duration / total_action_time

                            if progress < move_ratio:
                                route_coords = get_route_coords(G, store_lat, store_lng, market["lat"], market["lng"])
                                move_progress = progress / move_ratio
                                lat, lng = interpolate_on_route(route_coords, move_progress)
                                return lat, lng, "moving_market", None, route_coords, step5_action
                            elif progress < move_ratio + stay_ratio:
                                wander_path = generate_wander_path(G, market["lat"], market["lng"], num_nodes=8, seed=seed)
                                market_progress = (progress - move_ratio) / stay_ratio
                                lat, lng = interpolate_on_route(wander_path, market_progress)
                                return lat, lng, "market", None, wander_path, step5_action
                            else:
                                route_coords = get_route_coords(G, market["lat"], market["lng"], store_lat, store_lng)
                                return_progress = (progress - move_ratio - stay_ratio) / move_ratio
                                lat, lng = interpolate_on_route(route_coords, return_progress)
                                return lat, lng, "moving", None, route_coords, step5_action

                        elif step5_action == "집에서_쉬기":
                            move_ratio = move_time / total_action_time if total_action_time > 0 else 0.1
                            if progress < move_ratio:
                                route_coords = get_route_coords(G, store_lat, store_lng, home_lat, home_lng)
                                move_progress = progress / move_ratio
                                lat, lng = interpolate_on_route(route_coords, move_progress)
                                return lat, lng, "moving_home", None, route_coords, step5_action
                            else:
                                return home_lat, home_lng, "home", None, [], step5_action

                        elif step5_action == "회사_가기":
                            work = LANDMARKS["회사"]
                            move_ratio = move_time / total_action_time if total_action_time > 0 else 0.1
                            if progress < move_ratio:
                                route_coords = get_route_coords(G, store_lat, store_lng, work["lat"], work["lng"])
                                move_progress = progress / move_ratio
                                lat, lng = interpolate_on_route(route_coords, move_progress)
                                return lat, lng, "moving_work", None, route_coords, step5_action
                            else:
                                return work["lat"], work["lng"], "work", None, [], step5_action

                    else:
                        # 행동 완료 - 원래 위치에서 대기 (다음 식사 이동 전까지)
                        return store_lat, store_lng, "idle", None, [], None

                else:
                    # 다음 식사 시간이 됨 - 다음 식사를 위해 이동
                    if next_slot and next_slot in slot_data:
                        next_row = slot_data[next_slot]
                        if next_row['decision'] == 'visit':
                            next_store = stores_dict.get(next_row['visited_store'], {})
                            next_lat = next_store.get('lat', home_lat)
                            next_lng = next_store.get('lng', home_lng)

                            # 다음 식사 timeslot 시간에 출발
                            next_move_start = next_meal_hour
                            route_coords = get_route_coords(G, store_lat, store_lng, next_lat, next_lng)
                            next_travel_time = calculate_travel_time(route_coords, walking_speed)

                            if current_hour >= next_move_start:
                                progress = (current_hour - next_move_start) / next_travel_time if next_travel_time > 0 else 1.0
                                progress = min(1.0, max(0.0, progress))
                                lat, lng = interpolate_on_route(route_coords, progress)
                                return lat, lng, "moving", next_row, route_coords, None

                return store_lat, store_lng, "idle", None, [], None
        else:
            # 외식 안함 - 집에서 대기
            pass

    # 다음 슬롯으로 이동 확인 (timeslot 시간에 출발)
    if next_slot and next_slot in slot_data:
        next_row = slot_data[next_slot]
        next_hour = TIMESLOT_HOURS[next_slot]

        if next_row['decision'] == 'visit':
            next_store = stores_dict.get(next_row['visited_store'], {})
            next_lat = next_store.get('lat', home_lat)
            next_lng = next_store.get('lng', home_lng)

            # timeslot 시간에 출발 (30분 전이 아닌 정시 출발)
            move_start = next_hour
            if current_hour >= move_start:
                route_coords = get_route_coords(G, home_lat, home_lng, next_lat, next_lng)
                next_travel_time = calculate_travel_time(route_coords, walking_speed)
                progress = (current_hour - move_start) / next_travel_time if next_travel_time > 0 else 1.0
                progress = min(1.0, max(0.0, progress))
                lat, lng = interpolate_on_route(route_coords, progress)
                return lat, lng, "moving", next_row, route_coords, None

    return home_lat, home_lng, "idle", None, [], None


# 데이터 로드
results_df, visits_df, stores_dict = load_data()
cafe_stores = load_cafe_stores()

if results_df.empty:
    st.error("시뮬레이션 결과가 없습니다. 먼저 시뮬레이션을 실행하세요.")
    st.stop()

# OSM 네트워크 로드
with st.spinner("도로망 로드 중..."):
    G = load_street_network()

# 에이전트 정보
agent_name = results_df['agent_name'].iloc[0]
agent_segment = results_df['segment'].iloc[0]
agent_health = results_df['health_preference'].iloc[0]
agent_change = results_df['change_preference'].iloc[0]

# 헤더
st.title("🕐 24시간 에이전트 시뮬레이션")

# 걷는 속도 계산 (에이전트 이름 기반 시드로 일관성 유지)
walking_speed_display = get_walking_speed(agent_segment, agent_health, seed=hash(agent_name))

# 사이드바
st.sidebar.markdown(f"""
### 🧑 {agent_name}
- **세그먼트**: {agent_segment}
- **건강성향**: {agent_health}
- **변화성향**: {agent_change}
- **걷는 속도**: {walking_speed_display:.1f} km/h (LLM 판단)
""")

# 날짜 선택
results_df['date'] = results_df['timestamp'].dt.date
available_dates = sorted(results_df['date'].unique())

selected_date = st.sidebar.selectbox(
    "날짜 선택",
    available_dates,
    format_func=lambda x: f"{x} ({results_df[results_df['date']==x]['weekday'].iloc[0]}요일)"
)

# 애니메이션 컨트롤
st.sidebar.markdown("---")
st.sidebar.markdown("### 애니메이션 컨트롤")

if 'current_hour' not in st.session_state:
    st.session_state.current_hour = 6.0
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False

ctrl_col1, ctrl_col2, ctrl_col3 = st.sidebar.columns(3)

if ctrl_col1.button("⏮️"):
    st.session_state.current_hour = 6.0
    st.session_state.is_playing = False

if ctrl_col2.button("▶️" if not st.session_state.is_playing else "⏸️"):
    st.session_state.is_playing = not st.session_state.is_playing

if ctrl_col3.button("⏭️"):
    st.session_state.current_hour = 24.0
    st.session_state.is_playing = False

speed = st.sidebar.slider("속도 (배속)", 1, 60, 10, 1)

current_hour = st.sidebar.slider(
    "시간", 6.0, 24.0,
    st.session_state.current_hour, 1/3600,  # 1초 단위
    format="%.4f"
)
st.session_state.current_hour = current_hour

# 메인 영역
main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    hours = int(current_hour)
    remaining = (current_hour - hours) * 60
    minutes = int(remaining)
    seconds = int((remaining - minutes) * 60)
    st.markdown(f'<div class="time-display">🕐 {hours:02d}:{minutes:02d}:{seconds:02d}</div>', unsafe_allow_html=True)

    # 에이전트 상태 계산 (LLM이 페르소나 보고 판단한 걷는 속도 적용)
    agent_lat, agent_lng, status, current_activity, route_coords, step5_action = get_agent_state(
        results_df, stores_dict, G, cafe_stores, selected_date, current_hour,
        agent_segment, agent_health, agent_name
    )

    if agent_lat and agent_lng:
        m = folium.Map(
            location=[agent_lat, agent_lng],
            zoom_start=16,
            tiles='cartodbpositron'
        )

        # 랜드마크 표시
        for name, info in LANDMARKS.items():
            if name != "집":
                folium.CircleMarker(
                    location=[info["lat"], info["lng"]],
                    radius=8,
                    color='purple',
                    fill=True,
                    fillColor='purple',
                    fillOpacity=0.5,
                    tooltip=f"📍 {info['name']}"
                ).add_to(m)

        # 이동/배회 경로 표시 (발자취 - 현재 위치까지만)
        if route_coords and len(route_coords) > 1:
            if status == "wander":
                color = 'red'
            elif "park" in status:
                color = 'green'
            elif "market" in status:
                color = 'purple'
            else:
                color = 'blue'

            # 현재 위치까지의 경로만 추출 (발자취)
            traveled = [route_coords[0]]
            for i in range(1, len(route_coords)):
                coord = route_coords[i]
                # 현재 에이전트 위치에 도달했는지 확인
                dist_to_agent = ((coord[0] - agent_lat) ** 2 + (coord[1] - agent_lng) ** 2) ** 0.5
                if dist_to_agent < 0.0001:  # 거의 같은 위치
                    traveled.append(coord)
                    break
                traveled.append(coord)
                # 다음 좌표가 현재 위치를 지나쳤는지 확인
                if i < len(route_coords) - 1:
                    next_coord = route_coords[i + 1]
                    # 현재 세그먼트 내에 에이전트가 있는지
                    seg_len = ((next_coord[0] - coord[0]) ** 2 + (next_coord[1] - coord[1]) ** 2) ** 0.5
                    agent_dist = ((agent_lat - coord[0]) ** 2 + (agent_lng - coord[1]) ** 2) ** 0.5
                    if agent_dist < seg_len:
                        traveled.append((agent_lat, agent_lng))
                        break

            # 마지막에 현재 위치 추가 (없으면)
            if len(traveled) > 0 and traveled[-1] != (agent_lat, agent_lng):
                last = traveled[-1]
                if ((last[0] - agent_lat) ** 2 + (last[1] - agent_lng) ** 2) ** 0.5 > 0.00001:
                    traveled.append((agent_lat, agent_lng))

            if len(traveled) > 1:
                folium.PolyLine(
                    traveled,
                    color=color,
                    weight=4,
                    opacity=0.9
                ).add_to(m)

        # 오늘 방문한 매장들 표시
        day_visits = visits_df[visits_df['timestamp'].dt.date == selected_date]
        for _, row in day_visits.iterrows():
            visit_hour = TIMESLOT_HOURS.get(row['time_slot'], 0)
            if visit_hour + 0.5 <= current_hour:
                folium.CircleMarker(
                    location=[row['store_lat'], row['store_lng']],
                    radius=10,
                    color='green',
                    fill=True,
                    fillColor='green',
                    fillOpacity=0.6,
                    tooltip=f"✓ {row['visited_store']} ({row['time_slot']})"
                ).add_to(m)

        # 에이전트 마커
        if status == "eating":
            icon_html = '<div style="font-size: 28px;">🍽️</div>'
            tooltip = f"🍽️ {current_activity['visited_store']}에서 식사 중"
        elif status == "cafe":
            icon_html = '<div style="font-size: 28px;">☕</div>'
            cafe_name = current_activity.get('name', '카페') if (current_activity is not None and isinstance(current_activity, dict)) else '카페'
            tooltip = f"☕ {cafe_name}에서 휴식 중"
        elif status == "wander":
            icon_html = '<div style="font-size: 28px;">🚶</div>'
            tooltip = "🚶 망원동 거리 배회 중"
        elif status == "park":
            icon_html = '<div style="font-size: 28px;">🌳</div>'
            tooltip = "🌳 한강공원 산책 중"
        elif status == "market":
            icon_html = '<div style="font-size: 28px;">🛒</div>'
            tooltip = "🛒 망원시장 장보기 중"
        elif status == "home":
            icon_html = '<div style="font-size: 28px;">🏠</div>'
            tooltip = "🏠 집에서 휴식 중"
        elif status == "work":
            icon_html = '<div style="font-size: 28px;">💼</div>'
            tooltip = "💼 회사에서 근무 중"
        elif "moving" in status:
            icon_html = '<div style="font-size: 28px;">🚶</div>'
            if current_activity is not None:
                if isinstance(current_activity, dict):
                    dest = current_activity.get('visited_store') or current_activity.get('name', '?')
                else:
                    # pandas Series인 경우
                    dest = current_activity.get('visited_store', '?') if hasattr(current_activity, 'get') else '?'
            else:
                dest = "목적지"
            tooltip = f"🚶 {dest}(으)로 이동 중"
        else:
            icon_html = '<div style="font-size: 28px;">🏠</div>'
            tooltip = "🏠 집/대기"

        folium.Marker(
            location=[agent_lat, agent_lng],
            icon=folium.DivIcon(
                html=icon_html,
                icon_size=(35, 35),
                icon_anchor=(17, 17)
            ),
            tooltip=tooltip
        ).add_to(m)

        st_folium(m, width=700, height=500)

with main_col2:
    st.markdown("### 현재 상태")

    if status == "eating" and current_activity is not None:
        st.markdown(f"""
        <div class="status-box status-eating">
            <h4>🍽️ 식사 중</h4>
            <p><b>매장:</b> {current_activity['visited_store']}</p>
            <p><b>카테고리:</b> {current_activity['visited_category']}</p>
        </div>
        """, unsafe_allow_html=True)
    elif status == "cafe":
        cafe_name = current_activity.get('name', '카페') if (current_activity is not None and isinstance(current_activity, dict)) else '카페'
        st.markdown(f"""
        <div class="status-box status-cafe">
            <h4>☕ 카페에서 휴식</h4>
            <p><b>장소:</b> {cafe_name}</p>
        </div>
        """, unsafe_allow_html=True)
    elif status == "wander":
        st.markdown("""
        <div class="status-box status-wander">
            <h4>🚶 배회 중</h4>
            <p>망원동 거리를 걸으며 구경</p>
        </div>
        """, unsafe_allow_html=True)
    elif status == "park":
        st.markdown("""
        <div class="status-box status-park">
            <h4>🌳 한강공원 산책</h4>
            <p>망원한강공원에서 산책 중</p>
        </div>
        """, unsafe_allow_html=True)
    elif status == "market":
        st.markdown("""
        <div class="status-box status-market">
            <h4>🛒 망원시장 장보기</h4>
            <p>망원시장에서 장보기 중</p>
        </div>
        """, unsafe_allow_html=True)
    elif status == "home":
        st.markdown("""
        <div class="status-box status-idle">
            <h4>🏠 집에서 휴식</h4>
            <p>집에서 쉬는 중</p>
        </div>
        """, unsafe_allow_html=True)
    elif status == "work":
        st.markdown("""
        <div class="status-box status-work">
            <h4>💼 회사에서 근무</h4>
            <p>회사에서 일하는 중</p>
        </div>
        """, unsafe_allow_html=True)
    elif "moving" in status and current_activity is not None:
        if isinstance(current_activity, dict):
            dest = current_activity.get('visited_store') or current_activity.get('name', '?')
        elif hasattr(current_activity, 'get'):
            # pandas Series인 경우
            dest = current_activity.get('visited_store', '?')
        else:
            dest = "?"
        st.markdown(f"""
        <div class="status-box status-moving">
            <h4>🚶 이동 중</h4>
            <p><b>목적지:</b> {dest}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-box status-idle">
            <h4>🏠 대기 중</h4>
            <p>집 또는 망원동 외부</p>
        </div>
        """, unsafe_allow_html=True)

    # Step 5 행동 표시
    if step5_action:
        action_names = {
            "카페_가기": "☕ 카페 가기",
            "배회하기": "🚶 배회하기",
            "한강공원_산책": "🌳 한강공원 산책",
            "망원시장_장보기": "🛒 망원시장 장보기",
            "집에서_쉬기": "🏠 집에서 쉬기",
            "회사_가기": "💼 회사 가기"
        }
        st.info(f"**현재 행동:** {action_names.get(step5_action, step5_action)}")

    st.markdown("### 📅 오늘의 스케줄")
    day_data = results_df[results_df['date'] == selected_date].sort_values('timestamp')

    for _, row in day_data.iterrows():
        slot = row['time_slot']
        slot_hour = TIMESLOT_HOURS.get(slot, 0)
        is_past = slot_hour + 2 <= current_hour
        is_current = slot_hour <= current_hour < slot_hour + 2

        if row['decision'] == 'visit':
            if is_current:
                icon = "▶️"
            elif is_past:
                icon = "✅"
            else:
                icon = "⏳"
            st.markdown(f"**{icon} {slot} ({slot_hour}:00)** - {row['visited_store']}")
        else:
            icon = "⬜" if is_past else "⏳"
            st.markdown(f"**{icon} {slot} ({slot_hour}:00)** - 외부 식사")

    st.markdown("### 📊 통계")
    day_visits = visits_df[visits_df['timestamp'].dt.date == selected_date]
    visited_count = len(day_visits[day_visits['timestamp'].dt.hour + 2 <= current_hour])
    st.metric("방문 완료", f"{visited_count}개")

# 자동 재생 (1초 = 1/3600 시간)
if st.session_state.is_playing:
    time_module.sleep(0.1)  # 0.1초마다 업데이트
    st.session_state.current_hour += (1/3600) * speed  # speed초씩 증가
    if st.session_state.current_hour >= 24.0:
        st.session_state.current_hour = 6.0
        st.session_state.is_playing = False
    st.rerun()
