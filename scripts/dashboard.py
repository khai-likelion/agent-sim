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
from pathlib import Path
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

# CSS 스타일 — Tailwind-inspired 모던 디자인 시스템
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    /* ── 글로벌 리셋 + 폰트 ── */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #f8fafc;
    }

    /* ── 사이드바: 다크 슬레이트 ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 60%, #0f172a 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label {
        color: #64748b !important; font-size: 0.7rem; text-transform: uppercase;
        letter-spacing: 0.08em; font-weight: 600;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

    /* ── 메트릭 카드 ── */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02);
        transition: all 0.2s cubic-bezier(0.4,0,0.2,1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px -5px rgba(59,130,246,0.12), 0 4px 10px rgba(0,0,0,0.04);
        border-color: #bfdbfe;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e40af, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 6px;
    }

    /* ── 시간 디스플레이 ── */
    .time-display {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        color: #0f172a;
        padding: 12px 20px;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        letter-spacing: 6px;
        position: relative;
    }
    .time-display .time-period {
        font-size: 0.9rem; color: #94a3b8; margin-left: 8px;
        letter-spacing: 0.05em; font-weight: 500;
    }

    /* ── 플레이어 컨트롤 바 ── */
    .player-bar {
        display: flex; align-items: center; gap: 8px;
        background: white; border: 1px solid #e2e8f0;
        border-radius: 14px; padding: 10px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }

    /* ── 상태 박스 ── */
    .status-box {
        padding: 16px 20px;
        border-radius: 12px;
        margin: 8px 0;
        border: 1px solid transparent;
        position: relative;
        overflow: hidden;
    }
    .status-box::before {
        content: ''; position: absolute; top: 0; left: 0; bottom: 0; width: 3px;
    }
    .status-box h4 {
        margin: 0 0 6px 0; font-size: 0.95rem; font-weight: 600;
        letter-spacing: -0.01em;
    }
    .status-box p {
        margin: 2px 0; font-size: 0.82rem; color: #64748b; line-height: 1.6;
    }
    .status-eating {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border-color: #bbf7d0;
    }
    .status-eating::before { background: #22c55e; }
    .status-cafe {
        background: linear-gradient(135deg, #fffbeb, #fef3c7);
        border-color: #fde68a;
    }
    .status-cafe::before { background: #f59e0b; }
    .status-idle {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border-color: #e2e8f0;
    }
    .status-idle::before { background: #94a3b8; }
    .status-moving {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border-color: #bfdbfe;
    }
    .status-moving::before { background: #3b82f6; }
    .status-wander {
        background: linear-gradient(135deg, #fff7ed, #ffedd5);
        border-color: #fed7aa;
    }
    .status-wander::before { background: #f97316; }
    .status-park {
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
        border-color: #a7f3d0;
    }
    .status-park::before { background: #10b981; }
    .status-market {
        background: linear-gradient(135deg, #f5f3ff, #ede9fe);
        border-color: #c4b5fd;
    }
    .status-market::before { background: #8b5cf6; }
    .status-work {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border-color: #bae6fd;
    }
    .status-work::before { background: #0ea5e9; }

    /* ── 프로필 뱃지 ── */
    .profile-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: #eff6ff; color: #1e40af;
        padding: 5px 14px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600;
        border: 1px solid #bfdbfe;
    }

    /* ── 스케줄 타임라인 ── */
    .schedule-item {
        display: flex; align-items: center; gap: 12px;
        padding: 10px 14px; border-radius: 10px;
        margin: 3px 0; font-size: 0.85rem;
        transition: all 0.15s cubic-bezier(0.4,0,0.2,1);
        border: 1px solid transparent;
    }
    .schedule-item:hover { background: #f8fafc; border-color: #e2e8f0; }
    .schedule-current {
        background: #eff6ff !important; border: 1px solid #bfdbfe !important;
        font-weight: 600;
    }
    .schedule-past { color: #cbd5e1; }
    .schedule-future { color: #334155; }

    /* ── 탭: 세그먼트 컨트롤 스타일 ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px; background: #f1f5f9; border-radius: 12px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; padding: 8px 20px;
        font-weight: 500; font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
    }

    /* ── 헤더 타이포그래피 ── */
    h1 { letter-spacing: -0.03em; font-weight: 800; color: #0f172a; }
    h2 { letter-spacing: -0.02em; font-weight: 700; color: #1e293b; }
    h3 { font-weight: 600; font-size: 1.1rem; color: #334155; letter-spacing: -0.01em; }

    /* ── 버튼: 모던 필 + 아웃라인 ── */
    .stButton > button {
        border-radius: 10px; font-weight: 600; font-size: 0.82rem;
        border: 1px solid #e2e8f0; background: white; color: #334155;
        transition: all 0.15s cubic-bezier(0.4,0,0.2,1);
        padding: 8px 16px;
    }
    .stButton > button:hover {
        background: #f8fafc; border-color: #3b82f6;
        color: #1e40af; box-shadow: 0 2px 8px rgba(59,130,246,0.12);
    }
    .stButton > button:active {
        transform: scale(0.98); background: #eff6ff;
    }

    /* ── 슬라이더 커스텀 ── */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: white !important; border: 2px solid #3b82f6 !important;
        box-shadow: 0 2px 6px rgba(59,130,246,0.25) !important;
        width: 20px !important; height: 20px !important;
    }

    /* ── Folium 지도 컨테이너 ── */
    iframe[title="streamlit_folium.st_folium"] {
        border-radius: 14px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06) !important;
    }

    /* ── 셀렉트박스 ── */
    .stSelectbox > div > div {
        border-radius: 10px !important;
        border-color: #e2e8f0 !important;
    }

    /* ── expander ── */
    .streamlit-expanderHeader {
        font-weight: 600; font-size: 0.88rem;
        border-radius: 10px;
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

    # home_location이 없거나 [0,0]인 에이전트에 유형별 초기 위치 할당
    def _is_valid_loc(loc):
        return loc and loc != [0.0, 0.0] and loc != [0, 0]

    for agent in agents:
        if not _is_valid_loc(agent.get('home_location')):
            # 유동 에이전트: entry_point 우선
            ep = agent.get('entry_point')
            if _is_valid_loc(ep):
                agent['home_location'] = list(ep)
                continue
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
    json_dir = DATA_DIR / "raw" / "split_by_store_id_ver5"
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
        G = ox.graph_from_point((37.5564, 126.9053), dist=2000, network_type='walk')
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

    if home_location and home_location != [0.0, 0.0] and home_location != [0, 0]:
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


def create_map_with_routes(visits_df, stores_df, agents, selected_date=None,
                           store_filter=None, show_routes=False, G=None,
                           results_df=None):
    """Folium 지도 생성 (에이전트 위치 & 방문 현황)"""
    center_lat, center_lon = 37.5565, 126.9029
    m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles='cartodbpositron')

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
                # 방문 횟수에 비례한 크기 (최소 5, 최대 14)
                r = min(14, max(5, 3 + count * 0.8))
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=r,
                    color='#3b82f6',
                    fill=True,
                    fill_color='#3b82f6',
                    fill_opacity=0.55,
                    weight=1.5,
                    popup=f"<b>{store_name}</b><br>방문 {count}회",
                    tooltip=f"{store_name}: {count}회"
                ).add_to(m)

    return m


def main():
    # ── 시뮬레이션 폴더 선택 ──
    sim_folders = ["(기본)"]
    for d in sorted(OUTPUT_DIR.iterdir()):
        if d.is_dir() and (d / "visit_log.csv").exists():
            sim_folders.append(d.name)

    st.sidebar.markdown("### SIMULATION")
    selected_sim = st.sidebar.selectbox(
        "결과 폴더", sim_folders, index=0,
        help="before/after 비교 시뮬레이션 결과를 선택하세요"
    )
    sim_folder = "" if selected_sim == "(기본)" else selected_sim

    # 데이터 로드
    results_df, visits_df, agents, stores_df, stores_dict = load_simulation_data(sim_folder)
    cafe_stores = load_cafe_stores()

    # 사이드바 - 필터
    st.sidebar.markdown("---")
    st.sidebar.markdown("### FILTERS")

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
    st.sidebar.markdown("### STORE")

    if not visits_df.empty:
        all_stores = ["전체"] + sorted(visits_df['visited_store'].unique())
        store_filter = st.sidebar.selectbox("특정 매장만 보기", all_stores, index=0)

        if store_filter != "전체":
            filtered_visits = filtered_visits[filtered_visits['visited_store'] == store_filter]
    else:
        store_filter = "전체"

    # 에이전트 선택 (개별 추적용)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### AGENT TRACKING")

    if agents and not results_df.empty:
        active_pids = set(results_df['persona_id'].unique())
        agent_names = ["전체"] + sorted([a['persona_id'] for a in agents if a['persona_id'] in active_pids])
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
    st.sidebar.markdown("### OVERVIEW")

    if not results_df.empty:
        total_events = len(results_df)
        total_visits = len(visits_df)
        st.sidebar.markdown(f"이벤트 **{total_events:,}**건 · 방문 **{total_visits:,}**건")

    # 메인 콘텐츠 — 헤더
    sim_label = f" — {sim_folder}" if sim_folder else ""
    st.markdown(f"## 망원동 에이전트 시뮬레이션{sim_label}")

    # 날짜/필터 컨텍스트
    ctx_parts = []
    if selected_date:
        ctx_parts.append(f"{selected_date}")
    else:
        ctx_parts.append("전체 기간")
    if store_filter != "전체":
        ctx_parts.append(f"{store_filter}")
    st.caption(" · ".join(ctx_parts))

    # 메트릭 카드
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if store_filter != "전체" and not filtered_visits.empty:
            active_agents = filtered_visits['persona_id'].nunique()
        else:
            active_agents = len(agents)
        st.markdown(f'<div class="metric-card"><div class="metric-value">{active_agents}</div><div class="metric-label">에이전트</div></div>', unsafe_allow_html=True)

    with col2:
        total_visits = len(filtered_visits)
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_visits:,}</div><div class="metric-label">총 방문</div></div>', unsafe_allow_html=True)

    with col3:
        if not filtered_visits.empty:
            unique_stores = filtered_visits['visited_store'].nunique()
        else:
            unique_stores = 0
        st.markdown(f'<div class="metric-card"><div class="metric-value">{unique_stores}</div><div class="metric-label">방문 업체</div></div>', unsafe_allow_html=True)

    with col4:
        if not filtered_results.empty and len(filtered_results) > 0:
            conversion_rate = len(filtered_visits) / len(filtered_results) * 100
        else:
            conversion_rate = 0
        st.markdown(f'<div class="metric-card"><div class="metric-value">{conversion_rate:.1f}%</div><div class="metric-label">전환율</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ==================== 전체 보기: 방문 현황 지도 + 매장 평점 ====================
    if selected_agent == "전체":
        col_map, col_ratings = st.columns([2, 1])

        with col_map:
            st.markdown("### 방문 현황")
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
            ratings_path = (OUTPUT_DIR / sim_folder / "store_ratings.json") if sim_folder else (OUTPUT_DIR / "store_ratings.json")
            if ratings_path.exists():
                with open(ratings_path, 'r', encoding='utf-8') as f:
                    ratings_data = json.load(f)

                stats = ratings_data.get('statistics', {})
                st.markdown(f"**평점 보유 매장:** {stats.get('stores_with_agent_ratings', 0)}개 / {stats.get('total_stores', 0)}개")
                st.markdown(f"**총 평점 수:** {stats.get('total_agent_ratings', 0)}건")
                st.markdown(f"**평균 별점:** {stats.get('avg_star_rating', 0):.2f}점")
                st.markdown(f"**맛 태그:** {stats.get('total_taste_tags', 0)}건 · **가성비:** {stats.get('total_value_tags', 0)}건")
                st.markdown(f"**분위기 태그:** {stats.get('total_atmosphere_tags', 0)}건 · **서비스:** {stats.get('total_service_tags', 0)}건")

                st.markdown("---")
                st.markdown("#### 매장별 평점 TOP 10")
                stores_rated = ratings_data.get('stores', [])
                stores_rated_sorted = sorted(stores_rated, key=lambda x: x.get('agent_rating_count', 0), reverse=True)

                for i, store in enumerate(stores_rated_sorted[:10], 1):
                    with st.expander(f"{i}. {store['store_name']} ({store['agent_rating_count']}건)", expanded=(i <= 3)):
                        st.markdown(f"**카테고리:** {store.get('category', '-')}")
                        st.markdown(f"**평균가격:** {store.get('average_price', 0):,.0f}원")
                        st.markdown(f"**맛 태그:** {store.get('taste_count', 0)} / **가성비:** {store.get('value_count', 0)} / **분위기:** {store.get('atmosphere_count', 0)} / **서비스:** {store.get('service_count', 0)}")
                        st.markdown(f"**평균 별점:** {store.get('agent_avg_rating', 0):.2f}점")
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
                ctrl_cols = st.columns([1, 1, 1, 4])
                if ctrl_cols[0].button("⏮", key="anim_start", use_container_width=True, help="처음으로"):
                    st.session_state.current_hour = 6.0
                    st.session_state.anim_playing = False
                play_icon = "⏸" if st.session_state.anim_playing else "▶"
                if ctrl_cols[1].button(play_icon, key="anim_play", use_container_width=True, help="재생/정지"):
                    st.session_state.anim_playing = not st.session_state.anim_playing
                if ctrl_cols[2].button("⏭", key="anim_end", use_container_width=True, help="끝으로"):
                    st.session_state.current_hour = 24.0
                    st.session_state.anim_playing = False
                speed = ctrl_cols[3].slider("배속", 1, 60, 10, 1, key="anim_speed", label_visibility="collapsed")

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
                time_period = "AM" if hours < 12 else "PM"
                # 현재 타임슬롯 판별
                if current_hour < 10:
                    slot_label = "MORNING"
                elif current_hour < 15:
                    slot_label = "LUNCH"
                elif current_hour < 20:
                    slot_label = "DINNER"
                else:
                    slot_label = "LATE NIGHT"
                st.markdown(
                    f'<div class="time-display">'
                    f'{hours:02d}<span style="opacity:0.4">:</span>{minutes:02d}<span style="opacity:0.4">:</span>{seconds:02d}'
                    f'<span class="time-period">{time_period}</span>'
                    f'<div style="font-size:0.65rem;color:#94a3b8;letter-spacing:0.15em;margin-top:2px;font-weight:600">{slot_label}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # 에이전트 상태 계산 (home_location [0,0] → entry_point fallback)
                agent_home = agent_info.get('home_location')
                if not agent_home or agent_home == [0.0, 0.0] or agent_home == [0, 0]:
                    ep = agent_info.get('entry_point')
                    if ep and ep != [0.0, 0.0]:
                        agent_home = ep
                    else:
                        agent_home = [LANDMARKS["집"]["lat"], LANDMARKS["집"]["lng"]]
                agent_lat, agent_lng, status, current_activity, route_coords, step5_action = get_agent_state(
                    results_df[results_df['persona_id'] == selected_agent],
                    stores_dict, G_anim, cafe_stores, anim_selected_date, current_hour,
                    agent_segment, selected_agent, home_location=agent_home
                )

                # 지도 + 상태 표시
                map_col, status_col = st.columns([2, 1])

                with map_col:
                    if agent_lat and agent_lng:
                        # 상태별 이모지/색상 결정
                        status_info = {
                            "eating":  {"emoji": "🍽️", "label": "식사 중", "color": "red"},
                            "cafe":    {"emoji": "☕",  "label": "카페",   "color": "purple"},
                            "wander":  {"emoji": "🚶",  "label": "배회",   "color": "orange"},
                            "park":    {"emoji": "🌳",  "label": "공원",   "color": "green"},
                            "market":  {"emoji": "🛒",  "label": "시장",   "color": "purple"},
                            "home":    {"emoji": "🏠",  "label": "집",     "color": "blue"},
                            "work":    {"emoji": "💼",  "label": "출근",   "color": "darkblue"},
                            "idle":    {"emoji": "🏠",  "label": "대기",   "color": "gray"},
                        }
                        matched = {"emoji": "🚶", "label": "이동 중", "color": "blue"}
                        for key, info in status_info.items():
                            if key in status:
                                matched = info
                                break
                        if "moving" in status:
                            matched = {"emoji": "🚶", "label": "이동 중", "color": "blue"}

                        # 바운딩 박스 계산 → 줌/중심
                        all_lats = [agent_lat]
                        all_lngs = [agent_lng]
                        if route_coords:
                            for c in route_coords:
                                all_lats.append(c[0])
                                all_lngs.append(c[1])

                        # 방문 매장
                        agent_visits_anim = visits_df[
                            (visits_df['persona_id'] == selected_agent) &
                            (visits_df['timestamp'].dt.date == anim_selected_date)
                        ] if not visits_df.empty else pd.DataFrame()
                        visited_stores_list = []
                        if not agent_visits_anim.empty:
                            for _, row in agent_visits_anim.iterrows():
                                visit_hour = TIMESLOT_HOURS.get(row['time_slot'], 0)
                                if visit_hour + 0.5 <= current_hour:
                                    s_info = stores_dict.get(row['visited_store'], {})
                                    if s_info:
                                        visited_stores_list.append({
                                            "lat": s_info['lat'], "lng": s_info['lng'],
                                            "name": row['visited_store'],
                                            "slot": row['time_slot'],
                                            "category": row.get('visited_category', s_info.get('category', '')),
                                        })
                                        all_lats.append(s_info['lat'])
                                        all_lngs.append(s_info['lng'])

                        min_lat, max_lat = min(all_lats), max(all_lats)
                        min_lng, max_lng = min(all_lngs), max(all_lngs)
                        center_lat = (min_lat + max_lat) / 2
                        center_lng = (min_lng + max_lng) / 2
                        spread = max(max_lat - min_lat, max_lng - min_lng)
                        if not route_coords and len(visited_stores_list) == 0:
                            zoom = 15
                        elif spread < 0.001:
                            zoom = 16
                        elif spread < 0.005:
                            zoom = 15
                        elif spread < 0.01:
                            zoom = 14
                        else:
                            zoom = 13

                        m = folium.Map(
                            location=[center_lat, center_lng], zoom_start=zoom,
                            tiles='cartodbpositron',
                            control_scale=True,
                        )

                        # 경로 색상 매핑
                        route_colors = {
                            "eating": "#ef4444", "cafe": "#f59e0b", "wander": "#f97316",
                            "park": "#10b981", "market": "#8b5cf6", "home": "#6b7280",
                            "work": "#0ea5e9", "moving": "#3b82f6", "idle": "#94a3b8",
                        }
                        route_color = "#3b82f6"
                        for key, col in route_colors.items():
                            if key in status:
                                route_color = col
                                break

                        # 랜드마크 (미니멀 핀)
                        lm_icons = {"한강공원": "🌊", "망원시장": "🏪", "회사": "🏢"}
                        for k, v in LANDMARKS.items():
                            icon = lm_icons.get(k)
                            if not icon:
                                continue
                            folium.Marker(
                                [v["lat"], v["lng"]],
                                icon=folium.DivIcon(
                                    html=f'<div style="display:flex;flex-direction:column;align-items:center;gap:1px">'
                                         f'<span style="font-size:14px;filter:grayscale(0.3)">{icon}</span>'
                                         f'<span style="font-size:8px;color:#94a3b8;font-weight:500;font-family:Inter,sans-serif;white-space:nowrap">{v["name"]}</span></div>',
                                    icon_size=(70, 32), icon_anchor=(35, 16)),
                            ).add_to(m)

                        # 이동 경로 (그라데이션 느낌)
                        if route_coords and len(route_coords) > 1:
                            # 전체 예정 경로 (점선)
                            folium.PolyLine(route_coords, color=route_color, weight=2, opacity=0.2, dash_array='8 6').add_to(m)

                            # 이동 완료 구간 (실선)
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
                                folium.PolyLine(traveled, color=route_color, weight=3.5, opacity=0.85).add_to(m)

                        # 방문 매장 마커 (깔끔한 핀 + 라벨)
                        store_cat_icons = {"한식": "🍚", "양식": "🍝", "일식": "🍣", "중식": "🥟",
                                           "커피": "☕", "호프": "🍺", "치킨": "🍗", "제과": "🥐", "패스트": "🍔"}
                        for vs in visited_stores_list:
                            s_icon = "📍"
                            cat = vs.get("category", "")
                            for key, ico in store_cat_icons.items():
                                if key in cat:
                                    s_icon = ico
                                    break
                            folium.Marker(
                                [vs["lat"], vs["lng"]],
                                icon=folium.DivIcon(
                                    html=f'<div style="display:flex;flex-direction:column;align-items:center;gap:0">'
                                         f'<div style="width:28px;height:28px;border-radius:50%;background:white;border:2px solid #ef4444;'
                                         f'display:flex;align-items:center;justify-content:center;font-size:14px;'
                                         f'box-shadow:0 2px 8px rgba(239,68,68,0.3)">{s_icon}</div>'
                                         f'<span style="font-size:9px;color:#1e293b;font-weight:600;font-family:Inter,sans-serif;'
                                         f'white-space:nowrap;background:rgba(255,255,255,0.9);padding:1px 5px;border-radius:4px;'
                                         f'margin-top:2px;box-shadow:0 1px 2px rgba(0,0,0,0.08)">{vs["name"]}</span></div>',
                                    icon_size=(100, 46), icon_anchor=(50, 23)),
                                tooltip=f'{vs["name"]} ({vs["slot"]})',
                            ).add_to(m)

                        # 에이전트 마커 (원형 아바타 + 상태 이모지)
                        agent_color = route_color
                        folium.Marker(
                            [agent_lat, agent_lng],
                            icon=folium.DivIcon(
                                html=f'<div style="position:relative;display:flex;align-items:center;justify-content:center">'
                                     f'<div style="width:36px;height:36px;border-radius:50%;background:{agent_color};'
                                     f'display:flex;align-items:center;justify-content:center;font-size:18px;'
                                     f'box-shadow:0 3px 12px {agent_color}55,0 0 0 3px white;'
                                     f'animation:pulse 2s infinite">{matched["emoji"]}</div></div>'
                                     f'<style>@keyframes pulse{{0%,100%{{box-shadow:0 3px 12px {agent_color}55,0 0 0 3px white}}'
                                     f'50%{{box-shadow:0 3px 16px {agent_color}88,0 0 0 5px white}}}}</style>',
                                icon_size=(42, 42), icon_anchor=(21, 21)),
                            tooltip=f'{matched["emoji"]} {matched["label"]}',
                        ).add_to(m)

                        st_folium(m, width=None, height=480, key="anim_map", returned_objects=[])
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

                    # 오늘 스케줄 (타임라인 스타일)
                    st.markdown("### 오늘의 타임라인")
                    day_data_anim = results_df[
                        (results_df['persona_id'] == selected_agent) &
                        (results_df['timestamp'].dt.date == anim_selected_date)
                    ].sort_values('timestamp')

                    slot_icons = {"아침": "🌅", "점심": "☀️", "저녁": "🌆", "야식": "🌙"}
                    for _, row in day_data_anim.iterrows():
                        slot = row['time_slot']
                        slot_hour = TIMESLOT_HOURS.get(slot, 0)
                        is_past = slot_hour + 2 <= current_hour
                        is_current = slot_hour <= current_hour < slot_hour + 2
                        s_icon = slot_icons.get(slot, "⏰")

                        if is_current:
                            css_cls = "schedule-current"
                        elif is_past:
                            css_cls = "schedule-past"
                        else:
                            css_cls = "schedule-future"

                        if row['decision'] == 'visit':
                            indicator = "●" if is_current else ("✓" if is_past else "○")
                            label = row['visited_store']
                        else:
                            indicator = "●" if is_current else ("–" if is_past else "○")
                            label = "외출 안 함"

                        st.markdown(
                            f'<div class="schedule-item {css_cls}">'
                            f'<span style="font-size:0.75rem;width:16px;text-align:center">{indicator}</span>'
                            f'<span style="font-size:0.75rem;color:#94a3b8;min-width:32px">{slot_hour}:00</span>'
                            f'{s_icon} <span style="font-weight:500">{label}</span></div>',
                            unsafe_allow_html=True
                        )

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
                            rating = row.get('rating', '-')
                            reason = row.get('reason', '')

                            with st.expander(f"🍽️ {time_slot} ({timestamp}) → {store_name}", expanded=False):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**매장:** {store_name}")
                                    st.markdown(f"**카테고리:** {category}")
                                    st.markdown(f"**평점:** {rating}점")
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
                col_stat1, col_stat2, col_stat3 = st.columns(3)

                with col_stat1:
                    st.metric("총 방문 횟수", f"{len(agent_visits)}회")
                with col_stat2:
                    unique_stores = agent_visits['visited_store'].nunique()
                    st.metric("방문 매장 수", f"{unique_stores}개")
                with col_stat3:
                    avg_rating = agent_visits['rating'].mean()
                    st.metric("평균 평점", f"{avg_rating:.1f}점")

                # 방문 매장 목록
                st.markdown("#### 🍽️ 방문한 매장")
                store_summary = agent_visits.groupby('visited_store').agg({
                    'rating': 'mean',
                    'timestamp': 'count'
                }).reset_index()
                store_summary.columns = ['매장', '평균 평점', '방문횟수']
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
        if not filtered_visits.empty and 'rating' in filtered_visits.columns:
            rating_labels = {1: '매우별로(1)', 2: '별로(2)', 3: '보통(3)', 4: '좋음(4)', 5: '매우좋음(5)'}
            rating_colors = ['#ff6b6b', '#ffa06b', '#ffd93d', '#a8e063', '#6bcb77']

            rating_dist = filtered_visits['rating'].value_counts().reset_index()
            rating_dist.columns = ['rating', 'count']
            rating_dist['rating_label'] = rating_dist['rating'].map(rating_labels)
            rating_dist = rating_dist.dropna(subset=['rating_label'])

            fig_rating = px.pie(
                rating_dist,
                values='count',
                names='rating_label',
                title="평점 분포",
                color_discrete_sequence=rating_colors
            )
            st.plotly_chart(fig_rating, use_container_width=True)
        else:
            st.info("평점 데이터 없음")

    # 방문 로그 테이블
    st.markdown("---")
    st.markdown("### 📋 방문 로그")

    if not filtered_visits.empty:
        # 사용 가능한 컬럼 확인
        available_cols = filtered_visits.columns.tolist()
        base_cols = ['timestamp', 'persona_id', 'generation', 'segment',
                    'visited_store', 'visited_category', 'rating']
        base_names = ['시간', '에이전트', '세대', '세그먼트',
                     '방문매장', '카테고리', '평점']

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

                st.markdown(f"**평점:** {selected_row['평점']}점")

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
