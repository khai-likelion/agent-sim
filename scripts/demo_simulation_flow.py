"""
Simulation Demo Script - showcase key functions in action.
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import required modules
from config import get_settings
from src.simulation_layer.persona.generative_agent import GenerativeAgentFactory
from src.data_layer.global_store import get_global_store, GlobalStore
from src.simulation_layer.persona.cognitive_modules.action_algorithm import ActionAlgorithm
from src.data_layer.street_network import StreetNetwork, StreetNetworkConfig

def print_box(text, color="\033[94m"):
    border = "=" * 60
    print(f"\n{color}{border}\n{text}\n{border}\033[0m")

def main():
    settings = get_settings()
    
    # 1. Scene 1: Agent Generation
    print_box("SCENE 1: AGENT GENERATION (페르소나 생성)")
    factory = GenerativeAgentFactory()
    agent = factory.generate_unique_agents(max_count=1)[0]
    print(agent.get_persona_summary())
    time.sleep(1)

    # 2. Scene 2: Environment Exploration
    print_box("SCENE 2: ENVIRONMENT EXPLORATION (주변 탐색)")
    global_store = get_global_store()
    json_dir = settings.paths.data_dir / "split_by_store_id"
    if json_dir.exists():
        global_store.load_from_json_files(json_dir)
        print(f"매장 데이터 로드 완료: {len(global_store.stores)}개 매장")
    
    # Use first store's location for demonstration
    if global_store.stores:
        sample_store = list(global_store.stores.values())[0]
        current_lat, current_lng = sample_store.coordinates
        print(f"에이전트 현재 위치 (샘플 매장 '{sample_store.store_name}' 근처): ({current_lat}, {current_lng})")
    else:
        current_lat, current_lng = 37.556, 126.905 
        print(f"에이전트 현재 위치: ({current_lat}, {current_lng})")
    
    nearby_stores = global_store.get_stores_in_radius(current_lat, current_lng, radius_km=1.0)
    print(f"반경 1km 내 발견된 매장: {len(nearby_stores)}개")
    for s in nearby_stores[:3]:
        print(f"  - {s.store_name} ({s.category}): {s.average_price:,}원")
    time.sleep(1)

    # 3. Scene 3: Action Algorithm Process
    print_box("SCENE 3: ACTION ALGORITHM (LLM 의사결정 프로세스)")
    algorithm = ActionAlgorithm(rate_limit_delay=0.5)
    
    print("\n[Step 1] 망원동 내 식사 여부 결정 중...")
    try:
        step1 = algorithm.step1_eat_in_mangwon(agent, "점심", "토")
        print(f"결정: {'외출함' if step1['eat_in_mangwon'] else '집에 있음'}")
        print(f"이유: {step1['reason']}")
        
        if step1['eat_in_mangwon']:
            print("\n[Step 2] 선호 업종 선택 중...")
            step2 = algorithm.step2_category_selection(agent, "식당", "점심")
            category = step2['category']
            print(f"선택 업종: {category}")
            print(f"이유: {step2['reason']}")
            
            print("\n[Step 3] 특정 매장 선택 중...")
            step3 = algorithm.step3_store_selection(agent, category, nearby_stores, "점심")
            store_name = step3['store_name']
            print(f"선택 매장: {store_name}")
            print(f"이유: {step3['reason']}")
            
            if store_name:
                print("\n[Step 4] 방문 후 평가 생성 중...")
                store_obj = global_store.get_by_name(store_name)
                step4 = algorithm.step4_evaluate_and_feedback(agent, store_obj, "2026-02-14 12:00:00")
                print(f"리뷰: {step4['comment']}")
                print(f"평점: 맛({step4['taste_rating']}) 가성비({step4['value_rating']}) 분위기({step4['atmosphere_rating']})")
    except Exception as e:
        print(f"LLM 호출 중 오류 (API 키 확인 필요): {e}")

    # 4. Scene 4: Movement on Network
    print_box("SCENE 4: MOVEMENT (도로 네트워크 기반 이동)")
    print("도로 데이터(OSM) 로드 중... (잠시만 기다려주세요)")
    config = StreetNetworkConfig(
        center_lat=current_lat,
        center_lng=current_lng,
        radius_m=500.0,
        network_type="walk"
    )
    network = StreetNetwork(config)
    network.load_graph()
    
    start_node = network.get_random_start_node()
    node_data = network.graph_proj.nodes[start_node]
    agent_lat, agent_lng = network.xy_to_latlng(node_data["x"], node_data["y"])
    location = network.initialize_agent_location(agent_lat, agent_lng)
    
    print(f"출발지 노드: {start_node} ({agent_lat:.4f}, {agent_lng:.4f})")
    
    # Move 100 meters
    print("에이전트가 100m 이동 중...")
    new_location = network.move_agent(location, distance_m=100.0)
    print(f"이동 후 위치: ({new_location.lat:.4f}, {new_location.lng:.4f})")
    print(f"현재 노드: {new_location.current_node}, 다음 노드: {new_location.next_node}")

    print_box("DEMO COMPLETED", color="\033[92m")

if __name__ == "__main__":
    main()
