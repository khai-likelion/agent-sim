"""
Perceive module: Generates observation data for an agent at a location.
Extracts the observation prompt logic from the original Environment class.

Supports two modes:
1. StreetNetwork mode: Uses graph-based neighborhood for visible stores
2. Legacy H3 mode: Uses H3 hexagonal grid

Current: rule-based observation prompt.
Future: LLM-enhanced perception with memory context.
"""

from typing import Union, List, Dict, Any, Optional

import pandas as pd

from .base import CognitiveModule
from src.data_layer.spatial_index import Environment
from src.data_layer.street_network import StreetNetwork, AgentLocation


class PerceiveModule(CognitiveModule):
    """
    Generates the agent's perception of surrounding stores.
    Supports both StreetNetwork and legacy H3 Environment.
    """

    def __init__(self, environment: Union[Environment, StreetNetwork]):
        self.use_street_network = isinstance(environment, StreetNetwork)
        if self.use_street_network:
            self.street_network = environment
            self.environment = None
        else:
            self.street_network = None
            self.environment = environment

    def process(
        self,
        current_lat: float,
        current_lng: float,
        k_ring: int | None = None,
        agent_location: Optional[AgentLocation] = None,
    ) -> dict:
        """
        Returns:
            dict with keys: 'visible_stores' (list or DataFrame), 'observation_prompt' (str)
        """
        if self.use_street_network and agent_location is not None:
            # Street network mode: use agent's location on graph
            visible_stores = self.street_network.get_nearby_stores(
                agent_location, k_ring or 1
            )
            prompt = self.generate_observation_prompt_from_list(
                current_lat, current_lng, visible_stores
            )
        else:
            # Legacy H3 mode
            visible_stores = self.environment.get_visible_stores(
                current_lat, current_lng, k_ring
            )
            prompt = self.generate_observation_prompt(
                current_lat, current_lng, visible_stores
            )

        return {
            "visible_stores": visible_stores,
            "observation_prompt": prompt,
        }

    def generate_observation_prompt(
        self,
        current_lat: float,
        current_lng: float,
        visible_stores: pd.DataFrame,
    ) -> str:
        """
        Generate a natural language observation prompt for the agent (DataFrame input).
        Future: load from prompt_templates/observation.txt and render with variables.
        """
        if visible_stores.empty:
            return "현재 위치 주변에 매장이 없습니다."

        prompt = f"=== 현재 위치 주변 매장 정보 ===\n"
        prompt += f"좌표: ({current_lat:.6f}, {current_lng:.6f})\n"
        prompt += f"발견된 매장 수: {len(visible_stores)}개\n\n"

        for idx, store in visible_stores.iterrows():
            prompt += f"[{idx+1}] {store['장소명']}\n"
            prompt += f"    - 카테고리: {store['카테고리']}\n"
            prompt += f"    - 주소: {store['주소']}\n"
            prompt += f"    - 상권: {store['상권']}\n"
            prompt += f"    - 업종: {store['업종']}\n"
            if pd.notna(store["전화번호"]) and store["전화번호"] != "없음":
                prompt += f"    - 전화: {store['전화번호']}\n"
            prompt += "\n"

        return prompt

    def generate_observation_prompt_from_list(
        self,
        current_lat: float,
        current_lng: float,
        visible_stores: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a natural language observation prompt for the agent (list input).
        Used with StreetNetwork mode where stores are returned as list of dicts.
        """
        if not visible_stores:
            return "현재 위치 주변에 매장이 없습니다."

        prompt = f"=== 현재 위치 주변 매장 정보 ===\n"
        prompt += f"좌표: ({current_lat:.6f}, {current_lng:.6f})\n"
        prompt += f"발견된 매장 수: {len(visible_stores)}개\n\n"

        for idx, store in enumerate(visible_stores):
            store_name = store.get("장소명", store.get("store_name", "Unknown"))
            category = store.get("카테고리", store.get("category", "Unknown"))
            address = store.get("주소", store.get("address", "Unknown"))
            area = store.get("상권", store.get("commercial_area", "Unknown"))
            business_type = store.get("업종", store.get("business_type", "Unknown"))
            phone = store.get("전화번호", store.get("phone", "없음"))

            prompt += f"[{idx+1}] {store_name}\n"
            prompt += f"    - 카테고리: {category}\n"
            prompt += f"    - 주소: {address}\n"
            prompt += f"    - 상권: {area}\n"
            prompt += f"    - 업종: {business_type}\n"
            if phone and phone != "없음" and pd.notna(phone):
                prompt += f"    - 전화: {phone}\n"
            prompt += "\n"

        return prompt
