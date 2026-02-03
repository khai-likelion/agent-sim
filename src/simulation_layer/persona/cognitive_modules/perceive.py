"""
Perceive module: Generates observation data for an agent at a location.
Extracts the observation prompt logic from the original Environment class.

Current: rule-based observation prompt.
Future: LLM-enhanced perception with memory context.
"""

import pandas as pd

from .base import CognitiveModule
from src.data_layer.spatial_index import Environment


class PerceiveModule(CognitiveModule):
    """
    Generates the agent's perception of surrounding stores.
    """

    def __init__(self, environment: Environment):
        self.environment = environment

    def process(
        self, current_lat: float, current_lng: float, k_ring: int | None = None
    ) -> dict:
        """
        Returns:
            dict with keys: 'visible_stores' (DataFrame), 'observation_prompt' (str)
        """
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
        Generate a natural language observation prompt for the agent.
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
