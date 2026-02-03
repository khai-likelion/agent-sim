"""
Population statistics data manager.
Parses Mangwon-dong demographic data for agent generation and simulation weighting.
"""

import json

from config import get_settings


class PopulationStatistics:
    """
    Manages population demographic data (age, gender, time, weekday distributions).
    """

    def __init__(
        self,
        json_path: str | None = None,
        area_code: str | None = None,
        quarter: str | None = None,
    ):
        settings = get_settings()
        json_path = json_path or str(settings.paths.population_json)
        area_code = area_code or settings.area.area_code
        quarter = quarter or settings.area.quarter

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        target_id = f"{area_code}_{quarter}_population"
        self.data = next((item for item in data if item["id"] == target_id), None)

        if not self.data:
            raise ValueError(f"Data not found: {target_id}")

        self.area_code = area_code
        self.quarter = quarter
        self._parse_statistics()

    def _parse_statistics(self):
        """Parse demographic breakdowns from raw data."""
        raw_data = self.data["raw_data_context"]["dimension_breakdowns"]

        self.age_distribution = {
            item["dimensions"]["age_group"]: item["share"]
            for item in raw_data["floating_by_age"]
        }

        self.time_distribution = {
            item["dimensions"]["time_slot"]: item["share"]
            for item in raw_data["floating_by_time"]
        }

        self.weekday_distribution = {
            item["dimensions"]["weekday"]: item["share"]
            for item in raw_data["floating_by_weekday"]
        }

        gender_text = self.data["population_analysis"]["gender_structure"]
        male_ratio = (
            float(gender_text.split("남성")[1].split("%")[0].split("(")[1]) / 100
        )
        female_ratio = (
            float(gender_text.split("여성")[1].split("%")[0].split("(")[1]) / 100
        )
        self.gender_distribution = {"남성": male_ratio, "여성": female_ratio}

    def get_random_age_group(self) -> str:
        """Return a random age group weighted by population distribution."""
        import random

        age_groups = list(self.age_distribution.keys())
        weights = list(self.age_distribution.values())
        return random.choices(age_groups, weights=weights)[0]

    def get_random_gender(self) -> str:
        """Return a random gender weighted by population distribution."""
        import random

        genders = list(self.gender_distribution.keys())
        weights = list(self.gender_distribution.values())
        return random.choices(genders, weights=weights)[0]

    def get_time_weight(self, time_slot: str) -> float:
        """Return activity weight for a time slot."""
        return self.time_distribution.get(time_slot, 0.0)

    def get_weekday_weight(self, weekday: str) -> float:
        """Return activity weight for a weekday."""
        return self.weekday_distribution.get(weekday, 0.0)
