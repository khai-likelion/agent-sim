"""
Agent state management for realistic behavioral simulation.
Tracks hunger, fatigue, mood, and recent activities.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import random


@dataclass
class MealRecord:
    """Record of a meal/visit."""
    timestamp: str
    time_slot: str
    meal_type: str  # breakfast, lunch, dinner, snack, drink
    store_name: Optional[str]
    category: Optional[str]
    satisfaction: float  # 0~1


@dataclass
class AgentState:
    """
    Dynamic state of an agent that changes throughout simulation.
    Updated each time step based on actions and time passage.
    """
    agent_id: int

    # Physical state (0~100)
    hunger: float = 30.0  # 0=full, 100=starving
    fatigue: float = 20.0  # 0=energetic, 100=exhausted
    social_need: float = 30.0  # 0=satisfied, 100=lonely

    # Emotional state
    mood: str = "neutral"  # positive, neutral, negative
    stress_level: float = 30.0  # 0~100

    # Recent history
    last_meal_time: Optional[str] = None
    last_meal_type: Optional[str] = None
    meals_today: List[MealRecord] = field(default_factory=list)
    recent_visits: List[str] = field(default_factory=list)  # Last 5 store names
    recent_categories: List[str] = field(default_factory=list)  # Last 5 categories

    # Daily patterns
    woke_up_at: str = "07:00"
    commute_start: str = "08:30"
    commute_end: str = "18:30"
    sleep_time: str = "23:00"

    # Preferences for today (can vary by day)
    budget_today: str = "normal"  # tight, normal, generous
    time_pressure: str = "normal"  # rushed, normal, relaxed
    companion: Optional[str] = None  # alone, friend, family, date, colleague

    def update_for_timeslot(self, time_slot: str, weekday: str):
        """Update state based on time passage."""
        # Hunger increases over time
        hunger_rate = {
            "아침": 15,  # Wake up somewhat hungry
            "점심": 25,  # Most hungry at lunch
            "저녁": 20,  # Hungry for dinner
            "야간": 10,  # Late night cravings
        }
        self.hunger = min(100, self.hunger + hunger_rate.get(time_slot, 15))

        # Fatigue varies by time
        fatigue_map = {
            "아침": -10,  # Fresh in morning
            "점심": 5,   # Slight post-lunch dip
            "저녁": 15,  # Tired after work
            "야간": 25,  # Very tired at night
        }
        self.fatigue = min(100, max(0, self.fatigue + fatigue_map.get(time_slot, 0)))

        # Social need varies by weekday
        if weekday in ["토", "일"]:
            self.social_need = min(100, self.social_need + 15)
        else:
            self.social_need = min(100, self.social_need + 5)

        # Mood can shift randomly
        if random.random() < 0.2:
            self.mood = random.choice(["positive", "neutral", "negative"])

    def record_meal(self, time_slot: str, store_name: str, category: str, satisfaction: float = 0.7):
        """Record a meal and update state."""
        meal_type_map = {
            "아침": "breakfast",
            "점심": "lunch",
            "저녁": "dinner",
            "야간": "snack",
        }

        record = MealRecord(
            timestamp=datetime.now().isoformat(),
            time_slot=time_slot,
            meal_type=meal_type_map.get(time_slot, "meal"),
            store_name=store_name,
            category=category,
            satisfaction=satisfaction,
        )
        self.meals_today.append(record)

        # Update state after eating
        self.hunger = max(0, self.hunger - 60)  # Eating reduces hunger significantly
        self.last_meal_time = time_slot
        self.last_meal_type = record.meal_type

        # Update recent history
        if store_name:
            self.recent_visits.append(store_name)
            if len(self.recent_visits) > 5:
                self.recent_visits.pop(0)
        if category:
            self.recent_categories.append(category)
            if len(self.recent_categories) > 5:
                self.recent_categories.pop(0)

        # Social satisfaction if eating out
        if store_name:
            self.social_need = max(0, self.social_need - 20)

    def reset_for_new_day(self):
        """Reset daily state for a new day."""
        self.meals_today = []
        self.hunger = random.uniform(20, 40)
        self.fatigue = random.uniform(10, 30)
        self.budget_today = random.choice(["tight", "normal", "normal", "generous"])
        self.time_pressure = random.choice(["rushed", "normal", "normal", "relaxed"])
        self.companion = None

        # Weekend vs weekday patterns
        self.stress_level = random.uniform(20, 50)

    def get_hunger_level(self) -> str:
        """Get hunger as descriptive text."""
        if self.hunger < 20:
            return "배부름"
        elif self.hunger < 40:
            return "적당함"
        elif self.hunger < 60:
            return "약간 배고픔"
        elif self.hunger < 80:
            return "배고픔"
        else:
            return "매우 배고픔"

    def get_fatigue_level(self) -> str:
        """Get fatigue as descriptive text."""
        if self.fatigue < 20:
            return "활기참"
        elif self.fatigue < 40:
            return "보통"
        elif self.fatigue < 60:
            return "약간 피곤"
        elif self.fatigue < 80:
            return "피곤함"
        else:
            return "매우 피곤"

    def get_context_summary(self) -> str:
        """Get state summary for prompts."""
        meals_str = ", ".join([f"{m.meal_type}({m.store_name or '집밥'})" for m in self.meals_today]) or "아직 없음"
        recent_str = ", ".join(self.recent_visits[-3:]) if self.recent_visits else "없음"

        return f"""- 배고픔: {self.get_hunger_level()} ({self.hunger:.0f}/100)
- 피로도: {self.get_fatigue_level()} ({self.fatigue:.0f}/100)
- 기분: {self.mood}
- 오늘 식사: {meals_str}
- 최근 방문: {recent_str}
- 오늘 예산: {self.budget_today}
- 시간 여유: {self.time_pressure}
- 동행: {self.companion or '혼자'}"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "agent_id": self.agent_id,
            "hunger": self.hunger,
            "fatigue": self.fatigue,
            "social_need": self.social_need,
            "mood": self.mood,
            "stress_level": self.stress_level,
            "last_meal_time": self.last_meal_time,
            "meals_today": [
                {
                    "time_slot": m.time_slot,
                    "meal_type": m.meal_type,
                    "store_name": m.store_name,
                    "category": m.category,
                }
                for m in self.meals_today
            ],
            "recent_visits": self.recent_visits,
            "budget_today": self.budget_today,
            "companion": self.companion,
        }
