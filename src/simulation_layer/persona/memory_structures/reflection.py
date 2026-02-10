"""
Reflection: Periodic high-level insight generation from event memories.
Analyzes visit patterns and auto-updates preferences.

Supports both rule-based and LLM-driven reflection.
Rule-based: Pattern analysis (category frequency, satisfaction trends)
LLM-based: Uses reflection.txt template for richer insights.
"""

from typing import List, Optional, Dict
from datetime import datetime

from .base import MemoryStructure
from .event_memory import EventMemory, MemoryEntry


class Reflection(MemoryStructure):
    """
    Generates and stores high-level reflections from event memories.
    Triggers every REFLECTION_THRESHOLD visits, producing preference adjustments.
    """

    REFLECTION_THRESHOLD = 5  # Trigger after every N visits

    def __init__(self, event_memory: EventMemory, use_llm: bool = False):
        self.event_memory = event_memory
        self.reflections: List[MemoryEntry] = []
        self.use_llm = use_llm
        self._visits_since_reflection = 0
        self._llm_client = None
        self._prompt_cache: Optional[str] = None
        # Preference adjustments from reflections: category -> weight delta
        self.preference_adjustments: Dict[str, float] = {}

    def add(self, content, timestamp: datetime) -> None:
        entry = MemoryEntry(
            timestamp=timestamp,
            event_type="reflection",
            content=str(content),
            importance=0.8,
        )
        self.reflections.append(entry)
        # Also add to event memory so it appears in retrieval
        self.event_memory.add(entry, timestamp)

    def retrieve(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        return sorted(
            self.reflections, key=lambda m: m.timestamp, reverse=True
        )[:top_k]

    def record_visit(self) -> None:
        """Called after each visit to track reflection trigger."""
        self._visits_since_reflection += 1

    def should_reflect(self) -> bool:
        """Check if reflection should be triggered."""
        return self._visits_since_reflection >= self.REFLECTION_THRESHOLD

    def generate_reflection(
        self,
        agent,
        current_time: datetime,
    ) -> Optional[str]:
        """Generate reflection from recent memories.

        Returns reflection text if generated, None otherwise.
        """
        recent = self.event_memory.get_visit_history()
        if len(recent) < 3:
            return None

        self._visits_since_reflection = 0

        if self.use_llm:
            result = self._llm_reflection(agent, recent, current_time)
            if result:
                return result

        return self._rule_based_reflection(recent, current_time)

    def _rule_based_reflection(
        self, memories: List[MemoryEntry], current_time: datetime
    ) -> str:
        """Analyze patterns without LLM call."""
        cat_counts: Dict[str, int] = {}
        satisfaction_by_cat: Dict[str, List[float]] = {}
        store_counts: Dict[str, int] = {}

        for m in memories:
            if m.category:
                cat_counts[m.category] = cat_counts.get(m.category, 0) + 1
                if m.satisfaction is not None:
                    satisfaction_by_cat.setdefault(m.category, []).append(m.satisfaction)
            if m.store_name:
                store_counts[m.store_name] = store_counts.get(m.store_name, 0) + 1

        insights = []

        # Pattern: category repetition with satisfaction
        for cat, count in cat_counts.items():
            if count >= 3:
                sats = satisfaction_by_cat.get(cat, [0.5])
                avg_sat = sum(sats) / len(sats)
                if avg_sat >= 0.7:
                    insights.append(f"{cat}을(를) 자주 먹는데 만족스럽다. 단골이 되고 있다.")
                    self.preference_adjustments[cat] = self.preference_adjustments.get(cat, 0) + 0.1
                elif avg_sat < 0.4:
                    insights.append(f"{cat}을(를) 자주 먹었지만 만족도가 낮다. 다른 걸 시도해볼까.")
                    self.preference_adjustments[cat] = self.preference_adjustments.get(cat, 0) - 0.1
                else:
                    insights.append(f"{cat}을(를) {count}번 방문했다. 보통 수준의 경험이었다.")

        # Pattern: store loyalty
        for store, count in store_counts.items():
            if count >= 3:
                insights.append(f"{store}에 자주 간다. 단골 매장이다.")

        # Pattern: variety seeking
        unique_cats = len(cat_counts)
        total_visits = sum(cat_counts.values())
        if total_visits >= 5 and unique_cats >= 4:
            insights.append("다양한 종류의 음식을 골고루 먹고 있다.")

        reflection_text = " | ".join(insights[:3]) if insights else "특별한 패턴 없음"
        self.add(reflection_text, current_time)
        return reflection_text

    def _llm_reflection(
        self, agent, memories: List[MemoryEntry], current_time: datetime
    ) -> Optional[str]:
        """Use LLM to generate reflection insights."""
        try:
            from src.ai_layer.llm_client import create_llm_client

            if self._llm_client is None:
                self._llm_client = create_llm_client()

            if self._prompt_cache is None:
                from config import get_settings
                settings = get_settings()
                template_path = settings.paths.prompt_templates_dir / "reflection.txt"
                self._prompt_cache = template_path.read_text(encoding='utf-8')

            recent_text = "\n".join([f"- {m.content}" for m in memories[-15:]])

            prompt = self._prompt_cache.format(
                agent_name=agent.name,
                age=agent.age,
                gender=agent.gender,
                occupation=agent.occupation,
                recent_memories=recent_text,
            )

            response = self._llm_client.generate_sync(prompt)
            self.add(response.strip(), current_time)
            self._extract_adjustments_from_text(response, memories)
            return response.strip()

        except Exception as e:
            print(f"  Reflection LLM error: {e}")
            return None

    def _extract_adjustments_from_text(self, text: str, memories: List[MemoryEntry]) -> None:
        """Extract preference adjustments from reflection text heuristically."""
        cat_counts: Dict[str, int] = {}
        for m in memories:
            if m.category:
                cat_counts[m.category] = cat_counts.get(m.category, 0) + 1

        positive_words = ["좋", "만족", "맛있", "추천", "단골", "자주"]
        negative_words = ["별로", "불만", "실망", "안 가", "지겨", "질리"]

        for cat in cat_counts:
            if cat in text:
                if any(pw in text for pw in positive_words):
                    self.preference_adjustments[cat] = self.preference_adjustments.get(cat, 0) + 0.1
                elif any(nw in text for nw in negative_words):
                    self.preference_adjustments[cat] = self.preference_adjustments.get(cat, 0) - 0.1

    def get_adjustment(self, category: str) -> float:
        """Get preference adjustment for a category."""
        return self.preference_adjustments.get(category, 0.0)

    def to_dict(self) -> dict:
        """Serialize for JSON save."""
        return {
            "reflections": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "content": r.content,
                    "importance": r.importance,
                }
                for r in self.reflections
            ],
            "preference_adjustments": self.preference_adjustments,
            "visits_since_reflection": self._visits_since_reflection,
        }

    def load_from_dict(self, data: dict) -> None:
        """Load state from saved data."""
        self.preference_adjustments = data.get("preference_adjustments", {})
        self._visits_since_reflection = data.get("visits_since_reflection", 0)
        for r in data.get("reflections", []):
            self.reflections.append(MemoryEntry(
                timestamp=datetime.fromisoformat(r["timestamp"]),
                event_type="reflection",
                content=r["content"],
                importance=r.get("importance", 0.8),
            ))
