"""
Reflection: Periodic high-level insight generation from event memories.
Second step in the extension roadmap (Memory -> Reflection -> Planning).

Future: Uses LLM to synthesize patterns from recent experiences.
"""

from typing import List
from datetime import datetime

from .base import MemoryStructure
from .event_memory import EventMemory, MemoryEntry


class Reflection(MemoryStructure):
    """
    Generates and stores high-level reflections from event memories.
    E.g., "I've been visiting dessert cafes a lot lately" -> strengthen dessert preference.
    """

    def __init__(self, event_memory: EventMemory):
        self.event_memory = event_memory
        self.reflections: List[MemoryEntry] = []

    def add(self, content, timestamp: datetime) -> None:
        self.reflections.append(
            MemoryEntry(
                timestamp=timestamp,
                event_type="reflection",
                content=str(content),
                importance=0.8,  # Reflections are higher importance
            )
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        return sorted(
            self.reflections, key=lambda m: m.timestamp, reverse=True
        )[:top_k]

    def generate_reflection(self) -> str:
        """
        Placeholder for LLM-driven reflection.
        Future: feed recent event_memory entries to LLM
        with prompt_templates/reflection.txt to generate insights.
        """
        recent = self.event_memory.get_recent(20)
        if not recent:
            return "아직 충분한 경험이 없습니다."
        # TODO: LLM call with prompt_templates/reflection.txt
        return f"{len(recent)}개의 최근 경험을 바탕으로 한 반성 (LLM 연동 필요)"
