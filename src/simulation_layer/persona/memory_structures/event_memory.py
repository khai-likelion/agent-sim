"""
Event memory: Stanford Generative Agents style memory stream.
Stores factual records of agent experiences with structured retrieval.

Retrieval scoring: recency (0.4) × importance (0.3) × relevance (0.3)
Supports cross-day persistence and JSON serialization.
"""

import math
from typing import Any, List, Optional, Dict
from datetime import datetime
from dataclasses import dataclass, field

from .base import MemoryStructure


@dataclass
class MemoryEntry:
    """Single memory record with structured metadata."""

    timestamp: datetime
    event_type: str  # 'visit', 'observe', 'receive_report', 'reflection'
    content: str  # Natural language description
    location: Optional[tuple] = None  # (lat, lng)
    importance: float = 0.5  # 0-1, for retrieval ranking
    metadata: dict = field(default_factory=dict)
    # Structured fields for visit events
    store_name: Optional[str] = None
    category: Optional[str] = None
    satisfaction: Optional[float] = None
    day_number: int = 0


class EventMemory(MemoryStructure):
    """
    Chronological event memory with Stanford GA-style retrieval.
    Scores memories by recency × importance × relevance.
    """

    MAX_ENTRIES = 200

    def __init__(self):
        self.memories: List[MemoryEntry] = []

    def add(self, content: Any, timestamp: datetime) -> None:
        if isinstance(content, MemoryEntry):
            self.memories.append(content)
        else:
            self.memories.append(
                MemoryEntry(
                    timestamp=timestamp,
                    event_type="generic",
                    content=str(content),
                )
            )
        self._trim()

    def add_visit(
        self,
        timestamp: datetime,
        store_name: str,
        category: str,
        satisfaction: float,
        decision_reason: str,
        companion: Optional[str] = None,
        day_number: int = 0,
        importance: float = 0.6,
    ) -> None:
        """Record a store visit with structured data."""
        time_str = timestamp.strftime("%m/%d %H:%M")
        sat_text = "만족" if satisfaction >= 0.7 else ("보통" if satisfaction >= 0.4 else "불만족")
        content = f"{time_str} {store_name}({category}) 방문. {sat_text}. {decision_reason}"

        entry = MemoryEntry(
            timestamp=timestamp,
            event_type="visit",
            content=content,
            importance=importance,
            store_name=store_name,
            category=category,
            satisfaction=satisfaction,
            day_number=day_number,
            metadata={"companion": companion, "reason": decision_reason},
        )
        self.memories.append(entry)
        self._trim()

    def add_report_reception(
        self,
        timestamp: datetime,
        store_name: str,
        description: str,
        day_number: int = 0,
    ) -> None:
        """Record receiving a business report/promotion."""
        content = f"{timestamp.strftime('%m/%d %H:%M')} 프로모션 수신: {store_name} - {description}"
        entry = MemoryEntry(
            timestamp=timestamp,
            event_type="receive_report",
            content=content,
            importance=0.4,
            store_name=store_name,
            day_number=day_number,
            metadata={"description": description},
        )
        self.memories.append(entry)
        self._trim()

    def retrieve(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Retrieve memories by recency (backward compatible)."""
        return sorted(
            self.memories, key=lambda m: m.timestamp, reverse=True
        )[:top_k]

    def retrieve_relevant(
        self,
        query_context: Dict[str, str],
        top_k: int = 5,
        current_time: Optional[datetime] = None,
    ) -> List[MemoryEntry]:
        """Stanford GA-style retrieval: recency × importance × relevance."""
        if not self.memories:
            return []

        now = current_time or datetime.now()
        scored = []
        for m in self.memories:
            recency = self._recency_score(m.timestamp, now)
            importance = m.importance
            relevance = self._relevance_score(m, query_context)
            combined = 0.4 * recency + 0.3 * importance + 0.3 * relevance
            scored.append((combined, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        """Return the N most recent memories."""
        return self.memories[-n:]

    def get_visit_history(self) -> List[MemoryEntry]:
        """Return all visit memories."""
        return [m for m in self.memories if m.event_type == "visit"]

    def get_store_visits(self, store_name: str) -> List[MemoryEntry]:
        """Return all visits to a specific store."""
        return [m for m in self.memories if m.store_name == store_name and m.event_type == "visit"]

    def format_for_prompt(self, entries: List[MemoryEntry], max_chars: int = 500) -> str:
        """Format retrieved memories into compact text for prompt injection."""
        if not entries:
            return "기억 없음"

        lines = []
        char_count = 0
        for e in entries:
            line = f"- {e.content}"
            if char_count + len(line) > max_chars:
                break
            lines.append(line)
            char_count += len(line)

        return "\n".join(lines) if lines else "기억 없음"

    def _recency_score(self, memory_time: datetime, current_time: datetime) -> float:
        """Exponential decay: e^(-lambda * hours). ~50% at 35 hours."""
        hours = max(0, (current_time - memory_time).total_seconds() / 3600)
        return math.exp(-0.02 * hours)

    def _relevance_score(self, memory: MemoryEntry, context: Dict[str, str]) -> float:
        """Keyword-based relevance scoring."""
        score = 0.0
        if context.get("category") and memory.category:
            if context["category"] == memory.category:
                score += 0.5
            elif context["category"].lower() in (memory.category or "").lower():
                score += 0.3
        if context.get("store_name") and memory.store_name == context.get("store_name"):
            score += 0.5
        if context.get("time_slot") and context["time_slot"] in memory.content:
            score += 0.2
        return min(1.0, score)

    def _trim(self) -> None:
        """Keep memory within MAX_ENTRIES by removing lowest-importance old entries."""
        if len(self.memories) > self.MAX_ENTRIES:
            self.memories.sort(key=lambda m: m.importance)
            self.memories = self.memories[-self.MAX_ENTRIES:]
            # Re-sort by timestamp
            self.memories.sort(key=lambda m: m.timestamp)

    def to_dict(self) -> List[dict]:
        """Serialize for JSON save."""
        return [
            {
                "timestamp": m.timestamp.isoformat(),
                "event_type": m.event_type,
                "content": m.content,
                "importance": m.importance,
                "store_name": m.store_name,
                "category": m.category,
                "satisfaction": m.satisfaction,
                "metadata": m.metadata,
                "day_number": m.day_number,
            }
            for m in self.memories
        ]

    @classmethod
    def from_dict(cls, data: List[dict]) -> 'EventMemory':
        """Deserialize from saved JSON data."""
        em = cls()
        for d in data:
            em.memories.append(MemoryEntry(
                timestamp=datetime.fromisoformat(d["timestamp"]),
                event_type=d["event_type"],
                content=d["content"],
                importance=d.get("importance", 0.5),
                store_name=d.get("store_name"),
                category=d.get("category"),
                satisfaction=d.get("satisfaction"),
                metadata=d.get("metadata", {}),
                day_number=d.get("day_number", 0),
            ))
        return em
