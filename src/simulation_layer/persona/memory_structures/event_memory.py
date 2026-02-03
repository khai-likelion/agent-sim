"""
Event memory: Stores factual records of agent experiences.
First step in the extension roadmap (Memory -> Reflection -> Planning).

Current: in-memory list with recency-based retrieval.
Future: backed by vector store for semantic retrieval.
"""

from typing import Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from .base import MemoryStructure


@dataclass
class MemoryEntry:
    """Single memory record."""

    timestamp: datetime
    event_type: str  # 'visit', 'observe', 'receive_report'
    content: str  # Natural language description
    location: Optional[tuple] = None  # (lat, lng)
    importance: float = 0.5  # 0-1, for retrieval ranking
    metadata: dict = field(default_factory=dict)


class EventMemory(MemoryStructure):
    """
    Chronological event memory.
    Stores visit history, observations, and report receptions.
    """

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

    def retrieve(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Retrieve memories. Currently returns most recent; future: vector similarity."""
        return sorted(
            self.memories, key=lambda m: m.timestamp, reverse=True
        )[:top_k]

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        """Return the N most recent memories."""
        return self.memories[-n:]
