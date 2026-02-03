"""
Abstract base for memory structures.
Extension roadmap: Memory -> Reflection -> Planning (in that order).
"""

from abc import ABC, abstractmethod
from typing import Any, List
from datetime import datetime


class MemoryStructure(ABC):
    """Base class for agent memory systems."""

    @abstractmethod
    def add(self, content: Any, timestamp: datetime) -> None:
        """Store a new memory entry."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Any]:
        """Retrieve relevant memories given a query."""
        ...
