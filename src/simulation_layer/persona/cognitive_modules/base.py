"""
Abstract base class for cognitive modules.
Each module represents one stage of the Perceive -> Decide -> React cycle
(Stanford Generative Agents pattern).
"""

from abc import ABC, abstractmethod
from typing import Any


class CognitiveModule(ABC):
    """Base class for all cognitive modules."""

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Execute this cognitive step and return its output."""
        ...
