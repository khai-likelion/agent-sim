"""
RAG retriever for store and area context.
Skeleton for future vector store integration.
"""

from abc import ABC, abstractmethod
from typing import List


class RAGRetriever(ABC):
    """Abstract RAG retriever interface."""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        """Retrieve relevant context documents for a query."""
        ...


class SimpleRAGRetriever(RAGRetriever):
    """Placeholder: keyword-based retrieval from store data."""

    async def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        raise NotImplementedError(
            "RAG retrieval pending vector store setup"
        )
