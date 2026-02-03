"""
Store data repository.
Current: CSV-based. Future: vector store integration for RAG retrieval.
"""

from abc import ABC, abstractmethod

import pandas as pd


class StoreRepository(ABC):
    """Abstract base for store data access."""

    @abstractmethod
    def get_all_stores(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def get_stores_by_category(self, category: str) -> pd.DataFrame:
        ...

    @abstractmethod
    def search_stores(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """Semantic search. Future: vector store backed."""
        ...


class CsvStoreRepository(StoreRepository):
    """CSV-based store repository (current implementation)."""

    def __init__(self, stores_df: pd.DataFrame):
        self.stores_df = stores_df

    def get_all_stores(self) -> pd.DataFrame:
        return self.stores_df

    def get_stores_by_category(self, category: str) -> pd.DataFrame:
        return self.stores_df[
            self.stores_df["카테고리"].str.contains(category, na=False)
        ]

    def search_stores(self, query: str, top_k: int = 10) -> pd.DataFrame:
        mask = self.stores_df["장소명"].str.contains(query, na=False)
        return self.stores_df[mask].head(top_k)
