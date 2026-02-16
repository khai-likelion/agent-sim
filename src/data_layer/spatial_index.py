"""
H3 grid-based spatial indexing for store discovery.
Manages the Mangwon-dong store environment using Uber's H3 hexagonal grid system.
"""

import pandas as pd
import h3

from config import get_settings


class Environment:
    """
    H3 grid-based spatial environment.
    Allows agents to discover nearby stores from a given coordinate.
    """

    def __init__(self, stores_df: pd.DataFrame, h3_resolution: int | None = None):
        settings = get_settings()
        self.stores_df = stores_df
        self.h3_resolution = h3_resolution or settings.simulation.h3_resolution

        # Cache stores by H3 index for O(1) lookup
        self.h3_to_stores: dict[str, list] = {}
        for _, store in stores_df.iterrows():
            h3_idx = store["h3_index"]
            if h3_idx not in self.h3_to_stores:
                self.h3_to_stores[h3_idx] = []
            self.h3_to_stores[h3_idx].append(store)

    def get_visible_stores(
        self, current_lat: float, current_lng: float, k_ring: int | None = None
    ) -> pd.DataFrame:
        """
        Return stores visible from a given coordinate.

        Args:
            current_lat: Current latitude.
            current_lng: Current longitude.
            k_ring: Adjacent grid search radius (default from settings).

        Returns:
            DataFrame of nearby stores.
        """
        settings = get_settings()
        k_ring = k_ring if k_ring is not None else settings.simulation.k_ring

        current_h3 = h3.latlng_to_cell(
            current_lat, current_lng, self.h3_resolution
        )
        nearby_h3_cells = h3.grid_disk(current_h3, k_ring)

        visible_stores = []
        for h3_idx in nearby_h3_cells:
            if h3_idx in self.h3_to_stores:
                visible_stores.extend(self.h3_to_stores[h3_idx])

        if visible_stores:
            return pd.DataFrame(visible_stores)
        else:
            return pd.DataFrame()


def load_and_index_stores(
    csv_path: str | None = None, h3_resolution: int | None = None
) -> pd.DataFrame:
    """
    Load store data from CSV and add H3 spatial index.

    Args:
        csv_path: Path to stores.csv. Defaults to settings.paths.stores_csv.
        h3_resolution: H3 resolution. Defaults to settings.simulation.h3_resolution.

    Returns:
        DataFrame with h3_index column added.
    """
    settings = get_settings()
    csv_path = csv_path or str(settings.paths.stores_csv)
    h3_resolution = h3_resolution or settings.simulation.h3_resolution

    print(f"Loading store data... ({csv_path})")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"Loaded {len(df)} stores")

    df["h3_index"] = df.apply(
        lambda row: h3.latlng_to_cell(row["y"], row["x"], h3_resolution),
        axis=1,
    )
    print(f"H3 indexing complete (resolution={h3_resolution}, "
          f"{df['h3_index'].nunique()} unique cells)")

    return df
