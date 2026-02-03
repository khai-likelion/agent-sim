"""
Data Layer - Feature Store and Spatial Indexing.

Provides:
- StreetNetwork: OSMnx-based pedestrian network (primary)
- Environment: Legacy H3-based spatial indexing (deprecated)
- PopulationStatistics: Demographic data
- StoreRepository: Store data access
"""

from src.data_layer.street_network import (
    StreetNetwork,
    StreetNetworkConfig,
    AgentLocation,
    get_street_network,
    reset_street_network,
)
from src.data_layer.spatial_index import Environment, load_and_index_stores
from src.data_layer.population_stats import PopulationStatistics

__all__ = [
    # Street Network (primary)
    "StreetNetwork",
    "StreetNetworkConfig",
    "AgentLocation",
    "get_street_network",
    "reset_street_network",
    # Legacy H3 (deprecated)
    "Environment",
    "load_and_index_stores",
    # Population
    "PopulationStatistics",
]
