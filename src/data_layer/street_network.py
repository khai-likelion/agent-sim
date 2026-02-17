"""
OSMnx-based street network for pedestrian movement.
Replaces H3 grid-based spatial indexing with actual road network traversal.

Based on Agent_Street.ipynb approach:
- Loads pedestrian walking network from OpenStreetMap
- Agents move along actual street edges (not through buildings/rivers)
- Graph-based movement with edge geometry interpolation
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import osmnx as ox
import networkx as nx
import pandas as pd
from shapely.geometry import LineString
from pyproj import Transformer


@dataclass
class StreetNetworkConfig:
    """Configuration for street network loading."""
    center_lat: float
    center_lng: float
    radius_m: float = 2000.0
    network_type: str = "walk"
    simplify: bool = True


@dataclass
class AgentLocation:
    """Agent's current location on the street network."""
    prev_node: Optional[int] = None
    current_node: int = 0
    next_node: Optional[int] = None
    edge_data: Optional[Dict[str, Any]] = None
    edge_geometry: Optional[LineString] = None
    edge_length: float = 0.0
    distance_along_edge: float = 0.0  # meters traveled on current edge

    # Current position in lat/lng
    lat: float = 0.0
    lng: float = 0.0


class StreetNetwork:
    """
    OSMnx-based street network for pedestrian simulation.

    Provides:
    - Graph loading from OpenStreetMap
    - Coordinate transformations (WGS84 <-> projected)
    - Agent movement along street edges
    - Store snapping to nearest nodes
    """

    def __init__(self, config: StreetNetworkConfig):
        self.config = config
        self._graph: Optional[nx.MultiDiGraph] = None
        self._graph_proj: Optional[nx.MultiDiGraph] = None
        self._to_wgs84: Optional[Transformer] = None
        self._to_proj: Optional[Transformer] = None
        self._stores_by_node: Dict[int, List[Dict[str, Any]]] = {}

        # Configure OSMnx
        ox.settings.use_cache = True
        ox.settings.log_console = False

    def load_graph(self) -> None:
        """Load pedestrian walking network from OpenStreetMap."""
        # WGS84 graph (for visualization)
        self._graph = ox.graph_from_point(
            (self.config.center_lat, self.config.center_lng),
            dist=self.config.radius_m,
            network_type=self.config.network_type,
            simplify=self.config.simplify,
        )

        # Projected graph (for meter-based distance calculations)
        self._graph_proj = ox.project_graph(self._graph)

        # Setup coordinate transformers
        crs_proj = self._graph_proj.graph["crs"]
        self._to_wgs84 = Transformer.from_crs(crs_proj, "EPSG:4326", always_xy=True)
        self._to_proj = Transformer.from_crs("EPSG:4326", crs_proj, always_xy=True)

        print(f"[StreetNetwork] Loaded graph: {len(self._graph.nodes)} nodes, {len(self._graph.edges)} edges")

    @property
    def graph(self) -> nx.MultiDiGraph:
        """Get the WGS84 graph."""
        if self._graph is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")
        return self._graph

    @property
    def graph_proj(self) -> nx.MultiDiGraph:
        """Get the projected graph (meters)."""
        if self._graph_proj is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")
        return self._graph_proj

    def xy_to_latlng(self, x: float, y: float) -> Tuple[float, float]:
        """Convert projected coordinates to WGS84 lat/lng."""
        if self._to_wgs84 is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")
        lng, lat = self._to_wgs84.transform(x, y)
        return lat, lng

    def latlng_to_xy(self, lat: float, lng: float) -> Tuple[float, float]:
        """Convert WGS84 lat/lng to projected coordinates."""
        if self._to_proj is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")
        x, y = self._to_proj.transform(lng, lat)
        return x, y

    def snap_to_nearest_node(self, lat: float, lng: float) -> Tuple[int, float, float, float]:
        """
        Snap a coordinate to the nearest graph node.

        Returns:
            (node_id, snapped_lat, snapped_lng, snap_distance_m)
        """
        # OSMnx nearest_nodes expects X=lng, Y=lat for unprojected graph
        node_id = ox.distance.nearest_nodes(self._graph, X=lng, Y=lat)

        # Get snapped position
        node_data = self._graph_proj.nodes[node_id]
        sx, sy = node_data["x"], node_data["y"]
        snap_lat, snap_lng = self.xy_to_latlng(sx, sy)

        # Calculate snap distance
        ax, ay = self.latlng_to_xy(lat, lng)
        snap_dist = math.hypot(sx - ax, sy - ay)

        return node_id, snap_lat, snap_lng, snap_dist

    def get_random_start_node(self) -> int:
        """Get a random node from the graph as starting point."""
        return random.choice(list(self._graph_proj.nodes()))

    def choose_next_node(
        self,
        current_node: int,
        prev_node: Optional[int] = None,
        avoid_backtrack: bool = True
    ) -> Optional[int]:
        """
        Choose next node to move to from current node.

        Args:
            current_node: Current graph node ID
            prev_node: Previous node (to avoid backtracking)
            avoid_backtrack: If True, try not to return to prev_node

        Returns:
            Next node ID or None if dead-end
        """
        neighbors = list(self._graph_proj.successors(current_node))
        if not neighbors:
            neighbors = list(self._graph_proj.predecessors(current_node))
        if not neighbors:
            return None

        # Avoid going back to previous node if possible
        if avoid_backtrack and prev_node in neighbors and len(neighbors) > 1:
            neighbors = [n for n in neighbors if n != prev_node]

        return random.choice(neighbors)

    def get_edge_data(self, u: int, v: int) -> Optional[Dict[str, Any]]:
        """
        Get edge data between two nodes.
        For multi-edges, returns the shortest one.
        """
        data = self._graph_proj.get_edge_data(u, v)
        if not data:
            return None

        # For MultiDiGraph, pick edge with shortest length
        best_key = min(data, key=lambda k: data[k].get("length", float("inf")))
        return data[best_key]

    def get_edge_geometry(self, u: int, v: int, edge_data: Dict[str, Any]) -> LineString:
        """Get or create geometry for an edge."""
        geom = edge_data.get("geometry")
        if geom is not None:
            return geom

        # Create simple line if no geometry stored
        x1, y1 = self._graph_proj.nodes[u]["x"], self._graph_proj.nodes[u]["y"]
        x2, y2 = self._graph_proj.nodes[v]["x"], self._graph_proj.nodes[v]["y"]
        return LineString([(x1, y1), (x2, y2)])

    def initialize_agent_location(self, lat: float, lng: float) -> AgentLocation:
        """
        Initialize an agent's location on the street network.

        Args:
            lat, lng: Starting coordinates (will be snapped to nearest node)

        Returns:
            AgentLocation with initialized position and first edge
        """
        # Snap to nearest node
        node_id, snap_lat, snap_lng, snap_dist = self.snap_to_nearest_node(lat, lng)

        location = AgentLocation(
            current_node=node_id,
            lat=snap_lat,
            lng=snap_lng,
        )

        # Choose first edge
        next_node = self.choose_next_node(node_id, None)
        if next_node is None:
            print(f"[Warning] Start node {node_id} has no neighbors")
            return location

        edge_data = self.get_edge_data(node_id, next_node)
        if edge_data is None:
            return location

        location.next_node = next_node
        location.edge_data = edge_data
        location.edge_length = float(edge_data.get("length", 0.0))
        location.edge_geometry = self.get_edge_geometry(node_id, next_node, edge_data)
        location.distance_along_edge = 0.0

        return location

    def move_agent(
        self,
        location: AgentLocation,
        distance_m: float
    ) -> AgentLocation:
        """
        Move agent along street network by given distance.

        Args:
            location: Current agent location
            distance_m: Distance to move in meters

        Returns:
            Updated AgentLocation
        """
        if location.edge_geometry is None or location.edge_length == 0:
            return location

        location.distance_along_edge += distance_m

        # Traverse edges until we've consumed the distance
        while location.edge_length > 0 and location.distance_along_edge >= location.edge_length:
            location.distance_along_edge -= location.edge_length
            location.prev_node = location.current_node
            location.current_node = location.next_node

            # Choose next edge
            next_node = self.choose_next_node(
                location.current_node,
                location.prev_node
            )
            if next_node is None:
                location.edge_length = 0
                break

            edge_data = self.get_edge_data(location.current_node, next_node)
            if edge_data is None:
                location.edge_length = 0
                break

            location.next_node = next_node
            location.edge_data = edge_data
            location.edge_length = float(edge_data.get("length", 0.0))
            location.edge_geometry = self.get_edge_geometry(
                location.current_node, next_node, edge_data
            )

        # Interpolate position along current edge
        if location.edge_geometry is not None and location.edge_length > 0:
            interp_dist = min(location.distance_along_edge, location.edge_geometry.length)
            point = location.edge_geometry.interpolate(interp_dist)
            location.lat, location.lng = self.xy_to_latlng(point.x, point.y)

        return location

    def load_stores(self, stores_df: pd.DataFrame) -> None:
        """
        Load stores and associate them with nearest graph nodes.

        Args:
            stores_df: DataFrame with coordinates. Supports multiple naming conventions:
                       - 'lat', 'lng' (standard)
                       - 'y', 'x' (Kakao convention: y=lat, x=lng)
        """
        self._stores_by_node.clear()

        for idx, row in stores_df.iterrows():
            # Support multiple column naming conventions
            # Kakao uses: x=lng, y=lat
            lat = row.get("lat") if "lat" in row.index else row.get("y")
            lng = row.get("lng") if "lng" in row.index else row.get("x")

            if pd.isna(lat) or pd.isna(lng):
                continue

            # Convert to float
            lat = float(lat)
            lng = float(lng)

            node_id, _, _, snap_dist = self.snap_to_nearest_node(lat, lng)

            store_info = row.to_dict()
            store_info["_snap_distance_m"] = snap_dist
            store_info["_graph_node"] = node_id
            # Normalize lat/lng fields for consistent access
            store_info["lat"] = lat
            store_info["lng"] = lng

            if node_id not in self._stores_by_node:
                self._stores_by_node[node_id] = []
            self._stores_by_node[node_id].append(store_info)

        total_stores = sum(len(s) for s in self._stores_by_node.values())
        print(f"[StreetNetwork] Loaded {total_stores} stores across {len(self._stores_by_node)} nodes")

    def get_nearby_stores(
        self,
        location: AgentLocation,
        k_ring: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get stores near agent's current location.

        Uses graph-based neighborhood (k-hop neighbors) instead of H3 k-ring.

        Args:
            location: Agent's current location
            k_ring: Number of hops to search (similar to H3 k-ring concept)

        Returns:
            List of nearby store dictionaries
        """
        nearby_stores = []
        visited_nodes = set()
        current_level = {location.current_node}

        for _ in range(k_ring + 1):
            for node in current_level:
                if node in visited_nodes:
                    continue
                visited_nodes.add(node)

                if node in self._stores_by_node:
                    nearby_stores.extend(self._stores_by_node[node])

            # Expand to next level
            next_level = set()
            for node in current_level:
                next_level.update(self._graph_proj.successors(node))
                next_level.update(self._graph_proj.predecessors(node))
            current_level = next_level - visited_nodes

        return nearby_stores

    def calculate_distance_to_store(
        self,
        location: AgentLocation,
        store: Dict[str, Any]
    ) -> float:
        """
        Calculate approximate distance from agent to store.

        Uses network distance if on same connected component,
        otherwise falls back to Euclidean distance.
        """
        store_node = store.get("_graph_node")
        if store_node is None:
            # Fallback to Euclidean
            store_lat = store.get("lat", location.lat)
            store_lng = store.get("lng", location.lng)
            ax, ay = self.latlng_to_xy(location.lat, location.lng)
            sx, sy = self.latlng_to_xy(store_lat, store_lng)
            return math.hypot(ax - sx, ay - sy)

        try:
            # Try network distance
            return nx.shortest_path_length(
                self._graph_proj,
                location.current_node,
                store_node,
                weight="length"
            )
        except nx.NetworkXNoPath:
            # Fallback to Euclidean
            ax, ay = self.latlng_to_xy(location.lat, location.lng)
            sx = self._graph_proj.nodes[store_node]["x"]
            sy = self._graph_proj.nodes[store_node]["y"]
            return math.hypot(ax - sx, ay - sy)


# Module-level factory function
_network_instance: Optional[StreetNetwork] = None


def get_street_network(config: Optional[StreetNetworkConfig] = None) -> StreetNetwork:
    """
    Get or create the singleton street network instance.

    Args:
        config: Configuration (required on first call)

    Returns:
        StreetNetwork instance
    """
    global _network_instance

    if _network_instance is None:
        if config is None:
            raise ValueError("Config required on first call to get_street_network()")
        _network_instance = StreetNetwork(config)
        _network_instance.load_graph()

    return _network_instance


def reset_street_network() -> None:
    """Reset the singleton instance (for testing)."""
    global _network_instance
    _network_instance = None
