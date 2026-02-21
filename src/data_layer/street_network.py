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
from typing import Optional, Tuple, Dict, Any

import osmnx as ox
import networkx as nx

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
    prev_node: Optional[int] = None # 직전에 지나온 노드 (역방향 방지용)
    current_node: int = 0           # 현재 있는 노드 ID
    next_node: Optional[int] = None # 향하고 있는 다음 노드
    edge_data: Optional[Dict[str, Any]] = None # 현재 이동 중인 엣지의 속성 (도로명, 길이 등)
    edge_geometry: Optional[LineString] = None # 엣지의 실제 선형 형상 (곡선 도로 표현)
    edge_length: float = 0.0 # 현재 엣지의 총 길이(미터)
    distance_along_edge: float = 0.0  # 현재 엣지에서 이미 이동한 거리(미터)

    # 현재 실제 위경도 좌표
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
        self._graph: Optional[nx.MultiDiGraph] = None # WGS84(위경도) 그래프 -> 시각화용
        self._graph_proj: Optional[nx.MultiDiGraph] = None # 투영 좌표(미터) 그래프 -> 거리 계산용
        self._to_wgs84: Optional[Transformer] = None # 두 좌표계 간 변환기
        self._to_proj: Optional[Transformer] = None # 두 좌표계 간 변환기

        # Configure OSMnx
        ox.settings.use_cache = True # OSM 데이터 로컬 캐시 (재실행 시 빠르게)
        ox.settings.log_console = False # OSMnx 로그 출력 끔
        ox.settings.requests_timeout = 600 # 타임아웃 10분으로 연장

    ## OSM에서 도로 네트워크 다운로드
    def load_graph(self) -> None:
        """Load pedestrian walking network from OpenStreetMap."""
        # WGS84 graph (for visualization)
        self._graph = ox.graph_from_point(
            (self.config.center_lat, self.config.center_lng),
            dist=self.config.radius_m,
            network_type=self.config.network_type,
            simplify=self.config.simplify,
        )

        ## 에이전트가 한강 위를 걸어다니는 버그 방지
        # 한강 위 노드 제거 (lat < 37.550)
        river_nodes = [n for n, d in self._graph.nodes(data=True) if d.get('y', 999) < 37.550]
        if river_nodes:
            self._graph.remove_nodes_from(river_nodes)
            print(f"[StreetNetwork] Removed {len(river_nodes)} river nodes (lat < 37.550)")

        # 투영 좌표계로 변환 (미터 단위 거리 계산 가능하게)
        self._graph_proj = ox.project_graph(self._graph)

        # 좌표 변환기 초기화
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

    ## 좌표 변환 메서드
    ## 투영 좌표(미터) -> WGS84 위경도
    def xy_to_latlng(self, x: float, y: float) -> Tuple[float, float]:
        """Convert projected coordinates to WGS84 lat/lng."""
        if self._to_wgs84 is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")
        lng, lat = self._to_wgs84.transform(x, y)
        return lat, lng

    ## WGS84 위경도 -> 투영 좌표(미터)
    def latlng_to_xy(self, lat: float, lng: float) -> Tuple[float, float]:
        """Convert WGS84 lat/lng to projected coordinates."""
        if self._to_proj is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")
        x, y = self._to_proj.transform(lng, lat)
        return x, y

    ## 좌표를 받아 가장 가까운 도로 노드에 스냅. 에이전트 초기 위치 설정, 매장 위치 매핑에 사용.
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

    ## 다음 이동 방향 설정
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
        ## 나가는 방향 이웃 노드
        neighbors = list(self._graph_proj.successors(current_node))
        if not neighbors: ## 없으면 들어오는 방향으로
            neighbors = list(self._graph_proj.predecessors(current_node))
        if not neighbors: 
            return None

        ## 역방향 방지 : 바로 직전에 온 노드로 되돌아가지 않게
        if avoid_backtrack and prev_node in neighbors and len(neighbors) > 1:
            neighbors = [n for n in neighbors if n != prev_node]

        return random.choice(neighbors) ## 남은 이웃 중 랜덤 선택

    ## 같은 두 노드 간 여러 엣지가 있을 수 있어서 가장 짧은 엣지 선택
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

    ## 엣지의 실제 선형 형상 반환
    def get_edge_geometry(self, u: int, v: int, edge_data: Dict[str, Any]) -> LineString:
        """Get or create geometry for an edge."""
        geom = edge_data.get("geometry")
        if geom is not None:
            return geom

        # Create simple line if no geometry stored
        x1, y1 = self._graph_proj.nodes[u]["x"], self._graph_proj.nodes[u]["y"]
        x2, y2 = self._graph_proj.nodes[v]["x"], self._graph_proj.nodes[v]["y"]
        return LineString([(x1, y1), (x2, y2)])

    ## 에이전트 초기 배치
    def initialize_agent_location(self, lat: float, lng: float) -> AgentLocation:
        """
        Initialize an agent's location on the street network.

        Args:
            lat, lng: Starting coordinates (will be snapped to nearest node)

        Returns:
            AgentLocation with initialized position and first edge
        """
        ## 집/진입점 좌표 -> 가장 가까운 도로 노드로 스냅
        node_id, snap_lat, snap_lng, snap_dist = self.snap_to_nearest_node(lat, lng)

        ## AgentLocation 생성
        location = AgentLocation(
            current_node=node_id,
            lat=snap_lat,
            lng=snap_lng,
        )

        ## 첫 번째 이동 엣지 설정
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

    ## 에이전트 이동
    def move_agent(
        self,
        location: AgentLocation,
        distance_m: float ## 만큼 도로를 따라 이동
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

        ## 이동 거리 누적
        location.distance_along_edge += distance_m

        ## 현재 엣지를 다 지나갔으면 다음 엣지로
        while location.edge_length > 0 and location.distance_along_edge >= location.edge_length:
            location.distance_along_edge -= location.edge_length
            location.prev_node = location.current_node
            location.current_node = location.next_node

            ## 다음 방향 선택
            next_node = self.choose_next_node(
                location.current_node,
                location.prev_node
            )
            if next_node is None:
                location.edge_length = 0
                break
            
            ## 다음 엣지 정보
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

        # 엣지 위 정확한 좌표 보간 (직선/곡선 따라 비례 위치 계산)
        if location.edge_geometry is not None and location.edge_length > 0:
            interp_dist = min(location.distance_along_edge, location.edge_geometry.length)
            point = location.edge_geometry.interpolate(interp_dist)
            location.lat, location.lng = self.xy_to_latlng(point.x, point.y)

        return location



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
        _network_instance.load_graph() ## 첫 호출 시 OSM 다운로드

    return _network_instance ## 이후 호출은 캐시 반환


def reset_street_network() -> None:
    """Reset the singleton instance (for testing)."""
    global _network_instance
    _network_instance = None ## 테스트용 리셋
