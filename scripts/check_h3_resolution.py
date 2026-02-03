"""
H3 Resolution grid size information utility.
"""

import sys
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import h3


def haversine_distance(lat1, lng1, lat2, lng2):
    """Haversine formula for distance in meters between two coordinates."""
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def calculate_h3_metrics(resolution=10):
    sample_lat = 37.5556
    sample_lng = 126.9104
    h3_index = h3.latlng_to_cell(sample_lat, sample_lng, resolution)

    print("=" * 60)
    print(f"H3 Resolution {resolution} Grid Info")
    print("=" * 60)
    print()

    area_km2 = h3.cell_area(h3_index, unit="km^2")
    area_m2 = h3.cell_area(h3_index, unit="m^2")
    print(f"Grid area: {area_m2:.2f} m^2 ({area_km2:.6f} km^2)")
    print(f"  ~{math.sqrt(area_m2):.1f}m x {math.sqrt(area_m2):.1f}m equivalent")
    print()

    boundary = h3.cell_to_boundary(h3_index)
    edge_lengths = []
    for i in range(len(boundary)):
        next_i = (i + 1) % len(boundary)
        lat1, lng1 = boundary[i]
        lat2, lng2 = boundary[next_i]
        edge_lengths.append(haversine_distance(lat1, lng1, lat2, lng2))

    edge_length_m = sum(edge_lengths) / len(edge_lengths)
    print(f"Edge length: {edge_length_m:.2f}m")
    print(f"Diameter: ~{edge_length_m * 2:.1f}m")
    print()

    nearby = h3.grid_disk(h3_index, 1)
    print(f"k_ring=1: {len(nearby)} cells, ~{edge_length_m * 2:.1f}m radius")
    print()

    print("=" * 60)
    print("Resolution Comparison")
    print("=" * 60)
    print(f"{'Res':<6} {'Area (m^2)':<16} {'Edge (m)':<12} {'Use case'}")
    print("-" * 60)

    for res, usage in [
        (8, "neighborhood"),
        (9, "block"),
        (10, "commercial (current)"),
        (11, "building"),
        (12, "detailed"),
    ]:
        temp_h3 = h3.latlng_to_cell(sample_lat, sample_lng, res)
        temp_area = h3.cell_area(temp_h3, unit="m^2")
        temp_boundary = h3.cell_to_boundary(temp_h3)
        temp_edges = []
        for i in range(len(temp_boundary)):
            next_i = (i + 1) % len(temp_boundary)
            lat1, lng1 = temp_boundary[i]
            lat2, lng2 = temp_boundary[next_i]
            temp_edges.append(haversine_distance(lat1, lng1, lat2, lng2))
        temp_edge = sum(temp_edges) / len(temp_edges)
        marker = " *" if res == resolution else "  "
        print(f"{marker}{res:<4} {temp_area:>14,.1f} {temp_edge:>10.1f}   {usage}")
    print()


if __name__ == "__main__":
    calculate_h3_metrics(resolution=10)
