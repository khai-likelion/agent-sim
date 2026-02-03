"""
CLI entry point for running simulation with census-based agents.
Uses CensusBasedAgentGenerator for realistic demographic distribution.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from src.data_layer.spatial_index import load_and_index_stores
from src.data_layer.population_stats import PopulationStatistics
from src.simulation_layer.persona.census_agent_generator import CensusBasedAgentGenerator
from src.simulation_layer.engine import SimulationEngine
from src.simulation_layer.scenario.mangwon_scenario import (
    create_default_reports,
    get_default_start_date,
)


def create_street_network_environment(stores_df):
    """Create OSMnx-based street network environment."""
    from src.data_layer.street_network import StreetNetwork, StreetNetworkConfig

    settings = get_settings()
    config = StreetNetworkConfig(
        center_lat=settings.area.center_lat,
        center_lng=settings.area.center_lng,
        radius_m=settings.simulation.network_radius_m,
        network_type="walk",
        simplify=True,
    )
    network = StreetNetwork(config)
    network.load_graph()
    network.load_stores(stores_df)
    return network


def create_h3_environment(stores_df):
    """Create legacy H3-based environment."""
    from src.data_layer.spatial_index import Environment

    return Environment(stores_df)


def main():
    parser = argparse.ArgumentParser(description="Mangwon-dong Agent Simulation (Census-based)")
    parser.add_argument(
        "--h3-legacy",
        action="store_true",
        help="Use legacy H3 grid mode instead of street network",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=3500,
        help="Number of agents to generate (default: 3500)",
    )
    args = parser.parse_args()

    settings = get_settings()
    use_street_network = not args.h3_legacy

    print("=" * 80)
    print("Mangwon-dong Agent Simulation (Census-Based)")
    print("=" * 80)
    mode_str = "StreetNetwork (OSMnx)" if use_street_network else "H3 Grid (Legacy)"
    print(f"Mode: {mode_str}")
    print(f"Agents: {args.num_agents} (census-based demographic distribution)")
    print()

    # 1. Load and index stores
    print("[1/5] Loading store data...")
    stores_df = load_and_index_stores()
    print(f"  Loaded {len(stores_df)} stores")
    print()

    # 2. Create environment
    print("[2/5] Creating environment...")
    if use_street_network:
        env = create_street_network_environment(stores_df)
        print(f"  Street network loaded: {env.graph.number_of_nodes()} nodes, {env.graph.number_of_edges()} edges")
    else:
        env = create_h3_environment(stores_df)
        print(f"  H3 grid environment created")
    print()

    # 3. Generate census-based agents
    print("[3/5] Generating census-based agents...")
    print("  Using real census data from area_summary.csv")
    print("  - 72 census areas")
    print("  - 35,589 total population")
    print(f"  - Scaling to {args.num_agents} agents")
    generator = CensusBasedAgentGenerator(target_agents=args.num_agents)
    agents = generator.generate_agents()

    # Save agents
    agents_path = settings.paths.output_dir / "census_agents.json"
    agents_path.parent.mkdir(parents=True, exist_ok=True)
    with open(agents_path, "w", encoding="utf-8") as f:
        json.dump([a.to_dict() for a in agents], f, ensure_ascii=False, indent=2)
    print(f"  {len(agents)} agents generated -> {agents_path}")
    print()

    # 4. Run simulation
    print("[4/5] Running simulation...")
    # Use PopulationStatistics for time/weekday weights (still needed for activity patterns)
    stats = PopulationStatistics()
    reports = create_default_reports()
    engine = SimulationEngine(env, stats, agents)
    results_df = engine.run_simulation(
        start_date=get_default_start_date(),
        reports=reports,
    )
    print()

    # 5. Save results
    print("[5/5] Saving results...")
    settings.paths.output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = settings.paths.output_dir / "census_simulation_result.csv"
    results_df.to_csv(
        results_csv,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"  Full log -> {results_csv}")

    visit_stats = results_df[results_df["decision"] == "visit"][
        [
            "timestamp", "agent_name", "age_group", "gender",
            "weekday", "time_slot", "visited_store", "visited_category",
            "report_received", "decision_reason",
        ]
    ]
    visit_csv = settings.paths.output_dir / "census_visit_log.csv"
    visit_stats.to_csv(
        visit_csv,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"  Visit log -> {visit_csv}")

    print()
    print("=" * 80)
    print("Simulation complete!")
    print("=" * 80)

    # Quick summary
    total = len(results_df)
    active = len(results_df[results_df["is_active"] == True])
    visits = len(results_df[results_df["decision"] == "visit"])
    print(f"  Total events: {total:,}")
    print(f"  Active events: {active:,} ({active/total*100:.1f}%)")
    print(f"  Visit events: {visits:,} ({visits/total*100:.1f}%)")
    if active > 0:
        print(f"  Conversion rate: {visits/active*100:.1f}%")

    # Demographic breakdown
    print()
    print("Visit Demographics:")
    if visits > 0:
        visit_df = results_df[results_df["decision"] == "visit"]
        print("\n  By Age Group:")
        age_counts = visit_df["age_group"].value_counts()
        for age_group in ["10대", "20대", "30대", "40대", "50대", "60대+"]:
            count = age_counts.get(age_group, 0)
            pct = count / visits * 100 if visits > 0 else 0
            print(f"    {age_group}: {count} ({pct:.1f}%)")

        print("\n  By Gender:")
        gender_counts = visit_df["gender"].value_counts()
        for gender in ["남성", "여성"]:
            count = gender_counts.get(gender, 0)
            pct = count / visits * 100 if visits > 0 else 0
            print(f"    {gender}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
