"""
CLI entry point for running the full simulation pipeline.
Replaces: run_simulation.bat + main() from agent_generator.py + simulation_engine.py

Modes:
    --street-network (default): Use OSMnx street network for realistic pedestrian movement
    --h3-legacy: Use H3 grid-based spatial indexing (deprecated)
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
from src.simulation_layer.persona.agent_generator import AgentGenerator
from src.simulation_layer.engine import SimulationEngine
from src.simulation_layer.scenario.mangwon_scenario import (
    create_default_reports,
    get_default_start_date,
)
from src.simulation_layer.persona.cognitive_modules.decide import DecideModule


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
    parser = argparse.ArgumentParser(description="Mangwon-dong Agent Simulation")
    parser.add_argument(
        "--h3-legacy",
        action="store_true",
        help="Use legacy H3 grid mode instead of street network",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for behavior decisions (slower but more realistic)",
    )
    parser.add_argument(
        "--llm-delay",
        type=float,
        default=2.5,
        help="Delay between LLM calls in seconds (default: 2.5 for Groq free tier)",
    )
    args = parser.parse_args()

    settings = get_settings()
    use_street_network = not args.h3_legacy
    use_llm = args.use_llm

    print("=" * 80)
    print("Mangwon-dong Agent Simulation")
    print("=" * 80)
    mode_str = "StreetNetwork (OSMnx)" if use_street_network else "H3 Grid (Legacy)"
    decision_mode = "LLM (Groq)" if use_llm else "Rule-based"
    print(f"Spatial Mode: {mode_str}")
    print(f"Decision Mode: {decision_mode}")
    print()

    # 1. Load and index stores
    print("[1/4] Loading store data...")
    stores_df = load_and_index_stores()
    print()

    # 2. Create environment
    print("[2/5] Creating environment...")
    if use_street_network:
        env = create_street_network_environment(stores_df)
    else:
        env = create_h3_environment(stores_df)
    print()

    # 3. Load population stats + generate agents
    print("[3/5] Generating agents...")
    stats = PopulationStatistics()
    generator = AgentGenerator(stats)
    agents = generator.generate_agents()

    # Save agents
    agents_path = settings.paths.agents_json
    agents_path.parent.mkdir(parents=True, exist_ok=True)
    with open(agents_path, "w", encoding="utf-8") as f:
        json.dump([a.to_dict() for a in agents], f, ensure_ascii=False, indent=2)
    print(f"  {len(agents)} agents generated -> {agents_path}")
    print()

    # 4. Run simulation
    print("[4/5] Running simulation...")
    reports = create_default_reports()

    # Create decide module with LLM or rule-based mode
    decide_module = DecideModule(use_llm=use_llm, rate_limit_delay=args.llm_delay)

    engine = SimulationEngine(env, stats, agents, decide_module=decide_module)
    results_df = engine.run_simulation(
        start_date=get_default_start_date(),
        reports=reports,
    )
    print()

    # 5. Save results
    print("[5/5] Saving results...")
    settings.paths.output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        settings.paths.simulation_result_csv,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"  Full log -> {settings.paths.simulation_result_csv}")

    visit_stats = results_df[results_df["decision"] == "visit"][
        [
            "timestamp", "agent_name", "age_group", "gender",
            "weekday", "time_slot", "visited_store", "visited_category",
            "report_received", "decision_reason",
        ]
    ]
    visit_stats.to_csv(
        settings.paths.visit_log_csv,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"  Visit log -> {settings.paths.visit_log_csv}")

    print()
    print("=" * 80)
    print("Simulation complete!")
    print("=" * 80)

    # Quick summary
    total = len(results_df)
    active = len(results_df[results_df["is_active"] == True])
    visits = len(results_df[results_df["decision"] == "visit"])
    print(f"  Total events: {total}")
    print(f"  Active events: {active} ({active/total*100:.1f}%)")
    print(f"  Visit events: {visits} ({visits/total*100:.1f}%)")
    if active > 0:
        print(f"  Conversion rate: {visits/active*100:.1f}%")


if __name__ == "__main__":
    main()
