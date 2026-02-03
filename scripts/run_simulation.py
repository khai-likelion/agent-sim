"""
CLI entry point for running the full simulation pipeline.
Replaces: run_simulation.bat + main() from agent_generator.py + simulation_engine.py
"""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from src.data_layer.spatial_index import Environment, load_and_index_stores
from src.data_layer.population_stats import PopulationStatistics
from src.simulation_layer.persona.agent_persona import AgentPersona
from src.simulation_layer.persona.agent_generator import AgentGenerator
from src.simulation_layer.engine import SimulationEngine
from src.simulation_layer.scenario.mangwon_scenario import (
    create_default_reports,
    get_default_start_date,
)


def main():
    settings = get_settings()

    print("=" * 80)
    print("Mangwon-dong Agent Simulation")
    print("=" * 80)
    print()

    # 1. Load and index stores
    print("[1/4] Loading store data...")
    stores_df = load_and_index_stores()
    env = Environment(stores_df)
    print()

    # 2. Load population stats + generate agents
    print("[2/4] Generating agents...")
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

    # 3. Run simulation
    print("[3/4] Running simulation...")
    reports = create_default_reports()
    engine = SimulationEngine(env, stats, agents)
    results_df = engine.run_simulation(
        start_date=get_default_start_date(),
        reports=reports,
    )
    print()

    # 4. Save results
    print("[4/4] Saving results...")
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
