"""
Test script for census-based agent generation.
Verifies that agents are generated with proper proportions.
"""

import sys
from pathlib import Path
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation_layer.persona.census_agent_generator import CensusBasedAgentGenerator


def analyze_agents(agents):
    """Analyze and print agent statistics."""
    print(f"\n{'='*60}")
    print(f"AGENT GENERATION STATISTICS")
    print(f"{'='*60}")
    print(f"Total agents generated: {len(agents)}\n")

    # Gender distribution
    print("Gender Distribution:")
    gender_counts = Counter(a.gender for a in agents)
    for gender, count in sorted(gender_counts.items()):
        pct = count / len(agents) * 100
        print(f"  {gender}: {count} ({pct:.1f}%)")

    # Age group distribution
    print("\nAge Group Distribution:")
    age_group_counts = Counter(a.age_group for a in agents)
    for age_group in ["10대", "20대", "30대", "40대", "50대", "60대+"]:
        count = age_group_counts[age_group]
        pct = count / len(agents) * 100
        print(f"  {age_group}: {count} ({pct:.1f}%)")

    # Residence type distribution
    print("\nResidence Type Distribution:")
    residence_counts = Counter(a.residence_type for a in agents)
    for rtype, count in sorted(residence_counts.items(), key=lambda x: -x[1]):
        pct = count / len(agents) * 100
        print(f"  {rtype}: {count} ({pct:.1f}%)")

    # Household type distribution
    print("\nHousehold Type Distribution:")
    household_counts = Counter(a.household_type for a in agents)
    for htype in ["1인가구", "2세대가구", "3세대가구", "4세대가구"]:
        count = household_counts.get(htype, 0)
        pct = count / len(agents) * 100 if count > 0 else 0
        print(f"  {htype}: {count} ({pct:.1f}%)")

    # Income level distribution
    print("\nIncome Level Distribution:")
    income_counts = Counter(a.income_level for a in agents)
    for level in ["하", "중하", "중", "중상", "상"]:
        count = income_counts.get(level, 0)
        pct = count / len(agents) * 100 if count > 0 else 0
        print(f"  {level}: {count} ({pct:.1f}%)")

    # Occupation distribution (top 10)
    print("\nTop 10 Occupations:")
    occupation_counts = Counter(a.occupation for a in agents)
    for occupation, count in occupation_counts.most_common(10):
        pct = count / len(agents) * 100
        print(f"  {occupation}: {count} ({pct:.1f}%)")

    # Sample agents
    print(f"\n{'='*60}")
    print("Sample Agents:")
    print(f"{'='*60}")
    for i, agent in enumerate(agents[:5], 1):
        print(f"\nAgent {i}:")
        print(f"  Name: {agent.name}")
        print(f"  Age: {agent.age} ({agent.age_group})")
        print(f"  Gender: {agent.gender}")
        print(f"  Occupation: {agent.occupation}")
        print(f"  Income: {agent.income_level}")
        print(f"  Residence: {agent.residence_type}")
        print(f"  Household: {agent.household_type}")
        print(f"  Census Area: {agent.census_area_code}")
        print(f"  Store Prefs: {', '.join(agent.store_preferences[:3])}")


def main():
    print("Initializing Census-Based Agent Generator...")
    generator = CensusBasedAgentGenerator(target_agents=3500)

    print("\nGenerating 3500 agents...")
    agents = generator.generate_agents(3500)

    analyze_agents(agents)

    # Save to CSV for inspection
    import pandas as pd
    df = pd.DataFrame([a.to_dict() for a in agents])
    output_path = project_root / "data" / "output" / "census_agents_3500.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved agent data to: {output_path}")


if __name__ == "__main__":
    main()
