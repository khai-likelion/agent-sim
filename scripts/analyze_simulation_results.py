"""
Analyze and visualize census-based simulation results.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import get_settings

# Set Korean font for matplotlib
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False


def analyze_results():
    """Analyze simulation results."""
    settings = get_settings()

    # Load results
    results_path = settings.paths.output_dir / "census_simulation_result.csv"
    visits_path = settings.paths.output_dir / "census_visit_log.csv"

    if not results_path.exists():
        print(f"Error: {results_path} not found")
        print("Run the simulation first with: python scripts/run_census_simulation.py")
        return

    print("Loading simulation results...")
    df = pd.read_csv(results_path, encoding='utf-8-sig')
    visits_df = pd.read_csv(visits_path, encoding='utf-8-sig')

    print(f"\nLoaded {len(df):,} events, {len(visits_df):,} visits")

    # Basic statistics
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)

    total = len(df)
    active = len(df[df['is_active'] == True])
    visits = len(df[df['decision'] == 'visit'])

    print(f"Total events: {total:,}")
    print(f"Active events: {active:,} ({active/total*100:.1f}%)")
    print(f"Visit events: {visits:,} ({visits/total*100:.1f}%)")
    print(f"Conversion rate: {visits/active*100:.1f}%")

    # Demographics
    print("\n" + "="*60)
    print("VISIT DEMOGRAPHICS")
    print("="*60)

    visit_df = df[df['decision'] == 'visit']

    print("\nBy Age Group:")
    age_counts = visit_df['age_group'].value_counts()
    for age_group in ["10대", "20대", "30대", "40대", "50대", "60대+"]:
        count = age_counts.get(age_group, 0)
        pct = count / visits * 100
        print(f"  {age_group}: {count:,} ({pct:.1f}%)")

    print("\nBy Gender:")
    gender_counts = visit_df['gender'].value_counts()
    for gender in ["남성", "여성"]:
        count = gender_counts.get(gender, 0)
        pct = count / visits * 100
        print(f"  {gender}: {count:,} ({pct:.1f}%)")

    # Temporal patterns
    print("\n" + "="*60)
    print("TEMPORAL PATTERNS")
    print("="*60)

    print("\nBy Day of Week:")
    weekday_counts = visit_df['weekday'].value_counts()
    for weekday in ["월", "화", "수", "목", "금", "토", "일"]:
        count = weekday_counts.get(weekday, 0)
        pct = count / visits * 100
        print(f"  {weekday}요일: {count:,} ({pct:.1f}%)")

    print("\nBy Time Slot:")
    time_counts = visit_df['time_slot'].value_counts().sort_index()
    for time_slot, count in time_counts.items():
        pct = count / visits * 100
        print(f"  {time_slot}: {count:,} ({pct:.1f}%)")

    # Store categories
    print("\n" + "="*60)
    print("POPULAR STORE CATEGORIES (Top 15)")
    print("="*60)

    category_counts = visit_df['visited_category'].value_counts().head(15)
    for category, count in category_counts.items():
        pct = count / visits * 100
        print(f"  {category}: {count:,} ({pct:.1f}%)")

    # Top visited stores
    print("\n" + "="*60)
    print("TOP VISITED STORES (Top 20)")
    print("="*60)

    store_counts = visit_df['visited_store'].value_counts().head(20)
    for i, (store, count) in enumerate(store_counts.items(), 1):
        pct = count / visits * 100
        print(f"  {i:2d}. {store}: {count:,} ({pct:.1f}%)")

    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    output_dir = settings.paths.output_dir

    # 1. Age group distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    age_order = ["10대", "20대", "30대", "40대", "50대", "60대+"]
    age_data = [age_counts.get(age, 0) for age in age_order]
    ax.bar(age_order, age_data, color='skyblue', edgecolor='black')
    ax.set_title('연령대별 방문 분포', fontsize=16, fontweight='bold')
    ax.set_xlabel('연령대', fontsize=12)
    ax.set_ylabel('방문 횟수', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(age_data):
        ax.text(i, v + 100, f'{v:,}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(output_dir / 'visits_by_age.png', dpi=150)
    print("  ✓ visits_by_age.png")
    plt.close()

    # 2. Gender distribution
    fig, ax = plt.subplots(figsize=(8, 8))
    gender_data = [gender_counts.get('남성', 0), gender_counts.get('여성', 0)]
    colors = ['#3498db', '#e74c3c']
    wedges, texts, autotexts = ax.pie(
        gender_data,
        labels=['남성', '여성'],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 14}
    )
    ax.set_title('성별 방문 비율', fontsize=16, fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'visits_by_gender.png', dpi=150)
    print("  ✓ visits_by_gender.png")
    plt.close()

    # 3. Weekday distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    weekday_order = ["월", "화", "수", "목", "금", "토", "일"]
    weekday_data = [weekday_counts.get(day, 0) for day in weekday_order]
    ax.plot(weekday_order, weekday_data, marker='o', linewidth=2, markersize=10, color='#e74c3c')
    ax.set_title('요일별 방문 추이', fontsize=16, fontweight='bold')
    ax.set_xlabel('요일', fontsize=12)
    ax.set_ylabel('방문 횟수', fontsize=12)
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(weekday_data):
        ax.text(i, v + 200, f'{v:,}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(output_dir / 'visits_by_weekday.png', dpi=150)
    print("  ✓ visits_by_weekday.png")
    plt.close()

    # 4. Time slot distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    time_order = sorted(time_counts.index)
    time_data = [time_counts.get(t, 0) for t in time_order]
    ax.bar(time_order, time_data, color='lightgreen', edgecolor='black')
    ax.set_title('시간대별 방문 분포', fontsize=16, fontweight='bold')
    ax.set_xlabel('시간대', fontsize=12)
    ax.set_ylabel('방문 횟수', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(time_data):
        ax.text(i, v + 100, f'{v:,}', ha='center', va='bottom')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'visits_by_timeslot.png', dpi=150)
    print("  ✓ visits_by_timeslot.png")
    plt.close()

    # 5. Top categories
    fig, ax = plt.subplots(figsize=(12, 8))
    top_categories = category_counts.head(10)
    ax.barh(range(len(top_categories)), top_categories.values, color='coral', edgecolor='black')
    ax.set_yticks(range(len(top_categories)))
    ax.set_yticklabels(top_categories.index)
    ax.set_title('인기 업종 Top 10', fontsize=16, fontweight='bold')
    ax.set_xlabel('방문 횟수', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    for i, v in enumerate(top_categories.values):
        ax.text(v + 50, i, f'{v:,}', va='center')
    plt.tight_layout()
    plt.savefig(output_dir / 'top_categories.png', dpi=150)
    print("  ✓ top_categories.png")
    plt.close()

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    analyze_results()
