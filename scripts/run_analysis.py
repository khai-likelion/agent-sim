"""
CLI entry point for analyzing simulation results.
Refactored from the original analyze_results.py.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from config import get_settings


def print_header(title: str):
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def analyze_basic_stats(df: pd.DataFrame):
    print_header("Basic Statistics")
    active_df = df[df["is_active"] == True]
    visit_df = df[df["decision"] == "visit"]
    print(f"Active events: {len(active_df)} ({len(active_df)/len(df)*100:.1f}%)")
    print(f"Visit events: {len(visit_df)} ({len(visit_df)/len(df)*100:.1f}%)")
    print(f"Conversion rate: {len(visit_df)/len(active_df)*100:.1f}% (vs active)")
    print()
    return active_df, visit_df


def analyze_time_slots(active_df: pd.DataFrame, visit_df: pd.DataFrame):
    print_header("Time Slot Analysis")
    ts_activity = active_df.groupby("time_slot").size()
    ts_visits = visit_df.groupby("time_slot").size()
    print(f"{'Time Slot':<12} {'Active':<10} {'Visits':<10} {'Rate'}")
    print("-" * 50)
    for slot in sorted(ts_activity.index):
        a = ts_activity.get(slot, 0)
        v = ts_visits.get(slot, 0)
        r = (v / a * 100) if a > 0 else 0
        print(f"{slot:<12} {a:<10} {v:<10} {r:.1f}%")
    print()


def analyze_weekdays(active_df: pd.DataFrame, visit_df: pd.DataFrame):
    print_header("Weekday Analysis")
    wd_activity = active_df.groupby("weekday").size()
    wd_visits = visit_df.groupby("weekday").size()
    print(f"{'Weekday':<12} {'Active':<10} {'Visits':<10} {'Rate'}")
    print("-" * 50)
    for day in ["월", "화", "수", "목", "금", "토", "일"]:
        a = wd_activity.get(day, 0)
        v = wd_visits.get(day, 0)
        r = (v / a * 100) if a > 0 else 0
        print(f"{day:<12} {a:<10} {v:<10} {r:.1f}%")
    print()


def analyze_age_groups(active_df: pd.DataFrame, visit_df: pd.DataFrame):
    print_header("Age Group Analysis")
    age_order = {"10대": 1, "20대": 2, "30대": 3, "40대": 4, "50대": 5, "60대+": 6}
    age_activity = active_df.groupby("age_group").size()
    age_visits = visit_df.groupby("age_group").size()
    print(f"{'Age Group':<12} {'Active':<10} {'Visits':<10} {'Rate'}")
    print("-" * 50)
    for age in sorted(age_activity.index, key=lambda x: age_order.get(x, 7)):
        a = age_activity.get(age, 0)
        v = age_visits.get(age, 0)
        r = (v / a * 100) if a > 0 else 0
        print(f"{age:<12} {a:<10} {v:<10} {r:.1f}%")
    print()


def analyze_popular_stores(visit_df: pd.DataFrame):
    print_header("Popular Stores TOP 20")
    store_counts = visit_df["visited_store"].value_counts().head(20)
    print(f"{'Rank':<5} {'Store':<40} {'Visits'}")
    print("-" * 80)
    for idx, (store, count) in enumerate(store_counts.items(), 1):
        print(f"{idx:<5} {store:<40} {count}")
    print()


def analyze_categories(visit_df: pd.DataFrame):
    print_header("Category Analysis")
    vdf = visit_df.copy()
    vdf["main_category"] = vdf["visited_category"].apply(
        lambda x: x.split(">")[0].strip() if pd.notna(x) else "Unknown"
    )
    cat_counts = vdf["main_category"].value_counts().head(15)
    total = len(visit_df)
    print(f"{'Rank':<5} {'Category':<30} {'Visits':<10} {'Rate'}")
    print("-" * 80)
    for idx, (cat, count) in enumerate(cat_counts.items(), 1):
        print(f"{idx:<5} {cat:<30} {count:<10} {count/total*100:.1f}%")
    print()


def analyze_report_effectiveness(df: pd.DataFrame):
    print_header("Business Report Effectiveness")
    report_df = df[df["report_received"].notna()]
    report_visit_df = report_df[report_df["decision"] == "visit"]

    if len(report_df) == 0:
        print("No report events found.")
        print()
        return

    print(f"Report received: {len(report_df)}")
    print(f"Report -> visit: {len(report_visit_df)}")
    print(f"Report effectiveness: {len(report_visit_df)/len(report_df)*100:.1f}%")
    print()

    for report_text in report_df["report_received"].unique():
        rdf = report_df[report_df["report_received"] == report_text]
        rv = len(rdf[rdf["decision"] == "visit"])
        eff = rv / len(rdf) * 100
        print(f"  {report_text}")
        print(f"    Received: {len(rdf)}, Visits: {rv}, Rate: {eff:.1f}%")

        age_response = rdf.groupby("age_group").apply(
            lambda x: len(x[x["decision"] == "visit"]) / len(x) * 100
            if len(x) > 0 else 0,
            include_groups=False,
        ).sort_values(ascending=False)
        print(f"    Top age group response:")
        for age, rate in age_response.head(3).items():
            print(f"      {age}: {rate:.1f}%")
        print()


def analyze_agents(df: pd.DataFrame):
    print_header("Agent Activity (TOP 10)")
    agent_stats = df.groupby("agent_name").agg(
        {"is_active": "sum", "decision": lambda x: (x == "visit").sum()}
    ).rename(columns={"is_active": "activity_count", "decision": "visit_count"})
    agent_stats["conversion_rate"] = (
        agent_stats["visit_count"] / agent_stats["activity_count"] * 100
    )
    agent_stats = agent_stats.sort_values("visit_count", ascending=False)

    print(f"{'Rank':<5} {'Agent':<15} {'Active':<12} {'Visits':<12} {'Rate'}")
    print("-" * 80)
    for idx, (name, row) in enumerate(agent_stats.head(10).iterrows(), 1):
        print(
            f"{idx:<5} {name:<15} {int(row['activity_count']):<12} "
            f"{int(row['visit_count']):<12} {row['conversion_rate']:.1f}%"
        )
    print()


def analyze_daily_trend(df: pd.DataFrame):
    print_header("Daily Trend")
    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["timestamp"]).dt.date
    daily = df_copy.groupby("date").agg(
        {"is_active": "sum", "decision": lambda x: (x == "visit").sum()}
    ).rename(columns={"is_active": "activity_count", "decision": "visit_count"})
    daily["conversion_rate"] = daily["visit_count"] / daily["activity_count"] * 100

    wd_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
    print(f"{'Date':<15} {'Day':<5} {'Active':<10} {'Visits':<10} {'Rate'}")
    print("-" * 80)
    for date, row in daily.iterrows():
        wd = wd_map[pd.to_datetime(date).weekday()]
        print(
            f"{date!s:<15} {wd:<5} {int(row['activity_count']):<10} "
            f"{int(row['visit_count']):<10} {row['conversion_rate']:.1f}%"
        )
    print()


def main():
    settings = get_settings()

    print("=" * 80)
    print("Mangwon-dong Simulation Result Analysis")
    print("=" * 80)
    print()

    csv_path = settings.paths.simulation_result_csv
    print(f"Loading data... ({csv_path})")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"Total events: {len(df)}")
    print()

    active_df, visit_df = analyze_basic_stats(df)
    analyze_time_slots(active_df, visit_df)
    analyze_weekdays(active_df, visit_df)
    analyze_age_groups(active_df, visit_df)
    analyze_popular_stores(visit_df)
    analyze_categories(visit_df)
    analyze_report_effectiveness(df)
    analyze_agents(df)
    analyze_daily_trend(df)

    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
