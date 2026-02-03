"""
Check household type data in area_summary.csv
"""

import pandas as pd
from pathlib import Path

csv_path = Path(__file__).parent.parent / "data" / "raw" / "area_summary.csv"
df = pd.read_csv(csv_path, encoding='cp949')

print("Household Type Statistics:")
print("="*60)

household_cols = ['1세대가구', '2세대가구', '3세대가구', '4세대가구']

for col in household_cols:
    total = df[col].sum()
    mean = df[col].mean()
    print(f"{col}:")
    print(f"  Total: {total}")
    print(f"  Mean per area: {mean:.1f}")
    print(f"  Min: {df[col].min()}, Max: {df[col].max()}")

print("\n1인가구 column:")
print(f"  Total: {df['1인가구'].sum()}")
print(f"  Mean: {df['1인가구'].mean():.1f}")

print("\nSample row (first area):")
for col in household_cols + ['1인가구']:
    print(f"  {col}: {df.iloc[0][col]}")
