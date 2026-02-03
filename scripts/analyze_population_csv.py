"""
Analyze area_summary.csv to understand the population data structure.
"""

import pandas as pd
from pathlib import Path

# Read CSV with proper encoding
csv_path = Path(__file__).parent.parent / "data" / "raw" / "area_summary.csv"
df = pd.read_csv(csv_path, encoding='cp949')  # Try Korean encoding

print("CSV Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head(3))
print("\nData types:")
print(df.dtypes)
print("\nTotal population sum:")
if '총인구수_합계' in df.columns:
    print(df['총인구수_합계'].sum())
elif df.columns[-1] not in ['TOT_OA_CD']:
    print(f"Last column ({df.columns[-1]}) sum:", df.iloc[:, -1].sum())
