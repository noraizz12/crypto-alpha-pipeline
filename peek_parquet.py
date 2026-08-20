import sys

import pandas as pd

MAX_ROWS = 800

pd.options.display.max_rows = MAX_ROWS
pd.options.display.width = 250
pd.options.display.max_columns = 25

file = sys.argv[1]
df = pd.read_parquet(file)
print(f"Reading {file} Length: {len(df)}")

print(df.dtypes.sort_index())
print()
print(df.iloc[0].sort_index())
print()
print(df.iloc[-1].sort_index())
print()

if len(df) < MAX_ROWS:
    print(df)
else:
    print(df.head())
    print(df.tail())

for col in df.columns:
    print(col)
    print()
    print(df[col].describe())
