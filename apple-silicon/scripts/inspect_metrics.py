import pandas as pd

df = pd.read_csv("runs/run_20260514_183932/metrics.csv")
print("Metrics Shape:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10))
print("\nLast 10 rows:")
print(df.tail(10))
print("\nSummary Statistics:")
print(df.describe())
