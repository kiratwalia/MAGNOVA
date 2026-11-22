
import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/public/dst_labels.csv")

print("DST TARGET ANALYSIS")
print("=" * 50)

# Basic statistics
print("\nStatistical summary:")
print(df["dst"].describe())

# Extreme values
print("\nMinimum Dst:", df["dst"].min())
print("Maximum Dst:", df["dst"].max())

# Storm intensity distribution
# Conventional Dst storm thresholds
print("\nStorm category counts:")

print("Moderate (-50 to -100 nT):",
      ((df["dst"] <= -50) & (df["dst"] > -100)).sum())

print("Intense (-100 to -200 nT):",
      ((df["dst"] <= -100) & (df["dst"] > -200)).sum())

print("Severe (<= -200 nT):",
      (df["dst"] <= -200).sum())

# Distribution by period
print("\nDst statistics by period:")
print(df.groupby("period")["dst"].agg(
    ["count", "mean", "std", "min", "max"]
))

print("\nTARGET ANALYSIS COMPLETED")
