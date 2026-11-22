
import pandas as pd
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "processed" / "forecasting_temporal_dataset.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "forecasting_temporal_splits.csv"
)

# Split configuration
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
GAP = 2

# Load dataset
df = pd.read_csv(INPUT_FILE)

df["hour"] = pd.to_timedelta(df["hour"])

df = df.sort_values(
    ["period", "hour"]
).reset_index(drop=True)

df["split"] = ""

# Create chronological splits independently per period
for period, group in df.groupby("period", sort=False):

    indices = group.index.to_list()
    n = len(indices)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_indices = indices[:train_end - GAP]

    val_indices = indices[
        train_end:val_end - GAP
    ]

    test_indices = indices[val_end:]

    df.loc[train_indices, "split"] = "train"
    df.loc[val_indices, "split"] = "validation"
    df.loc[test_indices, "split"] = "test"

# Save
df.to_csv(OUTPUT_FILE, index=False)

# Report
print("=" * 60)
print("TEMPORAL DATASET SPLITS CREATED")
print("=" * 60)

print("\nTotal records:", len(df))

print("\nOverall split counts:")
print(df["split"].value_counts())

print("\nSplit counts by period:")
print(
    pd.crosstab(
        df["period"],
        df["split"]
    )
)

print(
    "\nUnassigned gap rows:",
    (df["split"] == "").sum()
)

print("\nSaved to:", OUTPUT_FILE)

print("\nTEMPORAL SPLITTING COMPLETED!")
