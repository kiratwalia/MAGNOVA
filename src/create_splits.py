
import pandas as pd

INPUT_PATH = "data/processed/forecasting_dataset.csv"
OUTPUT_PATH = "data/processed/forecasting_splits.csv"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
GAP = 2

df = pd.read_csv(INPUT_PATH)

df["hour"] = pd.to_timedelta(df["hour"])

# Sort each period chronologically
df = df.sort_values(
    ["period", "hour"]
).reset_index(drop=True)

# Assign chronological splits independently per period
df["split"] = ""

for period, group in df.groupby("period", sort=False):

    indices = group.index.to_list()
    n = len(indices)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    # Training partition, excluding boundary gap
    train_indices = indices[:train_end - GAP]

    # Validation partition, excluding boundary gaps
    val_indices = indices[train_end:val_end - GAP]

    # Testing partition
    test_indices = indices[val_end:]

    df.loc[train_indices, "split"] = "train"
    df.loc[val_indices, "split"] = "validation"
    df.loc[test_indices, "split"] = "test"

# Save split information
df.to_csv(OUTPUT_PATH, index=False)

print("=" * 60)
print("CHRONOLOGICAL DATA SPLITTING")
print("=" * 60)

print("\nSplit distribution:")
print(df["split"].value_counts())

print("\nSplit distribution by period:")
print(
    pd.crosstab(
        df["period"],
        df["split"]
    )
)

print("\nMissing split assignments:")
print((df["split"] == "").sum())

print("\nSaved to:", OUTPUT_PATH)
