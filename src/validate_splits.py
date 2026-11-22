
import pandas as pd

PATH = "data/processed/forecasting_splits.csv"

df = pd.read_csv(PATH)
df["hour"] = pd.to_timedelta(df["hour"])

print("=" * 60)
print("CHRONOLOGICAL SPLIT VALIDATION")
print("=" * 60)

all_valid = True

for period, group in df.groupby("period"):

    group = group.sort_values("hour").reset_index(drop=True)

    print(f"\n{period}")

    # Find the indices belonging to each split
    train_idx = group.index[group["split"] == "train"]
    val_idx = group.index[group["split"] == "validation"]
    test_idx = group.index[group["split"] == "test"]

    # Find unassigned gap rows
    gap_idx = group.index[
    group["split"].isna() | (group["split"] == "")
]
    print("Training records:", len(train_idx))
    print("Validation records:", len(val_idx))
    print("Testing records:", len(test_idx))
    print("Gap records:", len(gap_idx))

    # Verify two rows between each split
    train_val_gap = (
        val_idx[0] - train_idx[-1] - 1
        if len(train_idx) and len(val_idx) else -1
    )

    val_test_gap = (
        test_idx[0] - val_idx[-1] - 1
        if len(val_idx) and len(test_idx) else -1
    )

    print("Training-validation gap:", train_val_gap)
    print("Validation-testing gap:", val_test_gap)

    valid = (
        train_val_gap == 2
        and val_test_gap == 2
        and train_idx[-1] < val_idx[0] < test_idx[0]
    )

    print("Period split valid:", valid)

    if not valid:
        all_valid = False

print("\n" + "=" * 60)

if all_valid:
    print("ALL SPLIT BOUNDARIES VALIDATED!")
else:
    print("SPLIT VALIDATION FAILED!")

print("=" * 60)
