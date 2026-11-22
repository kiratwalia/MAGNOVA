
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "processed" / "forecasting_temporal_splits.csv"
)

df = pd.read_csv(INPUT_FILE)

df["hour"] = pd.to_timedelta(df["hour"])

print("=" * 60)
print("TEMPORAL SPLIT VALIDATION")
print("=" * 60)

all_valid = True

for period, group in df.groupby("period"):

    group = group.sort_values("hour").reset_index(drop=True)

    train_idx = group.index[group["split"] == "train"]
    val_idx = group.index[group["split"] == "validation"]
    test_idx = group.index[group["split"] == "test"]

    gap_idx = group.index[
    group["split"].isna() | (group["split"] == "")
]

    train_val_gap = val_idx[0] - train_idx[-1] - 1
    val_test_gap = test_idx[0] - val_idx[-1] - 1

    chronological = (
        train_idx[-1] < val_idx[0] < test_idx[0]
    )

    valid = (
        len(train_idx) > 0
        and len(val_idx) > 0
        and len(test_idx) > 0
        and train_val_gap == 2
        and val_test_gap == 2
        and len(gap_idx) == 4
        and chronological
    )

    print(f"\nPeriod: {period}")
    print("Training:", len(train_idx))
    print("Validation:", len(val_idx))
    print("Testing:", len(test_idx))
    print("Gap rows:", len(gap_idx))
    print("Train-validation gap:", train_val_gap)
    print("Validation-test gap:", val_test_gap)
    print("Chronological order:", chronological)
    print("Period valid:", valid)

    if not valid:
        all_valid = False

print("\n" + "=" * 60)

if all_valid:
    print("ALL TEMPORAL SPLITS VALIDATED!")
else:
    print("TEMPORAL SPLIT VALIDATION FAILED!")

print("=" * 60)

assert all_valid, "Split validation failed!"
