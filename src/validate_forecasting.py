
import pandas as pd
import numpy as np

# File path
PATH = "data/processed/forecasting_dataset.csv"

# Load forecasting dataset
df = pd.read_csv(PATH)

# Convert hour column
df["hour"] = pd.to_timedelta(df["hour"])

print("=" * 60)
print("FORECASTING DATASET VALIDATION")
print("=" * 60)

# 1. Check duplicate timestamps
duplicates = df.duplicated(["period", "hour"]).sum()

print("\n1. Duplicate timestamps:", duplicates)

# 2. Check chronological ordering
ordered = all(
    group["hour"].is_monotonic_increasing
    for _, group in df.groupby("period")
)

print("2. Chronological ordering:", ordered)

# 3. Validate T+0 target consistency
t0_valid = True

for period, group in df.groupby("period"):
    group = group.sort_values("hour")

    actual_t0 = group["dst_t0"].iloc[:-1].to_numpy()
    expected_t0 = group["dst"].iloc[1:].to_numpy()

    if not np.array_equal(actual_t0, expected_t0):
        t0_valid = False

print("3. T+0 target consistency:", t0_valid)

# 4. Validate T+1 target consistency
t1_valid = True

for period, group in df.groupby("period"):
    group = group.sort_values("hour")

    actual_t1 = group["dst_t1"].iloc[:-2].to_numpy()
    expected_t1 = group["dst"].iloc[2:].to_numpy()

    if not np.array_equal(actual_t1, expected_t1):
        t1_valid = False

print("4. T+1 target consistency:", t1_valid)

# 5. Check consecutive hourly timestamps
consecutive = True

for period, group in df.groupby("period"):
    differences = group["hour"].diff().dropna()

    if not (differences == pd.Timedelta(hours=1)).all():
        consecutive = False

print("5. Consecutive hourly timestamps:", consecutive)

# 6. Check missing target values
print("\n6. Missing target values:")

print(
    df[["dst_t0", "dst_t1"]].isnull().sum()
)

# 7. Check rows per period
print("\n7. Rows per period:")

print(df.groupby("period").size())

# 8. Check ML feature columns
excluded_columns = [
    "period",
    "hour",
    "dst",
    "dst_t0",
    "dst_t1"
]

features = [
    col for col in df.columns
    if col not in excluded_columns
]

print("\n8. Potential ML features:", len(features))

# 9. Verify target exclusion
targets_excluded = all(
    col not in features
    for col in ["dst", "dst_t0", "dst_t1"]
)

print("9. Target columns excluded:", targets_excluded)

# 10. Final validation status
all_checks_passed = (
    duplicates == 0
    and ordered
    and t0_valid
    and t1_valid
    and consecutive
    and df[["dst_t0", "dst_t1"]].isnull().sum().sum() == 0
    and targets_excluded
)

print("\n" + "=" * 60)

if all_checks_passed:
    print("ALL VALIDATION CHECKS PASSED!")
else:
    print("VALIDATION FAILED! CHECK THE RESULTS ABOVE.")

print("=" * 60)
