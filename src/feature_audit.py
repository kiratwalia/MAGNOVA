
import pandas as pd

PATH = "data/processed/forecasting_dataset.csv"

df = pd.read_csv(PATH)

# Columns that must never enter the ML feature matrix
excluded = [
    "period",
    "hour",
    "dst",
    "dst_t0",
    "dst_t1"
]

features = [
    col for col in df.columns
    if col not in excluded
]

print("=" * 65)
print("FEATURE AUDIT")
print("=" * 65)

print("\n1. ML FEATURES:")

for i, feature in enumerate(features, start=1):
    print(f"{i}. {feature}")

print("\nTotal ML features:", len(features))

print("\n2. TARGET LEAKAGE CHECK:")

for target in ["dst_t0", "dst_t1"]:
    print(f"{target} excluded:", target not in features)

print("\n3. MISSING VALUES PER FEATURE:")

missing = df[features].isnull().sum()

missing_summary = pd.DataFrame({
    "missing_count": missing,
    "missing_percentage": (
        missing / len(df) * 100
    ).round(2)
})

print(
    missing_summary.sort_values(
        "missing_percentage",
        ascending=False
    )
)

print("\n4. PERIOD-WISE TIME RANGES:")

for period, group in df.groupby("period"):
    print(f"\n{period}")
    print("Start:", group["hour"].iloc[0])
    print("End:", group["hour"].iloc[-1])
    print("Records:", len(group))

print("\n5. MISSING VALUES PER PERIOD:")

for period, group in df.groupby("period"):
    missing_count = group[features].isnull().sum().sum()
    total_cells = len(group) * len(features)

    percentage = missing_count / total_cells * 100

    print(
        f"{period}: {missing_count} missing cells "
        f"({percentage:.2f}%)"
    )

print("\n6. FEATURE STATISTICS:")
print(df[features].describe().round(2))

print("\nFEATURE AUDIT COMPLETED!")
