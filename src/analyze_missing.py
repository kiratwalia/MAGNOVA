
import pandas as pd

df = pd.read_csv("data/processed/hourly_dataset.csv")

print("=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

# Missing values in each column
missing = df.isnull().sum()

summary = pd.DataFrame({
    "Missing_Count": missing,
    "Missing_Percentage": (missing / len(df) * 100).round(2)
})

summary = summary[summary["Missing_Count"] > 0]
summary = summary.sort_values(
    "Missing_Count",
    ascending=False
)

print("\nMissing values by column:")
print(summary.to_string())

# Overall missing statistics
print("\nOVERALL STATISTICS")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("Total missing cells:", df.isnull().sum().sum())
print("Rows containing missing values:", df.isnull().any(axis=1).sum())
print("Rows with no missing values:", (~df.isnull().any(axis=1)).sum())

# Missing values by period
print("\nMISSING VALUES BY PERIOD")

for period, group in df.groupby("period"):
    total = group.isnull().sum().sum()
    print(
        period,
        "| Rows:", len(group),
        "| Missing cells:", total,
        "| Missing percentage:",
        round(total / (len(group) * (len(df.columns) - 3)) * 100, 2),
        "%"
    )

# Check target column
print("\nDST MISSING VALUES:", df["dst"].isnull().sum())

print("\nMISSING VALUE ANALYSIS COMPLETED")
