
import pandas as pd

INPUT_PATH = "data/processed/hourly_dataset.csv"
OUTPUT_PATH = "data/processed/forecasting_dataset.csv"

# 1. Load original hourly dataset
df = pd.read_csv(INPUT_PATH)

# 2. Convert time column
df["hour"] = pd.to_timedelta(df["hour"])

# 3. Sort within each period
df = df.sort_values(
    ["period", "hour"]
).reset_index(drop=True)

# 4. Create forecasting targets
df["dst_t0"] = df.groupby("period")["dst"].shift(-1)
df["dst_t1"] = df.groupby("period")["dst"].shift(-2)

# 5. Remove rows where future targets are unavailable
df = df.dropna(
    subset=["dst_t0", "dst_t1"]
).copy()

# 6. Convert targets to integers
df["dst_t0"] = df["dst_t0"].astype(int)
df["dst_t1"] = df["dst_t1"].astype(int)

# 7. Save forecasting dataset
df.to_csv(OUTPUT_PATH, index=False)

# 8. Display results
print("FORECASTING DATASET CREATED")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(
    df[["period", "hour", "dst", "dst_t0", "dst_t1"]].head()
)

print("\nRows per period:")
print(df.groupby("period").size())

print("\nMissing target values:")
print(df[["dst_t0", "dst_t1"]].isnull().sum())

print("\nSaved to:", OUTPUT_PATH)
