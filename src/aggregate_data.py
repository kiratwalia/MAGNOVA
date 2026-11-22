
import pandas as pd

SOLAR_PATH = "data/raw/public/solar_wind.csv"
DST_PATH = "data/raw/public/dst_labels.csv"

FEATURES = [
    "bx_gsm",
    "by_gsm",
    "bz_gsm",
    "bt",
    "density",
    "speed",
    "temperature"
]

print("Loading solar wind data...")

solar = pd.read_csv(SOLAR_PATH)

# Convert timedelta into pandas duration
solar["timedelta"] = pd.to_timedelta(solar["timedelta"])

# Create hourly timestamp
solar["hour"] = solar["timedelta"].dt.floor("h")

print("Aggregating minute-level measurements...")

# Aggregate measurements separately for every period and hour
hourly = solar.groupby(
    ["period", "hour"]
)[FEATURES].agg(["mean", "min", "max", "std"])

# Flatten column names
hourly.columns = [
    f"{column}_{stat}"
    for column, stat in hourly.columns
]

hourly = hourly.reset_index()

print("Loading Dst labels...")

dst = pd.read_csv(DST_PATH)

dst["timedelta"] = pd.to_timedelta(dst["timedelta"])

dst["hour"] = dst["timedelta"].dt.floor("h")

# Merge hourly features with target
dataset = pd.merge(
    hourly,
    dst[["period", "hour", "dst"]],
    on=["period", "hour"],
    how="inner",
    validate="one_to_one"
)

# Save processed dataset
output_path = "data/processed/hourly_dataset.csv"

dataset.to_csv(output_path, index=False)

print("\nAGGREGATION COMPLETED")
print("Dataset shape:", dataset.shape)
print("Saved to:", output_path)

print("\nFirst 5 rows:")
print(dataset.head())

print("\nMissing values:")
print(dataset.isnull().sum().sum())
