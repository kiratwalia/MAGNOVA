
import pandas as pd
import os

DATA_PATH = "data/raw/public"

files = [
    "dst_labels.csv",
    "solar_wind.csv",
    "satellite_positions.csv",
    "sunspots.csv"
]

for file in files:
    path = os.path.join(DATA_PATH, file)

    df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print("FILE:", file)
    print("=" * 60)

    # Dataset dimensions
    print("Total rows:", len(df))
    print("Total columns:", len(df.columns))

    # Column data types
    print("\nData types:")
    print(df.dtypes)

    # Missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Records per period
    print("\nRecords per period:")
    print(df["period"].value_counts().sort_index())

    # Time range
    df["timedelta"] = pd.to_timedelta(df["timedelta"])

    print("\nTime range:")
    print("Start:", df["timedelta"].min())
    print("End:", df["timedelta"].max())

    # Duplicate records
    print("\nDuplicate rows:", df.duplicated().sum())

print("\nDATASET AUDIT COMPLETED")
