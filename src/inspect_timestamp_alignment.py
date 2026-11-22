
import pandas as pd

SOLAR_PATH = "data/raw/public/solar_wind.csv"
DST_PATH = "data/raw/public/dst_labels.csv"

solar = pd.read_csv(SOLAR_PATH)
dst = pd.read_csv(DST_PATH)

solar["timedelta"] = pd.to_timedelta(solar["timedelta"])
dst["timedelta"] = pd.to_timedelta(dst["timedelta"])

solar["hour"] = solar["timedelta"].dt.floor("h")
dst["hour"] = dst["timedelta"].dt.floor("h")

# --------------------------------------------------
# 1. Solar-wind records per hourly bucket
# --------------------------------------------------

counts = (
    solar.groupby(["period", "hour"])
    .size()
)

print("\nSOLAR-WIND RECORDS PER HOUR")
print("-----------------------------------")
print(counts.describe())

print("\nHourly bucket counts:")
print(counts.value_counts().sort_index())

# --------------------------------------------------
# 2. First and last timestamps
# --------------------------------------------------

print("\nFIRST AND LAST SOLAR TIMESTAMPS")
print("-----------------------------------")

for period in solar["period"].unique():

    part = solar[solar["period"] == period]

    print(f"\n{period}")
    print("First:", part["timedelta"].min())
    print("Last:", part["timedelta"].max())

# --------------------------------------------------
# 3. Dst timestamp intervals
# --------------------------------------------------

print("\nDST TIMESTAMP INTERVALS")
print("-----------------------------------")

for period in dst["period"].unique():

    part = dst[dst["period"] == period].sort_values("timedelta")

    differences = part["timedelta"].diff().dropna()

    print(f"\n{period}")
    print(differences.value_counts().head())

# --------------------------------------------------
# 4. Compare solar and Dst hourly coverage
# --------------------------------------------------

solar_hours = set(
    zip(solar["period"], solar["hour"])
)

dst_hours = set(
    zip(dst["period"], dst["hour"])
)

print("\nHOURLY COVERAGE")
print("-----------------------------------")
print("Solar hourly buckets:", len(solar_hours))
print("Dst hourly labels:", len(dst_hours))
print("Common buckets:", len(solar_hours & dst_hours))
print("Solar-only buckets:", len(solar_hours - dst_hours))
print("Dst-only buckets:", len(dst_hours - solar_hours))

print("\nTIMESTAMP DIAGNOSTIC COMPLETED")
