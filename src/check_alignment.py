
import pandas as pd
from collections import Counter

DST_PATH = "data/raw/public/dst_labels.csv"
SOLAR_PATH = "data/raw/public/solar_wind.csv"

print("CHECKING SOLAR WIND AND DST ALIGNMENT")
print("=" * 55)

# Load hourly Dst timestamps
dst = pd.read_csv(DST_PATH, usecols=["period", "timedelta"])

dst["hour"] = (
    pd.to_timedelta(dst["timedelta"])
    .dt.floor("h")
)

expected = set(zip(dst["period"], dst["hour"]))

print("Total Dst hourly records:", len(dst))
print("Expected hourly timestamps:", len(expected))

# Read solar wind timestamps in chunks
counts = Counter()

for chunk in pd.read_csv(
    SOLAR_PATH,
    usecols=["period", "timedelta"],
    chunksize=500_000
):
    chunk["hour"] = (
        pd.to_timedelta(chunk["timedelta"])
        .dt.floor("h")
    )

    grouped = chunk.groupby(
        ["period", "hour"]
    ).size()

    counts.update(grouped.to_dict())

print("\nSOLAR WIND CHECK")
print("Total hourly buckets:", len(counts))

# Compare expected and observed timestamps
observed = set(counts.keys())

missing_hours = expected - observed
extra_hours = observed - expected

# Check number of minute records per hour
record_distribution = Counter(counts.values())

incorrect_hours = sum(
    count for count in counts.values()
    if count != 60
)

print("\nALIGNMENT RESULTS")
print("Missing hourly buckets:", len(missing_hours))
print("Unexpected hourly buckets:", len(extra_hours))
print("Hours with other than 60 records:", incorrect_hours)

print("\nRecords per hour distribution:")
for records, hours in sorted(record_distribution.items()):
    print(f"{records} records: {hours} hours")

print("\nPERIOD-WISE RESULTS")

for period in sorted(dst["period"].unique()):
    period_hours = {
        key: value
        for key, value in counts.items()
        if key[0] == period
    }

    print(
        period,
        "| Dst hours:",
        (dst["period"] == period).sum(),
        "| Solar hours:",
        len(period_hours)
    )

print("\nALIGNMENT CHECK COMPLETED")
