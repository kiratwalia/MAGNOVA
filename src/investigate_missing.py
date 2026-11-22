
import pandas as pd

solar = pd.read_csv("data/raw/public/solar_wind.csv")

solar["timedelta"] = pd.to_timedelta(solar["timedelta"])
solar["hour"] = solar["timedelta"].dt.floor("h")

features = [
    "bx_gsm",
    "by_gsm",
    "bz_gsm",
    "bt",
    "density",
    "speed",
    "temperature"
]

# Count valid minute-level observations per hour
counts = solar.groupby(
    ["period", "hour"]
)[features].count()

print("=" * 65)
print("MISSING DATA INVESTIGATION")
print("=" * 65)

for feature in features:
    print(f"\nFEATURE: {feature}")
    print("-" * 45)

    valid_counts = counts[feature]

    print("Hours with 0 valid readings:",
          (valid_counts == 0).sum())

    print("Hours with 1 valid reading:",
          (valid_counts == 1).sum())

    print("Hours with 2-59 valid readings:",
          ((valid_counts >= 2) & (valid_counts < 60)).sum())

    print("Hours with all 60 readings:",
          (valid_counts == 60).sum())

    print("Total hours:", len(valid_counts))

# Investigate temperature separately
print("\n\nTEMPERATURE MISSING DATA BY PERIOD")
print("=" * 65)

for period, group in counts.groupby(level="period"):
    values = group["temperature"]

    print(
        period,
        "| Zero:", (values == 0).sum(),
        "| One:", (values == 1).sum(),
        "| Partial:", ((values >= 2) & (values < 60)).sum(),
        "| Complete:", (values == 60).sum()
    )

print("\nINVESTIGATION COMPLETED")
