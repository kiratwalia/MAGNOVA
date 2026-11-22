
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/hourly_dataset.csv"
REPORT_DIR = "reports/period_analysis"

os.makedirs(REPORT_DIR, exist_ok=True)

print("\nLoading hourly dataset...")

df = pd.read_csv(DATA_PATH)

# Original seven solar-wind variables
variables = [
    "bx_gsm_mean",
    "by_gsm_mean",
    "bz_gsm_mean",
    "bt_mean",
    "density_mean",
    "speed_mean",
    "temperature_mean"
]

# =====================================================
# 1. Dst DISTRIBUTION
# =====================================================

dst_results = []

for period, group in df.groupby("period"):

    dst = group["dst"]

    dst_results.append({
        "period": period,
        "samples": len(group),
        "dst_mean": dst.mean(),
        "dst_std": dst.std(),
        "dst_min": dst.min(),
        "dst_max": dst.max(),
        "moderate_storm": ((dst <= -50) & (dst > -100)).sum(),
        "intense_storm": ((dst <= -100) & (dst > -200)).sum(),
        "severe_storm": (dst <= -200).sum()
    })

dst_summary = pd.DataFrame(dst_results)

dst_summary.to_csv(
    f"{REPORT_DIR}/dst_distribution.csv",
    index=False
)

print("\nDST DISTRIBUTION")
print(dst_summary.to_string(index=False))

# =====================================================
# 2. SOLAR-WIND DISTRIBUTIONS
# =====================================================

solar_results = []

for period, group in df.groupby("period"):

    for variable in variables:

        values = group[variable]

        solar_results.append({
            "period": period,
            "feature": variable,
            "mean": values.mean(),
            "std": values.std(),
            "median": values.median(),
            "missing_percentage": values.isna().mean() * 100
        })

solar_summary = pd.DataFrame(solar_results)

solar_summary.to_csv(
    f"{REPORT_DIR}/solar_wind_distribution.csv",
    index=False
)

print("\nSOLAR-WIND DISTRIBUTION")
print(solar_summary.to_string(index=False))

# =====================================================
# 3. Dst BOXPLOT
# =====================================================

plt.figure(figsize=(10, 6))

data = [
    df[df["period"] == period]["dst"].dropna()
    for period in sorted(df["period"].unique())
]

plt.boxplot(
    data,
    tick_labels=sorted(df["period"].unique()),
    showfliers=False
)

plt.axhline(-50, linestyle="--", label="Storm threshold")
plt.title("Dst Distribution Across Historical Periods")
plt.ylabel("Dst (nT)")
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{REPORT_DIR}/dst_period_boxplot.png",
    dpi=300
)

plt.close()

# =====================================================
# 4. SOLAR-WIND FEATURE BOXPLOTS
# =====================================================

for variable in variables:

    plt.figure(figsize=(10, 6))

    data = [
        df[df["period"] == period][variable].dropna()
        for period in sorted(df["period"].unique())
    ]

    plt.boxplot(
        data,
        tick_labels=sorted(df["period"].unique()),
        showfliers=False
    )

    plt.title(f"{variable} Distribution Across Periods")
    plt.ylabel(variable)
    plt.tight_layout()

    plt.savefig(
        f"{REPORT_DIR}/{variable}_period_boxplot.png",
        dpi=300
    )

    plt.close()

# =====================================================
# 5. MISSING DATA COMPARISON
# =====================================================

missing_results = []

for period, group in df.groupby("period"):

    for variable in variables:

        missing_results.append({
            "period": period,
            "feature": variable,
            "missing_count": int(group[variable].isna().sum()),
            "missing_percentage": group[variable].isna().mean() * 100
        })

missing_summary = pd.DataFrame(missing_results)

missing_summary.to_csv(
    f"{REPORT_DIR}/missing_data_comparison.csv",
    index=False
)

print("\nMISSING DATA COMPARISON")
print(missing_summary.to_string(index=False))

print("\n" + "=" * 65)
print("PERIOD DISTRIBUTION ANALYSIS COMPLETED")
print("=" * 65)

print("Reports saved in:", REPORT_DIR)
