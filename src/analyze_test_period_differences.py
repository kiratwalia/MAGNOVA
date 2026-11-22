
import pandas as pd
import numpy as np
from pathlib import Path

print("\nLoading datasets...")

hourly = pd.read_csv("data/processed/hourly_dataset.csv")
pred = pd.read_csv("reports/final_test_predictions.csv")

output_dir = Path("reports/test_period_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Match test predictions with hourly solar-wind data
# --------------------------------------------------

print("\nMatching test predictions with hourly data...")

data = pred.merge(
    hourly,
    on=["period", "hour", "dst"],
    how="left",
    validate="one_to_one"
)

if len(data) != len(pred):
    raise ValueError("Row count changed after merging!")

if data["bx_gsm_mean"].isna().all():
    raise ValueError("Solar-wind data matching failed!")

print("Merged shape:", data.shape)
print("Unmatched rows:", data["bx_gsm_mean"].isna().sum())

# --------------------------------------------------
# 2. Test-period Dst distribution
# --------------------------------------------------

print("\nTEST PERIOD DST DISTRIBUTION")

dst_results = []

for period, group in data.groupby("period"):

    row = {
        "period": period,
        "samples": len(group),
        "dst_mean": group["dst"].mean(),
        "dst_std": group["dst"].std(),
        "dst_t0_mean": group["dst_t0"].mean(),
        "dst_t1_mean": group["dst_t1"].mean(),
        "moderate_storm": ((group["dst"] <= -50) & (group["dst"] > -100)).sum(),
        "intense_storm": ((group["dst"] <= -100) & (group["dst"] > -200)).sum(),
        "severe_storm": (group["dst"] <= -200).sum()
    }

    dst_results.append(row)

dst_df = pd.DataFrame(dst_results)
print(dst_df.to_string(index=False))
dst_df.to_csv(output_dir / "test_dst_distribution.csv", index=False)

# --------------------------------------------------
# 3. Solar-wind distribution in test segments
# --------------------------------------------------

print("\nTEST PERIOD SOLAR-WIND DISTRIBUTION")

variables = [
    "bx_gsm_mean",
    "by_gsm_mean",
    "bz_gsm_mean",
    "bt_mean",
    "density_mean",
    "speed_mean",
    "temperature_mean"
]

solar_results = []

for period, group in data.groupby("period"):

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

solar_df = pd.DataFrame(solar_results)

print(solar_df.to_string(index=False))
solar_df.to_csv(
    output_dir / "test_solar_wind_distribution.csv",
    index=False
)

# --------------------------------------------------
# 4. Prediction error by period
# --------------------------------------------------

print("\nTEST PERIOD MODEL PERFORMANCE")

performance_results = []

for period, group in data.groupby("period"):

    for target in ["dst_t0", "dst_t1"]:

        actual = group[target]

        for model in ["xgb", "persistence"]:

            prediction = group[f"{target}_{model}_prediction"]

            errors = actual - prediction

            performance_results.append({
                "period": period,
                "target": target,
                "model": model,
                "samples": len(group),
                "rmse": np.sqrt(np.mean(errors ** 2)),
                "mae": np.mean(np.abs(errors))
            })

performance_df = pd.DataFrame(performance_results)

print(performance_df.to_string(index=False))

performance_df.to_csv(
    output_dir / "test_period_model_performance.csv",
    index=False
)

# --------------------------------------------------
# 5. Error by actual Dst magnitude
# --------------------------------------------------

print("\nERROR BY ACTUAL DST MAGNITUDE")

bins = [-np.inf, -200, -100, -50, 0, np.inf]
labels = [
    "Severe (<= -200)",
    "Intense (-200 to -100)",
    "Moderate (-100 to -50)",
    "Mild (-50 to 0)",
    "Non-negative (>= 0)"
]

magnitude_results = []

for target in ["dst_t0", "dst_t1"]:

    data[f"{target}_band"] = pd.cut(
        data[target],
        bins=bins,
        labels=labels,
        right=True
    )

    for band, group in data.groupby(
        f"{target}_band", observed=True
    ):

        if len(group) == 0:
            continue

        actual = group[target]

        for model in ["xgb", "persistence"]:

            prediction = group[f"{target}_{model}_prediction"]
            errors = actual - prediction

            magnitude_results.append({
                "target": target,
                "dst_band": str(band),
                "model": model,
                "samples": len(group),
                "rmse": np.sqrt(np.mean(errors ** 2)),
                "mae": np.mean(np.abs(errors))
            })

magnitude_df = pd.DataFrame(magnitude_results)

print(magnitude_df.to_string(index=False))

magnitude_df.to_csv(
    output_dir / "error_by_dst_magnitude.csv",
    index=False
)

# --------------------------------------------------
# 6. Missing data comparison
# --------------------------------------------------

print("\nMISSING DATA IN TEST SEGMENTS")

missing_results = []

for period, group in data.groupby("period"):

    for variable in variables:

        missing_results.append({
            "period": period,
            "feature": variable,
            "missing_count": group[variable].isna().sum(),
            "missing_percentage": group[variable].isna().mean() * 100
        })

missing_df = pd.DataFrame(missing_results)

print(missing_df.to_string(index=False))

missing_df.to_csv(
    output_dir / "test_missing_data_comparison.csv",
    index=False
)

print("\n" + "=" * 60)
print("TEST PERIOD DIAGNOSTICS COMPLETED")
print("=" * 60)
print("Reports saved in:", output_dir)
