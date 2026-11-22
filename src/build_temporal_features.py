
import pandas as pd
from pathlib import Path

# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "processed" / "hourly_dataset.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "forecasting_temporal_dataset.csv"
)

# ==================================================
# 2. FEATURE CONFIGURATION
# ==================================================

BASE_VARIABLES = [
    "bz_gsm_mean",
    "bz_gsm_min",
    "speed_mean",
    "density_mean",
    "bt_mean"
]

ROLLING_VARIABLES = [
    "bz_gsm_mean",
    "bz_gsm_min",
    "speed_mean",
    "density_mean",
    "bt_mean"
]

LAG_VALUES = [1, 2, 3, 4, 5]

ROLLING_WINDOWS = [6, 12, 24]

# ==================================================
# 3. LOAD DATA
# ==================================================

print("Loading hourly dataset...")

df = pd.read_csv(INPUT_FILE)

df["hour"] = pd.to_timedelta(df["hour"])

df = df.sort_values(
    ["period", "hour"]
).reset_index(drop=True)

original_feature_columns = [
    col for col in df.columns
    if col not in ["period", "hour", "dst"]
]

print("Original dataset shape:", df.shape)
print("Original features:", len(original_feature_columns))

# ==================================================
# 4. MISSINGNESS INDICATORS
# Generate before imputation
# ==================================================

print("\nCreating missingness indicators...")

df["missing_feature_count"] = (
    df[original_feature_columns]
    .isnull()
    .sum(axis=1)
)

df["missing_feature_fraction"] = (
    df["missing_feature_count"]
    / len(original_feature_columns)
)

for variable in BASE_VARIABLES:

    df[f"{variable}_missing"] = (
        df[variable].isnull().astype(int)
    )

# ==================================================
# 5. LAG FEATURES
# Calculated separately within each period
# ==================================================

print("Creating lag features...")

for variable in BASE_VARIABLES:

    for lag in LAG_VALUES:

        df[f"{variable}_lag_{lag}"] = (
            df.groupby("period")[variable]
            .shift(lag)
        )

# ==================================================
# 6. ROLLING FEATURES
# Current hour + previous observations
# ==================================================

print("Creating rolling features...")

for variable in ROLLING_VARIABLES:

    grouped = df.groupby("period")[variable]

    for window in ROLLING_WINDOWS:

        df[f"{variable}_rolling_mean_{window}h"] = (
            grouped.transform(
                lambda series: series.rolling(
                    window=window,
                    min_periods=1
                ).mean()
            )
        )

        df[f"{variable}_rolling_min_{window}h"] = (
            grouped.transform(
                lambda series: series.rolling(
                    window=window,
                    min_periods=1
                ).min()
            )
        )

# ==================================================
# 7. PHYSICS-INSPIRED FEATURES
# ==================================================

print("Creating physics-inspired features...")

# Southward Bz proxy
df["southward_bz"] = (
    -df["bz_gsm_mean"]
).clip(lower=0)

# Solar-wind electric-field-related proxy
df["vbs_proxy"] = (
    df["speed_mean"] * df["southward_bz"]
)

# Dynamic-pressure-related proxy
df["dynamic_pressure_proxy"] = (
    df["density_mean"] * (df["speed_mean"] ** 2)
)

# ==================================================
# 8. FORECASTING TARGETS
# ==================================================

print("Creating forecasting targets...")

df["dst_t0"] = (
    df.groupby("period")["dst"].shift(-1)
)

df["dst_t1"] = (
    df.groupby("period")["dst"].shift(-2)
)

# ==================================================
# 9. REMOVE ONLY INCOMPLETE HISTORY
# ==================================================

print("Removing incomplete historical lookback...")

# First 23 rows do not have 24 historical hourly records
df["_period_row"] = (
    df.groupby("period").cumcount()
)

df = df[
    df["_period_row"] >= 23
].copy()

# Remove only rows without future targets
df = df.dropna(
    subset=["dst_t0", "dst_t1"]
).copy()

df["dst_t0"] = df["dst_t0"].astype(int)
df["dst_t1"] = df["dst_t1"].astype(int)

df = df.drop(
    columns=["_period_row"]
)

df = df.reset_index(drop=True)

# ==================================================
# 10. SAVE DATASET
# ==================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ==================================================
# 11. FINAL REPORT
# ==================================================

excluded_columns = [
    "period",
    "hour",
    "dst",
    "dst_t0",
    "dst_t1"
]

feature_columns = [
    col for col in df.columns
    if col not in excluded_columns
]

print("\n" + "=" * 60)
print("TEMPORAL FORECASTING DATASET CREATED")
print("=" * 60)

print("\nFinal dataset shape:", df.shape)

print("Total ML features:", len(feature_columns))

print("\nRows per period:")
print(df.groupby("period").size())

print("\nMissing target values:")
print(df[["dst_t0", "dst_t1"]].isnull().sum())

print(
    "\nRemaining missing feature values:",
    df[feature_columns].isnull().sum().sum()
)

print("\nFeature categories:")

print("Original features:", len(original_feature_columns))

print(
    "Lag features:",
    len(BASE_VARIABLES) * len(LAG_VALUES)
)

print(
    "Rolling features:",
    len(ROLLING_VARIABLES) * len(ROLLING_WINDOWS) * 2
)

print("Physics-inspired features: 3")

print(
    "Missingness indicators:",
    2 + len(BASE_VARIABLES)
)

print("\nSaved to:", OUTPUT_FILE)

# ==================================================
# 12. SAFETY CHECKS
# ==================================================

assert df["dst_t0"].notnull().all()
assert df["dst_t1"].notnull().all()

assert len(feature_columns) == 93

assert (
    df.groupby("period").size() > 0
).all()

assert (
    df.groupby("period")["hour"]
    .apply(lambda x: x.is_monotonic_increasing)
).all()

print("\nAll safety checks passed!")

print("\nFEATURE GENERATION COMPLETED SUCCESSFULLY!")
