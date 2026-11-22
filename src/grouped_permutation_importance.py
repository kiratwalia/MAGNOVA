
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# =====================================================
# 1. CONFIGURATION
# =====================================================

VAL_PATH = "data/processed/validation_temporal_prepared.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"

TARGETS = ["dst_t0", "dst_t1"]
N_REPEATS = 5
RANDOM_STATE = 42

os.makedirs(REPORT_DIR, exist_ok=True)

# =====================================================
# 2. LOAD DATA
# =====================================================

val_df = pd.read_csv(VAL_PATH)

with open(
    f"{MODEL_DIR}/temporal_xgboost_metadata.json", "r"
) as f:
    metadata = json.load(f)

feature_columns = metadata["features"]
X_val = val_df[feature_columns].copy()

assert len(feature_columns) == 93
assert X_val.isnull().sum().sum() == 0

# =====================================================
# 3. FEATURE GROUPS
# =====================================================

def get_feature_category(feature):

    if "_lag_" in feature:
        return "Lag"

    if "_rolling_" in feature:
        return "Rolling"

    if feature in [
        "southward_bz",
        "vbs_proxy",
        "dynamic_pressure_proxy"
    ]:
        return "Physics"

    if feature.startswith("missing_") or feature.endswith("_missing"):
        return "Missingness"

    return "Original"


groups = {}

for feature in feature_columns:

    category = get_feature_category(feature)

    groups.setdefault(category, []).append(feature)

print("\nFEATURE GROUP COUNTS")

for category, features in groups.items():
    print(f"{category}: {len(features)}")

assert sum(len(features) for features in groups.values()) == 93

# =====================================================
# 4. GROUPED PERMUTATION IMPORTANCE
# =====================================================

all_results = []

for target_index, target in enumerate(TARGETS):

    print("\n" + "=" * 65)
    print("GROUPED PERMUTATION:", target.upper())
    print("=" * 65)

    y_val = val_df[target].to_numpy()

    model = XGBRegressor()
    model.load_model(
        f"{MODEL_DIR}/temporal_xgboost_{target}.json"
    )

    baseline_pred = model.predict(X_val)

    baseline_rmse = np.sqrt(
        mean_squared_error(y_val, baseline_pred)
    )

    print(f"Baseline RMSE: {baseline_rmse:.4f}")

    rng = np.random.default_rng(
        RANDOM_STATE + target_index
    )

    target_results = []

    for category, features in groups.items():

        rmse_increases = []

        for repeat in range(N_REPEATS):

            shuffled_X = X_val.copy()

            # Same row permutation for every feature
            # in this group preserves within-group relationships.
            permutation = rng.permutation(len(X_val))

            shuffled_X.loc[:, features] = (
                X_val[features].iloc[permutation].to_numpy()
            )

            shuffled_pred = model.predict(shuffled_X)

            shuffled_rmse = np.sqrt(
                mean_squared_error(y_val, shuffled_pred)
            )

            rmse_increases.append(
                shuffled_rmse - baseline_rmse
            )

        mean_increase = np.mean(rmse_increases)
        std_increase = np.std(rmse_increases, ddof=1)

        target_results.append({
            "target": target,
            "category": category,
            "feature_count": len(features),
            "baseline_rmse": baseline_rmse,
            "mean_rmse_increase": mean_increase,
            "std_rmse_increase": std_increase
        })

    results_df = pd.DataFrame(target_results)

    results_df = results_df.sort_values(
        "mean_rmse_increase",
        ascending=False
    )

    all_results.append(results_df)

    print("\nGROUPED IMPORTANCE RESULTS:")

    print(
        results_df.round(5).to_string(index=False)
    )

    results_df.to_csv(
        f"{REPORT_DIR}/{target}_grouped_permutation.csv",
        index=False
    )

    # Plot
    plot_df = results_df.sort_values(
        "mean_rmse_increase"
    )

    plt.figure(figsize=(9, 5))

    plt.barh(
        plot_df["category"],
        plot_df["mean_rmse_increase"],
        xerr=plot_df["std_rmse_increase"],
        capsize=4
    )

    plt.axvline(0, linewidth=1)

    plt.xlabel("Increase in Validation RMSE (nT)")
    plt.ylabel("Feature Group")

    plt.title(
        f"Grouped Permutation Importance - {target.upper()}"
    )

    plt.tight_layout()

    plt.savefig(
        f"{REPORT_DIR}/{target}_grouped_permutation.png",
        dpi=300
    )

    plt.close()

# =====================================================
# 5. SAVE COMBINED RESULTS
# =====================================================

combined = pd.concat(all_results, ignore_index=True)

combined.to_csv(
    f"{REPORT_DIR}/grouped_permutation_results.csv",
    index=False
)

print("\n" + "=" * 65)
print("GROUPED PERMUTATION ANALYSIS COMPLETED")
print("=" * 65)

print("\nSaved:")
print("- reports/dst_t0_grouped_permutation.csv")
print("- reports/dst_t1_grouped_permutation.csv")
print("- reports/grouped_permutation_results.csv")
print("- T+0 and T+1 grouped importance plots")

print("\nExisting models were not retrained.")
print("Test data was not used.")
