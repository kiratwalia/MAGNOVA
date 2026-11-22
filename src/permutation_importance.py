
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
N_REPEATS = 3
RANDOM_STATE = 42

os.makedirs(REPORT_DIR, exist_ok=True)

# =====================================================
# 2. LOAD DATA AND METADATA
# =====================================================

print("\nLoading validation data and metadata...")

val_df = pd.read_csv(VAL_PATH)

with open(
    f"{MODEL_DIR}/temporal_xgboost_metadata.json", "r"
) as f:
    metadata = json.load(f)

feature_columns = metadata["features"]

excluded = [
    "period", "hour", "dst",
    "dst_t0", "dst_t1", "split"
]

X_val = val_df[feature_columns].copy()

assert len(feature_columns) == 93
assert list(X_val.columns) == feature_columns
assert not set(excluded).intersection(X_val.columns)

print("Validation shape:", val_df.shape)
print("Number of features:", len(feature_columns))
print("Permutation repeats:", N_REPEATS)

# =====================================================
# 3. FEATURE CATEGORY
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

    if feature.startswith("missing_"):
        return "Missingness"

    return "Original"

# =====================================================
# 4. PERMUTATION IMPORTANCE
# =====================================================

all_results = []

for target in TARGETS:

    print("\n" + "=" * 65)
    print("PERMUTATION IMPORTANCE:", target.upper())
    print("=" * 65)

    y_val = val_df[target].to_numpy()

    # Load existing model
    model = XGBRegressor()

    model.load_model(
        f"{MODEL_DIR}/temporal_xgboost_{target}.json"
    )

    # Original validation performance
    baseline_pred = model.predict(X_val)

    baseline_rmse = np.sqrt(
        mean_squared_error(y_val, baseline_pred)
    )

    print(f"\nBaseline RMSE: {baseline_rmse:.4f}")

    target_results = []

    rng = np.random.default_rng(RANDOM_STATE)

    for feature in feature_columns:

        rmse_increases = []

        for repeat in range(N_REPEATS):

            shuffled_X = X_val.copy()

            shuffled_X[feature] = rng.permutation(
                X_val[feature].to_numpy()
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
            "feature": feature,
            "category": get_feature_category(feature),
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

    print("\nTOP 20 FEATURES:")
    print(
        results_df.head(20).round(5).to_string(index=False)
    )

    results_df.to_csv(
        f"{REPORT_DIR}/{target}_permutation_importance.csv",
        index=False
    )

    # Plot top 20
    top20 = results_df.head(20).sort_values(
        "mean_rmse_increase"
    )

    plt.figure(figsize=(11, 8))

    plt.barh(
        top20["feature"],
        top20["mean_rmse_increase"],
        xerr=top20["std_rmse_increase"],
        capsize=3
    )

    plt.axvline(0, linewidth=1)

    plt.xlabel("Increase in Validation RMSE (nT)")
    plt.ylabel("Feature")

    plt.title(
        f"Permutation Importance - {target.upper()}"
    )

    plt.tight_layout()

    plt.savefig(
        f"{REPORT_DIR}/{target}_permutation_importance.png",
        dpi=300
    )

    plt.close()

# =====================================================
# 5. COMBINED RESULTS
# =====================================================

combined = pd.concat(
    all_results,
    ignore_index=True
)

combined.to_csv(
    f"{REPORT_DIR}/all_permutation_importance.csv",
    index=False
)

print("\n" + "=" * 65)
print("PERMUTATION IMPORTANCE ANALYSIS COMPLETED")
print("=" * 65)

print("\nSaved reports:")
print("- dst_t0_permutation_importance.csv")
print("- dst_t1_permutation_importance.csv")
print("- all_permutation_importance.csv")
print("- T+0 and T+1 PNG plots")

print("\nNo model retraining performed.")
print("No test data used.")
