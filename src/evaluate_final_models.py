
import os
import json
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =====================================================
# 1. CONFIGURATION
# =====================================================

TEST_PATH = "data/processed/test_temporal_prepared.csv"
METADATA_PATH = "models/final/final_model_metadata.json"

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

TARGETS = ["dst_t0", "dst_t1"]

# =====================================================
# 2. LOAD TEST DATA AND FINAL MODELS
# =====================================================

print("\nLoading untouched test dataset...")

test_df = pd.read_csv(TEST_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

features = metadata["features"]

print("Test dataset shape:", test_df.shape)
print("Selected features:", len(features))

assert len(features) == 83
assert test_df[features].isna().sum().sum() == 0

# =====================================================
# 3. METRIC FUNCTION
# =====================================================

def calculate_metrics(actual, predicted):

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mae = mean_absolute_error(
        actual, predicted
    )

    return rmse, mae


def calculate_subset_metrics(actual, predicted, mask):

    if mask.sum() == 0:
        return None, None, 0

    rmse, mae = calculate_metrics(
        actual[mask],
        predicted[mask]
    )

    return rmse, mae, int(mask.sum())


# =====================================================
# 4. LOAD MODELS AND GENERATE PREDICTIONS
# =====================================================

X_test = test_df[features]

prediction_df = test_df[
    ["period", "hour", "dst", "dst_t0", "dst_t1"]
].copy()

results = []

for target in TARGETS:

    print("\n" + "=" * 65)
    print("EVALUATING:", target.upper())
    print("=" * 65)

    model_path = metadata["models"][target]["model_path"]

    model = XGBRegressor()
    model.load_model(model_path)

    actual = test_df[target].to_numpy()

    # XGBoost predictions
    predicted = model.predict(X_test)

    # Persistence predictions
    persistence_pred = test_df["dst"].to_numpy()

    # Overall metrics
    xgb_rmse, xgb_mae = calculate_metrics(
        actual, predicted
    )

    persistence_rmse, persistence_mae = calculate_metrics(
        actual, persistence_pred
    )

    print("\nOVERALL PERFORMANCE")

    print(
        f"XGBoost RMSE: {xgb_rmse:.4f} | "
        f"MAE: {xgb_mae:.4f}"
    )

    print(
        f"Persistence RMSE: {persistence_rmse:.4f} | "
        f"MAE: {persistence_mae:.4f}"
    )

    # Storm samples: target Dst <= -50 nT
    storm_mask = actual <= -50

    storm_xgb_rmse, storm_xgb_mae, storm_count = (
        calculate_subset_metrics(
            actual, predicted, storm_mask
        )
    )

    storm_p_rmse, storm_p_mae, _ = (
        calculate_subset_metrics(
            actual, persistence_pred, storm_mask
        )
    )

    print("\nSTORM PERFORMANCE (DST <= -50 nT)")
    print("Storm samples:", storm_count)

    if storm_count > 0:
        print(
            f"XGBoost RMSE: {storm_xgb_rmse:.4f} | "
            f"MAE: {storm_xgb_mae:.4f}"
        )

        print(
            f"Persistence RMSE: {storm_p_rmse:.4f} | "
            f"MAE: {storm_p_mae:.4f}"
        )

    # Storm onset:
    # Current Dst > -50 and target Dst <= -50
    onset_mask = (
        (test_df["dst"].to_numpy() > -50) &
        (actual <= -50)
    )

    onset_xgb_rmse, onset_xgb_mae, onset_count = (
        calculate_subset_metrics(
            actual, predicted, onset_mask
        )
    )

    onset_p_rmse, onset_p_mae, _ = (
        calculate_subset_metrics(
            actual, persistence_pred, onset_mask
        )
    )

    print("\nSTORM ONSET PERFORMANCE")
    print("Onset samples:", onset_count)

    if onset_count > 0:
        print(
            f"XGBoost RMSE: {onset_xgb_rmse:.4f} | "
            f"MAE: {onset_xgb_mae:.4f}"
        )

        print(
            f"Persistence RMSE: {onset_p_rmse:.4f} | "
            f"MAE: {onset_p_mae:.4f}"
        )

    # Save predictions
    prediction_df[f"{target}_xgb_prediction"] = predicted
    prediction_df[f"{target}_persistence_prediction"] = persistence_pred

    results.append({
        "target": target,
        "test_samples": len(actual),

        "xgb_rmse": xgb_rmse,
        "xgb_mae": xgb_mae,

        "persistence_rmse": persistence_rmse,
        "persistence_mae": persistence_mae,

        "storm_samples": storm_count,
        "storm_xgb_rmse": storm_xgb_rmse,
        "storm_xgb_mae": storm_xgb_mae,
        "storm_persistence_rmse": storm_p_rmse,
        "storm_persistence_mae": storm_p_mae,

        "onset_samples": onset_count,
        "onset_xgb_rmse": onset_xgb_rmse,
        "onset_xgb_mae": onset_xgb_mae,
        "onset_persistence_rmse": onset_p_rmse,
        "onset_persistence_mae": onset_p_mae
    })


# =====================================================
# 5. SAVE RESULTS
# =====================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(REPORT_DIR, "final_test_results.csv"),
    index=False
)

prediction_df.to_csv(
    os.path.join(REPORT_DIR, "final_test_predictions.csv"),
    index=False
)

print("\n" + "=" * 65)
print("FINAL TEST EVALUATION COMPLETED")
print("=" * 65)

print(results_df.to_string(index=False))

print("\nSaved:")
print("reports/final_test_results.csv")
print("reports/final_test_predictions.csv")

print("\nNo training or tuning was performed.")
print("Test dataset was used only for final evaluation.")
