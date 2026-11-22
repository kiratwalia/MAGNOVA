
import os
import json
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# =====================================================
# 1. PATHS AND CONFIGURATION
# =====================================================

TRAIN_PATH = "data/processed/train_temporal_prepared.csv"
VAL_PATH = "data/processed/validation_temporal_prepared.csv"

MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TARGETS = ["dst_t0", "dst_t1"]

PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist"
}


# =====================================================
# 2. LOAD DATA
# =====================================================

print("\nLoading prepared datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

print("Training shape:", train_df.shape)
print("Validation shape:", val_df.shape)


# =====================================================
# 3. SELECT FEATURES AND TARGETS
# =====================================================

excluded_columns = [
    "period", "hour", "dst",
    "dst_t0", "dst_t1", "split"
]

feature_columns = [
    col for col in train_df.columns
    if col not in excluded_columns
]

X_train = train_df[feature_columns]
X_val = val_df[feature_columns]

print("\nNumber of features:", len(feature_columns))

assert len(feature_columns) == 93
assert list(X_train.columns) == list(X_val.columns)
assert "dst" not in X_train.columns


# =====================================================
# 4. TRAIN AND EVALUATE
# =====================================================

results = []
predictions = val_df[
    ["period", "hour", "dst", "dst_t0", "dst_t1"]
].copy()

for target in TARGETS:

    print("\n" + "=" * 60)
    print(f"TRAINING TEMPORAL XGBOOST FOR {target.upper()}")
    print("=" * 60)

    y_train = train_df[target]
    y_val = val_df[target]

    model = XGBRegressor(**PARAMS)

    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    # Persistence baseline
    persistence_pred = val_df["dst"].to_numpy()

    # Save model
    model_path = os.path.join(
        MODEL_DIR,
        f"temporal_xgboost_{target}.json"
    )

    model.save_model(model_path)

    print("Model saved:", model_path)

    # -------------------------------------------------
    # METRIC FUNCTION
    # -------------------------------------------------

    def calculate_metrics(y_true, y_pred):

        rmse = np.sqrt(
            mean_squared_error(y_true, y_pred)
        )

        mae = mean_absolute_error(
            y_true, y_pred
        )

        return rmse, mae

    # -------------------------------------------------
    # OVERALL METRICS
    # -------------------------------------------------

    xgb_rmse, xgb_mae = calculate_metrics(
        y_val, val_pred
    )

    persistence_rmse, persistence_mae = calculate_metrics(
        y_val, persistence_pred
    )

    train_rmse, train_mae = calculate_metrics(
        y_train, train_pred
    )

    print("\nOVERALL PERFORMANCE")
    print("-" * 45)

    print(
        f"XGBoost     RMSE: {xgb_rmse:.4f} | "
        f"MAE: {xgb_mae:.4f}"
    )

    print(
        f"Persistence RMSE: {persistence_rmse:.4f} | "
        f"MAE: {persistence_mae:.4f}"
    )

    print(
        f"Training    RMSE: {train_rmse:.4f} | "
        f"MAE: {train_mae:.4f}"
    )

    # -------------------------------------------------
    # STORM PERIOD METRICS
    # Actual target Dst <= -50 nT
    # -------------------------------------------------

    storm_mask = y_val.to_numpy() <= -50

    print("\nSTORM PERIOD PERFORMANCE")
    print("-" * 45)

    if storm_mask.sum() > 0:

        storm_xgb_rmse, storm_xgb_mae = calculate_metrics(
            y_val.to_numpy()[storm_mask],
            val_pred[storm_mask]
        )

        storm_p_rmse, storm_p_mae = calculate_metrics(
            y_val.to_numpy()[storm_mask],
            persistence_pred[storm_mask]
        )

        print("Storm samples:", storm_mask.sum())

        print(
            f"XGBoost     RMSE: {storm_xgb_rmse:.4f} | "
            f"MAE: {storm_xgb_mae:.4f}"
        )

        print(
            f"Persistence RMSE: {storm_p_rmse:.4f} | "
            f"MAE: {storm_p_mae:.4f}"
        )

    else:
        print("No storm samples found.")

        storm_xgb_rmse = np.nan
        storm_xgb_mae = np.nan
        storm_p_rmse = np.nan
        storm_p_mae = np.nan

    # -------------------------------------------------
    # STORM ONSET METRICS
    # Current Dst > -50 and future Dst <= -50
    # -------------------------------------------------

    onset_mask = (
        (val_df["dst"].to_numpy() > -50) &
        (y_val.to_numpy() <= -50)
    )

    print("\nSTORM ONSET PERFORMANCE")
    print("-" * 45)

    if onset_mask.sum() > 0:

        onset_xgb_rmse, onset_xgb_mae = calculate_metrics(
            y_val.to_numpy()[onset_mask],
            val_pred[onset_mask]
        )

        onset_p_rmse, onset_p_mae = calculate_metrics(
            y_val.to_numpy()[onset_mask],
            persistence_pred[onset_mask]
        )

        print("Storm onset samples:", onset_mask.sum())

        print(
            f"XGBoost     RMSE: {onset_xgb_rmse:.4f} | "
            f"MAE: {onset_xgb_mae:.4f}"
        )

        print(
            f"Persistence RMSE: {onset_p_rmse:.4f} | "
            f"MAE: {onset_p_mae:.4f}"
        )

    else:
        print("No storm onset samples found.")

        onset_xgb_rmse = np.nan
        onset_xgb_mae = np.nan
        onset_p_rmse = np.nan
        onset_p_mae = np.nan

    # -------------------------------------------------
    # STORE RESULTS
    # -------------------------------------------------

    results.append({
        "target": target,
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "xgb_val_rmse": xgb_rmse,
        "xgb_val_mae": xgb_mae,
        "persistence_rmse": persistence_rmse,
        "persistence_mae": persistence_mae,
        "storm_samples": int(storm_mask.sum()),
        "storm_xgb_rmse": storm_xgb_rmse,
        "storm_xgb_mae": storm_xgb_mae,
        "storm_persistence_rmse": storm_p_rmse,
        "storm_persistence_mae": storm_p_mae,
        "onset_samples": int(onset_mask.sum()),
        "onset_xgb_rmse": onset_xgb_rmse,
        "onset_xgb_mae": onset_xgb_mae,
        "onset_persistence_rmse": onset_p_rmse,
        "onset_persistence_mae": onset_p_mae
    })

    # Store validation predictions
    predictions[f"{target}_xgb_pred"] = val_pred
    predictions[f"{target}_persistence_pred"] = persistence_pred


# =====================================================
# 5. SAVE RESULTS
# =====================================================

results_df = pd.DataFrame(results)

results_path = os.path.join(
    REPORT_DIR,
    "temporal_xgboost_results.csv"
)

predictions_path = os.path.join(
    REPORT_DIR,
    "temporal_validation_predictions.csv"
)

results_df.to_csv(results_path, index=False)
predictions.to_csv(predictions_path, index=False)

# Save feature metadata
metadata = {
    "model": "Temporal XGBoost",
    "number_of_features": len(feature_columns),
    "features": feature_columns,
    "parameters": PARAMS
}

with open(
    os.path.join(MODEL_DIR, "temporal_xgboost_metadata.json"),
    "w"
) as f:
    json.dump(metadata, f, indent=4)


# =====================================================
# 6. FINAL SUMMARY
# =====================================================

print("\n" + "=" * 60)
print("TEMPORAL XGBOOST EXPERIMENT COMPLETED")
print("=" * 60)

print("\nFINAL RESULTS:")
print(
    results_df[
        [
            "target",
            "xgb_val_rmse",
            "persistence_rmse",
            "xgb_val_mae",
            "persistence_mae"
        ]
    ].to_string(index=False)
)

print("\nSaved files:")
print("-", results_path)
print("-", predictions_path)
print("- Temporal XGBoost models in models/")

print("\nTEST DATA WAS NOT USED.")
