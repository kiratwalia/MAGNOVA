
import os
import json
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =====================================================
# 1. CONFIGURATION
# =====================================================

TRAIN_PATH = "data/processed/train_temporal_prepared.csv"
VAL_PATH = "data/processed/validation_temporal_prepared.csv"
METADATA_PATH = "models/temporal_xgboost_metadata.json"

os.makedirs("models/final", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Best configurations selected using validation RMSE
FINAL_CONFIGS = {
    "dst_t0": {
        "n_estimators": 600,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0
    },

    "dst_t1": {
        "n_estimators": 600,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 2.0
    }
}

COMMON_PARAMS = {
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist"
}

# =====================================================
# 2. LOAD DATA
# =====================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

all_features = metadata["features"]

def get_category(feature):

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


features = [
    feature for feature in all_features
    if get_category(feature) in ["Original", "Lag", "Rolling"]
]

assert len(features) == 83

print("Selected features:", len(features))

X_train = train_df[features]
X_val = val_df[features]

# =====================================================
# 3. TRAIN FINAL CANDIDATE MODELS
# =====================================================

results = []
saved_models = {}

for target, config in FINAL_CONFIGS.items():

    print("\n" + "=" * 65)
    print("TRAINING FINAL CANDIDATE:", target.upper())
    print("=" * 65)

    y_train = train_df[target].to_numpy()
    y_val = val_df[target].to_numpy()

    model = XGBRegressor(
        **COMMON_PARAMS,
        **config
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_pred)
    )

    val_rmse = np.sqrt(
        mean_squared_error(y_val, val_pred)
    )

    val_mae = mean_absolute_error(
        y_val, val_pred
    )

    persistence_pred = val_df["dst"].to_numpy()

    persistence_rmse = np.sqrt(
        mean_squared_error(y_val, persistence_pred)
    )

    persistence_mae = mean_absolute_error(
        y_val, persistence_pred
    )

    model_path = f"models/final/{target}_final.json"

    model.save_model(model_path)

    saved_models[target] = {
        "model_path": model_path,
        "configuration": config
    }

    results.append({
        "target": target,
        "feature_count": len(features),
        "train_rmse": train_rmse,
        "validation_rmse": val_rmse,
        "validation_mae": val_mae,
        "persistence_rmse": persistence_rmse,
        "persistence_mae": persistence_mae
    })

    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")
    print(f"Saved model: {model_path}")

# =====================================================
# 4. SAVE FINAL METADATA AND RESULTS
# =====================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "reports/final_candidate_results.csv",
    index=False
)

final_metadata = {
    "model_type": "XGBoost",
    "experiment": "83-feature temporal model",
    "feature_count": len(features),
    "features": features,
    "targets": FINAL_CONFIGS,
    "models": saved_models,
    "test_data_used": False
}

with open(
    "models/final/final_model_metadata.json",
    "w"
) as f:
    json.dump(final_metadata, f, indent=4)

print("\n" + "=" * 65)
print("FINAL CANDIDATE TRAINING COMPLETED")
print("=" * 65)

print(results_df.to_string(index=False))

print("\nModels saved in: models/final/")
print("Results saved in: reports/final_candidate_results.csv")
print("Test dataset was NOT used.")
