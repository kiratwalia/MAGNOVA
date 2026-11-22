
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

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

TARGETS = ["dst_t0", "dst_t1"]

# Same feature groups selected in Experiment D
SELECTED_CATEGORIES = ["Original", "Lag", "Rolling"]

# Controlled hyperparameter experiments
CONFIGS = {
    "Baseline": {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0
    },

    "Shallow_Trees": {
        "n_estimators": 600,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0
    },

    "Deeper_Trees": {
        "n_estimators": 500,
        "max_depth": 7,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0
    },

    "Strong_Regularization": {
        "n_estimators": 600,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0
    },

    "More_Sampling": {
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
    if get_category(feature) in SELECTED_CATEGORIES
]

print("\nSelected features:", len(features))

assert len(features) == 83, (
    f"Expected 83 features, found {len(features)}"
)

X_train = train_df[features]
X_val = val_df[features]

# =====================================================
# 3. TRAIN AND EVALUATE
# =====================================================

results = []

for target in TARGETS:

    print("\n" + "=" * 65)
    print("TARGET:", target.upper())
    print("=" * 65)

    y_train = train_df[target].to_numpy()
    y_val = val_df[target].to_numpy()

    persistence_pred = val_df["dst"].to_numpy()

    persistence_rmse = np.sqrt(
        mean_squared_error(y_val, persistence_pred)
    )

    persistence_mae = mean_absolute_error(
        y_val, persistence_pred
    )

    print(
        f"Persistence RMSE: {persistence_rmse:.4f}"
    )

    for config_name, config in CONFIGS.items():

        params = {**COMMON_PARAMS, **config}

        print("\nTraining:", config_name)

        model = XGBRegressor(**params)

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

        results.append({
            "target": target,
            "configuration": config_name,
            "feature_count": len(features),
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "persistence_rmse": persistence_rmse,
            "persistence_mae": persistence_mae,
            "rmse_difference_vs_persistence":
                val_rmse - persistence_rmse
        })

        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Validation RMSE: {val_rmse:.4f}")
        print(f"Validation MAE: {val_mae:.4f}")

# =====================================================
# 4. SAVE RESULTS
# =====================================================

results_df = pd.DataFrame(results)

output_path = os.path.join(
    REPORT_DIR,
    "xgboost_tuning_results.csv"
)

results_df.to_csv(output_path, index=False)

# =====================================================
# 5. BEST CONFIGURATION
# =====================================================

print("\n" + "=" * 65)
print("HYPERPARAMETER TUNING SUMMARY")
print("=" * 65)

for target in TARGETS:

    subset = results_df[
        results_df["target"] == target
    ].sort_values("val_rmse")

    print("\n", target.upper())
    print(
        subset[
            [
                "configuration",
                "train_rmse",
                "val_rmse",
                "val_mae"
            ]
        ].to_string(index=False)
    )

    best = subset.iloc[0]

    print("\nBest configuration:", best["configuration"])
    print(f"Best validation RMSE: {best['val_rmse']:.4f}")

print("\nSaved:", output_path)
print("\nTest dataset was NOT used.")
print("Existing models were NOT overwritten.")
