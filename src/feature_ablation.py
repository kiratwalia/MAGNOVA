
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

REPORT_DIR = "reports"
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

print("\nLoading training and validation datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

with open(
    "models/temporal_xgboost_metadata.json", "r"
) as f:
    metadata = json.load(f)

all_features = metadata["features"]

# =====================================================
# 3. FEATURE CLASSIFICATION
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

for feature in all_features:

    category = get_feature_category(feature)

    groups.setdefault(category, []).append(feature)

print("\nFEATURE GROUP COUNTS")

for category, features in groups.items():
    print(f"{category}: {len(features)}")

# =====================================================
# 4. DEFINE ABLATION EXPERIMENTS
# =====================================================

experiments = {
    "A_Original": [
        "Original"
    ],

    "B_Original_Lag": [
        "Original", "Lag"
    ],

    "C_Original_Rolling": [
        "Original", "Rolling"
    ],

    "D_Original_Lag_Rolling": [
        "Original", "Lag", "Rolling"
    ],

    "E_All_93": [
        "Original", "Lag", "Rolling",
        "Physics", "Missingness"
    ]
}

experiment_features = {}

for experiment, categories in experiments.items():

    selected = []

    for category in categories:
        selected.extend(groups[category])

    experiment_features[experiment] = selected

    print(
        f"{experiment}: {len(selected)} features"
    )

# =====================================================
# 5. TRAIN AND EVALUATE
# =====================================================

results = []

for target in TARGETS:

    print("\n" + "=" * 70)
    print("TARGET:", target.upper())
    print("=" * 70)

    y_train = train_df[target].to_numpy()
    y_val = val_df[target].to_numpy()

    # Persistence baseline
    persistence_pred = val_df["dst"].to_numpy()

    persistence_rmse = np.sqrt(
        mean_squared_error(y_val, persistence_pred)
    )

    persistence_mae = mean_absolute_error(
        y_val, persistence_pred
    )

    print(
        f"\nPersistence RMSE: {persistence_rmse:.4f}"
        f" | MAE: {persistence_mae:.4f}"
    )

    for experiment, features in experiment_features.items():

        print("\n" + "-" * 55)
        print(
            f"TRAINING {experiment} | "
            f"{len(features)} FEATURES"
        )

        X_train = train_df[features]
        X_val = val_df[features]

        assert list(X_train.columns) == list(X_val.columns)

        model = XGBRegressor(**PARAMS)

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
            "experiment": experiment,
            "feature_count": len(features),
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "persistence_rmse": persistence_rmse,
            "persistence_mae": persistence_mae,
            "rmse_difference_vs_persistence":
                val_rmse - persistence_rmse
        })

        print(
            f"Train RMSE: {train_rmse:.4f}"
        )

        print(
            f"Validation RMSE: {val_rmse:.4f}"
        )

        print(
            f"Validation MAE: {val_mae:.4f}"
        )

# =====================================================
# 6. SAVE RESULTS
# =====================================================

results_df = pd.DataFrame(results)

results_path = os.path.join(
    REPORT_DIR,
    "feature_ablation_results.csv"
)

results_df.to_csv(results_path, index=False)

# =====================================================
# 7. FINAL COMPARISON
# =====================================================

print("\n" + "=" * 70)
print("FEATURE ABLATION EXPERIMENT COMPLETED")
print("=" * 70)

for target in TARGETS:

    print(f"\nRESULTS FOR {target.upper()}")

    target_results = results_df[
        results_df["target"] == target
    ]

    print(
        target_results[
            [
                "experiment",
                "feature_count",
                "val_rmse",
                "val_mae",
                "persistence_rmse"
            ]
        ].to_string(index=False)
    )

print("\nSaved:", results_path)

print("\nTest data was NOT used.")
print("Existing models were NOT overwritten.")
