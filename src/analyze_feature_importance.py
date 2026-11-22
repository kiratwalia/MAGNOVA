
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

# =====================================================
# 1. PATHS
# =====================================================

MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

TARGETS = ["dst_t0", "dst_t1"]

# =====================================================
# 2. LOAD FEATURE METADATA
# =====================================================

metadata_path = os.path.join(
    MODEL_DIR,
    "temporal_xgboost_metadata.json"
)

with open(metadata_path, "r") as f:
    metadata = json.load(f)

feature_columns = metadata["features"]

print("\nTotal features:", len(feature_columns))

# =====================================================
# 3. FEATURE CATEGORY CLASSIFICATION
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
# 4. LOAD MODELS AND EXTRACT IMPORTANCE
# =====================================================

all_importance = []

for target in TARGETS:

    print("\n" + "=" * 55)
    print("FEATURE IMPORTANCE:", target.upper())
    print("=" * 55)

    model_path = os.path.join(
        MODEL_DIR,
        f"temporal_xgboost_{target}.json"
    )

    model = XGBRegressor()
    model.load_model(model_path)

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": importance
    })

    importance_df["target"] = target

    importance_df["category"] = (
        importance_df["feature"].apply(get_feature_category)
    )

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    all_importance.append(importance_df)

    print("\nTOP 20 FEATURES:")
    print(
        importance_df.head(20).to_string(index=False)
    )

    # Save individual CSV
    importance_df.to_csv(
        f"{REPORT_DIR}/{target}_feature_importance.csv",
        index=False
    )

    # Plot top 20
    top20 = importance_df.head(20).sort_values(
        "importance",
        ascending=True
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        top20["feature"],
        top20["importance"]
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Top 20 Feature Importance - {target.upper()}")

    plt.tight_layout()

    plt.savefig(
        f"{REPORT_DIR}/{target}_feature_importance.png",
        dpi=300
    )

    plt.close()

# =====================================================
# 5. CATEGORY-WISE IMPORTANCE
# =====================================================

combined = pd.concat(all_importance, ignore_index=True)

category_importance = (
    combined.groupby(["target", "category"])["importance"]
    .sum()
    .reset_index()
)

category_importance.to_csv(
    f"{REPORT_DIR}/category_feature_importance.csv",
    index=False
)

print("\nCATEGORY-WISE IMPORTANCE:")
print(
    category_importance.pivot(
        index="category",
        columns="target",
        values="importance"
    ).fillna(0).to_string()
)

# =====================================================
# 6. SAVE COMBINED RESULTS
# =====================================================

combined.to_csv(
    f"{REPORT_DIR}/all_feature_importance.csv",
    index=False
)

print("\nANALYSIS COMPLETED")

print("\nSaved files:")
print("- reports/dst_t0_feature_importance.csv")
print("- reports/dst_t1_feature_importance.csv")
print("- reports/category_feature_importance.csv")
print("- reports/all_feature_importance.csv")
print("- Feature importance PNG plots")

print("\nNo model retraining performed.")
print("No test data used.")
