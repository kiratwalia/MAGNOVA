
import pandas as pd
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "processed" / "forecasting_splits.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(INPUT_FILE)

# Define columns
TARGET_COLUMNS = ["dst_t0", "dst_t1"]
NON_FEATURE_COLUMNS = ["period", "hour", "dst", "dst_t0", "dst_t1", "split"]

FEATURE_COLUMNS = [
    col for col in df.columns
    if col not in NON_FEATURE_COLUMNS
]

print("Original dataset shape:", df.shape)
print("Number of features:", len(FEATURE_COLUMNS))

# Separate train, validation and test
train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

# Check that all partitions exist
assert len(train_df) > 0
assert len(val_df) > 0
assert len(test_df) > 0

# Fit imputer ONLY on training data
imputer = SimpleImputer(strategy="median")

train_df[FEATURE_COLUMNS] = imputer.fit_transform(
    train_df[FEATURE_COLUMNS]
)

# Transform validation and test using training medians
val_df[FEATURE_COLUMNS] = imputer.transform(
    val_df[FEATURE_COLUMNS]
)

test_df[FEATURE_COLUMNS] = imputer.transform(
    test_df[FEATURE_COLUMNS]
)

# Save prepared datasets
train_df.to_csv(OUTPUT_DIR / "train_prepared.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "validation_prepared.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test_prepared.csv", index=False)

# Save fitted imputer for future inference
joblib.dump(imputer, MODEL_DIR / "median_imputer.pkl")

# Final checks
for name, data in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:
    missing = data[FEATURE_COLUMNS].isnull().sum().sum()
    print(f"{name} shape: {data.shape}")
    print(f"{name} remaining feature missing values: {missing}")
    assert missing == 0

# Verify targets are unchanged
print("\nTarget columns preserved:")
print(train_df[TARGET_COLUMNS].isnull().sum())

print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")
