
import pandas as pd
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer

# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "processed" / "forecasting_temporal_splits.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# 2. LOAD DATA
# ==================================================

print("Loading temporal dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

# ==================================================
# 3. DEFINE FEATURES AND TARGETS
# ==================================================

TARGET_COLUMNS = ["dst_t0", "dst_t1"]

NON_FEATURE_COLUMNS = [
    "period",
    "hour",
    "dst",
    "dst_t0",
    "dst_t1",
    "split"
]

FEATURE_COLUMNS = [
    col for col in df.columns
    if col not in NON_FEATURE_COLUMNS
]

print("Number of ML features:", len(FEATURE_COLUMNS))

assert len(FEATURE_COLUMNS) == 93

# ==================================================
# 4. SEPARATE DATASETS
# ==================================================

train_df = df[df["split"] == "train"].copy()

val_df = df[df["split"] == "validation"].copy()

test_df = df[df["split"] == "test"].copy()

assert len(train_df) > 0
assert len(val_df) > 0
assert len(test_df) > 0

print("\nBefore imputation:")
print("Training:", train_df.shape)
print("Validation:", val_df.shape)
print("Testing:", test_df.shape)

# ==================================================
# 5. CHECK INFINITE VALUES
# ==================================================

print("\nChecking infinite values...")

for name, data in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:
    infinite_count = (
        data[FEATURE_COLUMNS]
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )

    print(f"{name} infinite values:", infinite_count)

    assert infinite_count == 0, (
        f"Infinite values found in {name}"
    )

# ==================================================
# 6. MEDIAN IMPUTATION
# Fit ONLY on training data
# ==================================================

print("\nFitting median imputer on training data...")

imputer = SimpleImputer(strategy="median")

train_df.loc[:, FEATURE_COLUMNS] = (
    imputer.fit_transform(
        train_df[FEATURE_COLUMNS]
    )
)

val_df.loc[:, FEATURE_COLUMNS] = (
    imputer.transform(
        val_df[FEATURE_COLUMNS]
    )
)

test_df.loc[:, FEATURE_COLUMNS] = (
    imputer.transform(
        test_df[FEATURE_COLUMNS]
    )
)

# ==================================================
# 7. SAVE PREPARED DATASETS
# ==================================================

train_path = OUTPUT_DIR / "train_temporal_prepared.csv"

val_path = OUTPUT_DIR / "validation_temporal_prepared.csv"

test_path = OUTPUT_DIR / "test_temporal_prepared.csv"

train_df.to_csv(train_path, index=False)

val_df.to_csv(val_path, index=False)

test_df.to_csv(test_path, index=False)

# Save separate imputer
imputer_path = MODEL_DIR / "temporal_median_imputer.pkl"

joblib.dump(imputer, imputer_path)

# ==================================================
# 8. FINAL VALIDATION
# ==================================================

print("\n" + "=" * 60)
print("TEMPORAL PREPROCESSING REPORT")
print("=" * 60)

for name, data in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:

    missing = (
        data[FEATURE_COLUMNS]
        .isnull()
        .sum()
        .sum()
    )

    print(f"\n{name} shape:", data.shape)

    print(f"{name} remaining missing features:", missing)

    assert missing == 0

    assert data[TARGET_COLUMNS].notnull().all().all()

# Verify targets were not included in features
assert not set(TARGET_COLUMNS).intersection(FEATURE_COLUMNS)

# Verify original Dst is excluded from solar-wind-only features
assert "dst" not in FEATURE_COLUMNS

print("\nFeature count:", len(FEATURE_COLUMNS))

print("Targets preserved:", TARGET_COLUMNS)

print("\nSaved files:")
print(train_path)
print(val_path)
print(test_path)
print(imputer_path)

print("\nTEMPORAL PREPROCESSING COMPLETED SUCCESSFULLY!")
