
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
MODEL_FILE = ROOT / "models" / "temporal_median_imputer.pkl"

FILES = {
    "Train": DATA_DIR / "train_temporal_prepared.csv",
    "Validation": DATA_DIR / "validation_temporal_prepared.csv",
    "Test": DATA_DIR / "test_temporal_prepared.csv"
}

TARGETS = ["dst_t0", "dst_t1"]

EXCLUDED = [
    "period", "hour", "dst",
    "dst_t0", "dst_t1", "split"
]

print("=" * 60)
print("TEMPORAL PREPROCESSING VERIFICATION")
print("=" * 60)

for name, path in FILES.items():

    df = pd.read_csv(path)

    features = [
        col for col in df.columns
        if col not in EXCLUDED
    ]

    missing = df[features].isnull().sum().sum()

    infinite = (
        df[features]
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )

    targets_valid = df[TARGETS].notnull().all().all()

    print(f"\n{name}")
    print("Shape:", df.shape)
    print("Feature count:", len(features))
    print("Missing feature values:", missing)
    print("Infinite feature values:", infinite)
    print("Targets valid:", targets_valid)

    assert len(features) == 93
    assert missing == 0
    assert infinite == 0
    assert targets_valid

imputer = joblib.load(MODEL_FILE)

print("\nImputer loaded:", type(imputer).__name__)
print("Imputer statistics:", len(imputer.statistics_))

assert len(imputer.statistics_) == 93

print("\nALL PREPROCESSING CHECKS PASSED!")
