
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

train = pd.read_csv(DATA_DIR / "train_prepared.csv")
val = pd.read_csv(DATA_DIR / "validation_prepared.csv")

excluded = [
    "period", "hour", "dst",
    "dst_t0", "dst_t1", "split"
]

features = [c for c in train.columns if c not in excluded]

for target in ["dst_t0", "dst_t1"]:

    model = xgb.XGBRegressor()
    model.load_model(MODEL_DIR / f"xgboost_{target}.json")

    print("\n" + "=" * 70)
    print(f"PERIOD ANALYSIS: {target}")
    print("=" * 70)

    for period in sorted(val["period"].unique()):

        train_p = train[train["period"] == period]
        val_p = val[val["period"] == period]

        y_train = train_p[target]
        y_val = val_p[target]

        predictions = model.predict(val_p[features])

        # Persistence baseline: latest historical Dst
        baseline = val_p["dst"]

        print(f"\n--- {period} ---")

        print("Training target:")
        print(
            f"Mean={y_train.mean():.2f}, "
            f"Std={y_train.std():.2f}, "
            f"Min={y_train.min()}, "
            f"Max={y_train.max()}"
        )

        print("Validation target:")
        print(
            f"Mean={y_val.mean():.2f}, "
            f"Std={y_val.std():.2f}, "
            f"Min={y_val.min()}, "
            f"Max={y_val.max()}"
        )

        print(
            "Validation storm records (Dst <= -50):",
            int((y_val <= -50).sum())
        )

        print(
            "Validation intense/severe records (Dst <= -100):",
            int((y_val <= -100).sum())
        )

        xgb_rmse = np.sqrt(mean_squared_error(y_val, predictions))
        xgb_mae = mean_absolute_error(y_val, predictions)

        base_rmse = np.sqrt(mean_squared_error(y_val, baseline))
        base_mae = mean_absolute_error(y_val, baseline)

        print("\nXGBoost:")
        print(f"RMSE={xgb_rmse:.4f}, MAE={xgb_mae:.4f}")

        print("Persistence:")
        print(f"RMSE={base_rmse:.4f}, MAE={base_mae:.4f}")

print("\nPERIOD SHIFT ANALYSIS COMPLETED!")
