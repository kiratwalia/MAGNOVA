
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

train_df = pd.read_csv(DATA_DIR / "train_prepared.csv")
val_df = pd.read_csv(DATA_DIR / "validation_prepared.csv")

targets = ["dst_t0", "dst_t1"]

for target in targets:

    print("\n" + "=" * 55)
    print(f"DIAGNOSTICS FOR {target}")
    print("=" * 55)

    model = xgb.XGBRegressor()
    model.load_model(MODEL_DIR / f"xgboost_{target}.json")

    excluded = [
        "period", "hour", "dst",
        "dst_t0", "dst_t1", "split"
    ]

    features = [
        col for col in train_df.columns
        if col not in excluded
    ]

    X_train = train_df[features]
    X_val = val_df[features]

    y_train = train_df[target]
    y_val = val_df[target]

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    print("\nTARGET DISTRIBUTION")
    print("Train mean:", round(y_train.mean(), 3))
    print("Validation mean:", round(y_val.mean(), 3))
    print("Train std:", round(y_train.std(), 3))
    print("Validation std:", round(y_val.std(), 3))

    print("\nPREDICTION DISTRIBUTION")
    print("Train prediction mean:", round(train_pred.mean(), 3))
    print("Validation prediction mean:", round(val_pred.mean(), 3))
    print("Validation prediction std:", round(val_pred.std(), 3))

    print("\nMODEL PERFORMANCE")
    print(
        "Train RMSE:",
        round(np.sqrt(mean_squared_error(y_train, train_pred)), 4)
    )
    print(
        "Validation RMSE:",
        round(np.sqrt(mean_squared_error(y_val, val_pred)), 4)
    )
    print(
        "Train MAE:",
        round(mean_absolute_error(y_train, train_pred), 4)
    )
    print(
        "Validation MAE:",
        round(mean_absolute_error(y_val, val_pred), 4)
    )

    print("\nVALIDATION PERFORMANCE BY PERIOD")

    for period in val_df["period"].unique():

        mask = val_df["period"] == period

        period_rmse = np.sqrt(
            mean_squared_error(y_val[mask], val_pred[mask])
        )

        period_mae = mean_absolute_error(
            y_val[mask], val_pred[mask]
        )

        print(
            f"{period}: RMSE={period_rmse:.4f}, "
            f"MAE={period_mae:.4f}"
        )

print("\nDIAGNOSTICS COMPLETED!")
