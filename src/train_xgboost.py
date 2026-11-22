
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load prepared datasets
train_df = pd.read_csv(DATA_DIR / "train_prepared.csv")
val_df = pd.read_csv(DATA_DIR / "validation_prepared.csv")

# Define columns
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

# Starting hyperparameters
params = {
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

# Train and evaluate both forecasting models
results = {}

for target in ["dst_t0", "dst_t1"]:

    print(f"\nTraining XGBoost for {target}...")

    y_train = train_df[target]
    y_val = val_df[target]

    model = xgb.XGBRegressor(**params)

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    mae = mean_absolute_error(y_val, predictions)

    results[target] = {
        "RMSE": rmse,
        "MAE": mae
    }

    # Save trained model
    model_path = MODEL_DIR / f"xgboost_{target}.json"
    model.save_model(model_path)

    print(f"{target} Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"Model saved: {model_path}")

print("\n========== VALIDATION SUMMARY ==========")

for target, metrics in results.items():
    print(
        f"{target}: "
        f"RMSE = {metrics['RMSE']:.4f}, "
        f"MAE = {metrics['MAE']:.4f}"
    )

print("\nXGBOOST TRAINING COMPLETED!")
