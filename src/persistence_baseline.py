
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "validation_prepared.csv"

df = pd.read_csv(DATA_FILE)

# Historical Dst is used only for the persistence baseline.
# It is NOT an ML feature.

y_true_t0 = df["dst_t0"]
y_true_t1 = df["dst_t1"]

# Persistence predictions
y_pred_t0 = df["dst"]
y_pred_t1 = df["dst"]

# Evaluation function
def evaluate(y_true, y_pred, horizon):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"\n{horizon}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")

    return rmse, mae

print("PERSISTENCE BASELINE - VALIDATION RESULTS")

rmse_t0, mae_t0 = evaluate(
    y_true_t0, y_pred_t0, "Dst T+0"
)

rmse_t1, mae_t1 = evaluate(
    y_true_t1, y_pred_t1, "Dst T+1"
)

print("\nBASELINE EVALUATION COMPLETED!")
