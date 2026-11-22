
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

# =====================================================
# 1. CONFIGURATION
# =====================================================

PREDICTION_PATH = "reports/final_test_predictions.csv"

REPORT_DIR = "reports/test_analysis"
PLOT_DIR = os.path.join(REPORT_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

TARGETS = ["dst_t0", "dst_t1"]

# =====================================================
# 2. LOAD TEST PREDICTIONS
# =====================================================

print("\nLoading saved test predictions...")

df = pd.read_csv(PREDICTION_PATH)

print("Dataset shape:", df.shape)
print("Periods:", df["period"].unique())

required_columns = [
    "period", "dst", "dst_t0", "dst_t1",
    "dst_t0_xgb_prediction",
    "dst_t1_xgb_prediction",
    "dst_t0_persistence_prediction",
    "dst_t1_persistence_prediction"
]

assert all(col in df.columns for col in required_columns)

# =====================================================
# 3. METRIC FUNCTION
# =====================================================

def calculate_metrics(actual, predicted):

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mae = mean_absolute_error(
        actual, predicted
    )

    return rmse, mae


# =====================================================
# 4. PER-PERIOD PERFORMANCE
# =====================================================

period_results = []

print("\n" + "=" * 65)
print("PERIOD-WISE PERFORMANCE")
print("=" * 65)

for period in sorted(df["period"].unique()):

    period_df = df[df["period"] == period]

    print("\nPERIOD:", period)
    print("Samples:", len(period_df))

    for target in TARGETS:

        actual = period_df[target].to_numpy()

        xgb_pred = period_df[
            f"{target}_xgb_prediction"
        ].to_numpy()

        persistence_pred = period_df[
            f"{target}_persistence_prediction"
        ].to_numpy()

        xgb_rmse, xgb_mae = calculate_metrics(
            actual, xgb_pred
        )

        p_rmse, p_mae = calculate_metrics(
            actual, persistence_pred
        )

        print(
            f"{target.upper()} | "
            f"XGB RMSE: {xgb_rmse:.4f} | "
            f"Persistence RMSE: {p_rmse:.4f}"
        )

        period_results.append({
            "period": period,
            "target": target,
            "samples": len(period_df),
            "xgb_rmse": xgb_rmse,
            "xgb_mae": xgb_mae,
            "persistence_rmse": p_rmse,
            "persistence_mae": p_mae
        })

period_results_df = pd.DataFrame(period_results)

period_results_df.to_csv(
    os.path.join(REPORT_DIR, "period_wise_results.csv"),
    index=False
)

# =====================================================
# 5. ERROR DISTRIBUTION
# =====================================================

for target in TARGETS:

    actual = df[target].to_numpy()

    xgb_pred = df[
        f"{target}_xgb_prediction"
    ].to_numpy()

    persistence_pred = df[
        f"{target}_persistence_prediction"
    ].to_numpy()

    xgb_error = xgb_pred - actual
    persistence_error = persistence_pred - actual

    plt.figure(figsize=(10, 6))

    plt.hist(
        xgb_error,
        bins=60,
        alpha=0.6,
        label="XGBoost"
    )

    plt.hist(
        persistence_error,
        bins=60,
        alpha=0.6,
        label="Persistence"
    )

    plt.axvline(0, linestyle="--")

    plt.title(f"Prediction Error Distribution - {target.upper()}")
    plt.xlabel("Prediction Error (Predicted - Actual), nT")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            f"{target}_error_distribution.png"
        ),
        dpi=300
    )

    plt.close()

# =====================================================
# 6. ACTUAL VS PREDICTED
# =====================================================

for target in TARGETS:

    actual = df[target].to_numpy()

    xgb_pred = df[
        f"{target}_xgb_prediction"
    ].to_numpy()

    persistence_pred = df[
        f"{target}_persistence_prediction"
    ].to_numpy()

    lower = min(actual.min(), xgb_pred.min(), persistence_pred.min())
    upper = max(actual.max(), xgb_pred.max(), persistence_pred.max())

    plt.figure(figsize=(8, 8))

    plt.scatter(
        actual,
        xgb_pred,
        alpha=0.25,
        s=10,
        label="XGBoost"
    )

    plt.scatter(
        actual,
        persistence_pred,
        alpha=0.25,
        s=10,
        label="Persistence"
    )

    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        label="Perfect Prediction"
    )

    plt.title(f"Actual vs Predicted - {target.upper()}")
    plt.xlabel("Actual Dst (nT)")
    plt.ylabel("Predicted Dst (nT)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            f"{target}_actual_vs_predicted.png"
        ),
        dpi=300
    )

    plt.close()

# =====================================================
# 7. STORM-PERIOD ANALYSIS
# =====================================================

storm_results = []

for target in TARGETS:

    actual = df[target].to_numpy()

    xgb_pred = df[
        f"{target}_xgb_prediction"
    ].to_numpy()

    persistence_pred = df[
        f"{target}_persistence_prediction"
    ].to_numpy()

    storm_mask = actual <= -50

    xgb_rmse, xgb_mae = calculate_metrics(
        actual[storm_mask],
        xgb_pred[storm_mask]
    )

    p_rmse, p_mae = calculate_metrics(
        actual[storm_mask],
        persistence_pred[storm_mask]
    )

    storm_results.append({
        "target": target,
        "samples": int(storm_mask.sum()),
        "xgb_rmse": xgb_rmse,
        "xgb_mae": xgb_mae,
        "persistence_rmse": p_rmse,
        "persistence_mae": p_mae
    })

storm_results_df = pd.DataFrame(storm_results)

storm_results_df.to_csv(
    os.path.join(REPORT_DIR, "storm_analysis.csv"),
    index=False
)

# =====================================================
# 8. STORM ONSET ANALYSIS
# =====================================================

onset_results = []

for target in TARGETS:

    actual = df[target].to_numpy()

    xgb_pred = df[
        f"{target}_xgb_prediction"
    ].to_numpy()

    persistence_pred = df[
        f"{target}_persistence_prediction"
    ].to_numpy()

    onset_mask = (
        (df["dst"].to_numpy() > -50) &
        (actual <= -50)
    )

    xgb_rmse, xgb_mae = calculate_metrics(
        actual[onset_mask],
        xgb_pred[onset_mask]
    )

    p_rmse, p_mae = calculate_metrics(
        actual[onset_mask],
        persistence_pred[onset_mask]
    )

    onset_results.append({
        "target": target,
        "samples": int(onset_mask.sum()),
        "xgb_rmse": xgb_rmse,
        "xgb_mae": xgb_mae,
        "persistence_rmse": p_rmse,
        "persistence_mae": p_mae
    })

onset_results_df = pd.DataFrame(onset_results)

onset_results_df.to_csv(
    os.path.join(REPORT_DIR, "storm_onset_analysis.csv"),
    index=False
)

# =====================================================
# 9. RMSE COMPARISON CHART
# =====================================================

overall = []

for target in TARGETS:

    actual = df[target].to_numpy()

    xgb_pred = df[
        f"{target}_xgb_prediction"
    ].to_numpy()

    persistence_pred = df[
        f"{target}_persistence_prediction"
    ].to_numpy()

    xgb_rmse, _ = calculate_metrics(actual, xgb_pred)
    p_rmse, _ = calculate_metrics(actual, persistence_pred)

    overall.append({
        "target": target.upper(),
        "XGBoost": xgb_rmse,
        "Persistence": p_rmse
    })

overall_df = pd.DataFrame(overall)

overall_df.to_csv(
    os.path.join(REPORT_DIR, "overall_comparison.csv"),
    index=False
)

x = np.arange(len(overall_df))
width = 0.35

plt.figure(figsize=(9, 6))

plt.bar(
    x - width / 2,
    overall_df["XGBoost"],
    width,
    label="XGBoost"
)

plt.bar(
    x + width / 2,
    overall_df["Persistence"],
    width,
    label="Persistence"
)

plt.xticks(x, overall_df["target"])
plt.ylabel("RMSE (nT)")
plt.title("Final Test RMSE Comparison")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(PLOT_DIR, "overall_rmse_comparison.png"),
    dpi=300
)

plt.close()

# =====================================================
# 10. SAMPLE TIME-SERIES VISUALIZATION
# =====================================================

# Plot the first 500 test rows from each period.
for period in sorted(df["period"].unique()):

    period_df = df[
        df["period"] == period
    ].head(500)

    for target in TARGETS:

        plt.figure(figsize=(14, 6))

        plt.plot(
            period_df[target].to_numpy(),
            label="Actual Dst",
            linewidth=1.2
        )

        plt.plot(
            period_df[
                f"{target}_xgb_prediction"
            ].to_numpy(),
            label="XGBoost",
            alpha=0.8
        )

        plt.plot(
            period_df[
                f"{target}_persistence_prediction"
            ].to_numpy(),
            label="Persistence",
            alpha=0.8
        )

        plt.axhline(-50, linestyle="--", label="Storm Threshold")

        plt.title(
            f"{period} - {target.upper()} "
            f"(First 500 Test Samples)"
        )

        plt.xlabel("Test Sample Index")
        plt.ylabel("Dst (nT)")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                PLOT_DIR,
                f"{period}_{target}_timeseries.png"
            ),
            dpi=300
        )

        plt.close()

# =====================================================
# 11. FINAL SUMMARY
# =====================================================

print("\n" + "=" * 65)
print("TEST ERROR ANALYSIS COMPLETED")
print("=" * 65)

print("\nPERIOD-WISE RESULTS")
print(period_results_df.to_string(index=False))

print("\nSTORM RESULTS")
print(storm_results_df.to_string(index=False))

print("\nSTORM ONSET RESULTS")
print(onset_results_df.to_string(index=False))

print("\nOVERALL COMPARISON")
print(overall_df.to_string(index=False))

print("\nReports saved in:", REPORT_DIR)
print("Plots saved in:", PLOT_DIR)

print("\nNo model was retrained.")
print("No hyperparameters were changed.")
