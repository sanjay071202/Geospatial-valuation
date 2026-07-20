"""
Week 2 — Evaluate the XGBoost baseline: MAPE, RMSE, and error analysis.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

MODEL_PATH = Path("models/xgb_baseline.pkl")
TEST_SET_PATH = Path("data/processed/test_set.csv")
METRICS_PATH = Path("outputs/week2_baseline_metrics.json")

TARGET_COL = "price"

FEATURE_COLS = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_above",
    "sqft_basement", "house_age", "was_renovated", "yrs_since_update",
    "dist_to_center_km",
]


def main():
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_SET_PATH)

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    y_pred = model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Baseline MAPE: {mape:.4f} ({mape * 100:.2f}%)")
    print(f"Baseline RMSE: ${rmse:,.2f}")

    metrics = {
        "model": "XGBoost baseline",
        "mape": round(mape, 4),
        "rmse": round(rmse, 2),
        "n_test_samples": len(y_test),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()