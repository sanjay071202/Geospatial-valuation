"""
Week 2 — Train an XGBoost baseline regressor on tabular features.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

FEATURED_DATA_PATH = Path("data/processed/kc_house_features.csv")
MODEL_PATH = Path("models/xgb_baseline.pkl")
TEST_SET_PATH = Path("data/processed/test_set.csv")

TARGET_COL = "price"

FEATURE_COLS = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_above",
    "sqft_basement", "house_age", "was_renovated", "yrs_since_update",
    "dist_to_center_km",
]


def main():
    df = pd.read_csv(FEATURED_DATA_PATH)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")

    # Save test set (features + actual price) for the evaluation script
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test
    test_df.to_csv(TEST_SET_PATH, index=False)
    print(f"Saved test set -> {TEST_SET_PATH}")


if __name__ == "__main__":
    main()