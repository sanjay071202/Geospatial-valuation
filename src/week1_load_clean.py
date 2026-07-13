"""
Week 1 — Load, clean, and outlier-filter the housing dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/kc_house_data.csv")
CLEAN_DATA_PATH = Path("data/processed/kc_house_clean.csv")

LAT_COL = "lat"
LONG_COL = "long"
PRICE_COL = "price"

LAT_BOUNDS = (47.1, 47.8)
LONG_BOUNDS = (-122.6, -121.3)


def load_data(path: Path) -> pd.DataFrame:
    print(f"Loading raw data from {path} ...")
    df = pd.read_csv(path)
    print(f"  -> {len(df):,} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()

    for col in (LAT_COL, LONG_COL, PRICE_COL):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[LAT_COL, LONG_COL, PRICE_COL])
    df = df[df[LAT_COL].between(*LAT_BOUNDS) & df[LONG_COL].between(*LONG_BOUNDS)]
    df = df[df[PRICE_COL] > 0]

    print(f"Cleaning: {before:,} -> {len(df):,} rows ({before - len(df):,} removed)")
    return df.reset_index(drop=True)


def remove_price_outliers(df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
    before = len(df)

    if method == "iqr":
        q1, q3 = df[PRICE_COL].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df = df[df[PRICE_COL].between(lower, upper)]
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Outlier filtering ({method}): {before:,} -> {len(df):,} rows ({before - len(df):,} removed)")
    df["price_log"] = np.log1p(df[PRICE_COL])
    return df.reset_index(drop=True)


def main():
    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    df = remove_price_outliers(df, method="iqr")

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Saved cleaned dataset -> {CLEAN_DATA_PATH} ({len(df):,} rows)")


if __name__ == "__main__":
    main()